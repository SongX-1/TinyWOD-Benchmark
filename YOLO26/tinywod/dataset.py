"""复用上传配置的 OWT YOLO 数据，或按原 split 转换 COCO；不重新划分。"""
import json
import math
import os
import shutil
import uuid
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from config import CLASS_NAMES, resolved_path
from .common import digest, link_or_copy, read_json, sha256, write_json

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# 归一化框边界容忍的浮点舍入：源标签以 6 位小数写入，cx±w/2 最多溢出约 5e-7；
# 留足余量裁剪回 [0,1]，同时仍能拒绝真实越界（如 cx=1.5）。
BOUND_TOLERANCE = 1e-4


def relative_name(value):
    p = PurePosixPath(str(value).replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or ":" in str(p):
        raise ValueError(f"不安全的图像相对路径: {value}")
    return Path(*p.parts)


def image_files(folder):
    return sorted((p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
                  key=lambda p: p.relative_to(folder).as_posix()) if Path(folder).is_dir() else []


def ordered_names(value):
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return value
    if isinstance(value, dict) and value and all(str(k).isdigit() for k in value):
        keys = sorted(value, key=lambda k: int(k))
        if [int(k) for k in keys] == list(range(len(keys))):
            return [value[k] for k in keys]
    return None


def confirm_names(root, cfg):
    """不能仅从 cls 数字推断语义。支持 manifest 和已有 data YAML 的类别声明。"""
    import yaml
    proofs = []
    for name in ("manifest.json", "dataset.yaml", "data.yaml", "tinywod.yaml"):
        p = root / name
        if not p.is_file():
            continue
        obj = read_json(p) if p.suffix == ".json" else yaml.safe_load(p.read_text(encoding="utf-8-sig"))
        if not isinstance(obj, dict):
            continue
        for key in ("names", "class_names", "classes"):
            found = ordered_names(obj.get(key))
            if found is not None:
                if found != list(CLASS_NAMES):
                    raise ValueError(f"{p} 的 {key} 类别顺序与 config.CLASS_NAMES 不一致: {found}")
                proofs.append({"file": str(p), "sha256": sha256(p), "key": key})
        # 原 OWT 可能把类别列表放在配置快照内。
        for key in ("config", "dataset"):
            if isinstance(obj.get(key), dict):
                found = ordered_names(obj[key].get("names", obj[key].get("class_names")))
                if found is not None:
                    if found != list(CLASS_NAMES):
                        raise ValueError(f"{p}/{key} 类别顺序错误: {found}")
                    proofs.append({"file": str(p), "sha256": sha256(p), "key": key})
    if not proofs and not cfg.accept_unverified_yolo_names:
        raise ValueError("已有 YOLO 数据缺少可验证的类别声明。核对 cls 0..4 后，在唯一配置文件 config.py "
                         "设 accept_unverified_yolo_names=True；不要凭数字猜类别。")
    return proofs or [{"manual_confirmation": True, "names": list(CLASS_NAMES)}]


def convert_coco(root, destination, cfg):
    """只转换标注及建立图像链接；框按边界裁剪，保留 crowd，不混入 OWT 网络。"""
    from PIL import Image
    source_manifest = {"mode": "coco", "root": str(root), "annotation_sha256": {}, "images": []}
    payloads = {}
    for split in ("train", "val", "test"):
        annotation = root / "annotations" / f"instances_{split}.json"
        if split != "test" and not annotation.is_file():
            raise FileNotFoundError(f"缺少官方划分标注: {annotation}")
        if annotation.is_file() and not (split == "test" and cfg.test_policy == "unlabeled"):
            payload = read_json(annotation)
            payloads[split] = payload
            source_manifest["annotation_sha256"][split] = sha256(annotation)
            files = [root / "images" / split / relative_name(i["file_name"]) for i in payload["images"]]
        else:
            files = image_files(root / "images" / split)
        if split != "test" and not files:
            raise ValueError(f"{split} 划分为空")
        for p in files:
            if not p.is_file():
                raise FileNotFoundError(f"COCO 标注引用的图像不存在: {p}")
            s = p.stat()
            row = {"file": p.relative_to(root).as_posix(), "bytes": s.st_size, "mtime_ns": s.st_mtime_ns}
            if cfg.hash_images:
                row["sha256"] = sha256(p)
            source_manifest["images"].append(row)
    signature = digest(source_manifest)
    target = destination / signature[:20]
    marker = target / "manifest.json"
    if marker.is_file() and read_json(marker).get("source_fingerprint") == signature:
        return target, source_manifest
    temporary = destination / (".prepare_" + uuid.uuid4().hex)
    temporary.mkdir(parents=True, exist_ok=False)
    counters = Counter()
    try:
        for split in ("train", "val", "test"):
            payload = payloads.get(split)
            if payload is None:
                for src in image_files(root / "images" / split):
                    link_or_copy(src, temporary / "images" / split / src.relative_to(root / "images" / split))
                continue
            categories = {int(c["id"]): c["name"] for c in payload["categories"]}
            if len(categories) != len(payload["categories"]):
                raise ValueError(f"{split} 存在重复 COCO category_id")
            mapping = {cid: CLASS_NAMES.index(name) for cid, name in categories.items() if name in CLASS_NAMES}
            if set(mapping.values()) != set(range(len(CLASS_NAMES))):
                raise ValueError(f"{split} COCO categories 与类别声明不一致: {categories}")
            unexpected = {cid: name for cid, name in categories.items() if name not in CLASS_NAMES and cid != 0}
            if unexpected:
                raise ValueError(f"发现未配置的非背景类别，拒绝静默丢弃: {unexpected}")
            grouped = defaultdict(list)
            ids = {i["id"] for i in payload["images"]}
            if len(ids) != len(payload["images"]):
                raise ValueError(f"{split} COCO image_id 重复")
            for annotation in payload["annotations"]:
                if annotation["image_id"] not in ids:
                    raise ValueError(f"{split} 标注引用不存在的 image_id")
                grouped[annotation["image_id"]].append(annotation)
            seen = set()
            for item in payload["images"]:
                rel = relative_name(item["file_name"])
                key = rel.with_suffix(".txt").as_posix().casefold()
                if key in seen:
                    raise ValueError(f"{split} 图像会覆盖同名标签: {rel}")
                seen.add(key)
                src = root / "images" / split / rel
                with Image.open(src) as im:
                    width, height = im.size
                if (width, height) != (item["width"], item["height"]):
                    raise ValueError(f"COCO 尺寸与原图不符: {src}")
                labels = []
                for ann in grouped[item["id"]]:
                    cid = int(ann["category_id"])
                    if cid == 0 and cid not in mapping:
                        counters["ignored_category_0"] += 1
                        continue
                    if cid not in mapping:
                        raise ValueError(f"标注引用未知类别 {cid}")
                    x, y, w, h = map(float, ann["bbox"])
                    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
                        raise ValueError(f"非法 COCO bbox: {ann}")
                    x1, y1, x2, y2 = max(0, x), max(0, y), min(width, x+w), min(height, y+h)
                    if x2 <= x1 or y2 <= y1:
                        counters["outside_boxes_removed"] += 1
                        continue
                    counters["clipped_boxes"] += (x1, y1, x2, y2) != (x, y, x+w, y+h)
                    counters["crowd_kept"] += bool(ann.get("iscrowd", 0))
                    labels.append(f"{mapping[cid]} {(x1+x2)/(2*width):.10f} {(y1+y2)/(2*height):.10f} "
                                  f"{(x2-x1)/width:.10f} {(y2-y1)/height:.10f}")
                label_path = temporary / "labels" / split / rel.with_suffix(".txt")
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
                link_or_copy(src, temporary / "images" / split / rel)
        write_json(temporary / "manifest.json", {"names": list(CLASS_NAMES), "source_fingerprint": signature,
                   "conversion": dict(counters), "test_labeled": "test" in payloads})
        write_json(temporary / "source_manifest.json", source_manifest)
        if target.exists():
            raise FileExistsError(f"不完整的旧转换目录，请移走后重试: {target}")
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target, source_manifest


def clip_box_to_unit(cx, cy, w, h, ref):
    """把归一化框裁剪回 [0,1]；超出 BOUND_TOLERANCE 视为真实越界并抛错（带 ref 上下文）。"""
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    if min(x1, y1) < -BOUND_TOLERANCE or max(x2, y2) > 1 + BOUND_TOLERANCE:
        raise ValueError(f"框超出图像边界 {ref}")
    if x1 >= 0 and y1 >= 0 and x2 <= 1 and y2 <= 1:
        return cx, cy, w, h  # 无需裁剪，保持原值避免浮点噪声
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(1.0, x2), min(1.0, y2)
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def inspect_yolo(root, cfg, name_proof):
    from PIL import Image
    rows, splits, physical, hashes, cleaned_labels = [], {}, {}, {}, {}
    for split in ("train", "val", "test"):
        files = image_files(root / "images" / split)
        if split != "test" and not files:
            raise FileNotFoundError(f"没有 {split} 图像: {root / 'images' / split}")
        label_files = sorted((root / "labels" / split).rglob("*.txt")) if (root / "labels" / split).is_dir() else []
        present = [(root / "labels" / split / p.relative_to(root / "images" / split).with_suffix(".txt")).is_file()
                   for p in files]
        if split == "test" and cfg.test_policy == "unlabeled":
            labeled = False
        elif present and any(present) and not all(present):
            raise ValueError(f"{split} 只有部分图像有标签，拒绝把缺失标签当背景")
        else:
            labeled = bool(files and all(present))
        if files and (split != "test" or cfg.test_policy == "required") and not labeled:
            raise ValueError(f"{split} 缺少完整标签，每张背景图也必须有空 txt")
        if split == "test" and cfg.test_policy == "required" and not files:
            raise ValueError("test_policy=required，但没有独立测试图片")
        cls_count, seen_names, used_labels = Counter(), set(), set()
        for p in files:
            rel = p.relative_to(root / "images" / split)
            label = root / "labels" / split / rel.with_suffix(".txt")
            key = rel.with_suffix(".txt").as_posix().casefold()
            if key in seen_names:
                raise ValueError(f"重复文件名/标签覆盖: {split}/{rel}")
            seen_names.add(key)
            stat = p.stat()
            inode = (stat.st_dev, stat.st_ino) if stat.st_ino else str(p.resolve())
            if inode in physical and physical[inode] != split:
                raise ValueError(f"不同数据划分引用同一物理图像: {p}")
            physical[inode] = split
            row = {"split": split, "image": p.relative_to(root).as_posix(), "bytes": stat.st_size,
                   "mtime_ns": stat.st_mtime_ns, "labeled": labeled}
            if cfg.verify_images:
                with Image.open(p) as im:
                    row["width"], row["height"] = im.size
                    if im.mode not in ("RGB", "L"):
                        raise ValueError(f"图像模式须 RGB/L，当前为 {im.mode}: {p}")
                    im.verify()
            if cfg.hash_images:
                h = sha256(p)
                if h in hashes and hashes[h] != split:
                    raise ValueError(f"不同划分图像字节重复: {p}")
                hashes[h] = split
                row["sha256"] = h
            if labeled:
                seen_boxes = set()
                cleaned = []
                for line_no, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
                    if not line.strip():
                        continue
                    values = list(map(float, line.split()))
                    if len(values) != 5 or not all(math.isfinite(v) for v in values):
                        raise ValueError(f"非法检测标签 {label}:{line_no}，应为 cls cx cy w h")
                    cls, cx, cy, w, h = values
                    if cls != int(cls) or not 0 <= cls < len(CLASS_NAMES) or not (0 < w <= 1 and 0 < h <= 1):
                        raise ValueError(f"非法类别/框尺寸 {label}:{line_no}")
                    # 裁剪写入时的浮点舍入（真实越界仍抛错），逐文件去重后写入视图。
                    cx, cy, w, h = clip_box_to_unit(cx, cy, w, h, f"{label}:{line_no}")
                    box = (int(cls), cx, cy, w, h)
                    if box in seen_boxes:
                        continue
                    seen_boxes.add(box)
                    cleaned.append(f"{box[0]} {cx:.10f} {cy:.10f} {w:.10f} {h:.10f}")
                    cls_count[int(cls)] += 1
                used_labels.add(label.resolve())
                row["label_sha256"] = sha256(label)
                cleaned_labels[label.relative_to(root).as_posix()] = "\n".join(cleaned) + ("\n" if cleaned else "")
            rows.append(row)
        if labeled and {p.resolve() for p in label_files} != used_labels:
            raise ValueError(f"{split} 存在无对应图像的多余标签")
        if split in ("train", "val") and not sum(cls_count.values()):
            raise ValueError(f"{split} 整个划分没有目标，无法进行检测实验")
        splits[split] = {"images": len(files), "labeled": labeled, "instances": sum(cls_count.values()),
                         "per_class": {CLASS_NAMES[k]: cls_count[k] for k in range(len(CLASS_NAMES))}}
    # 类别来源文件 mtime 不参与指纹；源数据内容和路径元信息参与。
    manifest = {"names": list(CLASS_NAMES), "splits": splits, "files": rows, "name_proof": name_proof,
                "image_hash_mode": "sha256" if cfg.hash_images else "path+size+mtime_ns"}
    manifest["fingerprint"] = digest({"names": manifest["names"], "files": rows})
    return manifest, cleaned_labels


def prepare_dataset(cfg, inputs_dir):
    import yaml
    if cfg.dataset_mode not in {"auto", "yolo", "coco"} or cfg.test_policy not in {"auto", "unlabeled", "required"}:
        raise ValueError("dataset_mode/test_policy 配置无效")
    inputs_dir = Path(inputs_dir)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    existing, raw = resolved_path(cfg.existing_yolo_root), resolved_path(cfg.raw_dataset_root)
    mode = cfg.dataset_mode
    if mode == "auto":
        mode = "yolo" if existing.exists() else "coco"
    if mode == "yolo":
        if not existing.is_dir():
            raise FileNotFoundError(f"已有 YOLO 数据不存在: {existing}；只需修改 config.py")
        root, source_manifest = existing, {"mode": "yolo", "root": str(existing)}
        proof = confirm_names(root, cfg)
    else:
        if not raw.is_dir():
            raise FileNotFoundError(f"数据路径均不可用。已有 YOLO: {existing}；COCO 原图: {raw}。请在 config.py 配置。")
        root, source_manifest = convert_coco(raw, resolved_path(cfg.prepared_dataset_root), cfg)
        proof = [{"conversion_manifest": str(root / "manifest.json"), "names": list(CLASS_NAMES)}]
    manifest, cleaned_labels = inspect_yolo(root, cfg, proof)
    write_json(inputs_dir / "dataset_manifest.json", manifest)
    write_json(inputs_dir / "source_manifest.json", source_manifest)
    view = inputs_dir / "dataset_view"
    view.mkdir(exist_ok=False)
    for row in manifest["files"]:
        rel = Path(row["image"])
        link_or_copy(root / rel, view / rel)
        if row["labeled"]:
            lab = Path("labels") / row["split"] / rel.relative_to(Path("images") / row["split"]).with_suffix(".txt")
            # 写入裁剪/去重后的标签，源数据保持不变。
            (view / lab).parent.mkdir(parents=True, exist_ok=True)
            (view / lab).write_text(cleaned_labels[lab.as_posix()], encoding="utf-8")
    dataset = {"path": str(view.resolve()), "train": "images/train", "val": "images/val",
               "names": dict(enumerate(CLASS_NAMES)), "nc": len(CLASS_NAMES)}
    if manifest["splits"]["test"]["images"]:
        dataset["test"] = "images/test"
    yaml_path = inputs_dir / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump(dataset, allow_unicode=True, sort_keys=False), encoding="utf-8")
    info = {"yaml": str(yaml_path.resolve()), "view": str(view.resolve()), "source": str(root),
            "fingerprint": manifest["fingerprint"], "splits": manifest["splits"],
            "final_evaluation_split": "test" if manifest["splits"]["test"]["labeled"] else "val",
            "manifest": str((inputs_dir / "dataset_manifest.json").resolve())}
    write_json(inputs_dir / "dataset_info.json", info)
    print(f"[数据] {yaml_path}\n[数据] 划分统计: {json.dumps(info['splits'], ensure_ascii=False)}\n"
          f"[数据] 指纹: {info['fingerprint']}；最终评价: {info['final_evaluation_split']}", flush=True)
    return info
