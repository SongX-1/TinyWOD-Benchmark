"""Readonly input, original split, experiment-specific native COCO view."""
import math
from pathlib import Path, PurePosixPath
from PIL import Image
from config import CLASS_NAMES, resolve
from .common import digest, read_json, write_json, sha256, write_csv

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)


def safe_relative(value):
    p = PurePosixPath(str(value).replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or not p.parts or any(":" in x for x in p.parts):
        raise ValueError(f"非法图像相对路径：{value}")
    return Path(*p.parts)


def read_names(root):
    for p in [root/"manifest.json", *root.glob("*.yaml"), *root.glob("*.yml")]:
        if not p.is_file(): continue
        if p.suffix == ".json": obj = read_json(p)
        else:
            import yaml
            obj = yaml.safe_load(p.read_text(encoding="utf-8-sig"))
        if not isinstance(obj, dict): continue
        names = obj.get("names", obj.get("class_names", obj.get("classes")))
        if names is None: continue
        if isinstance(names, dict):
            if sorted(map(int, names)) != list(range(len(CLASS_NAMES))): raise ValueError(f"非法类别编号：{p}")
            names = [names.get(i, names.get(str(i))) for i in range(len(CLASS_NAMES))]
        if list(names) != list(CLASS_NAMES): raise ValueError(f"类别顺序不匹配：{p}，names={names}")
        return str(p)
    return None


def coco_layout(cfg, root, split):
    """优先显式配置；自动候选有歧义时停止，不猜测划分。"""
    def choose(explicit, candidates, required):
        if explicit:
            path = Path(explicit)
            path = path if path.is_absolute() else root/path
            if not path.exists(): raise FileNotFoundError(path)
            return path
        found = [p for p in candidates if p.exists()]
        if len(found) > 1: raise ValueError(f"COCO 路径有歧义，请在 config.py 指定 {split}：{found}")
        if not found and required: raise FileNotFoundError(f"未发现 {split}：{candidates}")
        return found[0] if found else candidates[0]
    images = choose(cfg.coco_image_dirs.get(split),
                    [root/'images'/split, root/split], split != 'test')
    annotation = choose(cfg.coco_annotations.get(split),
        [root/'annotations'/f'instances_{split}.json', root/'annotations'/f'{split}.json',
         root/f'{split}.json', root/split/'annotations.json'], split != 'test')
    return images, annotation


def prepare_dataset(cfg, inputs):
    from .duplicate_annotations import DuplicateAnnotationAudit, DUPLICATE_POLICY_VERSION
    inputs = Path(inputs); view = inputs/"dataset_view"; view.mkdir(parents=True, exist_ok=False)
    duplicate_audit = DuplicateAnnotationAudit(inputs, CLASS_NAMES)
    yolo, coco = resolve(cfg.yolo_root), resolve(cfg.coco_root)
    mode = ("yolo" if yolo.is_dir() else "coco") if cfg.dataset_mode == "auto" else cfg.dataset_mode
    source = yolo if mode == "yolo" else coco
    if not source.is_dir():
        raise FileNotFoundError(f"缺少TinyWOD数据：{source}。请在config.py核对yolo_root/coco_root。")
    proof = read_names(source) if mode == "yolo" else None
    if mode == "yolo" and not proof and not cfg.accept_unverified_yolo_names:
        raise ValueError("YOLO目录缺少声明类别顺序的manifest/YAML。核实编号后才可显式设置accept_unverified_yolo_names。")
    if mode not in ('yolo', 'coco'): raise ValueError(f'未知 dataset_mode={mode}')
    if cfg.test_policy not in ('auto', 'unlabeled', 'required'): raise ValueError('未知 test_policy')
    if mode == 'yolo' and not proof:
        print(f'[类别声明] 源目录未提供类别 YAML；按 config.py 的 CLASS_NAMES={CLASS_NAMES} 读取，已记录该限制。', flush=True)
    splits, records, source_hashes, physical, content = {}, [], {}, {}, {}
    next_id = 1
    for split in ("train", "val", "test"):
        image_root = source/"images"/split
        ann = source/"annotations"/f"instances_{split}.json"
        if mode == "coco":
            image_root, ann = coco_layout(cfg, source, split)
            if split != "test" and not ann.is_file(): raise FileNotFoundError(ann)
            payload = read_json(ann) if ann.is_file() else None
            # 只有 images 的官方 test 清单不等于有 GT；保留清单的精确图像子集。
            labeled = payload is not None and not (split == "test" and
                (cfg.test_policy == "unlabeled" or not payload.get('annotations')))
            if payload is not None: source_hashes[str(ann)] = sha256(ann)
            if payload is not None:
                categories = {}
                for cat in payload.get("categories", []) if labeled else []:
                    cid = int(cat["id"]); name = str(cat["name"]).casefold().replace(" ", "_")
                    if cid in categories: raise ValueError(f"重复类别编号：{ann}")
                    if name in CLASS_NAMES: categories[cid] = CLASS_NAMES.index(name)
                    elif cid == 0 and name in ("ignored", "ignore", "background"): categories[cid] = None
                    else: raise ValueError(f"未知类别：{cat}")
                if labeled and set(v for v in categories.values() if v is not None) != set(range(len(CLASS_NAMES))):
                    raise ValueError(f"必须声明全部类别：{ann}")
                entries = payload.get("images", []); grouped = {int(e["id"]): [] for e in entries}
                if len(grouped) != len(entries): raise ValueError(f"重复image_id：{ann}")
                ids = set()
                for a in payload.get("annotations", []) if labeled else []:
                    if int(a["image_id"]) not in grouped: raise ValueError("标注引用未知图像")
                    if "id" in a and a["id"] in ids: raise ValueError("重复annotation_id")
                    if "id" in a: ids.add(a["id"])
                    grouped[int(a["image_id"])].append(a)
            else: entries = [{"file_name": p.relative_to(image_root).as_posix()} for p in image_files(image_root)]
        else:
            entries = [{"file_name": p.relative_to(image_root).as_posix()} for p in image_files(image_root)]
            paths = [source/"labels"/split/Path(e["file_name"]).with_suffix(".txt") for e in entries]
            count = sum(p.is_file() for p in paths)
            if split != "test" and count != len(entries): raise FileNotFoundError(f"{split}缺标签：{count}/{len(entries)}；背景图须有空txt")
            if split == "test" and count not in (0, len(entries)): raise ValueError("test仅部分有标签，不能将缺标签视为背景")
            labeled = split != "test" or (count > 0 and cfg.test_policy != "unlabeled")
        if not entries:
            if split != "test": raise ValueError(f"{split}为空：{image_root}")
            labeled = False
        if split == "test" and cfg.test_policy == "required" and (not entries or not labeled):
            raise ValueError("test_policy=required需要完整测试图像和GT")
        out = {"info": {"description": "TinyWOD experiment view"}, "images": [], "annotations": [],
               "categories": [{"id": i, "name": n} for i,n in enumerate(CLASS_NAMES)]}
        counts, seen, clipped, skipped, crowds = [0]*len(CLASS_NAMES), set(), 0, 0, 0
        for e in entries:
            rel = safe_relative(e["file_name"]); key = rel.as_posix().casefold()
            if key in seen: raise ValueError(f"重复图像名：{split}/{rel}")
            seen.add(key); path = image_root/rel
            if not path.is_file(): raise FileNotFoundError(path)
            st = path.stat(); identity = (st.st_dev, st.st_ino) if st.st_ino else str(path.resolve())
            if identity in physical and physical[identity] != split: raise ValueError(f"跨划分同一物理图像：{path}")
            physical[identity] = split
            with Image.open(path) as im:
                width, height = im.size
                if cfg.verify_images: im.verify()
            if mode == "coco" and labeled and (int(e["width"]), int(e["height"])) != (width, height):
                raise ValueError(f"COCO与图像尺寸不符：{path}")
            record = {"split": split, "relative": rel.as_posix(), "source": str(path.resolve()),
                      "width": width, "height": height, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
            if cfg.hash_images:
                value = sha256(path); record["sha256"] = value
                if value in content and content[value] != split: raise ValueError(f"跨划分图像字节相同：{path}")
                content[value] = split
            image_id = next_id; next_id += 1
            out["images"].append({"id": image_id, "file_name": rel.as_posix(), "width": width, "height": height})
            boxes = []
            if labeled and mode == "yolo":
                txt = source/"labels"/split/rel.with_suffix(".txt"); record["label_sha256"] = sha256(txt)
                for line_no, line in enumerate(txt.read_text(encoding="utf-8-sig").splitlines(),1):
                    parts = line.split()
                    if not parts: continue
                    if len(parts) != 5: raise ValueError(f"YOLO行格式错误：{txt}:{line_no}")
                    c,cx,cy,w,h = map(float,parts)
                    if not all(math.isfinite(v) for v in (c,cx,cy,w,h)) or c != int(c) or not 0 <= c < len(CLASS_NAMES):
                        raise ValueError(f"非法类别/数值：{txt}:{line_no}")
                    if w <= 0 or h <= 0 or min(cx-w/2,cy-h/2) < -1e-6 or max(cx+w/2,cy+h/2) > 1+1e-6:
                        raise ValueError(f"非法归一化框：{txt}:{line_no}")
                    boxes.append({"mapped": int(c), "_source_line": line_no, "bbox": [(cx-w/2)*width,(cy-h/2)*height,w*width,h*height]})
            elif labeled:
                for a in grouped[int(e["id"])]:
                    cid = int(a["category_id"])
                    if cid not in categories: raise ValueError(f"未声明类别：{cid}")
                    if categories[cid] is None: skipped += 1; continue
                    boxes.append({**a, "mapped": categories[cid]})
            unique = {}
            for source_box_index, a in enumerate(boxes, 1):
                x,y,w,h = map(float,a["bbox"])
                if not all(math.isfinite(v) for v in (x,y,w,h)) or min(w,h) <= 0: raise ValueError(f"无效框：{path}")
                x1,y1,x2,y2 = max(0.,x),max(0.,y),min(float(width),x+w),min(float(height),y+h)
                if x2 <= x1 or y2 <= y1: skipped += 1; continue
                changed = (x1,y1,x2,y2) != (x,y,x+w,y+h); clipped += int(changed)
                c = a["mapped"]
                area = float(a.get("area",(x2-x1)*(y2-y1))) if not changed else (x2-x1)*(y2-y1)
                if not math.isfinite(area) or area <= 0: raise ValueError("无效area")
                crowd = int(bool(a.get("iscrowd",0)))
                if not duplicate_audit.keep(unique, split=split, image_rel=rel.as_posix(),
                    source_image=path, source_labels=txt if mode == "yolo" else ann,
                    class_id=c, xyxy=(x1,y1,x2,y2), area=area, iscrowd=crowd,
                    source_box_index=source_box_index, source_annotation=a,
                    next_annotation_id=len(out["annotations"])+1):
                    continue
                crowds += crowd
                out["annotations"].append({"id": len(out["annotations"])+1, "image_id": image_id, "category_id": c,
                                           "bbox": [x1,y1,x2-x1,y2-y1], "area": area, "iscrowd": crowd})
                if not crowd: counts[c] += 1
            record["duplicate_boxes_removed"] = duplicate_audit.image_count(split, rel.as_posix())
            records.append(record)  # 只生成标注视图，直接读取源图片，不复制整个数据集。
        if labeled and sum(counts) == 0: raise ValueError(f"{split}没有非crowd有效GT，AP不可定义")
        target = view/"annotations"/f"instances_{split}.json"
        write_json(target,out)  # 未标注 test 也保存精确图片清单，不作为 GT 评价。
        splits[split] = {"images": len(entries), "instances": len(out["annotations"]), "class_counts": counts,
                         "labeled": labeled, "crowd": crowds, "clipped_boxes": clipped, "skipped_boxes": skipped,
                         "duplicate_boxes_removed": duplicate_audit.split_count(split),
                         "images_path": str(image_root.resolve()), "annotation": str(target.resolve()),
                         "source_annotation": str(ann) if mode == 'coco' else None}
        print(f"[数据] {split}: Images={len(entries)} Instances={len(out['annotations'])} GT={labeled} classes={counts} crowd={crowds} duplicate_removed={duplicate_audit.split_count(split)}",flush=True)
    duplicate_summary = duplicate_audit.finish()
    fingerprint = digest({"records": records, "names": CLASS_NAMES, "annotation_hashes": source_hashes,
                          "annotation_processing": DUPLICATE_POLICY_VERSION})
    info = {"mode": mode, "source": str(source), "fingerprint": fingerprint, "names": list(CLASS_NAMES),
            "class_proof": proof, "splits": splits, "final_split": "test" if splits["test"]["labeled"] else "val",
            "crowd_policy": "native training/OWT metrics exclude crowd targets; native COCOeval retains crowd ignore"}
    info["annotation_processing"] = DUPLICATE_POLICY_VERSION
    info["duplicate_annotations"] = duplicate_summary
    write_json(inputs/"dataset_info.json",info)
    write_json(inputs/"dataset_manifest.json", {"fingerprint": fingerprint,"files": records,"annotation_hashes": source_hashes,
                         "annotation_processing": DUPLICATE_POLICY_VERSION, "duplicate_annotations": duplicate_summary})
    write_csv(inputs/'dataset_statistics.csv', [dict(split=k, **{x: v[x] for x in
        ('images','instances','labeled','crowd','clipped_boxes','skipped_boxes','duplicate_boxes_removed')},
        **{f'class_{i}': n for i,n in enumerate(v['class_counts'])}) for k,v in splits.items()],
        ['split','images','instances','labeled','crowd','clipped_boxes','skipped_boxes','duplicate_boxes_removed',
         *[f'class_{i}' for i in range(len(CLASS_NAMES))]])
    return info

# NOTE: DUPLICATE_POLICY_VERSION value intentionally kept as 'sds_exact_duplicate_keep_first_v1'
# (hashed into dataset fingerprints; must not change or old checkpoints fail fingerprint match).
