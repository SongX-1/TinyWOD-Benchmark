# -*- coding: utf-8 -*-
"""沿用 OWT 的原始数据来源、划分、类别映射；DINO 直接读取 COCO JSON。"""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

from experiment_config import CLASS_NAMES, CATEGORY_MAP


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
    tmp.replace(path)


def _resolve_image(split_dir, raw_name):
    name = PurePosixPath(str(raw_name).replace("\\", "/"))
    if name.is_absolute() or ".." in name.parts:
        raise ValueError(f"图片 file_name 不能越出数据集目录: {raw_name}")
    root = split_dir.resolve()
    candidates = [split_dir / str(name), split_dir.parent / str(name),
                  split_dir.parent.parent / str(name), split_dir / name.name]
    for candidate in candidates:
        p = candidate.resolve()
        if p.is_relative_to(root) and p.is_file():
            return p, p.relative_to(root).as_posix()
    raise FileNotFoundError(f"找不到标注引用的图片: {split_dir} / {raw_name}")


def prepare_dataset(config):
    root = Path(config.dataset_root).expanduser().resolve()
    cache = Path(config.cache_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"TinyWOD 数据源不存在: {root}\n"
                                "请仅修改 experiment_config.py 的 dataset_root；"
                                "目录应含 images/train、images/val 和 coco/instances_train.json。")
    if cache.is_relative_to(root):
        raise ValueError("cache_root 必须在数据源之外，保持 OWT 原始数据只读。")
    manifest = {"version": "dino-sds-owt-aligned-v1", "dataset_root": str(root),  # 缓存版本保持旧值
                "image_source": "original compressed images; no background filtering",
                "category_map": {str(k): v for k, v in CATEGORY_MAP.items()},
                "names": CLASS_NAMES, "splits": {}}
    for split in ("train", "val"):
        source = root / "coco" / f"instances_{split}.json"
        image_dir = root / "images" / split
        if not source.is_file() or not image_dir.is_dir():
            raise FileNotFoundError(f"缺少 {source} 或 {image_dir}，请核对 experiment_config.py。")
        raw = source.read_bytes()
        data = json.loads(raw.decode("utf-8-sig"))
        categories = {int(x["id"]): x.get("name", "") for x in data.get("categories", [])}
        if not set(CATEGORY_MAP).issubset(categories):
            raise ValueError(f"{source} 的 categories 必须含 TinyWOD 类别 1(target)；实际 {categories}")
        images, image_ids, file_names = [], set(), set()
        for im in data.get("images", []):
            im = dict(im)
            iid = int(im["id"])
            if iid in image_ids:
                raise ValueError(f"{source} 图片ID重复: {iid}")
            full, relative = _resolve_image(image_dir, im["file_name"])
            if relative in file_names:
                raise ValueError(f"{source} 多个ID引用同一图片: {relative}")
            w, h = int(im["width"]), int(im["height"])
            if w <= 0 or h <= 0:
                raise ValueError(f"图片尺寸不合法: {full}: {w}x{h}")
            if config.validate_image_sizes:
                from PIL import Image
                with Image.open(full) as pic:
                    if pic.size != (w, h):
                        raise ValueError(f"标注/图片尺寸不一致: {full}: JSON={(w,h)}, 实图={pic.size}")
            im.update(id=iid, width=w, height=h, file_name=relative)
            images.append(im)
            image_ids.add(iid)
            file_names.add(relative)
        if not images:
            raise ValueError(f"{source} 没有图片，不启动空数据训练。")
        sizes = {im["id"]: (im["width"], im["height"]) for im in images}
        annotations, ann_ids = [], set()
        unknown = clipped = invalid = crowd = 0
        for a in data.get("annotations", []):
            a = dict(a)
            aid, iid, cat = int(a["id"]), int(a["image_id"]), int(a["category_id"])
            if aid in ann_ids:
                raise ValueError(f"{source} 标注ID重复: {aid}")
            ann_ids.add(aid)
            if iid not in image_ids:
                raise ValueError(f"标注 {aid} 引用了不存在的 image_id={iid}")
            if cat not in CATEGORY_MAP:
                unknown += 1
                continue
            x, y, w, h = map(float, a["bbox"])
            if not all(math.isfinite(v) for v in (x, y, w, h)):
                raise ValueError(f"标注 {aid} 的 bbox 有 NaN/Inf")
            iw, ih = sizes[iid]
            x1, y1 = max(0., x), max(0., y)
            x2, y2 = min(float(iw), x+w), min(float(ih), y+h)
            if x2 <= x1 or y2 <= y1 or w <= 0 or h <= 0:
                invalid += 1
                continue
            if (x1, y1, x2-x1, y2-y1) != (x, y, w, h):
                clipped += 1
            iscrowd = int(a.get("iscrowd", 0))
            # OWT 的 YOLO 标签不表达 crowd：统一转成普通框以保持参考口径，并记录数量。
            crowd += bool(iscrowd)
            a.update(id=aid, image_id=iid, category_id=CATEGORY_MAP[cat],
                     bbox=[x1, y1, x2-x1, y2-y1], area=(x2-x1)*(y2-y1), iscrowd=0)
            a.pop("segmentation", None)
            annotations.append(a)
        if split == "train" and not annotations:
            raise ValueError("训练集没有有效目标框，不能进行有监督目标检测训练。")
        counts = {str(k): sum(a["category_id"] == k for a in annotations) for k in range(len(CLASS_NAMES))}
        cached = cache / "annotations" / f"instances_{split}.json"
        payload = {"info": data.get("info", {}), "licenses": data.get("licenses", []),
                   "images": images, "annotations": annotations,
                   "categories": [{"id": i, "name": n} for i, n in enumerate(CLASS_NAMES)]}
        atomic_json(cached, payload)
        stats = {"images": len(images), "boxes": len(annotations), "per_class": counts,
                 "source_json": str(source), "source_sha256": hashlib.sha256(raw).hexdigest(),
                 "cache_json": str(cached), "image_dir": str(image_dir),
                 "skipped_unknown_cat": unknown, "skipped_invalid_box": invalid,
                 "clipped_boxes": clipped, "crowd_as_regular": crowd}
        manifest["splits"][split] = stats
        print(f"[prepare_tinywod] {split}: 原图 {len(images)} 张 / 标签框 {len(annotations)}；"
              f"忽略类别={unknown} 无效框={invalid} 截断框={clipped} crowd转普通框={crowd}", flush=True)
    test_dir = root / "images" / "test"
    test_images = sorted(p for p in test_dir.rglob("*") if p.is_file() and
                         p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}) if test_dir.is_dir() else []
    manifest["splits"]["test"] = {"images": len(test_images), "image_dir": str(test_dir),
                                     "has_ground_truth": False, "role": "inference only"}
    atomic_json(cache / "manifest.json", manifest)
    print(f"[prepare_tinywod] test: {len(test_images)} 张，无标签，仅推理；数据源文件未修改。", flush=True)
    return manifest


class LetterBox:
    """与 OWT 的640方形letterbox相同：保持比例，不拉伸，不过滤背景。"""
    def __init__(self, size):
        self.size = int(size)

    def __call__(self, image, target):
        import torch
        from PIL import Image, ImageOps
        width, height = image.size
        r = min(self.size / height, self.size / width)
        nw, nh = round(width*r), round(height*r)
        left, top = round((self.size-nw)/2-0.1), round((self.size-nh)/2-0.1)
        image = image.resize((nw, nh), Image.Resampling.BILINEAR)
        image = ImageOps.expand(image, (left, top, self.size-nw-left, self.size-nh-top), fill=(114,114,114))
        target = dict(target)
        target["boxes"] = target["boxes"] * r + target["boxes"].new_tensor([left,top,left,top])
        target["area"] = target["area"] * r*r
        target["size"] = torch.as_tensor([self.size,self.size])
        return image, target


def make_transforms(split, size):
    import datasets.transforms as T
    steps = [T.RandomHorizontalFlip()] if split == "train" else []
    return T.Compose(steps + [LetterBox(size), T.ToTensor(),
                             T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])


class OriginalSpacePostProcess:
    """把DINO输出从letterbox坐标映射回原图；验证、APsmall、test共用同一逻辑。"""
    def __init__(self, postprocessor, size):
        self.base = postprocessor
        self.size = size

    def __call__(self, outputs, target_sizes):
        padded = target_sizes.new_full(target_sizes.shape, self.size)
        result = self.base(outputs, padded)
        for item, (height, width) in zip(result, target_sizes.tolist()):
            r = min(self.size/height, self.size/width)
            left = round((self.size-round(width*r))/2-0.1)
            top = round((self.size-round(height*r))/2-0.1)
            b = item["boxes"].clone()
            b[:,[0,2]] = (b[:,[0,2]]-left)/r
            b[:,[1,3]] = (b[:,[1,3]]-top)/r
            # OWT 的指标使用映射后的框，不提前裁剪，保持同样的 IoU 和面积定义。
            item["boxes"] = b
        return result
