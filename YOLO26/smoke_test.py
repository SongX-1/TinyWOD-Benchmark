"""PyCharm 可直接运行：合成数据、CPU、128px、各 1 Epoch；绝不是正式实验。"""
import json
import random
import time
import uuid
from dataclasses import replace
from pathlib import Path

from config import CLASS_NAMES, CONFIG, MODEL_ORDER, PROJECT_ROOT


def make_synthetic_dataset(root, labeled_test=False):
    from PIL import Image, ImageDraw
    root = Path(root)
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    next_id = 1
    for split, count in (("train", 4), ("val", 2), ("test", 2)):
        payload = {"images": [], "annotations": [],
                   "categories": [{"id": i+1, "name": name} for i, name in enumerate(CLASS_NAMES)]}
        folder = root / "images" / split
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            width, height = 160+index*16, 96+index*8
            image = Image.new("RGB", (width, height), tuple(rng.randrange(20, 90) for _ in range(3)))
            draw = ImageDraw.Draw(image)
            filename = f"{split}_{index:03d}.jpg"
            payload["images"].append({"id": next_id, "file_name": filename, "width": width, "height": height})
            for cls in range(5):
                x, y, w, h = 8+cls*27, 12+cls*8, 16, 18
                draw.rectangle((x, y, x+w, y+h), fill=tuple(rng.randrange(100, 250) for _ in range(3)))
                payload["annotations"].append({"id": len(payload["annotations"])+1, "image_id": next_id,
                    "category_id": cls+1, "bbox": [x, y, w, h], "area": w*h, "iscrowd": 0})
            image.save(folder / filename)
            next_id += 1
        if split != "test" or labeled_test:
            (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def main():
    from tinywod.runner import launch
    root = PROJECT_ROOT / "verification" / ("smoke_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    raw = make_synthetic_dataset(root / "raw")
    cfg = replace(CONFIG, dataset_mode="coco", raw_dataset_root=str(raw), prepared_dataset_root=str(root / "converted"),
                  output_root=str(root / "runs"), epochs=1, imgsz=128, batch=2, workers=0, device="cpu", amp=False,
                  initialization="scratch", resume_checkpoints={"s": "", "x": ""}, startup_delay_hours={"s": 0, "x": 0},
                  benchmark_images=2, benchmark_repeats=1, benchmark_warmup=1, save_period=-1)
    print("注意：这是合成数据功能检查，结果禁止用作 TinyWOD 论文指标。", flush=True)
    code = launch(MODEL_ORDER, cfg=cfg, formal=False)
    print(f"冒烟测试输出: {root}；退出码={code}", flush=True)
    return code


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    raise SystemExit(main())
