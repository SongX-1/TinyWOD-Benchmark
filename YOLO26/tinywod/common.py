"""UTF-8 记录、原子写入及可复现指纹。仅使用标准库。"""
import csv
import dataclasses
import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def clean(value):
    if dataclasses.is_dataclass(value):
        return clean(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    return value


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_name(p.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, p)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_csv(path, rows, fieldnames=None):
    """Excel 可识别的 UTF-8 BOM；空值表示未计算，不伪造为 0。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = [clean(r) for r in rows]
    fields = fieldnames or list(dict.fromkeys(k for r in rows for k in r))
    temporary = p.with_name(p.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, p)


def append_jsonl(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(clean(value), ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(clean(value), ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def link_or_copy(src, dst):
    import shutil
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def f1_from_pr(p, r):
    return 2 * float(p) * float(r) / (float(p) + float(r)) if p + r else 0.0


def experiment_signature(cfg):
    """恢复必须匹配同一实验设置；不把存放目录或模型编号当作训练超参数。"""
    from dataclasses import asdict
    ignore = {"output_root", "weights_dir", "resume_checkpoints", "startup_delay_hours", "continue_on_failure",
              "existing_yolo_root", "raw_dataset_root", "prepared_dataset_root", "dataset_mode", "download_pretrained"}
    return digest({k: v for k, v in asdict(cfg).items() if k not in ignore})
