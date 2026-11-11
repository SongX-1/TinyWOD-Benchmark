import csv
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=_json, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=_json).encode()).hexdigest()


def write_csv(path, rows, fields):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields)); writer.writeheader()
        writer.writerows(rows); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


class CSV:
    """UTF-8 BOM；严格列名、唯一键、每次写完 flush/fsync/close。"""
    def __init__(self, path, fields, keys):
        self.path = Path(path); self.fields = list(fields); self.keys = tuple(keys)
        self.seen = set(); self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file():
            with self.path.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames != self.fields:
                    raise ValueError(f"CSV 列结构不匹配：{self.path}")
                for row in reader:
                    key = tuple(str(row[k]) for k in self.keys)
                    if key in self.seen:
                        raise ValueError(f"CSV 已有重复键：{self.path} {key}")
                    self.seen.add(key)
        else:
            write_csv(self.path, [], self.fields)

    def append(self, row):
        self.extend([row])

    def extend(self, rows):
        """A complete batch is flushed once, including ALL individual loss terms."""
        rows = list(rows); keys = set()
        for row in rows:
            if set(row) - set(self.fields):
                raise ValueError(f"CSV 未声明的列：{set(row) - set(self.fields)}")
            key = tuple(str(row[k]) for k in self.keys)
            if key in self.seen or key in keys:
                raise ValueError(f"拒绝重复写入：{self.path} {key}")
            keys.add(key)
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerows(rows)
            f.flush(); os.fsync(f.fileno())
        self.seen.update(keys)

    def update(self, row):
        """Atomically replace one existing keyed row, without appending a duplicate."""
        key=tuple(str(row[k]) for k in self.keys)
        if key not in self.seen or set(row)-set(self.fields):
            raise ValueError(f"CSV 更新键/列不匹配：{self.path} {key}")
        with self.path.open(encoding="utf-8-sig",newline="") as file:
            rows=list(csv.DictReader(file))
        rows=[row if tuple(str(old[k]) for k in self.keys)==key else old for old in rows]
        write_csv(self.path,rows,self.fields)


class Tee:
    def __init__(self, console, log):
        self.console = console; self.log = log
    def write(self, value):
        self.console.write(value)
        if not self.log.closed: self.log.write(value); self.log.flush()
        return len(value)
    def flush(self):
        self.console.flush()
        if not self.log.closed: self.log.flush()
    def isatty(self):
        return self.console.isatty()
    def fileno(self):
        return self.console.fileno()
    @property
    def encoding(self):
        return getattr(self.console, "encoding", "utf-8")


def resource_snapshot(device):
    row = {"cpu_rss_gb": None, "peak_allocated_gb": None,
           "allocated_gb": None, "reserved_gb": None, "resource_note": ""}
    try:
        import psutil
        row["cpu_rss_gb"] = psutil.Process().memory_info().rss / 2**30
    except Exception as e:
        row["resource_note"] = f"CPU memory unavailable: {type(e).__name__}: {e}"
    import torch
    if str(device).startswith("cuda"):
        row.update(peak_allocated_gb=torch.cuda.max_memory_allocated(device) / 2**30,
                   allocated_gb=torch.cuda.memory_allocated(device) / 2**30,
                   reserved_gb=torch.cuda.memory_reserved(device) / 2**30)
    return row
