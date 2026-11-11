"""Atomic outputs, complete console capture and append-only keyed CSV records."""
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import threading
import uuid


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def link_or_copy(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def atomic_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class CSVTable:
    def __init__(self, path, fields, key=()):
        self.path = Path(path)
        self.fields = tuple(fields)
        self.key = tuple(key)
        self.seen = set()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with self.path.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if tuple(reader.fieldnames or ()) != self.fields:
                    raise ValueError(f"CSV 表头不兼容：{self.path}")
                for row in reader:
                    if self.key:
                        identity = tuple(row[k] for k in self.key)
                        if identity in self.seen:
                            raise ValueError(f"CSV 已存在重复记录：{self.path} {identity}")
                        self.seen.add(identity)
        else:
            with self.path.open("x", encoding="utf-8-sig", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fields).writeheader()

    def append(self, row):
        unexpected = set(row) - set(self.fields)
        if unexpected:
            raise ValueError(f"CSV 字段未声明 {unexpected}：{self.path}")
        identity = tuple(str(row.get(k, "")) for k in self.key)
        if self.key and identity in self.seen:
            raise ValueError(f"拒绝重复写入 CSV：{self.path} {identity}")
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"CSV 非有限数值 {key}={value}：{self.path}")
        # 每行关闭文件，立即刷新至操作系统；逐轮关键结果另做 fsync。
        with self.path.open("a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(row)
            f.flush()
        if self.key:
            self.seen.add(identity)

    def sync(self):
        with self.path.open("a", encoding="utf-8", newline="") as f:
            f.flush()
            os.fsync(f.fileno())


class Tee:
    def __init__(self, stream, log, lock=None):
        self.stream, self.log = stream, log
        self.lock = lock or threading.RLock()

    def write(self, value):
        with self.lock:
            self.stream.write(value)
            self.log.write(value)
            self.log.flush()
        return len(value)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def isatty(self):
        return bool(getattr(self.stream, "isatty", lambda: False)())

    @property
    def encoding(self):
        return getattr(self.stream, "encoding", "utf-8")


def atomic_torch_save(value, path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("wb") as f:
            torch.save(value, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
