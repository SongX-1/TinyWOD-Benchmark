# -*- coding: utf-8 -*-
"""OWT逐轮记录约定 + 每个batch落盘 + 不裁剪DINO原始损失字段。"""
import csv
import datetime
import json
import math
import os
from pathlib import Path
import sys

OWT_COLUMNS = [
    "epoch", "train/giou_loss", "train/cls_loss", "train/l1_loss",
    "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)",
    "metrics/mAP50-95(B)", "metrics/mAP50-95-small(B)",
    "complexity/Parameters", "complexity/GFLOPs-dense-upper",
    "speed/val-latency-ms", "speed/val-FPS", "memory/PeakAllocated-GB",
    "sparsity/active-token-ratio", "sparsity/compact-token-ratio",
    "val/giou_loss", "val/cls_loss", "val/l1_loss", "lr/pg0", "lr/pg1", "lr/pg2",
]


def scalar(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    value = scalar(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def append_csv(path, row, required=()):
    """保留已有字段；有新增字段时扩展表头并补齐旧行，绝不静默丢列。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: scalar(v) for k, v in row.items()}
    existing = []
    if path.is_file() and path.stat().st_size:
        with path.open(encoding="utf-8-sig", newline="") as f:
            existing = next(csv.reader(f))
    fields = existing + [k for k in list(required) + list(row) if k not in existing]
    fields = list(dict.fromkeys(fields))
    if existing and fields != existing:
        tmp = path.with_suffix(path.suffix + ".schema.tmp")
        with path.open(encoding="utf-8-sig", newline="") as src, tmp.open("w", encoding="utf-8-sig", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=fields)
            writer.writeheader()
            for old in csv.DictReader(src):
                writer.writerow(old)
        tmp.replace(path)
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        if not existing:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


class Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, text):
        self.original.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.original.flush()
        self.log.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return getattr(self.original, "encoding", "utf-8")

    def fileno(self):
        return self.original.fileno()


class ConsoleLog:
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a", encoding="utf-8", buffering=1)
        self.out, self.err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(self.out, self.file), Tee(self.err, self.file)
        return self

    def __exit__(self, *exc):
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = self.out, self.err
        self.file.close()


def reserve_run(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        path = root / f"{n}run"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            n += 1


def trim_for_resume(run_dir, completed_epoch):
    """last checkpoint之外的中断半轮日志先备份，再回退CSV，防止恢复后重复行。"""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for path in Path(run_dir).rglob("*.csv"):
        if "resume_backups" in path.parts:
            continue
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            if not fields or "epoch" not in fields:
                continue
            rows = list(reader)
        keep = [r for r in rows if not r.get("epoch") or int(float(r["epoch"])) <= completed_epoch]
        if len(keep) == len(rows):
            continue
        import shutil
        backup = Path(run_dir) / "resume_backups" / stamp / path.relative_to(run_dir)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
        tmp = path.with_suffix(".resume.tmp")
        with tmp.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(keep)
        tmp.replace(path)


class Recorder:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)

    def csv(self, name, row, required=()):
        append_csv(self.run_dir / name, row, required)

    def print_train_header(self, epoch, epochs):
        print(f"\n[train] Epoch {epoch}/{epochs}", flush=True)
        print(f"{'Epoch':>9} {'Iter':>11} {'GPU_mem':>9} {'giou_loss':>11} {'cls_loss':>11} "
              f"{'l1_loss':>11} {'loss(total)':>12} {'Instances':>10} {'Size':>9} {'LR':>11} {'ETA':>9}", flush=True)

    def record_train_batch(self, epoch, batch, batches, epochs, samples, targets, total_loss,
                           loss_dict, weights, optimizer, meters, data_s, batch_s, device):
        import torch
        lr = {f"lr/pg{i}": g["lr"] for i, g in enumerate(optimizer.param_groups)}
        mem = torch.cuda.memory_reserved(device)/1e9 if device.type == "cuda" else 0.
        instances = sum(len(t["labels"]) for t in targets)
        h, w = samples.tensors.shape[-2:]
        base = {"giou_loss": "loss_giou", "cls_loss": "loss_ce", "l1_loss": "loss_bbox"}
        losses = {f"train/{alias}": scalar(loss_dict[key])*weights[key] for alias,key in base.items()}
        row = {"epoch": epoch, "batch": batch, "batches": batches,
               "global_step": (epoch-1)*batches+batch, "images": len(targets), "instances": instances,
               "input_height": h, "input_width": w, "gpu_reserved_GB": mem,
               "data_seconds": data_s, "compute_seconds": batch_s,
               "iteration_seconds": data_s+batch_s, "train/loss": total_loss, **losses, **lr}
        row.update({f"unscaled/{k}": scalar(v) for k,v in loss_dict.items()})
        row.update({f"weighted/{k}": scalar(v)*weights[k] for k,v in loss_dict.items() if k in weights})
        self.csv("metrics/train_iterations.csv", row)
        avg = {alias: meters.meters[key].global_avg for alias,key in base.items()}
        eta = str(datetime.timedelta(seconds=int((data_s+batch_s)*(batches-batch))))
        # 窗口只在同一行原地刷新（回车覆盖），最后一个batch换行；CSV仍每batch落盘。
        print(f"\r{epoch:>4}/{epochs:<4} {batch:>5}/{batches:<5} {mem:>8.3f}G "
              f"{avg['giou_loss']:>11.5g} {avg['cls_loss']:>11.5g} {avg['l1_loss']:>11.5g} "
              f"{total_loss:>12.5g} {instances:>10} {h:>4}x{w:<4} {lr['lr/pg0']:>11.5g} {eta:>9}",
              end="\n" if batch == batches else "", flush=True)

    def record_epoch(self, epoch, train, val_loss, metrics, optimizer, seconds, elapsed, best_epoch):
        row = {"epoch": epoch, **metrics, "time/epoch-seconds": seconds,
               "time/elapsed-seconds": elapsed, "best_epoch": best_epoch,
               "train/loss": train.get("loss", float("nan")), "val/loss": val_loss.get("loss", float("nan"))}
        for prefix, stats in [("train",train), ("val",val_loss)]:
            for alias,key in {"giou_loss":"loss_giou", "cls_loss":"loss_ce", "l1_loss":"loss_bbox"}.items():
                row[f"{prefix}/{alias}"] = stats.get(key, float("nan"))
            row.update({f"{prefix}/dino/{k}": v for k,v in stats.items()})
        row.update({f"lr/pg{i}": group["lr"] for i,group in enumerate(optimizer.param_groups)})
        # DINO默认AdamW只有两个组；保留OWT pg2列，NaN明确表示不适用，绝不伪造第三组。
        for key in OWT_COLUMNS:
            row.setdefault(key, float("nan"))
        self.csv("results.csv", row, OWT_COLUMNS)
        self.csv("metrics/epochs.csv", row, OWT_COLUMNS)
        with (self.run_dir / "log.txt").open("a", encoding="utf-8") as f:
            f.write(json.dumps(safe_json(row), ensure_ascii=False, allow_nan=False)+"\n")
        print(f"[epoch] {epoch}: P={metrics['metrics/precision(B)']:.5f} "
              f"R={metrics['metrics/recall(B)']:.5f} mAP50={metrics['metrics/mAP50(B)']:.5f} "
              f"mAP50-95={metrics['metrics/mAP50-95(B)']:.5f} "
              f"APsmall={metrics['metrics/mAP50-95-small(B)']:.5f} "
              f"F1={metrics['metrics/F1(B)']:.5f} 用时={seconds:.1f}s 最佳轮={best_epoch}", flush=True)

    def save_schema(self):
        schema = {}
        for path in self.run_dir.rglob("*.csv"):
            if "resume_backups" in path.parts:
                continue
            with path.open(encoding="utf-8-sig", newline="") as f:
                schema[path.relative_to(self.run_dir).as_posix()] = next(csv.reader(f), [])
        from .data import atomic_json
        atomic_json(self.run_dir / "csv_columns.json", schema)


def plot_training(run_dir):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = Path(run_dir)/"results.csv"
    if not path.is_file():
        return
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    x = [int(r["epoch"]) for r in rows]
    panels = [("train/giou_loss","val/giou_loss"), ("train/cls_loss","val/cls_loss"),
              ("train/l1_loss","val/l1_loss"), ("metrics/precision(B)","metrics/recall(B)"),
              ("metrics/mAP50(B)","metrics/mAP50-95(B)","metrics/mAP50-95-small(B)"),
              ("speed/val-FPS",)]
    fig, axes = plt.subplots(2,3,figsize=(16,8),tight_layout=True)
    for ax, fields in zip(axes.flat, panels):
        for field in fields:
            ax.plot(x, [float(r.get(field) or np.nan) for r in rows], label=field)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    fig.savefig(Path(run_dir)/"results.png",dpi=150)
    plt.close(fig)
