"""Central CSV schema: no silently dropped native loss or validation fields."""
import csv
from pathlib import Path
import shutil

from .common import CSVTable, atomic_csv, write_json

METRIC_FIELDS = ["precision", "recall", "f1", "confidence", "AP50", "AP", "AP75", "APsmall", "APmedium", "APlarge",
                 "AR1", "AR10", "AR100", "ARsmall", "ARmedium", "ARlarge", "diagnostic_AP50", "diagnostic_AP"]
RESOURCE_FIELDS = ["gpu_allocated_gb", "gpu_reserved_gb", "peak_allocated_gb", "peak_reserved_gb", "process_rss_gb", "system_memory_percent", "resource_note"]


def resources(device):
    import os
    import psutil
    import torch
    row = {key: None for key in RESOURCE_FIELDS}
    row["resource_note"] = ""
    try:
        row["process_rss_gb"] = psutil.Process(os.getpid()).memory_info().rss/1024**3
    except (psutil.Error, OSError) as exc:
        row["resource_note"] += f"process_memory_unavailable:{type(exc).__name__};"
    try:
        row["system_memory_percent"] = psutil.virtual_memory().percent
    except (psutil.Error, OSError) as exc:
        row["resource_note"] += f"system_memory_unavailable:{type(exc).__name__};"
    if device.type == "cuda":
        row.update(gpu_allocated_gb=torch.cuda.memory_allocated(device)/1024**3,
                   gpu_reserved_gb=torch.cuda.memory_reserved(device)/1024**3,
                   peak_allocated_gb=torch.cuda.max_memory_allocated(device)/1024**3,
                   peak_reserved_gb=torch.cuda.max_memory_reserved(device)/1024**3)
    return row


def base_loss_name(key):
    return key.split("_aux_")[0].split("_dn_")[0]


class Recorder:
    def __init__(self, output, keys, groups, weights):
        self.output, self.keys, self.weights = Path(output), list(keys), dict(weights)
        self.lr_fields = [f"lr/pg{i}" for i in range(groups)]
        self.train_fields = [f"train/{k}" for k in keys]+[f"train_raw/{k}" for k in keys]
        self.val_fields = [f"val/{k}" for k in keys]+[f"val_raw/{k}" for k in keys]
        aliases = [f"{split}/{name}" for split in ("train", "val") for name in ("box_loss", "cls_loss", "giou_loss", "total_loss")]
        self.result_fields = ["epoch", "epochs", "timestamp", "images_train", "instances_train", "images_val", "instances_val",
                              *aliases, *METRIC_FIELDS, *self.lr_fields, "train_seconds", "val_seconds", "metric_seconds", "epoch_seconds",
                              "inference_ms", "inference_fps", "preprocess_ms", "postprocess_ms", *RESOURCE_FIELDS,
                              "parameters", "trainable_parameters", "gflops", "best_epoch", "best_AP", "is_best", "evaluation_weights",
                              "validation_loss_scope", *self.train_fields, *self.val_fields]
        self.results = CSVTable(self.output/"results.csv", self.result_fields, ("epoch",))
        common = ["phase", "epoch", "split", "batch", "batches", "images", "instances", "height", "width", "seconds",
                  "data_seconds", "forward_seconds", "gradient_norm", "total_loss", *RESOURCE_FIELDS, *self.lr_fields]
        details = [f"loss/{k}" for k in keys]+[f"loss_raw/{k}" for k in keys]
        self.train_batch = CSVTable(self.output/"metrics"/"train_batches.csv", common+details, ("phase", "epoch", "split", "batch"))
        self.val_batch = CSVTable(self.output/"metrics"/"val_batches.csv", common+details, ("phase", "epoch", "split", "batch"))
        self.losses = CSVTable(self.output/"metrics"/"losses.csv",
                               ["phase", "epoch", "split", "loss", "weighted", "unweighted", "coefficient", "observed_batches", "available"],
                               ("phase", "epoch", "split", "loss"))
        self.lr = CSVTable(self.output/"metrics"/"learning_rates.csv", ["epoch", "group", "lr_used", "lr_next", "weight_decay"], ("epoch", "group"))
        self.metric_table = None
        self.class_table = None
        self.image_table = None
        self.time_table = CSVTable(self.output/"metrics"/"epoch_timing.csv",
                                   ["epoch", "train_start", "train_end", "val_end", "epoch_end", "train_seconds", "val_seconds", "metric_seconds", "epoch_seconds"], ("epoch",))
        self.resource_table = CSVTable(self.output/"metrics"/"resources.csv", ["phase", "epoch", "split", *RESOURCE_FIELDS], ("phase", "epoch", "split"))
        write_json(self.output/"inputs"/"csv_schema.json", {
            "results.csv": self.result_fields,
            "train_batches.csv / val_batches.csv": common+details,
            "native_weighted_losses": keys, "native_loss_coefficients": weights,
            "raw_loss": "native weighted loss divided by its original criterion coefficient; does not change backpropagation",
            "epoch_loss_mean": "arithmetic mean over batches, matching original MetricLogger; optional DN keys average only observed batches",
            "validation_loss": "model.eval(), last decoder head only; auxiliary/DN fields are blank, never fabricated zeros",
            "precision_recall_f1": "validation macro means at maximum macro-F1 confidence; labeled test uses frozen validation confidence; COCO AP is separate",
            "flush": "batch CSV closes after every row; epoch result and key epoch tables additionally fsync",
            "unique_keys": {"results": ["epoch"], "batch": ["phase", "epoch", "split", "batch"],
                            "metrics": ["phase", "epoch", "split"], "class": ["phase", "epoch", "split", "class_id"]},
            "nulls": "unavailable / undefined values are blank in CSV and null in JSON",
        })

    def details(self, loss):
        row = {f"loss/{key}": value for key, value in loss.items()}
        row.update({f"loss_raw/{key}": value/self.weights[base_loss_name(key)] for key, value in loss.items()})
        return row

    def epoch_losses(self, phase, epoch, split, averages, counts):
        for key in self.keys:
            value = averages.get(key)
            self.losses.append({"phase": phase, "epoch": epoch, "split": split, "loss": key,
                                "weighted": value, "unweighted": value/self.weights[base_loss_name(key)] if value is not None else None,
                                "coefficient": self.weights[base_loss_name(key)], "observed_batches": counts.get(key, 0),
                                "available": key in averages})

    def record_report(self, phase, report):
        summary = {"phase": phase, **report["summary"]}
        if self.metric_table is None:
            self.metric_table = CSVTable(self.output/"metrics"/"metrics.csv", list(summary), ("phase", "epoch", "split"))
        self.metric_table.append(summary)
        for field, attr, path, keys in (("per_class", "class_table", "per_class_metrics.csv", ("phase", "epoch", "split", "class_id")),
                                       ("per_image", "image_table", "per_image_metrics.csv", ("phase", "epoch", "split", "image_id"))):
            for original in report[field]:
                row = {"phase": phase, **original}
                table = getattr(self, attr)
                if table is None:
                    table = CSVTable(self.output/"metrics"/path, list(row), keys)
                    setattr(self, attr, table)
                table.append(row)

    def result_loss_columns(self, split, averages, total):
        values = {f"{split}/{key}": value for key, value in averages.items()}
        values.update({f"{split}_raw/{key}": value/self.weights[base_loss_name(key)] for key, value in averages.items()})
        values.update({f"{split}/box_loss": averages.get("loss_bbox"), f"{split}/cls_loss": averages.get("loss_vfl"),
                       f"{split}/giou_loss": averages.get("loss_giou"), f"{split}/total_loss": total})
        return values

    def sync(self):
        for table in (self.results, self.losses, self.lr, self.metric_table, self.time_table, self.resource_table):
            if table is not None:
                table.sync()


def restore_csv_history(source, output, completed_epoch):
    """Copy only committed epochs into a NEW run; leave the interrupted run intact."""
    source, output = Path(source), Path(output)
    candidates = [source/"results.csv", *(source/"metrics").glob("*.csv")]
    for path in candidates:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            if not fields or "epoch" not in fields:
                continue
            rows = [row for row in reader if row.get("epoch") and int(row["epoch"]) <= completed_epoch
                    and row.get("phase", "epoch") == "epoch"]
        atomic_csv(output/path.relative_to(source), rows, fields)
    for path in (source/"metrics").glob("epoch_*"):
        if path.is_dir() and path.name.removeprefix("epoch_").isdigit() and int(path.name.removeprefix("epoch_")) <= completed_epoch:
            shutil.copytree(path, output/"metrics"/path.name)
    native_log = source/"log.txt"
    if native_log.is_file():
        import json
        records = [line for line in native_log.read_text(encoding="utf-8").splitlines()
                   if line.strip() and json.loads(line).get("epoch_one_based", json.loads(line)["epoch"]+1) <= completed_epoch]
        (output/"log.txt").write_text("\n".join(records)+("\n" if records else ""), encoding="utf-8")
