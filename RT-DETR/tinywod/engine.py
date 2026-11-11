"""Native RT-DETR forward/loss/backward with the requested progress and CSV outputs."""
from collections import Counter
from contextlib import nullcontext
import math
from pathlib import Path
import time

import torch
from tqdm import tqdm

from .common import now, write_json
from .metrics import evaluate_predictions
from .recording import resources


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast(cfg, device):
    if cfg.mixed_precision == "no":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16 if cfg.mixed_precision == "fp16" else torch.bfloat16)


def float_outputs(value):
    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: float_outputs(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(float_outputs(item) for item in value)
    return value


class Averages:
    def __init__(self):
        self.sums, self.counts = Counter(), Counter()
        self.total, self.batches, self.images, self.instances = 0., 0, 0, 0

    def update(self, values, images, instances):
        for key, value in values.items():
            if not math.isfinite(value):
                raise FloatingPointError(f"损失非有限数：{key}={value}")
            self.sums[key] += value
            self.counts[key] += 1
        self.total += sum(values.values())
        self.batches += 1
        self.images += images
        self.instances += instances

    @property
    def means(self):
        return {key: value/self.counts[key] for key, value in self.sums.items()}

    def result(self):
        if not self.batches:
            raise ValueError("数据加载器没有产生任何 batch")
        return {"losses": self.means, "loss_counts": dict(self.counts), "total_loss": self.total/self.batches,
                "images": self.images, "instances": self.instances}


def move_targets(targets, device):
    return [{key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in target.items()} for target in targets]


def train_epoch(model, criterion, loader, optimizer, ema, scaler, cfg, device, epoch, recorder):
    model.train()
    criterion.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    averages = Averages()
    started = now()
    begin, previous = time.perf_counter(), time.perf_counter()
    print(f"[{cfg.model_name}] Epoch {epoch}/{cfg.epochs}", flush=True)
    print(f"{'Epoch':>11}{'GPU_mem':>11}{'box_loss':>12}{'cls_loss':>12}{'giou_loss':>12}{'total_loss':>13}{'Instances':>12}{'Size':>10}", flush=True)
    progress = tqdm(loader, total=len(loader), file=__import__('sys').stdout, mininterval=1., ncols=155, leave=True)
    for batch_index, (samples, targets) in enumerate(progress, 1):
        batch_start = time.perf_counter()
        data_seconds = batch_start-previous
        samples = samples.to(device, non_blocking=cfg.pin_memory)
        targets = move_targets(targets, device)
        instances = sum(len(target["labels"]) for target in targets)
        optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        forward_start = time.perf_counter()
        with autocast(cfg, device):
            outputs = model(samples, targets)
        synchronize(device)
        forward_seconds = time.perf_counter()-forward_start
        with torch.autocast(device_type=device.type, enabled=False):
            losses = criterion(float_outputs(outputs), targets)
            total = sum(losses.values())
        values = {key: float(value.detach()) for key, value in losses.items()}
        if not torch.isfinite(total).item() or any(not math.isfinite(value) for value in values.values()):
            raise FloatingPointError(f"Epoch {epoch} batch {batch_index} 非有限损失：{values}")
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_max_norm if cfg.clip_max_norm > 0 else float('inf'), error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)
        averages.update(values, len(samples), instances)
        height, width = map(int, model.last_input_size)
        resource = resources(device)
        elapsed = time.perf_counter()-batch_start
        if cfg.save_batch_csv:
            recorder.train_batch.append({"phase": "epoch", "epoch": epoch, "split": "train", "batch": batch_index,
                                         "batches": len(loader), "images": len(samples), "instances": instances,
                                         "height": height, "width": width, "seconds": elapsed, "data_seconds": data_seconds,
                                         "forward_seconds": forward_seconds, "gradient_norm": float(gradient_norm),
                                         "total_loss": float(total.detach()), **resource,
                                         **{f"lr/pg{i}": group["lr"] for i, group in enumerate(optimizer.param_groups)},
                                         **recorder.details(values)})
        means = averages.means
        memory = f"{resource['gpu_reserved_gb']:.2f}G" if device.type == "cuda" else "CPU"
        description = (f"{epoch:>7}/{cfg.epochs:<3}{memory:>11}{means.get('loss_bbox', 0):>12.4g}"
                       f"{means.get('loss_vfl', 0):>12.4g}{means.get('loss_giou', 0):>12.4g}"
                       f"{averages.total/averages.batches:>13.4g}{instances:>12}{width:>10}")
        progress.set_description_str(description, refresh=False)
        if batch_index % cfg.print_freq == 0:
            progress.refresh()
        previous = time.perf_counter()
        del outputs, losses, total, samples, targets
    progress.close()
    synchronize(device)
    result = {**averages.result(), "start": started, "end": now(), "seconds": time.perf_counter()-begin, "resources": resources(device)}
    recorder.epoch_losses("epoch", epoch, "train", result["losses"], result["loss_counts"])
    recorder.resource_table.append({"phase": "epoch", "epoch": epoch, "split": "train", **result["resources"]})
    return result


@torch.no_grad()
def validate(model, criterion, postprocessor, loader, cfg, device, epoch, split, recorder, directory, phase="epoch", report_confidence=None):
    model.eval()
    criterion.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    averages, predictions = Averages(), {}
    preprocess_seconds = forward_seconds = postprocess_seconds = 0.
    begin = time.perf_counter()
    header = f"{'Class':>22}{'Images':>11}{'Instances':>11}{'Box(P':>11}{'R':>11}{'AP50':>11}{'AP':>11}{'APsmall':>12}"
    progress = tqdm(loader, desc=header, total=len(loader), file=__import__('sys').stdout, mininterval=1., ncols=170, leave=True)
    previous = time.perf_counter()
    for batch_index, (samples, targets) in enumerate(progress, 1):
        begin_batch = time.perf_counter()
        data_seconds = begin_batch-previous
        samples = samples.to(device, non_blocking=cfg.pin_memory)
        targets = move_targets(targets, device)
        synchronize(device)
        preprocess_seconds += time.perf_counter()-begin_batch
        forward_start = time.perf_counter()
        with autocast(cfg, device):
            outputs = model(samples)
        synchronize(device)
        forward_elapsed = time.perf_counter()-forward_start
        forward_seconds += forward_elapsed
        with torch.autocast(device_type=device.type, enabled=False):
            values_t = criterion(float_outputs(outputs), targets)
        values = {key: float(value.detach()) for key, value in values_t.items()}
        instances = sum(len(target["labels"]) for target in targets)
        averages.update(values, len(samples), instances)
        post_start = time.perf_counter()
        sizes = torch.stack([target["orig_size"] for target in targets])
        results = postprocessor(float_outputs(outputs), sizes)
        for target, result in zip(targets, results):
            image_id = int(target["image_id"].item())
            predictions[image_id] = {key: value.detach().cpu().tolist() for key, value in result.items()}
        synchronize(device)
        postprocess_seconds += time.perf_counter()-post_start
        if cfg.save_batch_csv:
            recorder.val_batch.append({"phase": phase, "epoch": epoch, "split": split, "batch": batch_index,
                                       "batches": len(loader), "images": len(samples), "instances": instances,
                                       "height": int(samples.shape[-2]), "width": int(samples.shape[-1]),
                                       "seconds": time.perf_counter()-begin_batch, "data_seconds": data_seconds,
                                       "forward_seconds": forward_elapsed, "total_loss": sum(values.values()),
                                       **resources(device), **recorder.details(values)})
        previous = time.perf_counter()
    progress.close()
    stats = {**averages.result(), "seconds": time.perf_counter()-begin, "end": now(), "resources": resources(device)}
    stats.update(inference_ms=forward_seconds/averages.images*1000,
                 inference_fps=averages.images/forward_seconds if forward_seconds > 0 else None,
                 preprocess_ms=preprocess_seconds/averages.images*1000,
                 postprocess_ms=postprocess_seconds/averages.images*1000)
    metric_start = time.perf_counter()
    report = evaluate_predictions(loader.dataset.coco, predictions, cfg, directory, epoch, split, report_confidence=report_confidence)
    stats["metric_seconds"] = time.perf_counter()-metric_start
    recorder.record_report(phase, report)
    recorder.epoch_losses(phase, epoch, split, stats["losses"], stats["loss_counts"])
    recorder.resource_table.append({"phase": phase, "epoch": epoch, "split": split, **stats["resources"]})
    summary = report["summary"]
    def number(value):
        return f"{value:>11.4g}" if value is not None else f"{'N/A':>11}"
    print(f"{'all':>22}{summary['images']:>11}{summary['instances']:>11}"+
          ''.join(number(summary[key]) for key in ('precision', 'recall', 'AP50', 'AP', 'APsmall')), flush=True)
    if phase == "final_best":
        for row in report["per_class"]:
            print(f"{row['class_name']:>22}{row['images']:>11}{row['instances']:>11}"+
                  ''.join(number(row[key]) for key in ('precision', 'recall', 'AP50', 'AP', 'APsmall')), flush=True)
    return stats, report


@torch.no_grad()
def predict_unlabeled(model, postprocessor, loader, cfg, device, directory):
    import csv
    from PIL import Image, ImageDraw
    from config import CLASS_NAMES
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/"predictions.csv"
    model.eval()
    rows_json = []
    image_count, count = 0, 0
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        fields = ["image", "class_id", "class_name", "score", "x1", "y1", "x2", "y2", "width", "height"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for samples, targets in tqdm(loader, desc="unlabeled test predictions", file=__import__('sys').stdout, ncols=130):
            with autocast(cfg, device):
                outputs = model(samples.to(device))
            sizes = torch.stack([target["orig_size"] for target in targets]).to(device)
            results = postprocessor(float_outputs(outputs), sizes)
            for target, prediction in zip(targets, results):
                relative = target["relative"]
                width, height = target["orig_size"].tolist()
                image = None
                if cfg.save_prediction_images:
                    image = Image.open(loader.dataset.folder/relative).convert("RGB")
                    draw = ImageDraw.Draw(image)
                image_count += 1
                for box, score, label in zip(prediction["boxes"].cpu().tolist(), prediction["scores"].cpu().tolist(), prediction["labels"].cpu().tolist()):
                    if score < cfg.prediction_conf:
                        continue
                    row = {"image": relative, "class_id": label, "class_name": CLASS_NAMES[label], "score": score,
                           **dict(zip(("x1", "y1", "x2", "y2"), box)), "width": width, "height": height}
                    writer.writerow(row)
                    rows_json.append(row)
                    count += 1
                    if image is not None:
                        draw.rectangle(box, outline="red", width=2)
                        draw.text((box[0], box[1]), f"{CLASS_NAMES[label]} {score:.2f}", fill="red")
                if image is not None:
                    destination = directory/"images"/relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    image.save(destination)
                f.flush()
    write_json(directory/"predictions.json", rows_json)
    status = {"status": "unlabeled_predictions_only", "images": image_count, "detections": count,
              "confidence": cfg.prediction_conf, "AP": None, "APsmall": None, "precision": None, "recall": None}
    write_json(directory/"status.json", status)
    print(f"[测试集] 无 GT，仅保存 {image_count} 张图的预测；AP/P/R 为空。位置：{directory}", flush=True)
    return status


@torch.no_grad()
def benchmark(model, loader, cfg, device, directory):
    from .common import atomic_csv
    import numpy as np
    images, identities = [], []
    for samples, targets in loader:
        for sample, target in zip(samples, targets):
            images.append(sample.unsqueeze(0).to(device))
            image_id = int(target["image_id"].item())
            image_name = loader.dataset.coco.imgs[image_id]["file_name"] if hasattr(loader.dataset, "coco") else target.get("relative", "")
            identities.append((image_id, image_name))
            if len(images) >= cfg.benchmark_images:
                break
        if len(images) >= cfg.benchmark_images:
            break
    if not images:
        raise ValueError("速度测试没有图像")
    model.eval()
    for index in range(cfg.benchmark_warmup):
        with autocast(cfg, device):
            model(images[index % len(images)])
    synchronize(device)
    rows = []
    for repeat in range(1, cfg.benchmark_repeats+1):
        for index, image in enumerate(images, 1):
            synchronize(device)
            start = time.perf_counter()
            with autocast(cfg, device):
                model(image)
            synchronize(device)
            rows.append({"repeat": repeat, "image_index": index, "image_id": identities[index-1][0], "image": identities[index-1][1],
                         "batch_size": 1, "height": image.shape[-2],
                         "width": image.shape[-1], "inference_ms": (time.perf_counter()-start)*1000,
                         "device": str(device), "mixed_precision": cfg.mixed_precision})
    values = [row["inference_ms"] for row in rows]
    result = {"scope": "model forward only, batch=1, preloaded tensors, synchronized device; excludes preprocessing/postprocessing",
              "images": len(images), "repeats": cfg.benchmark_repeats, "warmup": cfg.benchmark_warmup,
              "mean_ms": float(np.mean(values)), "median_ms": float(np.median(values)), "p90_ms": float(np.percentile(values, 90)),
              "fps": 1000/float(np.mean(values)), "device": str(device), "mixed_precision": cfg.mixed_precision}
    atomic_csv(Path(directory)/"inference_speed_samples.csv", rows, list(rows[0]))
    atomic_csv(Path(directory)/"inference_speed.csv", [result], list(result))
    write_json(Path(directory)/"inference_speed.json", result)
    return result
