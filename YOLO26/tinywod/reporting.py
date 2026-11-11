"""逐类/AP 曲线/混淆矩阵/固定样本测速 CSV；保留上传 README 的文件约定。"""
import time
from pathlib import Path

import numpy as np

from config import CLASS_NAMES
from .common import f1_from_pr, sha256, write_csv, write_json


def plot_history(rows, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    x = [r["epoch"] for r in rows]
    keys = list(rows[-1])
    for filename, select, ylabel in (
        ("loss_curves.png", lambda k: k.startswith(("train/", "val/")), "Loss"),
        ("metric_curves.png", lambda k: k.startswith("metrics/") or k == "F1_from_mean_PR", "Metric"),
        ("learning_rate.png", lambda k: k.startswith("lr/"), "Learning rate"),
    ):
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        for key in keys:
            if select(key):
                ax.plot(x, [r.get(key, np.nan) for r in rows], label=key)
        ax.set(xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        fig.savefig(folder / filename, dpi=160)
        plt.close(fig)


def save_evaluation(validator, folder, split):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    metric, box = validator.metrics, validator.metrics.box
    results = metric.results_dict
    p, r = float(results["metrics/precision(B)"]), float(results["metrics/recall(B)"])
    data = {"split": split, "precision": p, "recall": r, "mAP50": float(results["metrics/mAP50(B)"]),
            "mAP50_95": float(results["metrics/mAP50-95(B)"]), "F1_from_mean_PR": f1_from_pr(p, r),
            "mean_per_class_F1": float(np.mean(box.f1)) if len(box.f1) else None,
            "fitness": float(metric.fitness), "fitness_definition": "mAP50_95 (Ultralytics 8.4.182)",
            "speed_ms_per_image": validator.speed, "images": validator.seen,
            "names": list(CLASS_NAMES), "conf": validator.args.conf, "max_det": validator.args.max_det,
            "precision_recall_protocol": "native AP IoU=0.50:0.05:0.95; P/R/F1 at native max mean-F1 confidence",
            "confusion_matrix_protocol": {"rows": "predicted", "columns": "true", "conf": validator.args.conf,
                                          "matching_iou": .45, "background_index": len(CLASS_NAMES)},
            "nms": False, "precision_dtype": "float32"}
    indices = {int(c): i for i, c in enumerate(box.ap_class_index)}
    per_class = []
    for cls, name in enumerate(CLASS_NAMES):
        i = indices.get(cls)
        row = {"class_id": cls, "class_name": name, "instances": int(metric.nt_per_class[cls]),
               "images": int(metric.nt_per_image[cls]), "precision": None, "recall": None,
               "F1": None, "AP50": None, "AP50_95": None}
        if i is not None:
            row.update(precision=float(box.p[i]), recall=float(box.r[i]), F1=float(box.f1[i]),
                       AP50=float(box.ap50[i]), AP50_95=float(box.ap[i]))
        per_class.append(row)
    write_csv(folder / "per_class.csv", per_class)
    image_metrics = getattr(box, "image_metrics", {})
    if isinstance(image_metrics, dict):
        write_csv(folder / "per_image.csv", [{"image": k, **v} for k, v in image_metrics.items()])
    np.savez_compressed(folder / "ap_and_curves.npz", **{k: np.asarray(getattr(box, k, [])) for k in
                        ("p", "r", "f1", "all_ap", "ap_class_index", "p_curve", "r_curve", "f1_curve", "px", "prec_values")})
    labels = [*CLASS_NAMES, "background"]
    matrix = validator.confusion_matrix.matrix
    write_csv(folder / "confusion_matrix.csv", [{"predicted_class": labels[i],
              **{label: int(matrix[i, j]) for j, label in enumerate(labels)}} for i in range(len(labels))])
    write_json(folder / "confusion_matrix_axes.json", {"row_axis": "predicted", "column_axis": "true",
               "labels": labels, "conf": validator.args.conf, "matching_iou": .45})
    write_json(folder / "metrics.json", data)
    write_csv(folder / "metrics.csv", [{k: v for k, v in data.items() if isinstance(v, (int, float, str)) or v is None}])
    print(f"[评价 {split}] P={p:.6f} R={r:.6f} mAP50={data['mAP50']:.6f} "
          f"mAP50-95={data['mAP50_95']:.6f} F1(mean P/R)={data['F1_from_mean_PR']:.6f}\n"
          f"[评价 {split}] CSV/NPZ/JSON: {folder}；框架耗时(ms/图): {validator.speed}", flush=True)
    return data


def compute_gflops(model, imgsz):
    """计算 GFLOPs（MACs*2/1e9）。优先 fork 版 thop 的 get_flops，缺 fork 时用原版 thop 兜底。"""
    from ultralytics.utils.torch_utils import get_flops, unwrap_model
    flops = get_flops(model, imgsz)
    if flops and flops > 0:
        return float(flops)
    try:
        import torch
        import thop
        deployed = unwrap_model(model)
        p = next(deployed.parameters(), None)
        if p is None:  # AutoBackend 等包装器，真实模型在 .model 属性
            deployed = getattr(deployed, "model", deployed)
            p = next(deployed.parameters())
        size = int(imgsz[0]) if isinstance(imgsz, (list, tuple)) else int(imgsz)
        sample = torch.zeros(1, p.shape[1], size, size, device=p.device, dtype=p.dtype)
        with torch.inference_mode():
            macs, _ = thop.profile(deployed, inputs=(sample,), verbose=False)
        return float(macs * 2 / 1e9)
    except Exception:
        return 0.0


def model_complexity(model, cfg, output):
    count = sum(p.numel() for p in model.parameters())
    flops = compute_gflops(model, cfg.imgsz) if cfg.profile_flops else None
    data = {"parameters": count, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "GFLOPs": flops if flops and flops > 0 else None, "input_shape": [1, 3, cfg.imgsz, cfg.imgsz],
            "scope": "full unfused training model with one-to-many and one-to-one heads",
            "GFLOPs_method": "Ultralytics get_flops / ultralytics-thop; 2 operations per MAC",
            "GFLOPs_status": "measured" if flops and flops > 0 else "disabled or profiler unavailable/failed"}
    write_json(Path(output) / "metrics" / "complexity.json", data)
    return data


def benchmark(model, cfg, dataset_info, output, device):
    """统一 val 固定排序前 N 张，batch1，FP32，排除读盘；逐样本留档。"""
    import cv2
    import torch
    from ultralytics import YOLO
    from .dataset import image_files
    folder = Path(output) / "metrics"
    files = image_files(Path(dataset_info["view"]) / "images" / "val")[:cfg.benchmark_images]
    if not files or cfg.benchmark_images <= 0:
        write_json(folder / "benchmark.json", {"status": "disabled", "inference_ms": None, "pipeline_ms": None})
        return {"inference_ms": None, "pipeline_ms": None}
    if cfg.benchmark_repeats < 1 or cfg.benchmark_warmup < 0:
        raise ValueError("benchmark_repeats 必须>=1，warmup必须>=0")
    # 使用官方 predictor 的预处理/推理/后处理，保持实际部署的 NMS-free 行为。
    yolo = YOLO(str(model))
    args = dict(imgsz=cfg.imgsz, device=device, batch=1, quantize=None, nms=False, rect=False, conf=cfg.conf,
                max_det=cfg.max_det, verbose=False, save=False, augment=False, channels_last=False)
    first = cv2.imread(str(files[0]))
    if first is None:
        raise ValueError(f"测速图像无法解码: {files[0]}")
    for _ in range(max(1, cfg.benchmark_warmup)):
        yolo.predict(first, **args)
    samples = []
    # Results.speed 是框架在同步设备之后测得的分阶段耗时；pipeline_wall 包含调用开销。
    for p in files:
        image = cv2.imread(str(p))
        if image is None:
            raise ValueError(f"测速图像无法解码: {p}")
        for repetition in range(cfg.benchmark_repeats):
            if torch.cuda.is_available() and str(device) != "cpu":
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = yolo.predict(image, **args)[0]
            if torch.cuda.is_available() and str(device) != "cpu":
                torch.cuda.synchronize()
            wall_ms = (time.perf_counter()-start)*1000
            samples.append({"image": p.relative_to(dataset_info["view"]).as_posix(), "image_sha256": sha256(p),
                            "repeat": repetition+1, "preprocess_ms": result.speed["preprocess"],
                            "inference_ms": result.speed["inference"], "postprocess_ms": result.speed["postprocess"],
                            "pipeline_wall_ms": wall_ms})
    write_csv(folder / "benchmark_samples.csv", samples)
    data = {"status": "measured", "device": device, "batch": 1, "dtype": "float32", "imgsz": cfg.imgsz,
            "warmup": max(1, cfg.benchmark_warmup), "images": len(files), "repeats": cfg.benchmark_repeats,
            "split": "val", "selection": "sorted relative paths, first N", "disk_decode_included": False,
            "inference_ms": float(np.mean([s["inference_ms"] for s in samples])),
            "pipeline_ms": float(np.mean([s["pipeline_wall_ms"] for s in samples])),
            "inference_median_ms": float(np.median([s["inference_ms"] for s in samples]))}
    write_json(folder / "benchmark.json", data)
    write_csv(folder / "benchmark_summary.csv", [data])
    return data


def predict_unlabeled(best, cfg, info, output, device):
    from ultralytics import YOLO
    from .dataset import image_files
    from .common import append_jsonl
    model = YOLO(str(best))
    folder = Path(output) / "results" / "test_predict"
    folder.mkdir(parents=True, exist_ok=False)
    for image in image_files(Path(info["view"]) / "images" / "test"):
        rel = image.relative_to(Path(info["view"]) / "images" / "test")
        result = model.predict(str(image), imgsz=cfg.imgsz, batch=1, device=device, rect=False, nms=False,
                               conf=cfg.prediction_conf, max_det=cfg.max_det, quantize=None, verbose=False,
                               channels_last=False, save=False)[0]
        label = folder / "labels" / rel.with_suffix(".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for cls, xywh, confidence in zip(result.boxes.cls.cpu().tolist(), result.boxes.xywhn.cpu().tolist(),
                                         result.boxes.conf.cpu().tolist()):
            lines.append(" ".join(map(str, [int(cls), *xywh, confidence])))
        label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        append_jsonl(folder / "predictions.jsonl", {"image": rel.as_posix(), "names": list(CLASS_NAMES),
                      "xyxy": result.boxes.xyxy.cpu().tolist(), "class": result.boxes.cls.cpu().tolist(),
                      "confidence": result.boxes.conf.cpu().tolist()})
        if cfg.save_test_images:
            destination = folder / "images" / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            result.save(str(destination))
    write_json(folder / "README.json", {"split": "test", "ground_truth_available": False,
               "metrics_computed": False, "conf": cfg.prediction_conf, "label_format": "cls cx cy w h confidence"})
    print(f"[测试] test 无标签，仅保存预测，不计算/伪造测试指标: {folder}", flush=True)
