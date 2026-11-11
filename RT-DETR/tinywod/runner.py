"""PyCharm entry orchestration, full failure traces and experiment summaries."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
import csv
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import uuid

from config import CONFIG, PROJECT_ROOT, resolve, snapshot
from .common import CSVTable, Tee, atomic_csv, atomic_torch_save, now, write_json

SUMMARY_FIELDS = ["run_id", "model", "status", "phase", "start_time", "end_time", "total_seconds", "completed_epochs", "epochs",
                  "best_epoch", "best_AP", "final_split", "precision", "recall", "f1", "AP50", "AP", "APsmall", "APmedium", "APlarge",
                  "val_AP", "val_APsmall", "test_AP", "test_APsmall", "test_status", "parameters", "trainable_parameters", "gflops",
                  "benchmark_ms", "benchmark_fps", "device", "seed", "initialization", "weight_sha256", "dataset_fingerprint",
                  "output_dir", "best_weight", "last_weight", "results_csv", "log_file", "error"]


@contextmanager
def comparison_lock(path, timeout=10):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        if f.tell() == 0:
            f.write(b"0")
            f.flush()
        start = time.monotonic()
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic()-start > timeout:
                    raise TimeoutError(f"实验汇总正在被其他进程写入：{path}")
                time.sleep(.05)
        try:
            yield
        finally:
            if os.name == "nt":
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def launch(cfg=CONFIG, *, verification=False):
    # verification is an explicit testing API, not a hidden CLI override.
    if not verification and cfg.epochs != 72:
        print(f"[轮数配置] 当前 config.py 明确指定 epochs={cfg.epochs}；交付默认值为 72。", flush=True)
    started, begin = now(), time.perf_counter()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8]
    root = resolve(cfg.output_root)
    output = root/cfg.model_name/run_id
    for folder in ("weights", "logs", "metrics", "results", "inputs"):
        (output/folder).mkdir(parents=True, exist_ok=False)
    summary = {field: None for field in SUMMARY_FIELDS}
    summary.update(run_id=run_id, model=cfg.model_name, status="initializing", phase="initialization", start_time=started,
                   completed_epochs=0, epochs=cfg.epochs, seed=cfg.seed, initialization=cfg.initialization,
                   output_dir=str(output), best_weight=str(output/"weights"/"best.pth"), last_weight=str(output/"weights"/"last.pth"),
                   results_csv=str(output/"results.csv"), log_file=str(output/"logs"/"console.log"))
    write_json(output/"inputs"/"config.json", snapshot(cfg))
    (output/"inputs"/"config_source.py").write_bytes((PROJECT_ROOT/"config.py").read_bytes())
    write_json(output/"run_state.json", summary)
    writer = None
    code = 1
    with (output/"logs"/"console.log").open("x", encoding="utf-8", buffering=1) as log:
        lock = threading.RLock()
        with redirect_stdout(Tee(sys.stdout, log, lock)), redirect_stderr(Tee(sys.stderr, log, lock)):
            print(f"[实验] {'合成数据冒烟测试' if verification else '正式 TinyWOD 训练'} | {run_id}", flush=True)
            print(f"[配置] {PROJECT_ROOT/'config.py'}\n[数据路径] YOLO={cfg.yolo_root} COCO={cfg.coco_root}", flush=True)
            print(f"[模型配置] {resolve(cfg.model_config)}\n[轮数] {cfg.epochs}\n[输出] {output}\n[开始时间] {started}", flush=True)
            try:
                from .environment import setup_environment
                from .modeling import build_model, loss_keys, validate_config
                validate_config(cfg)
                device, environment = setup_environment(cfg, output)
                summary["device"] = environment["device_name"]
                import torch
                from .data import prepare_dataset
                from .loaders import make_loaders
                from .recording import METRIC_FIELDS, Recorder
                from .checkpointing import load_resume, rng_state, restore_rng, state_dict
                from .engine import benchmark, predict_unlabeled, train_epoch, validate
                from .plotting import plot_evaluation, plot_training
                model, criterion, postprocessor, optimizer, scheduler, ema, scaler, initialization, complexity = build_model(cfg, output, device)
                summary.update(parameters=complexity["parameters"], trainable_parameters=complexity["trainable_parameters"], gflops=complexity["gflops"],
                               weight_sha256=initialization["sha256"])
                summary["phase"] = "dataset_preparation"
                info = prepare_dataset(cfg, output/"inputs")
                summary["dataset_fingerprint"] = info["fingerprint"]
                loaders, generators = make_loaders(cfg, info)
                write_json(output/"inputs"/"data_loading.json", {
                    split: {"dataset_class": str(type(loader.dataset)), "images_path": info["splits"][split]["images_path"],
                            "annotation": info["splits"][split]["annotation"], "batch_size": loader.batch_size,
                            "num_workers": loader.num_workers, "pin_memory": loader.pin_memory,
                            "drop_last": loader.drop_last, "shuffle": split == "train",
                            "transforms": repr(getattr(loader.dataset, "_transforms", "resize RGB [0,1]"))}
                    for split, loader in loaders.items()})
                resume = {"completed_epochs": 0, "best_epoch": 0, "best_AP": None}
                if cfg.resume_checkpoint:
                    resume = load_resume(cfg, output, resolve(cfg.resume_checkpoint), model, optimizer, scheduler, ema, scaler, generators, info)
                    initialization = resume["initialization"]
                    summary.update(initialization=initialization["initialization"], weight_sha256=initialization.get("sha256"))
                    write_json(output/"inputs"/"resume.json", {key: value for key, value in resume.items() if key not in ("initialization", "rng")})
                recorder = Recorder(output, loss_keys(model), len(optimizer.param_groups), criterion.weight_dict)
                if cfg.tensorboard:
                    from torch.utils.tensorboard import SummaryWriter
                    writer = SummaryWriter(str(output/"logs"/"tensorboard"))
                if "rng" in resume:
                    restore_rng(resume.pop("rng"), generators)
                summary.update(completed_epochs=resume["completed_epochs"], best_epoch=resume["best_epoch"], best_AP=resume["best_AP"], status="running")
                print(f"[保存位置] 最佳={output/'weights'/'best.pth'} | 最后={output/'weights'/'last.pth'}", flush=True)
                print(f"[CSV] {output/'results.csv'} | 逐 batch/类别/图像/损失明细={output/'metrics'}", flush=True)
                print(f"[指标口径] COCO AP/APsmall：maxDets=100、原图面积；P/R/F1：IoU=0.5、验证集最大宏平均 F1 阈值。", flush=True)
                for epoch in range(summary["completed_epochs"]+1, cfg.epochs+1):
                    epoch_begin = time.perf_counter()
                    summary["phase"] = "training"
                    write_json(output/"run_state.json", summary)
                    lr_used = {f"lr/pg{i}": group["lr"] for i, group in enumerate(optimizer.param_groups)}
                    train = train_epoch(model, criterion, loaders["train"], optimizer, ema, scaler, cfg, device, epoch, recorder)
                    scheduler.step()  # preserve original epoch-level scheduler stepping
                    summary["phase"] = "validation"
                    eval_model = ema.module if ema is not None else model
                    epoch_dir = output/"metrics"/f"epoch_{epoch:04d}"
                    validation, report = validate(eval_model, criterion, postprocessor, loaders["val"], cfg, device, epoch, "val", recorder, epoch_dir)
                    metrics = report["summary"]
                    score = metrics["AP"]
                    if score is None:
                        raise ValueError("验证集 AP 未定义，不能选择最佳模型。")
                    is_best = summary["best_AP"] is None or score > summary["best_AP"]
                    if is_best:
                        summary.update(best_AP=score, best_epoch=epoch)
                    for i, group in enumerate(optimizer.param_groups):
                        recorder.lr.append({"epoch": epoch, "group": i, "lr_used": lr_used[f"lr/pg{i}"], "lr_next": group["lr"], "weight_decay": group["weight_decay"]})
                    row = {"epoch": epoch, "epochs": cfg.epochs, "timestamp": now(), "images_train": train["images"],
                           "instances_train": train["instances"], "images_val": metrics["images"], "instances_val": metrics["instances"],
                           **recorder.result_loss_columns("train", train["losses"], train["total_loss"]),
                           **recorder.result_loss_columns("val", validation["losses"], validation["total_loss"]),
                           **{key: metrics[key] for key in METRIC_FIELDS}, **lr_used,
                           "train_seconds": train["seconds"], "val_seconds": validation["seconds"], "metric_seconds": validation["metric_seconds"],
                           "epoch_seconds": time.perf_counter()-epoch_begin, "inference_ms": validation["inference_ms"],
                           "inference_fps": validation["inference_fps"], "preprocess_ms": validation["preprocess_ms"], "postprocess_ms": validation["postprocess_ms"],
                           **validation["resources"], "parameters": complexity["parameters"], "trainable_parameters": complexity["trainable_parameters"],
                           "gflops": complexity["gflops"], "best_epoch": summary["best_epoch"], "best_AP": summary["best_AP"], "is_best": is_best,
                           "evaluation_weights": "EMA" if ema is not None else "model", "validation_loss_scope": "eval_last_decoder_only"}
                    recorder.results.append(row)
                    coco_keys = ("AP", "AP50", "AP75", "APsmall", "APmedium", "APlarge", "AR1", "AR10", "AR100", "ARsmall", "ARmedium", "ARlarge")
                    native_record = {"epoch": epoch-1, "epoch_one_based": epoch, "evaluation_split": "val", "train_lr": lr_used["lr/pg0"],
                                     "train_loss": train["total_loss"], "n_parameters": complexity["trainable_parameters"],
                                     "test_coco_eval_bbox": [metrics[key] if metrics[key] is not None else -1. for key in coco_keys],
                                     **{f"train_{key}": value for key, value in train["losses"].items()},
                                     **{f"val_{key}": value for key, value in validation["losses"].items()}}
                    with (output/"log.txt").open("a", encoding="utf-8") as native_log:
                        native_log.write(json.dumps(native_record, ensure_ascii=False, allow_nan=False)+"\n")
                        native_log.flush()
                        os.fsync(native_log.fileno())
                    plot_evaluation(report)
                    plot_training(output/"results.csv", output/"results")
                    recorder.sync()
                    # Capture RNG after plotting and commit checkpoints only once
                    # the corresponding epoch CSV/metrics exist on disk.
                    checkpoint = state_dict(cfg, output, epoch, model, optimizer, scheduler, ema, scaler, generators, info,
                                            summary["best_epoch"], summary["best_AP"], initialization)
                    if is_best:
                        atomic_torch_save(checkpoint, output/"weights"/"best.pth")
                        print(f"[最佳模型更新] Epoch {epoch}/{cfg.epochs} | val AP={score:.6f} APsmall={metrics['APsmall']} | {output/'weights'/'best.pth'}", flush=True)
                    if cfg.save_period > 0 and (epoch % cfg.save_period == 0 or epoch == cfg.epochs):
                        atomic_torch_save(checkpoint, output/"weights"/f"epoch_{epoch:04d}.pth")
                    atomic_torch_save(checkpoint, output/"weights"/"last.pth")
                    del checkpoint
                    epoch_seconds = time.perf_counter()-epoch_begin
                    recorder.time_table.append({"epoch": epoch, "train_start": train["start"], "train_end": train["end"],
                                                "val_end": validation["end"], "epoch_end": now(), "train_seconds": train["seconds"],
                                                "val_seconds": validation["seconds"], "metric_seconds": validation["metric_seconds"], "epoch_seconds": epoch_seconds})
                    recorder.sync()
                    summary["completed_epochs"] = epoch
                    summary["phase"] = "epoch_complete"
                    write_json(output/"run_state.json", summary)
                    peak = validation["resources"]["peak_allocated_gb"]
                    flops = f"{complexity['gflops']:.3f}" if complexity["gflops"] is not None else "N/A"
                    print(f"Complexity: Params={complexity['parameters']}, GFLOPs={flops}, val inference={validation['inference_ms']:.3f} ms/image "
                          f"({validation['inference_fps']:.2f} FPS), PeakAllocated={f'{peak:.3f} GB' if peak is not None else 'N/A (CPU)'}", flush=True)
                    loss_print = {**{f"train/{key}": round(value, 6) for key, value in train["losses"].items()},
                                  **{f"val/{key}": round(value, 6) for key, value in validation["losses"].items()}}
                    def formatted(key):
                        return f"{metrics[key]:.5f}" if metrics[key] is not None else "N/A"
                    print(f"[{cfg.model_name}] Epoch {epoch}/{cfg.epochs} 完成 | Box(P={formatted('precision')} R={formatted('recall')} "
                          f"AP50={formatted('AP50')} AP={formatted('AP')} APsmall={formatted('APsmall')}) | lr={lr_used} | loss={loss_print}", flush=True)
                    print(f"[耗时] train={train['seconds']:.2f}s val={validation['seconds']:.2f}s metrics={validation['metric_seconds']:.2f}s "
                          f"epoch={epoch_seconds:.2f}s | 最佳 Epoch={summary['best_epoch']} AP={summary['best_AP']:.6f}", flush=True)
                    if writer is not None:
                        for key, value in row.items():
                            if isinstance(value, (float, int)) and not isinstance(value, bool):
                                writer.add_scalar(key, value, epoch)
                        writer.flush()
                summary["phase"] = "final_evaluation"
                best = torch.load(output/"weights"/"best.pth", map_location="cpu", weights_only=True)
                eval_model = ema.module if ema is not None else model
                eval_model.load_state_dict(best["ema"]["module"] if best["ema"] is not None else best["model"], strict=True)
                del best
                print(f"[最终验证] 载入最佳 Epoch {summary['best_epoch']} 的 {'EMA' if ema is not None else 'model'} 权重。", flush=True)
                val_stats, val_report = validate(eval_model, criterion, postprocessor, loaders["val"], cfg, device, summary["best_epoch"], "val", recorder,
                                                 output/"metrics"/"final"/"val", phase="final_best")
                plot_evaluation(val_report)
                final = val_report["summary"]
                summary.update(val_AP=final["AP"], val_APsmall=final["APsmall"], final_split="val")
                if info["splits"]["test"]["labeled"]:
                    _, test_report = validate(eval_model, criterion, postprocessor, loaders["test"], cfg, device, summary["best_epoch"], "test", recorder,
                                               output/"metrics"/"final"/"test", phase="final_best", report_confidence=val_report["summary"]["confidence"])
                    plot_evaluation(test_report)
                    final = test_report["summary"]
                    summary.update(final_split="test", test_AP=final["AP"], test_APsmall=final["APsmall"], test_status="evaluated_once_after_selection")
                elif info["splits"]["test"]["images"]:
                    if cfg.predict_unlabeled_test:
                        test_status = predict_unlabeled(eval_model, postprocessor, loaders["test"], cfg, device, output/"results"/"test_unlabeled")
                        summary["test_status"] = test_status["status"]
                    else:
                        summary["test_status"] = "unlabeled_prediction_disabled"
                else:
                    summary["test_status"] = "not_provided"
                speed = benchmark(eval_model, loaders["val"], cfg, device, output/"metrics")
                summary.update(benchmark_ms=speed["mean_ms"], benchmark_fps=speed["fps"], status="completed", phase="completed")
                summary.update({key: final[key] for key in ("precision", "recall", "f1", "AP50", "AP", "APsmall", "APmedium", "APlarge")})
                recorder.sync()
                code = 0
                print(f"[训练结束] {cfg.model_name} | Epoch={summary['completed_epochs']}/{cfg.epochs} | best_epoch={summary['best_epoch']} "
                      f"best_val_AP={summary['best_AP']:.6f} | 最终划分={summary['final_split']} AP={summary['AP']:.6f} APsmall={summary['APsmall']}", flush=True)
                print(f"[速度] batch=1 forward={speed['mean_ms']:.3f} ms/image ({speed['fps']:.2f} FPS)", flush=True)
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    summary["status"], code = "interrupted", 130
                else:
                    summary["status"], code = "failed", 1
                summary["error"] = f"{type(exc).__name__}: {exc}"
                detail = traceback.format_exc()
                (output/"logs"/"error.log").write_text(detail, encoding="utf-8")
                print(detail, file=sys.stderr, flush=True)
                print(f"[失败] phase={summary['phase']} | {output/'logs'/'error.log'}", flush=True)
                if "out of memory" in str(exc).lower():
                    print("[显存不足] 本次失败已记录，未自动修改 batch/输入尺寸/精度。先释放其他 GPU 进程；如修改 config.py，需为对比实验明确记录并统一关键参数。", flush=True)
            finally:
                if writer is not None:
                    writer.close()
                summary.update(end_time=now(), total_seconds=time.perf_counter()-begin)
                write_json(output/"run_state.json", summary)
                write_json(output/"experiment_summary.json", summary)
                atomic_csv(output/"experiment_summary.csv", [summary], SUMMARY_FIELDS)
                try:
                    with comparison_lock(root/".experiment_comparison.lock"):
                        table = CSVTable(root/"experiment_comparison.csv", SUMMARY_FIELDS, ("run_id",))
                        table.append(summary)
                        table.sync()
                except Exception as exc:
                    print(f"[汇总写入失败] {exc}。本次独立汇总已保存在 {output/'experiment_summary.csv'}。", file=sys.stderr, flush=True)
                    summary["error"] = ((summary.get("error") or "")+f"\nSummary export failed: {type(exc).__name__}: {exc}").strip()
                    if summary["status"] == "completed":
                        summary.update(status="failed", phase="summary_export")
                    write_json(output/"run_state.json", summary)
                    write_json(output/"experiment_summary.json", summary)
                    atomic_csv(output/"experiment_summary.csv", [summary], SUMMARY_FIELDS)
                    code = 1
                print(f"[结束时间] {summary['end_time']} | 总耗时={summary['total_seconds']:.2f}s", flush=True)
                print(f"[实验汇总] status={summary['status']} completed={summary['completed_epochs']}/{cfg.epochs}", flush=True)
                print(f"[本次] {output/'experiment_summary.csv'}\n[对比] {root/'experiment_comparison.csv'}", flush=True)
    return {"exit_code": code, "output": str(output), "summary": summary}
