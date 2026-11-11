"""沿用上传入口 launch(...) 接口，逐模型隔离、同步控制台和日志、汇总真实状态。"""
import contextlib
import csv
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path

from config import CONFIG, MODEL_FILES, MODEL_ORDER, PROJECT_ROOT, resolved_path, snapshot
from .common import now, read_json, write_csv, write_json

SUMMARY_FIELDS = ["experiment_id", "formal", "model", "status", "stage", "start_time", "end_time", "elapsed_s",
                  "completed_epochs", "evaluation_split", "precision", "recall", "mAP50", "mAP50_95",
                  "F1_from_mean_PR", "mean_per_class_F1", "parameters", "GFLOPs", "inference_ms",
                  "pipeline_ms", "dataset_fingerprint", "initialization", "seed", "imgsz", "batch",
                  "epochs", "output_dir", "best_weight", "last_weight", "error"]


class Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, text):
        try:
            self.stream.write(text)
        except OSError:
            pass  # 控制台写入失败不影响训练，日志文件仍完整
        self.file.write(text)
        self.file.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.file.flush()
    def isatty(self):
        return False
    @property
    def encoding(self):
        return "utf-8"


def validate_config(cfg, variants, formal):
    if not variants or len(set(variants)) != len(variants) or any(v not in MODEL_ORDER for v in variants):
        raise ValueError("仅支持互不重复的 s/x；run_all 顺序固定为 s → x")
    if formal and cfg.epochs != 200:
        raise ValueError("正式对比实验必须 200 Epoch；功能测试请运行 smoke_test.py")
    if cfg.epochs < 1 or not isinstance(cfg.batch, int) or cfg.batch < 1:
        raise ValueError("epochs、batch 必须是正整数；不允许 AutoBatch")
    if cfg.imgsz < 64 or cfg.imgsz % 32 or cfg.workers < 0 or cfg.cpu_threads < 1:
        raise ValueError("imgsz 须为不小于64的32倍数；workers>=0，cpu_threads>=1")
    if cfg.initialization not in {"scratch", "pretrained"} or str(cfg.optimizer).lower() == "auto":
        raise ValueError("初始化仅支持 scratch/pretrained；须指定固定优化器，不能 auto")
    if cfg.max_det < 1 or not 0 < cfg.conf < 1 or cfg.eval_half:
        raise ValueError("max_det/conf 无效，正式比较最终评价统一 FP32（eval_half=False）")
    for v in variants:
        if not MODEL_FILES[v].is_file():
            raise FileNotFoundError(MODEL_FILES[v])
        if cfg.startup_delay_hours.get(v, 0) < 0:
            raise ValueError("延迟小时不能为负数")
        resume = cfg.resume_checkpoints.get(v, "")
        if resume and not resolved_path(resume).is_file():
            raise FileNotFoundError(f"断点不存在: {resume}")
        weight = resolved_path(cfg.weights_dir) / f"yolo26{v}.pt"
        if cfg.initialization == "pretrained" and not resume and not cfg.download_pretrained and not weight.is_file():
            raise FileNotFoundError(f"缺少预训练权重 {weight}，不会回退随机初始化")


def child_environment(cfg, output):
    env = os.environ.copy()
    env.update({"PYTHONHASHSEED": str(cfg.seed), "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                "YOLO_CONFIG_DIR": str(output / "inputs" / "ultralytics_settings"), "YOLO_OFFLINE": "true",
                "YOLO_AUTOINSTALL": "false", "MPLBACKEND": "Agg", "TINYWOD_FIXED_EXPERIMENT": "1",
                "OMP_NUM_THREADS": str(cfg.cpu_threads), "MKL_NUM_THREADS": str(cfg.cpu_threads),
                "OPENBLAS_NUM_THREADS": str(cfg.cpu_threads), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    return env


def run_child(job_path, cfg, output):
    import codecs
    log = output / "logs" / "console.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([sys.executable, "-u", "-m", "tinywod.worker", str(job_path)],
                               cwd=PROJECT_ROOT, env=child_environment(cfg, output), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    # 用字节块增量读取并即时转发，让 TQDM 的 \r 进度条在原终端原位刷新，而不是按行缓冲后刷屏。
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    try:
        with log.open("a", encoding="utf-8") as f:
            while True:
                block = process.stdout.read1(8192)
                if not block:
                    break
                value = decoder.decode(block)
                sys.stdout.write(value)
                sys.stdout.flush()
                f.write(value)
                f.flush()
            tail = decoder.decode(b"", final=True)
            sys.stdout.write(tail)
            f.write(tail)
        return process.wait()
    except BaseException:
        # 清理本次启动的进程及其 DataLoader 后代，不扫描/结束其他训练任务。
        try:
            import psutil
            parent = psutil.Process(process.pid)
            children = parent.children(recursive=True)
            for child in children:
                child.terminate()
            parent.terminate()
            _, alive = psutil.wait_procs([*children, parent], timeout=5)
            for child in alive:
                child.kill()
        except Exception:
            process.terminate()
        process.wait(timeout=10)
        raise
    finally:
        if process.stdout:
            process.stdout.close()


def delay_start(cfg, variant):
    hours = cfg.startup_delay_hours.get(variant, 0)
    if hours <= 0:
        return
    delay_s = hours * 3600
    start_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + delay_s))
    print(f"[run_yolo26_{variant}] 延迟启动：等待 {hours} 小时（预计 {start_at} 开始）", flush=True)
    # 分段等待，可在 PyCharm Stop 中断；保持上传入口的延迟打印用语。
    deadline = time.monotonic() + delay_s
    while time.monotonic() < deadline:
        time.sleep(min(30, max(0, deadline - time.monotonic())))
    print(f"[run_yolo26_{variant}] 延迟结束，开始训练", flush=True)


def launch(variants=MODEL_ORDER, cfg=CONFIG, formal=True):
    variants = tuple(variants)
    identifier = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    output_root = resolved_path(cfg.output_root)
    session = output_root / "comparisons" / identifier
    session.mkdir(parents=True, exist_ok=False)
    rows = []
    for variant in variants:
        output = output_root / f"YOLO26-{variant.upper()}" / identifier
        for folder in ("weights", "logs", "metrics", "results", "inputs"):
            (output / folder).mkdir(parents=True, exist_ok=False)
        rows.append({"experiment_id": identifier, "formal": formal, "model": f"YOLO26-{variant.upper()}",
                     "status": "pending", "stage": "preflight", "epochs": cfg.epochs, "seed": cfg.seed,
                     "imgsz": cfg.imgsz, "batch": cfg.batch, "initialization": cfg.initialization,
                     "output_dir": str(output)})
    def persist():
        write_csv(session / "summary.csv", rows, SUMMARY_FIELDS)
        write_json(session / "summary.json", {"experiment_id": identifier, "formal": formal, "models": rows})
        # 从不可覆盖的会话表重新建立累计表，避免重复追加；并发只影响此便捷索引。
        cumulative = []
        for path in sorted((output_root / "comparisons").glob("*/summary.csv")):
            with path.open(encoding="utf-8-sig", newline="") as f:
                cumulative.extend(dict(r) for r in csv.DictReader(f))
        write_csv(output_root / "summary.csv", cumulative, SUMMARY_FIELDS)
    interrupted = False
    with (session / "console.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                print(f"\n[实验] {identifier}；顺序 {' → '.join(r['model'] for r in rows)}；各 {cfg.epochs} Epoch", flush=True)
                print(f"[实验] 配置: {PROJECT_ROOT / 'config.py'}；汇总: {session / 'summary.csv'}", flush=True)
                validate_config(cfg, variants, formal)
                write_json(session / "inputs" / "config.json", snapshot(cfg))
                shutil.copy2(PROJECT_ROOT / "config.py", session / "inputs" / "config.py")
                from .dataset import prepare_dataset
                info = prepare_dataset(cfg, session / "inputs")
                jobs = []
                # 所有选择模型都先构建和检查权重，避免训练 S 后才发现 X 结构错误。
                for variant, row in zip(variants, rows):
                    output = Path(row["output_dir"])
                    row["dataset_fingerprint"] = info["fingerprint"]
                    for filename in ("config.json", "config.py", "dataset.yaml", "dataset_manifest.json",
                                     "source_manifest.json", "dataset_info.json"):
                        shutil.copy2(session / "inputs" / filename, output / "inputs" / filename)
                    job = {"variant": variant, "config": asdict(cfg), "dataset": info, "output": str(output),
                           "formal": formal, "action": "inspect"}
                    path = output / "inputs" / "job.json"
                    write_json(path, job)
                    row["start_time"] = now()
                    code = run_child(path, cfg, output)
                    if code:
                        status = read_json(output / "status.json") if (output / "status.json").exists() else {}
                        raise RuntimeError(f"{row['model']} 初始化检查失败: {status.get('error', '子进程退出码 '+str(code))}")
                    jobs.append((job, path))
                for index, ((job, path), row) in enumerate(zip(jobs, rows)):
                    variant, output = job["variant"], Path(job["output"])
                    delay_start(cfg, variant)
                    job["action"] = "train"
                    write_json(path, job)
                    row.update(status="running", stage="training", start_time=now())
                    persist()
                    started = time.monotonic()
                    code = run_child(path, cfg, output)
                    status = read_json(output / "status.json") if (output / "status.json").exists() else {}
                    row.update({k: v for k, v in status.items() if k in SUMMARY_FIELDS})
                    row.update(end_time=now(), elapsed_s=round(time.monotonic()-started, 3))
                    if code != 0 or status.get("status") != "success":
                        row.update(status="failed", error=status.get("error", f"子进程异常退出，code={code}；见 console.log"))
                    persist()
                    print(f"[完成] {row['model']} 状态={row['status']}，用时={row['elapsed_s']}秒，输出={output}", flush=True)
                    if row["status"] != "success" and not cfg.continue_on_failure:
                        for remaining in rows[index+1:]:
                            remaining.update(status="skipped", stage="not_started", error="前一模型失败，continue_on_failure=False")
                        break
            except BaseException as exc:
                interrupted = isinstance(exc, KeyboardInterrupt)
                message = f"{type(exc).__name__}: {exc}"
                trace = traceback.format_exc()
                (session / "error.log").write_text(trace, encoding="utf-8")
                print(trace, flush=True)
                for row in rows:
                    if row["status"] in {"pending", "running"}:
                        row.update(status="interrupted" if interrupted else "failed", error=message, end_time=now())
                        write_json(Path(row["output_dir"]) / "status.json", row)
                        (Path(row["output_dir"]) / "logs" / "error.log").write_text(trace, encoding="utf-8")
            finally:
                persist()
                print("\n[最终实验汇总]", flush=True)
                for row in rows:
                    print(f"  {row['model']}: {row['status']} | P={row.get('precision', '未计算')} "
                          f"R={row.get('recall', '未计算')} mAP50={row.get('mAP50', '未计算')} "
                          f"mAP50-95={row.get('mAP50_95', '未计算')} | {row['output_dir']}", flush=True)
                print(f"[CSV] {session / 'summary.csv'}\n[CSV] 累计索引 {output_root / 'summary.csv'}", flush=True)
    return 130 if interrupted else (0 if rows and all(r["status"] == "success" for r in rows) else 1)
