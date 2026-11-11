"""PyCharm直接Run：合成数据S模型，中断1/3轮再恢复至3轮，并检查有GT的独立test。"""
import csv
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path

from config import CONFIG, PROJECT_ROOT
from tinywod.common import read_json, write_json


def main():
    from smoke_test import make_synthetic_dataset
    from tinywod.dataset import prepare_dataset
    from tinywod.runner import child_environment, run_child
    root = PROJECT_ROOT / "verification" / ("resume_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    raw = make_synthetic_dataset(root / "raw", labeled_test=True)
    cfg = replace(CONFIG, dataset_mode="coco", raw_dataset_root=str(raw), prepared_dataset_root=str(root / "converted"),
                  output_root=str(root / "runs"), epochs=3, imgsz=128, batch=2, workers=0, device="cpu", amp=False,
                  initialization="scratch", resume_checkpoints={"s": "", "x": ""}, startup_delay_hours={"s": 0, "x": 0},
                  benchmark_images=0, profile_flops=False, save_period=1)
    info = prepare_dataset(cfg, root / "inputs")
    def make_job(label, configuration):
        output = root / label
        for part in ("weights", "logs", "metrics", "results", "inputs"):
            (output / part).mkdir(parents=True, exist_ok=False)
        job = {"variant": "s", "config": asdict(configuration), "dataset": info, "output": str(output),
               "formal": False, "action": "train"}
        path = output / "inputs" / "job.json"
        write_json(path, job)
        return output, path
    first, job_path = make_job("interrupted", cfg)
    print("[恢复测试] 合成数据，不是论文实验。故意在保存第1轮后中断。", flush=True)
    with (first / "logs" / "console.log").open("w", encoding="utf-8") as f:
        result = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--interrupt-once", str(job_path)],
                                cwd=PROJECT_ROOT, env=child_environment(cfg, first), stdout=f, stderr=subprocess.STDOUT)
    assert result.returncode == 130, f"未在预期位置中断，见 {first / 'logs/console.log'}"
    assert read_json(first / "status.json")["completed_epochs"] == 1
    last = first / "weights" / "last.pt"
    resumed_cfg = replace(cfg, resume_checkpoints={"s": str(last), "x": ""})
    resumed, path = make_job("resumed", resumed_cfg)
    code = run_child(path, resumed_cfg, resumed)
    assert code == 0 and read_json(resumed / "status.json")["status"] == "success"
    for name in ("results.csv", "metrics/epochs.csv"):
        with (resumed / name).open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        assert [int(row["epoch"]) for row in rows] == [1, 2, 3], f"重复/缺失轮次: {name}"
        assert [float(row["time"]) for row in rows] == sorted(float(row["time"]) for row in rows)
    metrics = read_json(resumed / "metrics" / "final_metrics.json")
    assert metrics["evaluation_split"] == "test" and metrics["split"] == "test"
    for split in ("val", "test"):
        for name in ("metrics.json", "metrics.csv", "per_class.csv", "per_image.csv", "confusion_matrix.csv", "ap_and_curves.npz"):
            assert (resumed / "metrics" / split / name).is_file()
    os.environ.update({"YOLO_CONFIG_DIR": str(resumed / "inputs" / "ultralytics_settings"), "YOLO_OFFLINE": "true"})
    import torch
    checkpoint = torch.load(resumed / "weights" / "last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 2 and checkpoint["optimizer"] is not None and checkpoint["scheduler"] is not None
    assert checkpoint["model"] is not None and checkpoint["ema"] is not None
    report = {"passed": True, "synthetic_only": True, "interrupted_after_epoch": 1, "resumed_total_epochs": 3,
              "epoch_rows": [1, 2, 3], "labeled_test_evaluation": True, "native_and_extended_csv": True,
              "checkpoint_model_ema_optimizer_scheduler_rng": True, "bitwise_resume_equivalence": "not_claimed",
              "root": str(root), "first": str(first), "resumed": str(resumed)}
    write_json(root / "resume_test_report.json", report)
    print(f"[恢复测试] 通过，报告: {root / 'resume_test_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    if len(sys.argv) == 3 and sys.argv[1] == "--interrupt-once":
        from tinywod.trainer import ComparisonTrainer
        from tinywod.worker import execute
        original = ComparisonTrainer.epoch_end
        def interrupt_after_first_epoch(self, trainer):
            original(self, trainer)
            if self.epoch == 0:
                raise KeyboardInterrupt("synthetic test: intentional interruption after completed checkpoint")
        ComparisonTrainer.epoch_end = interrupt_after_first_epoch
        raise SystemExit(execute(read_json(sys.argv[2])))
    raise SystemExit(main())
