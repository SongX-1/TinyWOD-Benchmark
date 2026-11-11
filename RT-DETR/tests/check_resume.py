"""Real R50vd training: compare 3 uninterrupted epochs against 1 + resume 2.

Synthetic data and CPU only. Requires the bundled local backbone. Saves large
temporary checkpoints under verification_runs, never under formal runs.
"""
from dataclasses import replace
from datetime import datetime
import csv
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def equivalent(left, right, path=""):
    import torch
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right), path
        return 1
    if isinstance(left, dict):
        assert set(left) == set(right), path
        return sum(equivalent(left[key], right[key], path+"/"+str(key)) for key in left)
    if isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        return sum(equivalent(a, b, path+f"/{i}") for i, (a, b) in enumerate(zip(left, right)))
    assert left == right, f"{path}: {left!r} != {right!r}"
    return 0


def main():
    import torch
    from config import CONFIG
    from tinywod.common import write_json
    from tinywod.synthetic import make_dataset
    from tinywod.runner import launch
    import tinywod.engine as engine
    identifier = datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8]
    directory = ROOT/"verification_runs"/"resume"/identifier
    dataset = make_dataset(directory/"synthetic_dataset")
    cfg = replace(CONFIG, yolo_root=str(dataset), coco_root=str(dataset/"unused"), dataset_mode="yolo", epochs=3,
                  image_size=160, train_scales=(160, 192), batch_size=1, val_batch_size=1, num_workers=0,
                  cpu_threads=2, device="cpu", mixed_precision="no", initialization="backbone_pretrained", resume_checkpoint="",
                  output_root=str(directory/"baseline"), save_period=0, profile_flops=False, tensorboard=False,
                  benchmark_images=1, benchmark_warmup=1, benchmark_repeats=1, predict_unlabeled_test=False)
    baseline = launch(cfg, verification=True)
    assert baseline["exit_code"] == 0, baseline
    original_train = engine.train_epoch

    def interrupt(*args, **kwargs):
        # epoch is the 9th positional argument in train_epoch.
        epoch = kwargs.get("epoch", args[8] if len(args) > 8 else None)
        if epoch == 2:
            raise RuntimeError("INTENTIONAL_TEST_STOP_AFTER_COMMITTED_EPOCH_1")
        return original_train(*args, **kwargs)

    engine.train_epoch = interrupt
    try:
        interrupted = launch(replace(cfg, output_root=str(directory/"interrupted")), verification=True)
    finally:
        engine.train_epoch = original_train
    assert interrupted["exit_code"] == 1
    assert interrupted["summary"]["status"] == "failed"
    assert interrupted["summary"]["completed_epochs"] == 1
    resumed = launch(replace(cfg, output_root=str(directory/"resumed"),
                             resume_checkpoint=str(Path(interrupted["output"])/"weights"/"last.pth")), verification=True)
    assert resumed["exit_code"] == 0, resumed
    a = torch.load(Path(baseline["output"])/"weights"/"last.pth", map_location="cpu", weights_only=True, mmap=True)
    b = torch.load(Path(resumed["output"])/"weights"/"last.pth", map_location="cpu", weights_only=True, mmap=True)
    tensors = {}
    for key in ("model", "ema", "optimizer", "scheduler", "scaler", "rng"):
        tensors[key] = equivalent(a[key], b[key], key)
    def rows(output, name):
        with (Path(output)/name).open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    left, right = rows(baseline["output"], "results.csv"), rows(resumed["output"], "results.csv")
    assert [int(row["epoch"]) for row in right] == [1, 2, 3]
    checked_fields = [key for key in left[0] if key.startswith(("train/", "train_raw/", "val/", "val_raw/", "lr/"))]
    checked_fields += ["AP", "AP50", "APsmall", "precision", "recall", "f1", "best_epoch", "best_AP"]
    for epoch, (x, y) in enumerate(zip(left, right), 1):
        for field in checked_fields:
            assert x[field] == y[field], (epoch, field, x[field], y[field])
    train_rows = rows(resumed["output"], "metrics/train_batches.csv")
    keys = [(row["epoch"], row["batch"]) for row in train_rows]
    assert len(keys) == len(set(keys)) == 9
    report = {"status": "passed", "data": "synthetic", "device": "cpu", "epochs": 3,
              "baseline": baseline["output"], "intentional_failure": interrupted["output"], "resumed": resumed["output"],
              "bitwise_equal_tensor_counts": tensors, "identical_result_fields": checked_fields,
              "unique_train_batch_rows": len(keys), "note": "Not a TinyWOD accuracy experiment or GPU validation."}
    write_json(directory/"resume_check_report.json", report)
    print("PASS interrupted status + 3-epoch bitwise model/EMA/optimizer/scheduler/scaler/RNG resume + CSV history", tensors)
    print(directory/"resume_check_report.json")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
