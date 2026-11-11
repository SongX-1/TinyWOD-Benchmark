"""Restore optimizer/EMA/RNG and committed CSV history without overwriting a run."""
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from config import snapshot
from .common import digest
from .recording import restore_csv_history

FORMAT = "rtdetr_seadronesee_v1"  # 保持旧值以兼容既有检查点
CRITICAL = ("model_config", "epochs", "image_size", "train_scales", "augmentation", "batch_size", "val_batch_size",
            "num_workers", "pin_memory", "cpu_threads", "seed", "mixed_precision", "deterministic", "allow_tf32",
            "learning_rate", "backbone_learning_rate", "weight_decay", "betas", "lr_milestones", "lr_gamma",
            "clip_max_norm", "use_ema", "ema_decay", "ema_warmups", "max_det", "coco_max_dets", "metric_conf",
            "confusion_conf", "confusion_iou", "initialization")


def signature(cfg):
    values = snapshot(cfg)
    return digest({key: values[key] for key in CRITICAL})


def rng_state(generators):
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loaders": {key: generator.get_state() for key, generator in generators.items()}}


def restore_rng(state, generators):
    random.setstate(state["python"])
    a, b, c, d, e = state["numpy"]
    np.random.set_state((a, np.asarray(b, dtype=np.uint32), c, d, e))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("恢复文件的 CUDA 设备数量与当前环境不同，不能保证随机状态一致。")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    for key, generator in generators.items():
        generator.set_state(state["loaders"][key].cpu())


def state_dict(cfg, output, epoch, model, optimizer, scheduler, ema, scaler, generators, info, best_epoch, best_ap, initialization):
    return {"format": FORMAT, "run_dir": str(Path(output).resolve()), "epoch": epoch,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "ema": ema.state_dict() if ema is not None else None, "scaler": scaler.state_dict(),
            "rng": rng_state(generators), "config": snapshot(cfg), "critical_signature": signature(cfg),
            "dataset_fingerprint": info["fingerprint"], "class_names": list(info["names"]),
            "best_epoch": best_epoch, "best_AP": best_ap, "initialization": initialization}


def load_resume(cfg, output, checkpoint_path, model, optimizer, scheduler, ema, scaler, generators, info):
    checkpoint_path, output = Path(checkpoint_path), Path(output)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if state.get("format") != FORMAT:
        raise ValueError("resume_checkpoint 必须是本项目保存的完整恢复文件；官方预训练文件请使用 initialization 配置。")
    if state["critical_signature"] != signature(cfg):
        previous, current = state["config"], snapshot(cfg)
        changed = [key for key in CRITICAL if digest(previous[key]) != digest(current[key])]
        raise ValueError(f"恢复时关键实验参数改变：{changed}。请恢复原参数，避免实验条件被静默修改。")
    if state["dataset_fingerprint"] != info["fingerprint"] or state["class_names"] != info["names"]:
        raise ValueError("恢复文件的数据指纹或类别顺序与当前数据不一致。")
    epoch = int(state["epoch"])
    if not 0 < epoch <= cfg.epochs:
        raise ValueError(f"恢复轮数非法：{epoch}")
    nearby = checkpoint_path.parent.parent
    source = nearby if (nearby/"results.csv").is_file() else Path(state["run_dir"])
    if not (source/"results.csv").is_file():
        raise FileNotFoundError("恢复训练需要保留原运行目录及 CSV，以完整继承历史；当前只能找到单独权重，未修改原实验。")
    best_source = source/"weights"/"best.pth"
    best_state = torch.load(best_source, map_location="cpu", weights_only=True)
    if int(best_state["epoch"]) != int(state["best_epoch"]) or best_state["best_AP"] != state["best_AP"]:
        raise ValueError("来源 best.pth 与所选恢复轮次记录不匹配。请从最新 last.pth 或对应 best.pth 恢复，避免错误继承未来最佳结果。")
    del best_state
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    # Optimizer.load_state_dict casts tensor states to each parameter's device.
    scheduler.load_state_dict(state["scheduler"])
    if ema is not None:
        ema.load_state_dict(state["ema"])
    elif state["ema"] is not None:
        raise ValueError("恢复文件与当前 EMA 设置不一致")
    scaler.load_state_dict(state["scaler"])
    restore_csv_history(source, output, epoch)
    (output/"weights").mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_source, output/"weights"/"best.pth")
    # Also preserve a valid last checkpoint if no further training is needed.
    shutil.copy2(checkpoint_path, output/"weights"/"last.pth")
    # Restore after all initialization/history work, immediately before training.
    restore_rng(state["rng"], generators)
    result = {"completed_epochs": epoch, "best_epoch": int(state["best_epoch"]), "best_AP": float(state["best_AP"]),
              "source_run": str(source), "initialization": state["initialization"], "rng": state["rng"]}
    print(f"[断点恢复] 已提交 Epoch={epoch}/{cfg.epochs} | best_epoch={result['best_epoch']} AP={result['best_AP']:.6f}", flush=True)
    print(f"[历史继承] {source} → {output}；只继承已提交轮次，原目录不覆盖。", flush=True)
    return result
