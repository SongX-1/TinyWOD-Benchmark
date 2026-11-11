# -*- coding: utf-8 -*-
"""增量训练入口：从最近的 last.pth 续训到目标轮数，不从头开始。

用法：PyCharm 直接运行本文件，或在命令行：
    python train_incremental.py

它会自动：
  1. 在 runs/DINO-R50/*run/weights/ 里找到 mtime 最新的 last.pth；
  2. 把该 checkpoint 设为 resume，延续实验条件（lr_drop 等保持不变）；
  3. 从断点续训到 experiment_config.py 里的 epochs（当前 36）。
"""
import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

TARGET_EPOCHS = 48


def latest_checkpoint(output_root):
    from pathlib import Path
    root = Path(output_root)
    candidates = sorted(
        root.glob("*run/weights/last.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"在 {root} 下找不到任何 last.pth，请先用 train_dino_r50.py 跑出至少一个完整 checkpoint。"
        )
    return candidates[0]


def main():
    import multiprocessing
    multiprocessing.freeze_support()
    from experiment_config import CONFIG
    from tinywod.runner import run_experiment

    ckpt = latest_checkpoint(CONFIG.output_root)
    CONFIG.resume = str(ckpt)
    CONFIG.epochs = TARGET_EPOCHS
    print(f"[incremental] 从 {ckpt} 续训到 {TARGET_EPOCHS} 轮（lr_drop={CONFIG.lr_drop} 保持不变，以满足断点校验）",
          flush=True)
    return run_experiment(CONFIG)


if __name__ == "__main__":
    main()
