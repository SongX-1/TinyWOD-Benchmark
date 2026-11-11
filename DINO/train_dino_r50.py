# -*- coding: utf-8 -*-
"""PyCharm 绿色 Run：TinyWOD 原图输入的 DINO-R50 训练入口。"""
import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def main():
    import multiprocessing
    from experiment_config import CONFIG
    multiprocessing.freeze_support()
    from tinywod.runner import run_experiment
    return run_experiment(CONFIG)


if __name__ == "__main__":
    main()
