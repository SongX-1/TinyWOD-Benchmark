# -*- coding: utf-8 -*-
"""直接 Run：检查 OWT 同源 TinyWOD，写入本项目的 COCO 类别映射缓存。"""
from experiment_config import CONFIG
from tinywod.data import prepare_dataset

if __name__ == "__main__":
    prepare_dataset(CONFIG)
