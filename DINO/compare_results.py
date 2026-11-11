# -*- coding: utf-8 -*-
"""PyCharm直接Run：汇总DINO各次实验，及experiment_config.py中可选的OWT结果CSV。"""
from experiment_config import CONFIG
from tinywod.comparison import build_comparison

if __name__=='__main__':
    build_comparison(CONFIG)
