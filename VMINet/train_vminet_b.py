"""PyCharm 独立 B 入口；配置仍全部来自 config.py。"""
from multiprocessing import freeze_support

if __name__=="__main__":
    freeze_support()
    from dataclasses import replace
    from config import CONFIG
    from tinywod.runner import launch
    raise SystemExit(launch(replace(CONFIG,variant="b")))
