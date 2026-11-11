"""PyCharm：选择本文件，点击绿色 Run。全部参数来自 config.py。"""
from multiprocessing import freeze_support


if __name__ == "__main__":
    freeze_support()
    from tinywod.runner import launch
    raise SystemExit(launch())
