"""PyCharm 绿色 Run：按 config.py 启动训练；不检查/安装依赖。"""
from multiprocessing import freeze_support

if __name__=="__main__":
    freeze_support()
    from tinywod.runner import launch
    raise SystemExit(launch())
