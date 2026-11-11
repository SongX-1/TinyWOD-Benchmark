"""本次只训练配置指定的一个 VMINet 变体，默认 B。"""
from multiprocessing import freeze_support

if __name__=="__main__":
    freeze_support()
    from tinywod.runner import launch
    raise SystemExit(launch())
