"""当前交付只有 H-DETR-R50；与单模型入口使用同一训练流程。"""
from multiprocessing import freeze_support


if __name__ == "__main__":
    freeze_support()
    from tinywod.runner import launch
    raise SystemExit(launch())
