"""PyCharm 直接 Run：仅训练 YOLO26-X，共 200 Epoch。"""
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    from tinywod.zombie import cleanup_zombie_workers
    cleanup_zombie_workers(tag="run_yolo26_x")
    from tinywod.runner import launch
    raise SystemExit(launch(("x",)))
