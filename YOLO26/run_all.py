"""PyCharm 直接 Run：YOLO26-S → YOLO26-X，各 200 Epoch，独立子进程。"""
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    from config import MODEL_ORDER
    from tinywod.zombie import cleanup_zombie_workers
    cleanup_zombie_workers(tag="run_all")
    from tinywod.runner import launch
    raise SystemExit(launch(MODEL_ORDER))
