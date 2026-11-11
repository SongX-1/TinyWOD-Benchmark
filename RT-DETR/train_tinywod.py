"""PyCharm 正式入口：RT-DETR-R50vd / TinyWOD / 默认 72 Epoch。"""
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    from tinywod.runner import launch
    raise SystemExit(launch()["exit_code"])
