"""与 train_tinywod.py 相同，明确标识原版 RT-DETR 的 R50vd。"""
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    from tinywod.runner import launch
    raise SystemExit(launch()["exit_code"])
