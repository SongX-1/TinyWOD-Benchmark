"""本次仅要求一个 R50 版本；直接 Run 即执行这一项完整实验。"""
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    from tinywod.runner import launch
    raise SystemExit(launch()["exit_code"])
