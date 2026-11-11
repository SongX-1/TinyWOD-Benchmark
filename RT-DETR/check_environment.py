"""PyCharm Run：一次检查依赖、GPU 运算及本地权重路径，不启动正式训练。"""
if __name__ == "__main__":
    from datetime import datetime
    from pathlib import Path
    import traceback
    import uuid
    from config import CONFIG, PROJECT_ROOT, resolve
    from tinywod.environment import setup_environment
    output = PROJECT_ROOT/"checks"/(datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8])
    try:
        setup_environment(CONFIG, output)
        print(f"[模型配置] {resolve(CONFIG.model_config)}")
        print(f"[数据目录] YOLO={resolve(CONFIG.yolo_root)} COCO={resolve(CONFIG.coco_root)}")
        if not resolve(CONFIG.model_config).is_file():
            raise FileNotFoundError(resolve(CONFIG.model_config))
        if not any(resolve(path).is_dir() for path in (CONFIG.yolo_root, CONFIG.coco_root)):
            raise FileNotFoundError("数据目录尚未配置到有效位置；请在 config.py 修改 yolo_root / coco_root。")
        print(f"[检查通过] 依赖、当前设备基础运算、本地权重文件、模型配置及数据根目录。记录：{output}")
        print("完整类别、图像与标签检查在正式入口执行。")
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
