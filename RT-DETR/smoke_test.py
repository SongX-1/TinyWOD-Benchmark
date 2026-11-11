"""直接 Run：合成数据 / CPU / 1 Epoch，检查真实 R50vd 前后向及全部保存链路。

本入口明确使用随机初始化和小图，仅用于软件测试；正式入口仍为 72 Epoch 和本地预训练。
"""
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    from dataclasses import replace
    from datetime import datetime
    import uuid
    from config import CONFIG, PROJECT_ROOT
    from tinywod.synthetic import make_dataset
    from tinywod.runner import launch
    identifier = datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8]
    dataset = make_dataset(PROJECT_ROOT/"verification_runs"/"synthetic_inputs"/identifier)
    cfg = replace(CONFIG, yolo_root=str(dataset), coco_root=str(dataset/"not_used"), dataset_mode="yolo", epochs=1,
                  image_size=160, train_scales=(160, 192), batch_size=1, val_batch_size=1, num_workers=0,
                  cpu_threads=2, device="cpu", mixed_precision="no", initialization="scratch", resume_checkpoint="",
                  output_root=str(PROJECT_ROOT/"verification_runs"/"smoke"), save_period=0, profile_flops=True,
                  benchmark_images=2, benchmark_warmup=1, benchmark_repeats=1, tensorboard=True)
    result = launch(cfg, verification=True)
    print(f"[冒烟测试] status={result['summary']['status']} | {result['output']}")
    raise SystemExit(result["exit_code"])
