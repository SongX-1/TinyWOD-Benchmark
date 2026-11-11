"""内部子进程入口。用户只运行项目根目录的三个入口，不需要命令行参数。"""
import importlib.metadata
import os
import platform
import sys
import time
import traceback
from pathlib import Path

from config import MODEL_FILES, OFFICIAL_WEIGHTS, PROJECT_ROOT, ExperimentConfig, resolved_path
from .common import digest, now, read_json, sha256, write_json
from .zombie import terminate_own_children


def environment(cfg, output):
    output = Path(output)
    os.environ.update({"YOLO_CONFIG_DIR": str(output / "inputs" / "ultralytics_settings"),
                       "YOLO_OFFLINE": "true", "YOLO_AUTOINSTALL": "false", "MPLBACKEND": "Agg",
                       "TINYWOD_FIXED_EXPERIMENT": "1", "ULTRALYTICS_API_KEY": ""})
    import cv2
    import torch
    import ultralytics
    from ultralytics import settings
    from ultralytics.utils import USER_CONFIG_DIR
    if Path(ultralytics.__file__).resolve() != (PROJECT_ROOT / "ultralytics/__init__.py").resolve():
        raise ImportError("导入了其他 ultralytics 安装，必须使用本工程源码。请在 PyCharm 打开整个项目。")
    if ultralytics.__version__ != "8.4.182":
        raise ImportError(f"本工程固定8.4.182，实际{ultralytics.__version__}，请勿覆盖源码")
    settings.update({k: False for k in ("sync", "clearml", "comet", "dvc", "mlflow", "raytune", "tensorboard", "wandb")})
    settings.update({"api_key": ""})
    torch.set_num_threads(cfg.cpu_threads)
    torch.set_num_interop_threads(max(1, min(2, cfg.cpu_threads)))
    if hasattr(torch.backends, "nnpack"):
        torch.backends.nnpack.set_flags(False)
    cv2.setNumThreads(0)
    if cfg.device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    else:
        device = str(cfg.device).lower().removeprefix("cuda:")
    if device != "cpu" and (not device.isdigit() or not torch.cuda.is_available() or int(device) >= torch.cuda.device_count()):
        raise ValueError(f"设备 {cfg.device} 不可用；仅支持单张 CUDA GPU 或 CPU，不静默切换")
    packages = {}
    for name in ("torch", "torchvision", "numpy", "Pillow", "PyYAML", "scipy", "matplotlib",
                 "pandas", "polars", "psutil", "ultralytics-thop", "nvidia-ml-py"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    # opencv 实际以 opencv-python-headless 安装，取模块真实版本，不依赖发行包名。
    packages["opencv-python"] = cv2.__version__
    data = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "ultralytics": ultralytics.__version__, "ultralytics_source": ultralytics.__file__, "device": device,
            "device_name": torch.cuda.get_device_name(int(device)) if device != "cpu" else platform.processor() or "CPU",
            "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "requested_workers": cfg.workers, "effective_workers": cfg.workers if device != "cpu" else 0,
            "amp": cfg.amp and device != "cpu", "packages": packages, "captured_at": now()}
    write_json(output / "inputs" / "environment.json", data)
    # 使用 matplotlib 自带字体，禁止绘图阶段临时联网取 Arial。
    import shutil
    from matplotlib import font_manager
    font = font_manager.findfont("DejaVu Sans")
    directory = USER_CONFIG_DIR
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("Arial.ttf", "Arial.Unicode.ttf"):
        shutil.copy2(font, directory / name)
    print(f"[环境] Python {platform.python_version()}，Ultralytics {ultralytics.__version__}，"
          f"PyTorch {torch.__version__}\n[设备] {device}: {data['device_name']}，CUDA build={data['cuda_build']}", flush=True)
    return data


def structure(model):
    """实际模块类型、参数形状及数量；不同文件名不构成不同模型的证明。"""
    modules = [(name, type(module).__module__ + "." + type(module).__name__) for name, module in model.named_modules()]
    shapes = [(name, list(value.shape)) for name, value in model.state_dict().items()]
    return {"scale": model.yaml.get("scale"), "nc": len(model.names), "head": type(model.model[-1]).__name__,
            "reg_max": model.model[-1].reg_max, "end2end": model.end2end,
            "parameters": sum(p.numel() for p in model.parameters()), "signature": digest([modules, shapes]),
            "modules": modules, "parameter_shapes": shapes}


def choose_model(cfg, variant, output):
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(cfg.seed, deterministic=cfg.deterministic)
    model_file = MODEL_FILES[variant]
    expected = YOLO(str(model_file), task="detect")
    expected.model.end2end = True
    spec = structure(expected.model)
    if spec["scale"] != variant or spec["head"] != "Detect" or spec["reg_max"] != 1 or not spec["end2end"]:
        raise ValueError(f"不是正确的 YOLO26-{variant.upper()} 结构: {spec['scale']}, {spec['head']}")
    record = {"model": f"YOLO26-{variant.upper()}", "initialization": cfg.initialization,
              "model_yaml": str(model_file), "yaml_sha256": sha256(model_file), "expected_structure": spec,
              "source_version": "ultralytics==8.4.182", "weight": None, "weight_sha256": None}
    resume = cfg.resume_checkpoints.get(variant, "")
    if resume:
        record.update(initialization="resume", weight=str(resolved_path(resume)), weight_sha256=sha256(resolved_path(resume)))
        loaded = YOLO(record["weight"], task="detect")
        reference = DetectionModel(str(model_file), nc=len(loaded.names), verbose=False)
        if structure(reference)["signature"] != structure(loaded.model)["signature"]:
            raise ValueError("恢复权重实际网络结构与当前 S/X 配置不符")
        # 训练器还会检查数据和超参数指纹，使用完整训练状态继续。
        del reference, loaded
    elif cfg.initialization == "pretrained":
        destination = resolved_path(cfg.weights_dir) / f"yolo26{variant}.pt"
        if not destination.is_file() and cfg.download_pretrained:
            import urllib.request
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".pt.download")
            try:
                urllib.request.urlretrieve(OFFICIAL_WEIGHTS[variant], temporary)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        if not destination.is_file():
            raise FileNotFoundError(f"缺少预训练权重 {destination}")
        loaded = YOLO(str(destination), task="detect")
        reference = DetectionModel(str(model_file), nc=len(loaded.names), verbose=False)
        if structure(reference)["signature"] != structure(loaded.model)["signature"]:
            raise ValueError(f"{destination} 不是对应 YOLO26-{variant.upper()}，请勿重命名其他模型冒充")
        record.update(weight=str(destination), weight_sha256=sha256(destination), declared_source=OFFICIAL_WEIGHTS[variant],
                      checksum_note="本地文件SHA256留档；不声称官方签名认证")
        expected = loaded
        del reference
    write_json(Path(output) / "inputs" / "initialization.json", record)
    print(f"[模型] YOLO26-{variant.upper()}；scale={variant}；完整 COCO 头参数={spec['parameters']:,}\n"
          f"[模型配置] {model_file}\n[初始化] {record['initialization']}；权重={record['weight'] or '无（随机初始化）'}", flush=True)
    return expected, record


def train_arguments(cfg, info, output, env, variant):
    output = Path(output)
    common = ("epochs", "imgsz", "batch", "seed", "deterministic", "optimizer", "lr0", "lrf", "momentum",
              "weight_decay", "warmup_epochs", "warmup_momentum", "warmup_bias_lr", "cos_lr", "nbs", "box", "cls",
              "dfl", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective", "flipud",
              "fliplr", "mosaic", "mixup", "copy_paste", "save_period", "conf", "max_det")
    args = {key: getattr(cfg, key) for key in common}
    args.update(data=info["yaml"], project=str(output.parent), name=output.name, exist_ok=True,
                device=env["device"], workers=env["effective_workers"], amp=env["amp"], patience=0,
                pretrained=cfg.initialization == "pretrained", resume=str(resolved_path(cfg.resume_checkpoints[variant]))
                if cfg.resume_checkpoints.get(variant) else False,
                val=True, split="val", save=True, plots=True, rect=False, cache=False, close_mosaic=0,
                multi_scale=0.0, fraction=1.0, nms=False, quantize=None, channels_last=False, compile=False,
                save_json=False, save_txt=False, verbose=True)
    return args


def execute(job):
    from functools import partial
    from config import CLASS_NAMES
    cfg = ExperimentConfig(**job["config"])
    output, variant, info = Path(job["output"]), job["variant"], job["dataset"]
    state = {"model": f"YOLO26-{variant.upper()}", "status": "running", "stage": job["action"],
             "start_time": now(), "output_dir": str(output)}
    write_json(output / "status.json", state)
    started = time.monotonic()
    trainer = None
    try:
        env = environment(cfg, output)
        model, init = choose_model(cfg, variant, output)
        if job["action"] == "inspect":
            # 提前验证断点与所有配置，避免运行 S 后才发现 X 的恢复配置错误。
            if cfg.resume_checkpoints.get(variant):
                import torch
                from .common import experiment_signature
                ckpt = torch.load(resolved_path(cfg.resume_checkpoints[variant]), map_location="cpu", weights_only=False)
                meta = ckpt.get("sds", {})
                if (meta.get("variant"), meta.get("dataset_fingerprint"), meta.get("experiment_signature")) != (
                        variant, info["fingerprint"], experiment_signature(cfg)):
                    raise ValueError("断点与当前模型、数据或训练配置不一致")
                if ckpt.get("optimizer") is None or ckpt.get("model") is None or not 0 < ckpt.get("epoch", -1)+1 < cfg.epochs:
                    raise ValueError("该断点无法继续训练（已完成、缺少优化器或不是完整 last.pt）")
                companion = resolved_path(cfg.resume_checkpoints[variant]).parent / "best.pt"
                if not companion.is_file():
                    raise FileNotFoundError(f"恢复缺少匹配的最佳模型: {companion}")
                best_meta = torch.load(companion, map_location="cpu", weights_only=False)
                if best_meta.get("epoch", -1) > ckpt["epoch"] or any(
                        best_meta.get("sds", {}).get(k) != meta.get(k)
                        for k in ("variant", "dataset_fingerprint", "experiment_signature")):
                    raise ValueError("best.pt 与所选恢复断点不匹配")
            state.update(status="ready", stage="preflight_passed", end_time=now())
            write_json(output / "status.json", state)
            return 0
        import torch
        from .trainer import ComparisonTrainer, ComparisonValidator
        from .reporting import benchmark, model_complexity, predict_unlabeled, save_evaluation
        files = [*PROJECT_ROOT.glob("*.py"), *PROJECT_ROOT.glob("*.toml"), *PROJECT_ROOT.glob("requirements*.txt"),
                 *PROJECT_ROOT.joinpath("models").glob("*.yaml"), *PROJECT_ROOT.joinpath("ultralytics/cfg").rglob("*.yaml"),
                 *PROJECT_ROOT.joinpath("tinywod").glob("*.py"),
                 *PROJECT_ROOT.joinpath("ultralytics").rglob("*.py")]
        write_json(output / "inputs" / "code_hashes.json", {p.relative_to(PROJECT_ROOT).as_posix(): sha256(p)
                   for p in sorted(set(files)) if p.is_file()})
        args = train_arguments(cfg, info, output, env, variant)
        factory = partial(ComparisonTrainer, experiment_cfg=cfg, dataset_info=info, variant=variant)
        print(f"[开始] {state['model']} | {now()} | Epoch={cfg.epochs} | batch={cfg.batch} | imgsz={cfg.imgsz}\n"
              f"[输出] 权重: {output / 'weights'}\n[输出] 日志: {output / 'logs'}\n"
              f"[输出] CSV: {output / 'metrics/epochs.csv'}；原生: {output / 'results.csv'}", flush=True)
        model.train(trainer=factory, **args)
        trainer = model.trainer
        completed = len(trainer.history)
        if completed != cfg.epochs or trainer.epoch+1 != cfg.epochs:
            raise RuntimeError(f"仅完成 {completed}/{cfg.epochs} Epoch，不能标记成功")
        if len(trainer.model.names) != len(CLASS_NAMES) or list(trainer.model.names.values()) != list(CLASS_NAMES):
            raise ValueError("实际训练类别头与数据声明不一致")
        write_json(output / "inputs" / "trained_structure.json", structure(trainer.model))
        complexity = model_complexity(trainer.model, cfg, output)
        best, last = output / "weights" / "best.pt", output / "weights" / "last.pt"
        if not best.is_file() or not last.is_file():
            raise FileNotFoundError("训练未生成 best.pt 或 last.pt")
        state.update(stage="evaluation", completed_epochs=completed)
        write_json(output / "status.json", state)
        # 先按 val 选定 best，再一次性评价独立 test；从不把 test 传入训练器。
        evaluations = {}
        for split in ("val", "test"):
            if split == "test" and not info["splits"]["test"]["labeled"]:
                continue
            validator = ComparisonValidator(save_dir=output / "results" / f"{split}_best", args={
                "model": str(best), "data": info["yaml"], "split": split, "imgsz": cfg.imgsz, "batch": cfg.batch,
                "workers": env["effective_workers"], "device": env["device"], "conf": cfg.conf, "max_det": cfg.max_det,
                "nms": False, "rect": False, "quantize": None, "channels_last": False,
                "plots": True, "save_json": False, "save_txt": True, "save_conf": True, "verbose": True})
            validator(model=str(best))
            evaluations[split] = save_evaluation(validator, output / "metrics" / split, split)
            del validator
        if info["splits"]["test"]["images"] and not info["splits"]["test"]["labeled"] and cfg.predict_unlabeled_test:
            predict_unlabeled(best, cfg, info, output, env["device"])
        state.update(stage="benchmark")
        write_json(output / "status.json", state)
        speed = benchmark(best, cfg, info, output, env["device"])
        selected_split = info["final_evaluation_split"]
        final = {**evaluations[selected_split], **complexity, "benchmark": speed,
                 "inference_ms": speed["inference_ms"], "pipeline_ms": speed["pipeline_ms"], "evaluation_split": selected_split,
                 "model": state["model"], "completed_epochs": completed, "best_weight": str(best), "last_weight": str(last),
                 "dataset_fingerprint": info["fingerprint"], "evaluations": evaluations,
                 "initialization": init["initialization"], "formal": job["formal"]}
        write_json(output / "metrics" / "final_metrics.json", final)
        from .common import write_csv
        write_csv(output / "metrics" / "final_metrics.csv", [{k: v for k, v in final.items()
                  if isinstance(v, (str, int, float, bool)) or v is None}])
        state.update({k: v for k, v in final.items() if not isinstance(v, (dict, list))})
        state.update(status="success", stage="completed", end_time=now(), elapsed_s=round(time.monotonic()-started, 3))
        print(f"[结束] {state['model']} | {state['end_time']} | 总耗时={state['elapsed_s']}秒\n"
              f"[最终指标] split={selected_split}，P={final['precision']:.6f}，R={final['recall']:.6f}，"
              f"mAP50={final['mAP50']:.6f}，mAP50-95={final['mAP50_95']:.6f}\n"
              f"[最终记录] {output / 'metrics/final_metrics.json'}", flush=True)
        write_json(output / "status.json", state)
        return 0
    except BaseException as exc:
        trace = traceback.format_exc()
        (output / "logs").mkdir(parents=True, exist_ok=True)
        (output / "logs" / "error.log").write_text(trace, encoding="utf-8")
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=f"{type(exc).__name__}: {exc}",
                     end_time=now(), elapsed_s=round(time.monotonic()-started, 3))
        progress = output / "progress.json"
        if progress.exists():
            state["completed_epochs"] = read_json(progress).get("completed_epochs")
        write_json(output / "status.json", state)
        print(trace, flush=True)
        if "out of memory" in str(exc).lower():
            print("[显存不足] 未自动缩小 batch/imgsz。释放显存，或在 config.py 为 S/X 统一减 batch 后重开整组实验。", flush=True)
        print(f"[失败] {state['model']}；详见 {output / 'logs/error.log'}", flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    if len(sys.argv) != 2:
        raise SystemExit("内部 worker 需要 job.json；请在 PyCharm 运行 train_yolo26_s.py / train_yolo26_x.py / run_all.py")
    try:
        code = execute(read_json(sys.argv[1]))
    finally:
        terminate_own_children()
    raise SystemExit(code)
