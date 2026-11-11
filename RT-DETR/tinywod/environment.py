"""Check all training prerequisites together; never install or download during training."""
import importlib
import importlib.metadata
import os
from pathlib import Path
import platform
import random
import sys

from config import resolve
from .common import atomic_csv, now, write_json

REQUIRED = {
    "torch": "torch", "torchvision": "torchvision", "numpy": "numpy",
    "Pillow": "PIL", "PyYAML": "yaml", "scipy": "scipy",
    "pycocotools": "pycocotools.cocoeval", "matplotlib": "matplotlib",
    "tqdm": "tqdm", "psutil": "psutil", "packaging": "packaging",
}
BACKBONE_URL = "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth"
DETECTOR_URL = "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth"


def weight_preflight(cfg):
    if cfg.resume_checkpoint:
        path, source = resolve(cfg.resume_checkpoint), "本项目恢复文件"
    elif cfg.initialization == "backbone_pretrained":
        path, source = resolve(cfg.backbone_weight), BACKBONE_URL
    elif cfg.initialization == "detector_pretrained":
        path, source = resolve(cfg.detector_weight), DETECTOR_URL
    elif cfg.initialization == "scratch":
        return None
    else:
        raise ValueError("initialization 只能是 backbone_pretrained / detector_pretrained / scratch")
    if not path.is_file():
        raise FileNotFoundError(f"缺少本地权重：{path}\n请用浏览器下载后放到上述位置：{source}\n训练程序不会联网下载，也不会自动改为随机初始化。")
    if path.stat().st_size < 1024:
        raise ValueError(f"权重文件过小，可能下载成网页或未完成：{path}")
    return path


def setup_environment(cfg, output, check_weight=True):
    print(f"[解释器] {sys.executable}", flush=True)
    errors, versions = [], []
    required = dict(REQUIRED)
    if cfg.tensorboard:
        required["tensorboard"] = "torch.utils.tensorboard"
    if cfg.profile_flops:
        required["fvcore"] = "fvcore.nn"
    for distribution, module in required.items():
        version = "unknown"
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass  # report metadata independently; imports are the functional check
        try:
            imported = importlib.import_module(module)
            versions.append({"package": distribution, "version": version, "status": "ok",
                             "module_path": str(getattr(imported, "__file__", "")), "error": ""})
        except Exception as exc:
            message = f"{distribution}: {type(exc).__name__}: {exc}"
            errors.append(message)
            versions.append({"package": distribution, "version": version, "status": "failed",
                             "module_path": "", "error": str(exc)})
    atomic_csv(Path(output)/"logs"/"package_versions.csv", versions,
               ["package", "version", "status", "module_path", "error"])
    if errors:
        raise RuntimeError("依赖检查一次汇总如下：\n" + "\n".join(errors) +
                           "\n在同一个 PyCharm 解释器运行 install_dependencies.py；torch/torchvision 配对问题请按官方安装页处理。")

    import numpy as np
    import torch
    import torchvision
    from packaging.version import Version
    if Version(str(torch.__version__).split("+")[0]) < Version("2.7") or Version(torchvision.__version__.split("+")[0]) < Version("0.22"):
        raise RuntimeError("本适配使用现代 torchvision v2 接口，需要 torch>=2.7、torchvision>=0.22，且两者版本必须配套。")
    if cfg.cpu_threads < 1:
        raise ValueError("cpu_threads 必须大于 0")
    torch.set_num_threads(cfg.cpu_threads)
    try:
        torch.set_num_interop_threads(min(4, cfg.cpu_threads))
    except RuntimeError:
        pass
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = cfg.deterministic
    torch.use_deterministic_algorithms(cfg.deterministic)
    torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf32
    torch.backends.cudnn.allow_tf32 = cfg.allow_tf32
    device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("已指定 CUDA，但当前解释器的 torch 无可用 CUDA。不会静默切换 CPU。")
    if cfg.mixed_precision not in ("no", "fp16", "bf16"):
        raise ValueError("mixed_precision 只能是 no / fp16 / bf16")
    if device.type == "cpu" and cfg.mixed_precision != "no":
        raise ValueError("本项目的混合精度设置用于 CUDA；CPU 请明确设置 mixed_precision='no'。")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if cfg.mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前 GPU / torch 不支持 bf16。")
        # Detect an incompatible GPU build now, before reading thousands of images.
        x = torch.randn(1, 4, 8, 8, device=device, requires_grad=True)
        grid = torch.zeros(1, 2, 2, 2, device=device)
        torch.nn.functional.grid_sample(x, grid, align_corners=False).sum().backward()
        torch.nn.functional.conv2d(x.detach(), torch.ones(4, 4, 3, 3, device=device)).sum().item()
        torch.cuda.synchronize(device)
        del x, grid
        torch.cuda.empty_cache()
        name = torch.cuda.get_device_name(device)
    else:
        name = platform.processor() or platform.machine()
        # Some containers cannot expose CPU feature information to NNPACK.
        # This affects CPU acceleration only, never the model or CUDA execution.
        if hasattr(torch.backends.nnpack, "set_flags"):
            torch.backends.nnpack.set_flags(False)
    print(f"[设备] {device} | {name} | torch={torch.__version__} | CUDA={torch.version.cuda} | mixed_precision={cfg.mixed_precision}", flush=True)
    print(f"[随机种子] {cfg.seed} | deterministic={cfg.deterministic} | TF32={cfg.allow_tf32}", flush=True)
    print(f"[加载参数] batch={cfg.batch_size} val_batch={cfg.val_batch_size} workers={cfg.num_workers} pin_memory={cfg.pin_memory}", flush=True)
    info = {"time": now(), "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "device": str(device), "device_name": name,
            "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "seed": cfg.seed, "deterministic": cfg.deterministic, "allow_tf32": cfg.allow_tf32,
            "cpu_threads": cfg.cpu_threads, "packages": versions}
    if device.type == "cuda":
        info.update(cuda_arch_list=torch.cuda.get_arch_list(), device_capability=list(torch.cuda.get_device_capability(device)),
                    gpu_total_memory_bytes=torch.cuda.get_device_properties(device).total_memory)
    write_json(Path(output)/"logs"/"environment.json", info)
    if check_weight:
        weight_preflight(cfg)
    return device, info
