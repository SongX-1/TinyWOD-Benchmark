"""TinyWOD YOLO26 S/X：唯一用户配置文件，PyCharm 直接 Run。"""
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CLASS_NAMES = (["target"])  # TinyWOD 全图只有一个前景类 target
MODEL_ORDER = ("s", "x")
MODEL_FILES = {v: PROJECT_ROOT / "models" / f"yolo26{v}.yaml" for v in MODEL_ORDER}
OFFICIAL_WEIGHTS = {
    v: f"https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26{v}.pt" for v in MODEL_ORDER
}


@dataclass(frozen=True)
class ExperimentConfig:
    # 沿用本次上传 config.py 的路径；交付环境没有这些 Windows 数据，未验证实际内容。
    dataset_mode: str = "auto"             # auto / yolo / coco
    existing_yolo_root: str = "D:/dataset"
    raw_dataset_root: str = "D:/dataset"
    prepared_dataset_root: str = str(PROJECT_ROOT / "datasets/tinywod")
    # 没有 OWT manifest.json 时，只有确认 YOLO cls 0..4 就是 CLASS_NAMES 后才改 True。
    accept_unverified_yolo_names: bool = False
    verify_images: bool = True
    hash_images: bool = False              # False: 图像用相对路径/大小/mtime；标签始终 SHA256
    test_policy: str = "auto"              # auto / unlabeled / required（required 要求完整测试标签）

    epochs: int = 200                      # 正式入口强制为 200；smoke_test.py 单独使用测试配置
    imgsz: int = 640
    batch: int = 8                         # 所有模型同值；禁止自动 batch 或 OOM 后静默缩减
    workers: int = 2                       # Windows 下减少 spawn worker 数量以降低僵尸/崩溃；CPU 实际为 0
    device: str = "auto"                   # auto: CUDA:0 可用则使用，否则 CPU；也可 "0" / "cpu"
    seed: int = 0
    deterministic: bool = True
    amp: bool = True                       # 仅 CUDA；用当前模型做有限值检查，失败即报错
    cpu_threads: int = 4
    pin_memory: bool = False

    # 固定 YOLO 的原生优化参数，避免 optimizer=auto 按训练规模自动改变算法。
    optimizer: str = "SGD"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    cos_lr: bool = False
    nbs: int = 64
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5                      # 上游保留字段；YOLO26 reg_max=1 无 DFL，不能当作 DFL 损失
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 0.0                    # 保留本次上传配置，不做背景过滤
    mixup: float = 0.0
    copy_paste: float = 0.0

    initialization: str = "scratch"        # scratch / pretrained；保留上传配置的随机初始化
    weights_dir: str = str(PROJECT_ROOT / "weights")
    download_pretrained: bool = False      # True 才从上方官方地址下载，失败不会回退随机初始化
    # 恢复至总计 200 轮；必须是本工程保存的完整 last.pt 或 epochXXX.pt。
    resume_checkpoints: dict = field(default_factory=lambda: {"s": "", "x": ""})
    # 保留你入口中的可选延迟功能；默认立即运行，单位小时。原上传 S=4，X实际=2（旧注释10）。
    startup_delay_hours: dict = field(default_factory=lambda: {"s": 0, "x": 0})
    output_root: str = str(PROJECT_ROOT / "runs")
    save_period: int = 10
    continue_on_failure: bool = True
    conf: float = 0.001
    max_det: int = 300
    eval_half: bool = False                # 两模型最终验证统一 FP32
    predict_unlabeled_test: bool = True
    prediction_conf: float = 0.25          # 仅无标签测试图片的可视化/预测保存阈值
    save_test_images: bool = False         # txt/json 始终保存；大图渲染默认关闭
    profile_flops: bool = True
    benchmark_images: int = 32             # 固定排序 val 前 32 张，batch=1，FP32
    benchmark_repeats: int = 3
    benchmark_warmup: int = 5


CONFIG = ExperimentConfig()


def resolved_path(value):
    """项目相对路径始终相对 config.py；不依赖 PyCharm 的 Working directory。"""
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def snapshot(cfg=CONFIG):
    data = asdict(cfg)
    for key in ("existing_yolo_root", "raw_dataset_root", "prepared_dataset_root", "weights_dir", "output_root"):
        data[key] = str(resolved_path(data[key]))
    data["resume_checkpoints"] = {k: str(resolved_path(v)) if v else "" for k, v in data["resume_checkpoints"].items()}
    return data
