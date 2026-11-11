"""统一配置：PyCharm 直接运行 train_tinywod.py；只在本文件修改实验设置。"""
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ("target",)  # TinyWOD 全图只有一个前景类 target


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_mode: str = "auto"  # 优先现有 YOLO；可选 yolo / coco
    yolo_root: str = "D:/dataset"
    coco_root: str = "D:/dataset"
    accept_unverified_yolo_names: bool = False
    verify_images: bool = True
    hash_images: bool = False
    test_policy: str = "auto"  # auto / unlabeled / required

    model_name: str = "RT-DETR-R50vd"
    model_config: str = "rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml"
    epochs: int = 72
    image_size: int = 640
    # 原项目 RTDETR.multi_scale：保留三个 640 的采样权重。
    train_scales: tuple = (480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800)
    augmentation: bool = True
    batch_size: int = 8
    val_batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    cpu_threads: int = 4
    seed: int = 0
    device: str = "auto"  # auto / cpu / cuda:0
    mixed_precision: str = "bf16"  # 原配置 FP32；改用 bf16 利用 RTX5090 Tensor Core 提速
    deterministic: bool = False
    allow_tf32: bool = False
    learning_rate: float = 0.0001
    backbone_learning_rate: float = 0.00001
    weight_decay: float = 0.0001
    betas: tuple = (0.9, 0.999)
    # 原项目 72 Epoch 配置的里程碑为 1000，故这 72 轮不衰减；不擅自更换策略。
    lr_milestones: tuple = (1000,)
    lr_gamma: float = 0.1
    clip_max_norm: float = 0.1
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_warmups: int = 2000

    # 仅本地加载，训练代码不会联网。R50vd 不能用普通 torchvision ResNet50 权重。
    initialization: str = "backbone_pretrained"  # backbone_pretrained / detector_pretrained / scratch
    backbone_weight: str = "weights/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth"
    detector_weight: str = "weights/rtdetr_r50vd_6x_coco_from_paddle.pth"
    expected_weight_sha256: str = ""  # 如已知完整 SHA256 可填入；每次仍会记录实际 SHA256
    resume_checkpoint: str = ""  # 本项目生成的 weights/last.pth 或 best.pth

    output_root: str = "runs"
    save_period: int = 12
    print_freq: int = 50
    save_batch_csv: bool = True
    save_epoch_predictions: bool = False
    max_det: int = 300  # 原生后处理候选数量
    coco_max_dets: int = 100  # COCO 标准 AP；与 max_det 分开记录
    metric_conf: float = 0.001
    confusion_conf: float = 0.25
    confusion_iou: float = 0.5
    profile_flops: bool = True
    tensorboard: bool = True
    benchmark_images: int = 32
    benchmark_warmup: int = 5
    benchmark_repeats: int = 3
    predict_unlabeled_test: bool = True
    prediction_conf: float = 0.25
    save_prediction_images: bool = False


CONFIG = ExperimentConfig()


def resolve(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def snapshot(cfg=CONFIG):
    return asdict(cfg)
