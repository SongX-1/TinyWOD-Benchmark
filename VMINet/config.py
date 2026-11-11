"""唯一用户配置文件。PyCharm 直接 Run train_tinywod.py。"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CLASS_NAMES=("target",)  # TinyWOD 全图只有一个前景类 target


def resolve(value):
    path=Path(value).expanduser()
    return path if path.is_absolute() else ROOT/path


@dataclass
class Experiment:
    epochs: int = 36
    # 原仓库没有 L；B 为所提供四种实现中最大的真实变体。
    variant: str = "b"  # ti / xs / s / b；本次默认只训练 B
    yolo_root: str = "D:/dataset"
    coco_root: str = "D:/dataset"
    dataset_mode: str = "auto"
    coco_annotations: dict = field(default_factory=dict)
    coco_image_dirs: dict = field(default_factory=dict)
    test_policy: str = "auto"
    accept_unverified_yolo_names: bool = True
    verify_images: bool = True
    hash_images: bool = False
    output_root: str = "runs"
    device: str = "auto"  # auto / cpu / cuda:0
    seed: int = 42
    val_loss_seed: int = 12387  # 验证采样独立，不改变训练 RNG
    cpu_threads: int = 4
    workers: int = 0
    pin_memory: bool = False
    cudnn_deterministic: bool = True
    amp: bool = False
    batch_size: int = 4
    val_batch_size: int = 2
    image_size: int = 640  # 固定正方形 letterbox；须为 32 的倍数
    letterbox_value: int = 114
    horizontal_flip: float = 0.5
    lr: float = 1e-4
    weight_decay: float = 0.05
    lr_milestones: tuple = (27,33)  # 论文 12 Epoch 的 (9,11) 等比例延长至 36
    lr_gamma: float = 0.1
    clip_max_norm: float = 0.1
    # VMINet 最后特征 -> Simple Feature Pyramid(P2..P6) -> Faster R-CNN bbox 分支。
    pyramid_channels: int = 256
    roi_hidden_dim: int = 1024
    roi_pool_size: int = 7
    roi_sampling_ratio: int = 2
    anchor_sizes: tuple = (8,16,32,64,128)
    anchor_ratios: tuple = (0.5,1.0,2.0)
    rpn_pre_nms_train: int = 2000
    rpn_post_nms_train: int = 2000
    rpn_pre_nms_test: int = 1000
    rpn_post_nms_test: int = 1000
    rpn_batch_size: int = 256
    roi_batch_size: int = 512
    rpn_nms_iou: float = 0.7
    nms_iou: float = 0.5
    score_threshold: float = 0.001
    max_detections: int = 100
    pretrained: bool = True
    backbone_weights: dict = field(default_factory=lambda: {
        "ti":"weights/VMINet2.0-Ti-checkpoint.pth.tar",
        "xs":"weights/VMINet2.0-XS-checkpoint.pth.tar",
        "s":"weights/VMINet2.0-S-checkpoint.pth.tar",
        "b":"weights/B-ckpt.pth.tar",
    })
    pretrained_state_key: str = "state_dict"  # 原分类训练的真实保存字段；也支持纯 state_dict
    detector_weights: str = ""
    resume_from: str = ""
    checkpoint_interval: int = 5
    prediction_images: int = 12
    benchmark_images: int = 8
    benchmark_warmup: int = 3
    benchmark_repeats: int = 10
    make_plots: bool = True
    smoke_test: bool = False


CONFIG=Experiment()
