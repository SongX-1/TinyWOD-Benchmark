"""唯一的用户配置位置。PyCharm 直接运行 train_tinywod.py。"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ("target",)  # TinyWOD 全图只有一个前景类 target


def resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


@dataclass
class Experiment:
    # 采用原项目 two_stage / hybrid_branch / R50 / 36eps 官方配方。
    epochs: int = 36
    yolo_root: str = "D:/dataset"
    coco_root: str = "D:/dataset"
    dataset_mode: str = "coco"  # auto 优先 YOLO，否则 COCO；保持原 train/val/test 划分
    coco_annotations: dict = field(default_factory=lambda: {
        "train": "coco/instances_train.json",
        "val": "coco/instances_val.json",
        "test": "coco/instances_test.json",
    })  # 本数据集的 COCO json 位于 coco/ 子目录（非默认 annotations/ 布局）
    coco_image_dirs: dict = field(default_factory=dict)
    test_policy: str = "auto"  # auto / unlabeled / required
    accept_unverified_yolo_names: bool = True  # 无 YAML 时显式按 CLASS_NAMES，日志记录限制
    verify_images: bool = True
    hash_images: bool = False  # True 额外检查跨划分相同图片内容，较耗时
    output_root: str = "runs"
    device: str = "auto"  # auto / cpu / cuda:0；显式 cuda 不可用就报错
    seed: int = 42
    cpu_threads: int = 8
    workers: int = 8
    pin_memory: bool = True
    deterministic: bool = True
    # CUDA grid_sample 反向有非确定性；开启严格模式会明确拒绝不支持的算子。
    strict_determinism: bool = False
    amp: bool = True  # 混合精度：减半激活显存、利用 Tensor Core 提速，降低对其它训练进程的显存压力
    gpu_reserve_fraction: float = 0.0  # 单卡独占无需预留；5090 32G 上原 0.15 会把本进程上限封到 ~25G 导致 OOM
    batch_size: int = 8
    val_batch_size: int = 8
    # 保留 DETR 的等比例缩放、ImageNet 归一化、NestedTensor padding。
    # 短边 640、长边上限 1067（不是 YOLO 的 640x640 letterbox）。
    image_size: int = 640
    max_size: int = 1067
    horizontal_flip: float = 0.5
    # 默认不随机裁掉小目标；如需原论文多尺度/裁剪，在这里显式选择。
    augmentation: str = "small_object"  # small_object / native
    native_train_scales: tuple = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)
    native_max_size: int = 1333
    lr: float = 2e-4
    lr_backbone: float = 2e-5
    lr_linear_proj_mult: float = 0.1
    weight_decay: float = 1e-4
    lr_drop: int = 30
    lr_gamma: float = 0.1
    clip_max_norm: float = 0.1
    num_queries_one2one: int = 300
    num_queries_one2many: int = 1500
    k_one2many: int = 6
    lambda_one2many: float = 1.0
    enc_layers: int = 6
    dec_layers: int = 6
    dim_feedforward: int = 2048
    use_checkpoint: bool = True  # 激活检查点：省反向中间激活（batch=2 在参考实现下反向峰值 OOM 的正解）
    max_detections: int = 100  # 原生 H-DETR top-k；COCO maxDets=100
    ops_backend: str = "auto"  # auto / native / pytorch；不临时编译、不调用 cl/nvcc
    ops_query_chunk_size: int = 2048  # 参考实现按查询分块；调小降低 torch.stack 中间张量峰值（OOM 点），速度影响小
    pretrained: bool = True
    backbone_weights: str = "weights/resnet50-0676ba61.pth"  # 只读本地文件
    # 完整检测器权重可选；必须与本项目类别数、查询数及网络严格匹配。
    detector_weights: str = ""
    resume_from: str = ""  # 本项目的 last.pth/epoch_XXXX.pth；续训另建目录，保留父记录
    checkpoint_interval: int = 5
    prediction_images: int = 12
    benchmark_images: int = 8
    benchmark_warmup: int = 3
    benchmark_repeats: int = 10
    make_plots: bool = True
    smoke_test: bool = False  # 仅测试脚本使用；正式训练保持 False


CONFIG = Experiment()
