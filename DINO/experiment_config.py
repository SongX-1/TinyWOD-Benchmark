# -*- coding: utf-8 -*-
"""唯一日常配置入口。PyCharm 直接运行 train_dino_r50.py，无需命令行参数。"""
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class ExperimentConfig:
    # 与 OWT/prepare_tinywod_data.py 相同的只读原图、标注来源。
    dataset_root: Path = Path(r"D:/dataset")
    output_root: Path = PROJECT_ROOT / "runs" / "DINO-R50"
    cache_root: Path = PROJECT_ROOT / "data_cache" / "tinywod"
    model_config: Path = PROJECT_ROOT / "config" / "DINO" / "DINO_TinyWOD_R50.py"
    epochs: int = 54  # 按用户要求迭代次数改为54。
    imgsz: int = 640
    batch_size: int = 4
    val_batch_size: int = 4
    workers: int = 8  # 与DAB一致；工作站空闲时发挥更多算力。
    seed: int = 0
    device: str = "auto"  # 自动选择 cuda:0 或 cpu，也可填 cuda:1。
    amp: bool = True  # 仅 CUDA 开启；CPU 自动关闭。
    deterministic: bool = False  # 与 OWT 相同，grid_sample 的 CUDA 反传非确定性。
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    lr_drop: int = 23  # 与24轮计划一致，第24轮使用衰减学习率。
    clip_max_norm: float = 0.1
    save_period: int = 10
    reserve_fraction: float = 0.10
    validate_image_sizes: bool = True
    profile_flops: bool = True
    save_predictions: bool = True  # 每轮验证预测 JSON，保留全部300个候选。
    save_eval_arrays: bool = True
    predict_test_after_training: bool = True
    test_confidence: float = 0.25  # 仅无标签 test 导出，不用于 AP 计算。
    owt_results_csv: str = ""  # 可选：OWT真实训练生成的results.csv，供最终对比汇总。
    # 只接收你自己可信的本地 checkpoint。空字符串表示从随机初始化训练。
    resume: str = ""
    pretrain_model_path: str = ""
    backbone_pretrained: bool = False  # 不自动联网下载任何权重。
    backbone_pretrained_path: Path = PROJECT_ROOT / 'weights' / 'resnet50_imagenet.pth'  # 本地ImageNet骨干权重，避免联网下载；自行放置，不随源码分发
    # 仅供有明确实验目的时覆盖原始模型配置；默认保留原R50四尺度6+6层900queries。
    model_overrides: dict = field(default_factory=dict)


CONFIG = ExperimentConfig()
CLASS_NAMES = ["target"]  # TinyWOD 全图只有一个前景类 target
CATEGORY_MAP = {1: 0}
