"""Use the original CocoDetection + v2 augmentation, keeping normalized cxcywh targets."""
from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .data import image_files

# TinyWOD 含约 2 亿像素的超高分辨率图，超过 PIL 默认解压炸弹上限（约 8948 万像素）。
# 数据为固定已知数据集，禁用该保护以避免真实训练/验证时 DecompressionBombError。
Image.MAX_IMAGE_PIXELS = None


def seed_worker(worker_id):
    Image.MAX_IMAGE_PIXELS = None  # 每个 DataLoader worker 进程也要放开解压炸弹限制
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)


def collate(items):
    return torch.stack([image.as_subclass(torch.Tensor) for image, _ in items]), [target for _, target in items]


def make_transforms(cfg, training):
    from src.data.transforms import Compose
    ops = []
    if training and cfg.augmentation:
        ops.extend([
            {"type": "Resize", "size": 800, "max_size": 1333},  # 预缩放高分辨率图，避免 RandomZoomOut 生成超大 canvas
            {"type": "RandomPhotometricDistort", "p": 0.5},
            {"type": "RandomZoomOut", "fill": 0},
            {"type": "RandomIoUCrop", "p": 0.8},
            {"type": "SanitizeBoundingBox", "min_size": 1},
            {"type": "RandomHorizontalFlip"},
        ])
    ops.extend([
        {"type": "Resize", "size": [cfg.image_size, cfg.image_size]},
        {"type": "ToImageTensor"}, {"type": "ConvertDtype"},
    ])
    if training:
        ops.append({"type": "SanitizeBoundingBox", "min_size": 1})
    # Native val omitted this because its criterion was commented out. Normalize
    # val targets too so the added validation loss is numerically meaningful.
    ops.append({"type": "ConvertBox", "out_fmt": "cxcywh", "normalize": True})
    return Compose(deepcopy(ops))


class UnlabeledImages(Dataset):
    def __init__(self, folder, size, relative_files=None):
        self.folder, self.size = Path(folder), size
        self.files = [self.folder/path for path in relative_files] if relative_files is not None else image_files(self.folder)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        from torchvision.transforms.v2 import functional as F
        with Image.open(self.files[index]) as source:
            image = source.convert("RGB")
        width, height = image.size
        tensor = F.to_dtype(F.to_image(image.resize((self.size, self.size), Image.Resampling.BILINEAR)), torch.float32, scale=True)
        return tensor, {"image_id": torch.tensor([index]), "orig_size": torch.tensor([width, height]),
                        "relative": self.files[index].relative_to(self.folder).as_posix()}


def make_loaders(cfg, info):
    from src.data.coco.coco_dataset import CocoDetection
    loaders, generators = {}, {}
    for offset, split in enumerate(("train", "val", "test")):
        spec = info["splits"][split]
        if not spec["images"]:
            continue
        if spec["labeled"]:
            dataset = CocoDetection(spec["images_path"], spec["annotation"],
                                    make_transforms(cfg, split == "train"), False, False)
        else:
            dataset = UnlabeledImages(spec["images_path"], cfg.image_size, spec["image_files"])
        generator = torch.Generator().manual_seed(cfg.seed + offset)
        generators[split] = generator
        loaders[split] = DataLoader(dataset, batch_size=cfg.batch_size if split == "train" else cfg.val_batch_size,
                                    shuffle=split == "train", num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                                    drop_last=False, persistent_workers=False, worker_init_fn=seed_worker,
                                    generator=generator, collate_fn=collate)
    return loaders, generators
