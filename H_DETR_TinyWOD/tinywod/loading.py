"""COCO view -> original H-DETR dataset, transforms, target normalization and padding."""
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed); np.random.seed(seed)


def make_transforms(cfg, training):
    import datasets.transforms as T
    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
    resize = T.RandomResize([cfg.image_size], max_size=cfg.max_size)
    if not training: return T.Compose([resize, normalize])
    if cfg.augmentation == "native":
        resize = T.RandomSelect(
            T.RandomResize(list(cfg.native_train_scales), max_size=cfg.native_max_size),
            T.Compose([T.RandomResize([400,500,600]), T.RandomSizeCrop(384,600),
                       T.RandomResize(list(cfg.native_train_scales), max_size=cfg.native_max_size)]))
    return T.Compose([T.RandomHorizontalFlip(p=cfg.horizontal_flip), resize, normalize])


def make_dataset(cfg, split, info, training=False):
    from datasets.coco import CocoDetection
    item = info["splits"][split]
    return CocoDetection(item["images_path"], item["annotation"],
                         transforms=make_transforms(cfg, training), return_masks=False,
                         cache_mode=False, local_rank=0, local_size=1)


def make_loaders(cfg, info):
    from util.misc import collate_fn
    train = make_dataset(cfg, "train", info, training=True)
    val = make_dataset(cfg, "val", info)
    generator = torch.Generator().manual_seed(cfg.seed)
    # Separate worker generator: worker creation must not perturb global model RNG.
    worker_generator = torch.Generator().manual_seed(cfg.seed+1)
    sampler = RandomSampler(train, generator=generator)
    if len(train) < cfg.batch_size:
        raise ValueError(f"训练图片 {len(train)} 少于 batch_size={cfg.batch_size}；原配方 drop_last=True 将无 batch。")
    train_loader = DataLoader(train, batch_sampler=BatchSampler(sampler, cfg.batch_size, drop_last=True),
        collate_fn=collate_fn, num_workers=cfg.workers, pin_memory=cfg.pin_memory,
        worker_init_fn=seed_worker, generator=worker_generator, persistent_workers=False)
    val_loader = DataLoader(val, batch_size=cfg.val_batch_size, sampler=SequentialSampler(val),
        drop_last=False, collate_fn=collate_fn, num_workers=cfg.workers, pin_memory=cfg.pin_memory,
        worker_init_fn=seed_worker, generator=worker_generator, persistent_workers=False)
    return train_loader, val_loader, generator, worker_generator
