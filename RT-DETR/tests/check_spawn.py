"""Exercise Windows-style spawn for the actual native dataset and transforms."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    import torch
    from config import CONFIG
    from tinywod.data import prepare_dataset
    from tinywod.loaders import collate, make_loaders, seed_worker
    from tinywod.modeling import native_imports
    from tinywod.synthetic import make_dataset
    native_imports()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        dataset = make_dataset(root/"dataset")
        cfg = replace(CONFIG, yolo_root=str(dataset), dataset_mode="yolo", augmentation=False,
                      image_size=160, train_scales=(), batch_size=1, num_workers=0)
        info = prepare_dataset(cfg, root/"inputs")
        loaders, _ = make_loaders(cfg, info)
        spawned = torch.utils.data.DataLoader(loaders["train"].dataset, batch_size=1, shuffle=False,
                                              num_workers=2, multiprocessing_context="spawn", collate_fn=collate,
                                              timeout=30,
                                              worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(123))
        images, instances = 0, 0
        for samples, targets in spawned:
            assert samples.shape[1:] == (3, 160, 160)
            assert torch.isfinite(samples).all()
            images += len(samples)
            instances += sum(len(target["labels"]) for target in targets)
        assert images == 3 and instances == 5
        print("PASS Windows-style spawn: native CocoDetection + transforms, 2 workers, 3 images, 5 GT")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
