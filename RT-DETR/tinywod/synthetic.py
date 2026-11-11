"""Explicit synthetic fixtures for smoke/regression tests; never real experiment data."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from config import CLASS_NAMES


def make_dataset(root, labeled_test=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    (root/"manifest.json").write_text(json.dumps({"names": list(CLASS_NAMES), "synthetic": True}), encoding="utf-8")
    rng = np.random.default_rng(123)
    specs = {
        "train": ["0 0.25 0.25 0.05 0.05\n4 0.75 0.75 0.05 0.05\n4 0.75 0.75 0.05 0.05\n",
                  "1 0.5 0.5 0.3 0.3\n2 0.8 0.2 0.08 0.08\n3 0.2 0.8 0.08 0.08\n", ""],
        "val": ["0 0.4 0.3 0.05 0.05\n", "1 0.5 0.5 0.4 0.4\n"],
        "test": ["0 0.5 0.5 0.05 0.05\n", ""],
    }
    for split, labels in specs.items():
        images = root/"images"/split
        images.mkdir(parents=True)
        if split != "test" or labeled_test:
            (root/"labels"/split).mkdir(parents=True)
        for i, text in enumerate(labels):
            array = rng.integers(0, 70, (160, 192, 3), dtype=np.uint8)
            image = Image.fromarray(array)
            draw = ImageDraw.Draw(image)
            draw.rectangle((30+i*2, 30, 40+i*2, 40), fill="white")
            image.save(images/f"{split}_{i}.png")
            if split != "test" or labeled_test:
                (root/"labels"/split/f"{split}_{i}.txt").write_text(text, encoding="utf-8")
    return root
