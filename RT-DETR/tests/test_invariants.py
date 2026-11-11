"""Numerical/data/output invariants; runnable without TinyWOD or GPU."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from config import CLASS_NAMES, CONFIG
from tinywod.common import CSVTable, sha256, write_json
from tinywod.data import prepare_dataset
from tinywod.metrics import evaluate_predictions, fixed_confusion
from tinywod.synthetic import make_dataset


class Invariants(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self):
        source = make_dataset(self.directory/"中文 空格路径"/"data")
        cfg = replace(CONFIG, dataset_mode="yolo", yolo_root=str(source), coco_root=str(source/"unused"),
                      image_size=160, train_scales=(), num_workers=0)
        return source, cfg

    def prepared(self, cfg):
        with redirect_stdout(io.StringIO()):
            return prepare_dataset(cfg, self.directory/"inputs")

    def coco(self, annotations):
        from pycocotools.coco import COCO
        coco = COCO()
        coco.dataset = {"info": {}, "images": [{"id": 1, "file_name": "one.jpg", "width": 200, "height": 160}],
                        "categories": [{"id": i, "name": n} for i, n in enumerate(CLASS_NAMES)], "annotations": annotations}
        with redirect_stdout(io.StringIO()):
            coco.createIndex()
        return coco

    def small_gt(self):
        return [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 20, 10, 10], "area": 100, "iscrowd": 0}]

    def evaluate(self, annotations, prediction):
        with redirect_stdout(io.StringIO()):
            return evaluate_predictions(self.coco(annotations), {1: prediction}, CONFIG, self.directory/"evaluation", 1, "val")

    def test_perfect_detection_and_small_ap(self):
        report = self.evaluate(self.small_gt(), {"boxes": [[10, 20, 20, 30]], "labels": [0], "scores": [.9]})
        for key in ("AP", "AP50", "APsmall", "precision", "recall", "f1", "ARsmall"):
            self.assertAlmostEqual(report["summary"][key], 1.)
        self.assertIsNone(report["per_class"][1]["AP"])

    def test_no_predictions_has_zero_ap(self):
        report = self.evaluate(self.small_gt(), {"boxes": [], "labels": [], "scores": []})
        for key in ("AP", "APsmall", "precision", "recall", "f1"):
            self.assertEqual(report["summary"][key], 0.)

    def test_duplicate_predictions_are_not_silently_removed(self):
        prediction = {"boxes": [[10, 20, 20, 30], [10, 20, 20, 30]], "labels": [0, 0], "scores": [.9, .8]}
        matrix, row = fixed_confusion(prediction, self.small_gt(), .25, .5)
        self.assertEqual((row["tp"], row["fp"], row["fn"]), (1, 1, 0))
        self.assertEqual(matrix[0, 5], 1)

    def test_crowd_overlap_is_ignored(self):
        annotations = self.small_gt()+[{"id": 2, "image_id": 1, "category_id": 0, "bbox": [100, 100, 80, 50], "area": 4000, "iscrowd": 1}]
        prediction = {"boxes": [[10, 20, 20, 30], [110, 110, 120, 120]], "labels": [0, 0], "scores": [.9, .8]}
        report = self.evaluate(annotations, prediction)
        self.assertAlmostEqual(report["summary"]["AP"], 1.)
        self.assertAlmostEqual(report["summary"]["precision"], 1.)
        self.assertEqual(report["per_image"][0]["crowd_ignored_detections"], 1)

    def test_no_small_gt_is_null_not_zero(self):
        annotations = [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 20, 60, 60], "area": 3600, "iscrowd": 0}]
        report = self.evaluate(annotations, {"boxes": [[10, 20, 70, 80]], "labels": [0], "scores": [.9]})
        self.assertAlmostEqual(report["summary"]["AP"], 1.)
        self.assertIsNone(report["summary"]["APsmall"])

    def test_background_only_image_has_undefined_recall(self):
        _, row = fixed_confusion({"boxes": [], "labels": [], "scores": []}, [], .25, .5)
        self.assertIsNone(row["recall"])
        self.assertIsNone(row["precision"])

    def test_csv_unique_key_and_header(self):
        path = self.directory/"table.csv"
        table = CSVTable(path, ["epoch", "loss"], ["epoch"])
        table.append({"epoch": 1, "loss": .1})
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            table.append({"epoch": 1, "loss": .2})
        self.assertEqual(before, path.read_bytes())
        with self.assertRaises(ValueError):
            CSVTable(path, ["wrong"], ["wrong"])

    def test_exact_duplicates_are_audited_source_unchanged(self):
        source, cfg = self.dataset()
        hashes = {str(path): sha256(path) for path in source.rglob("*") if path.is_file()}
        info = self.prepared(cfg)
        self.assertEqual(info["duplicate_annotations"]["duplicates_removed"], 1)
        self.assertEqual(info["splits"]["train"]["instances"], 5)
        self.assertEqual(info["splits"]["train"]["images_path"], str((source/"images"/"train").resolve()))
        self.assertEqual(hashes, {str(path): sha256(path) for path in source.rglob("*") if path.is_file()})

    def test_nearby_distinct_boxes_are_retained(self):
        source, cfg = self.dataset()
        path = source/"labels"/"train"/"train_2.txt"
        path.write_text("0 0.5 0.5 0.05 0.05\n0 0.500000001 0.5 0.05 0.05\n", encoding="utf-8")
        info = self.prepared(cfg)
        self.assertEqual(info["splits"]["train"]["instances"], 7)
        self.assertEqual(info["duplicate_annotations"]["duplicates_removed"], 1)

    def test_partial_test_labels_rejected(self):
        source, cfg = self.dataset()
        labels = source/"labels"/"test"
        labels.mkdir()
        (labels/"test_0.txt").write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "部分"):
            self.prepared(cfg)

    def test_missing_training_label_is_not_background(self):
        source, cfg = self.dataset()
        (source/"labels"/"train"/"train_2.txt").unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepared(cfg)

    def test_wrong_class_order_rejected(self):
        source, cfg = self.dataset()
        write_json(source/"manifest.json", {"names": list(reversed(CLASS_NAMES))})
        with self.assertRaisesRegex(ValueError, "顺序"):
            self.prepared(cfg)

    def test_cross_split_physical_alias_rejected(self):
        source, cfg = self.dataset()
        target = source/"images"/"val"/"val_0.png"
        target.unlink()
        os.link(source/"images"/"train"/"train_0.png", target)
        with self.assertRaisesRegex(ValueError, "跨划分"):
            self.prepared(cfg)

    def test_native_dataset_targets_are_normalized(self):
        from tinywod.modeling import native_imports
        from tinywod.loaders import make_loaders
        native_imports()
        _, cfg = self.dataset()
        cfg = replace(cfg, augmentation=False, batch_size=1, val_batch_size=1)
        info = self.prepared(cfg)
        loaders, _ = make_loaders(cfg, info)
        samples, targets = next(iter(loaders["val"]))
        self.assertEqual(tuple(samples.shape), (1, 3, 160, 160))
        self.assertTrue(np.allclose(targets[0]["boxes"].numpy(), [[.4, .3, .05, .05]], atol=1e-6))
        self.assertEqual(targets[0]["orig_size"].tolist(), [192, 160])

    def coco_dataset(self):
        import shutil
        source, cfg = self.dataset()
        root = self.directory/"coco"
        shutil.copytree(source/"images", root/"images")
        categories = [{"id": 0, "name": "ignored"}]+[{"id": i+1, "name": n} for i, n in enumerate(CLASS_NAMES)]
        for split in ("train", "val", "test"):
            files = sorted((root/"images"/split).glob("*.png"))
            images = [{"id": i+1, "file_name": p.name, "width": 192, "height": 160} for i, p in enumerate(files)]
            annotations = [] if split == "test" else [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 8, 8], "area": 64, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 50, 20, 20], "area": 400, "iscrowd": 1},
                {"id": 3, "image_id": 1, "category_id": 0, "bbox": [80, 80, 10, 10], "area": 100, "iscrowd": 0},
            ]
            write_json(root/"annotations"/f"instances_{split}.json", {"images": images, "annotations": annotations, "categories": categories})
        return root, replace(cfg, dataset_mode="coco", coco_root=str(root))

    def test_coco_semantic_mapping_crowd_and_ignored(self):
        root, cfg = self.coco_dataset()
        info = self.prepared(cfg)
        self.assertEqual(info["splits"]["train"]["class_counts"], [1, 0, 0, 0, 0])
        self.assertEqual(info["splits"]["train"]["crowd"], 1)
        data = json.loads(Path(info["splits"]["train"]["annotation"]).read_text())
        self.assertEqual([a["category_id"] for a in data["annotations"]], [0, 1])

    def test_coco_empty_test_manifest_is_unlabeled(self):
        root, cfg = self.coco_dataset()
        info = self.prepared(cfg)
        self.assertFalse(info["splits"]["test"]["labeled"])
        self.assertEqual(info["final_split"], "val")
        self.assertIsNone(info["splits"]["test"]["annotation"])

    def test_unlabeled_loader_respects_manifest_subset(self):
        import shutil
        from tinywod.modeling import native_imports
        from tinywod.loaders import make_loaders
        native_imports()
        root, cfg = self.coco_dataset()
        shutil.copy2(root/"images"/"test"/"test_0.png", root/"images"/"test"/"extra_not_in_manifest.png")
        info = self.prepared(cfg)
        loaders, _ = make_loaders(cfg, info)
        self.assertEqual(len(loaders["test"].dataset), 2)
        self.assertEqual([p.name for p in loaders["test"].dataset.files], ["test_0.png", "test_1.png"])

    def test_local_backbone_chinese_path_never_downloads(self):
        from unittest.mock import patch
        import torch
        from tinywod.modeling import native_imports
        native_imports()
        from src.nn.backbone.presnet import PResNet
        original = ROOT/"weights"/"ResNet50_vd_ssld_v2_pretrained_from_paddle.pth"
        if not original.is_file():
            self.skipTest("bundled weight not present in this checkout")
        target = self.directory/"本地 权重"/original.name
        target.parent.mkdir()
        try:
            os.link(original, target)
        except OSError:
            import shutil
            shutil.copy2(original, target)
        with patch("torch.hub.load_state_dict_from_url", side_effect=AssertionError("unexpected network")), \
             patch("urllib.request.urlopen", side_effect=AssertionError("unexpected network")):
            backbone = PResNet(depth=50, variant="d", pretrained=str(target))
        state = torch.load(original, map_location="cpu", weights_only=True)
        self.assertTrue(torch.equal(backbone.state_dict()["conv1.conv1_1.conv.weight"], state["conv1.conv1_1.conv.weight"]))

    def test_native_pad_to_size_updates_canvas(self):
        import torch
        from PIL import Image
        from torchvision import tv_tensors
        from tinywod.modeling import native_imports
        native_imports()
        from src.data.transforms import PadToSize
        image = Image.new("RGB", (100, 80))
        target = {"boxes": tv_tensors.BoundingBoxes(torch.tensor([[10., 10., 20., 20.]]), format="XYXY", canvas_size=(80, 100)),
                  "labels": torch.tensor([0])}
        image, target = PadToSize(128)(image, target)
        self.assertEqual(image.size, (128, 128))
        self.assertEqual(target["boxes"].canvas_size, (128, 128))
        self.assertEqual(target["padding"].tolist(), [0, 0, 28, 48])

    def test_postprocessor_uses_original_width_height(self):
        import torch
        from tinywod.modeling import native_imports
        native_imports()
        from src.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor
        post = RTDETRPostProcessor(num_classes=5, num_top_queries=1)
        outputs = {"pred_boxes": torch.tensor([[[.5, .5, .2, .4]]]),
                   "pred_logits": torch.tensor([[[0., 0., 10., 0., 0.]]])}
        result = post(outputs, torch.tensor([[200, 100]]))[0]
        self.assertEqual(result["labels"].tolist(), [2])
        self.assertTrue(torch.allclose(result["boxes"], torch.tensor([[80., 30., 120., 70.]]), atol=1e-4))

    def test_test_confidence_is_frozen_from_validation(self):
        prediction = {1: {"boxes": [[10, 20, 20, 30]], "labels": [0], "scores": [.6]}}
        coco = self.coco(self.small_gt())
        with self.assertRaisesRegex(ValueError, "阈值"):
            evaluate_predictions(coco, prediction, CONFIG, self.directory/"no_threshold", 1, "test")
        report = evaluate_predictions(coco, prediction, CONFIG, self.directory/"fixed_threshold", 1, "test", report_confidence=.8)
        self.assertEqual(report["summary"]["confidence"], .8)
        self.assertEqual(report["summary"]["recall"], 0.)
        self.assertEqual(report["summary"]["confidence_source"], "fixed_from_validation")
        self.assertAlmostEqual(report["summary"]["AP"], 1.)  # AP remains threshold-free


if __name__ == "__main__":
    unittest.main(verbosity=2)
