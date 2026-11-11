"""Exact-duplicate audit for DQ TinyWOD input views; source data stay read-only."""
import csv
import json
import os
import uuid
from collections import Counter
from pathlib import Path

DUPLICATE_POLICY_VERSION = "sds_exact_duplicate_keep_first_v1"  # 保持旧值：已写入数据集指纹
DETAIL_FIELDS = (
    "split", "image", "source_image", "source_labels", "class_id", "class_name",
    "retained_source_box_index", "duplicate_source_box_index",
    "retained_source_line", "duplicate_source_line",
    "retained_source_annotation_id", "duplicate_source_annotation_id",
    "retained_output_annotation_id", "input_bbox_xywh", "canonical_bbox_xyxy",
    "area", "iscrowd", "action", "policy_version",
)


def _atomic_csv(path, rows, fields):
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class DuplicateAnnotationAudit:
    """Keep first identical canonical annotation; never merge nearby boxes.

    Identity is scoped to one image and includes class, clipped pixel xyxy,
    area and iscrowd. Different classes/crowd/area or any coordinate difference
    remain separate. No IoU, distance threshold or rounding is applied here.
    """

    def __init__(self, inputs, names):
        self.inputs = Path(inputs)
        self.inputs.mkdir(parents=True, exist_ok=True)
        self.names = tuple(names)
        self.detail_path = self.inputs / "duplicate_annotations.csv"
        self.split_counts = Counter()
        self.image_counts = Counter()
        self.class_counts = {}
        self.image_details = {}
        self.total_removed = 0
        # The caller creates a new dataset_view for each run. Never truncate
        # an existing report if a caller accidentally reuses an inputs folder.
        with self.detail_path.open("x", encoding="utf-8-sig", newline="") as file:
            csv.DictWriter(file, fieldnames=DETAIL_FIELDS).writeheader()

    def keep(self, seen, *, split, image_rel, source_image, source_labels,
             class_id, xyxy, area, iscrowd, source_box_index,
             source_annotation, next_annotation_id):
        key = (int(class_id), *map(float, xyxy), float(area), int(iscrowd))
        metadata = {
            "source_box_index": source_box_index,
            "source_line": source_annotation.get("_source_line", ""),
            "source_annotation_id": source_annotation.get("id", ""),
            "output_annotation_id": next_annotation_id,
        }
        if key not in seen:
            seen[key] = metadata
            return True
        first = seen[key]
        split, image_rel = str(split), str(image_rel)
        row = {
            "split": split, "image": image_rel, "source_image": str(source_image),
            "source_labels": str(source_labels), "class_id": int(class_id),
            "class_name": self.names[int(class_id)],
            "retained_source_box_index": first["source_box_index"],
            "duplicate_source_box_index": source_box_index,
            "retained_source_line": first["source_line"],
            "duplicate_source_line": metadata["source_line"],
            "retained_source_annotation_id": first["source_annotation_id"],
            "duplicate_source_annotation_id": metadata["source_annotation_id"],
            "retained_output_annotation_id": first["output_annotation_id"],
            "input_bbox_xywh": json.dumps(source_annotation["bbox"], ensure_ascii=False),
            "canonical_bbox_xyxy": json.dumps(list(map(float, xyxy))),
            "area": float(area), "iscrowd": int(iscrowd),
            "action": "remove_exact_duplicate_keep_first",
            "policy_version": DUPLICATE_POLICY_VERSION,
        }
        # A closed append on each duplicate also flushes records if later input
        # validation fails; no outstanding report handle is left in a worker.
        with self.detail_path.open("a", encoding="utf-8", newline="") as file:
            csv.DictWriter(file, fieldnames=DETAIL_FIELDS).writerow(row)
        self.total_removed += 1
        self.split_counts[split] += 1
        image_key = (split, image_rel)
        self.image_counts[image_key] += 1
        self.class_counts.setdefault(split, Counter())[int(class_id)] += 1
        self.image_details[image_key] = {
            "split": split, "image": image_rel, "source_image": str(source_image),
            "source_labels": str(source_labels),
        }
        if self.total_removed <= 5:
            print(f"[标注去重] {split}/{image_rel} class={self.names[int(class_id)]} "
                  f"保留框#{first['source_box_index']}，移除完全重复框#{source_box_index} "
                  f"| 明细={self.detail_path}", flush=True)
        return False

    def split_count(self, split):
        return int(self.split_counts[str(split)])

    def image_count(self, split, image_rel):
        return int(self.image_counts[(str(split), str(image_rel))])

    def finish(self):
        summary_rows = []
        for split in ("train", "val", "test"):
            summary_rows.append({
                "split": split,
                "images_with_duplicates": sum(s == split for s, _ in self.image_counts),
                "duplicate_boxes_removed": self.split_count(split),
                **{name: int(self.class_counts.get(split, {}).get(i, 0))
                   for i, name in enumerate(self.names)},
            })
        summary_path = self.inputs / "duplicate_annotations_summary.csv"
        _atomic_csv(summary_path, summary_rows, [
            "split", "images_with_duplicates", "duplicate_boxes_removed", *self.names])
        per_image_path = self.inputs / "duplicate_annotations_per_image.csv"
        per_image_rows = [
            {**self.image_details[key], "duplicate_boxes_removed": count}
            for key, count in sorted(self.image_counts.items())
        ]
        _atomic_csv(per_image_path, per_image_rows, [
            "split", "image", "source_image", "source_labels", "duplicate_boxes_removed"])
        result = {
            "policy_version": DUPLICATE_POLICY_VERSION,
            "key": ["image", "class_id", "clipped_pixel_xyxy", "area", "iscrowd"],
            "comparison": "exact numeric equality, no rounding or IoU merging",
            "source_modified": False,
            "duplicates_removed": self.total_removed,
            "images_with_duplicates": len(self.image_counts),
            "splits": {row["split"]: row for row in summary_rows},
            "detail_csv": str(self.detail_path.resolve()),
            "summary_csv": str(summary_path.resolve()),
            "per_image_csv": str(per_image_path.resolve()),
        }
        path = self.inputs / "duplicate_annotations_summary.json"
        temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
        print(f"[标注去重汇总] 移除={self.total_removed} 框，涉及={len(self.image_counts)} 张图 "
              f"| 源数据未修改 | {summary_path}", flush=True)
        return result
