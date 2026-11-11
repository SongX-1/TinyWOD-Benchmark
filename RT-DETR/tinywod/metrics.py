"""Standard COCO AP/small-object AP plus explicit confidence/IoU diagnostics.

Headline AP values are COCOeval (original image coordinates, maxDets=100).
P/R/F1 use score-ordered, same-class one-to-one matches at IoU 0.5 and the
confidence maximizing macro F1 on validation classes present in GT. Labeled
test uses that frozen validation confidence. This never modifies training or
checkpoint selection.
"""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path

import numpy as np

from config import CLASS_NAMES
from .common import atomic_csv, write_json

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
CONFIDENCES = np.linspace(0, 1, 1001)

nc = len(CLASS_NAMES)  # 类别数
N = nc + 1             # 类别 + 背景（混淆矩阵尺寸）


def iou_matrix(boxes1, boxes2, denominator="union"):
    a = np.asarray(boxes1, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes2, dtype=np.float64).reshape(-1, 4)
    low = np.maximum(a[:, None, :2], b[None, :, :2])
    high = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.maximum(high-low, 0).prod(2)
    area_a = np.maximum(a[:, 2:]-a[:, :2], 0).prod(1)
    area_b = np.maximum(b[:, 2:]-b[:, :2], 0).prod(1)
    denom = area_a[:, None] if denominator == "prediction" else area_a[:, None]+area_b[None, :]-inter
    return inter/np.maximum(denom, 1e-12)


def gt_arrays(annotations):
    boxes, labels, crowd, areas = [], [], [], []
    for ann in annotations:
        x, y, width, height = ann["bbox"]
        boxes.append([x, y, x+width, y+height])
        labels.append(ann["category_id"])
        crowd.append(bool(ann.get("iscrowd", 0)))
        areas.append(ann["area"])
    return (np.asarray(boxes, dtype=np.float64).reshape(-1, 4), np.asarray(labels, dtype=int),
            np.asarray(crowd, dtype=bool), np.asarray(areas, dtype=np.float64))


def match_predictions(prediction, annotations, min_conf):
    boxes, labels, crowds, _ = gt_arrays(annotations)
    order = np.argsort(-np.asarray(prediction["scores"]), kind="stable")
    scores = np.asarray(prediction["scores"], dtype=float)[order]
    order, scores = order[scores >= min_conf], scores[scores >= min_conf]
    pboxes = np.asarray(prediction["boxes"], dtype=float).reshape(-1, 4)[order]
    plabels = np.asarray(prediction["labels"], dtype=int)[order]
    overlaps = iou_matrix(pboxes, boxes)
    crowd_overlap = iou_matrix(pboxes, boxes[crowds], "prediction")
    crowd_labels = labels[crowds]
    correct = np.zeros((len(scores), len(IOU_THRESHOLDS)), dtype=bool)
    ignored = np.zeros_like(correct)
    for ti, threshold in enumerate(IOU_THRESHOLDS):
        used = set()
        for pi, category in enumerate(plabels):
            candidates = np.flatnonzero((labels == category) & ~crowds)
            candidates = [i for i in candidates if i not in used and overlaps[pi, i] >= threshold]
            if candidates:
                best = max(candidates, key=lambda i: overlaps[pi, i])
                used.add(best)
                correct[pi, ti] = True
            else:
                valid_crowd = crowd_labels == category
                ignored[pi, ti] = bool(np.any(crowd_overlap[pi, valid_crowd] >= threshold))
    return scores, plabels, correct, ignored


def fixed_confusion(prediction, annotations, conf, threshold):
    boxes, labels, crowds, areas = gt_arrays(annotations)
    pboxes = np.asarray(prediction["boxes"], dtype=float).reshape(-1, 4)
    plabels = np.asarray(prediction["labels"], dtype=int)
    scores = np.asarray(prediction["scores"], dtype=float)
    keep = scores >= conf
    pboxes, plabels, scores = pboxes[keep], plabels[keep], scores[keep]
    # Crowd-overlapping predictions are ignored only when not matching a normal GT.
    normal_overlap = iou_matrix(pboxes, boxes[~crowds])
    crowd_overlap = iou_matrix(pboxes, boxes[crowds], "prediction")
    ignored = np.zeros(len(pboxes), dtype=bool)
    for pi, label in enumerate(plabels):
        normal_same = labels[~crowds] == label
        crowd_same = labels[crowds] == label
        ignored[pi] = (not np.any(normal_overlap[pi, normal_same] >= threshold)
                       and np.any(crowd_overlap[pi, crowd_same] >= threshold))
    pboxes, plabels, scores = pboxes[~ignored], plabels[~ignored], scores[~ignored]
    gt_boxes, gt_labels = boxes[~crowds], labels[~crowds]
    overlaps = iou_matrix(pboxes, gt_boxes)
    pairs = sorted(((float(overlaps[p, g]), p, g) for p, g in zip(*np.where(overlaps >= threshold))), reverse=True)
    used_pred, used_gt, matches = set(), set(), []
    matrix = np.zeros((N, N), dtype=np.int64)  # rows=predicted, columns=true; last=background
    for overlap, pi, gi in pairs:
        if pi not in used_pred and gi not in used_gt:
            used_pred.add(pi)
            used_gt.add(gi)
            matches.append((overlap, pi, gi))
            matrix[plabels[pi], gt_labels[gi]] += 1
    for pi, label in enumerate(plabels):
        if pi not in used_pred:
            matrix[label, nc] += 1
    for gi, label in enumerate(gt_labels):
        if gi not in used_gt:
            matrix[nc, label] += 1
    tp = int(np.trace(matrix[:nc, :nc]))
    fp = int(matrix[:nc, :].sum()) - tp
    fn = int(matrix[:, :nc].sum()) - tp
    row = {"gt": len(gt_boxes), "gt_small": int(np.sum(areas[~crowds] <= 32**2)),
           "detections": len(pboxes), "crowd_ignored_detections": int(ignored.sum()),
           "tp": tp, "fp": fp, "fn": fn,
           "precision": tp/(tp+fp) if tp+fp else None, "recall": tp/(tp+fn) if tp+fn else None,
           "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
           "matched_iou_mean": float(np.mean([value for value, _, _ in matches])) if matches else None}
    return matrix, row


def mean_valid(value):
    array = np.asarray(value)
    valid = array[array >= 0]
    return float(valid.mean()) if valid.size else None


def evaluate_predictions(coco, predictions, cfg, directory, epoch, split, report_confidence=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if split == "test" and report_confidence is None:
        raise ValueError("测试集 P/R/F1 必须沿用最佳验证集确定的阈值，不能在 test 上重新选阈值。")
    converted, matching, per_image, matrix = [], [], [], np.zeros((N, N), dtype=np.int64)
    gt_counts, images_per_class = np.zeros(nc, dtype=int), np.zeros(nc, dtype=int)
    for image_id in sorted(coco.imgs):
        pred = predictions.get(image_id, {"boxes": [], "scores": [], "labels": []})
        annotations = coco.imgToAnns[image_id]
        boxes, labels, crowds, _ = gt_arrays(annotations)
        gt_counts += np.bincount(labels[~crowds], minlength=nc)
        images_per_class += np.asarray([np.any(labels[~crowds] == i) for i in range(nc)], dtype=int)
        matching.append(match_predictions(pred, annotations, cfg.metric_conf))
        cm, row = fixed_confusion(pred, annotations, cfg.confusion_conf, cfg.confusion_iou)
        matrix += cm
        per_image.append({"epoch": epoch, "split": split, "image_id": image_id,
                          "image": coco.imgs[image_id]["file_name"], **row})
        for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
            x1, y1, x2, y2 = map(float, box)
            converted.append({"image_id": image_id, "category_id": int(label),
                              "bbox": [x1, y1, max(0., x2-x1), max(0., y2-y1)], "score": float(score)})
    capture = io.StringIO()
    with redirect_stdout(capture):
        if converted:
            dt = coco.loadRes(converted)
        else:
            dt = COCO()
            dt.dataset = {"images": deepcopy(coco.dataset["images"]), "categories": deepcopy(coco.dataset["categories"]), "annotations": []}
            dt.createIndex()
        evaluator = COCOeval(coco, dt, "bbox")
        evaluator.params.imgIds = sorted(coco.imgs)
        evaluator.params.catIds = list(range(nc))
        evaluator.params.maxDets = [1, 10, cfg.coco_max_dets]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    (directory/"coco_evaluation.txt").write_text(capture.getvalue(), encoding="utf-8")
    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    np.savez_compressed(directory/"coco_precision_recall.npz", precision=precision, recall=recall,
                        scores=evaluator.eval["scores"], iou_thresholds=evaluator.params.iouThrs,
                        recall_thresholds=evaluator.params.recThrs)
    scores = np.concatenate([x[0] for x in matching])
    labels = np.concatenate([x[1] for x in matching])
    correct = np.concatenate([x[2] for x in matching])
    ignored = np.concatenate([x[3] for x in matching])
    order = np.argsort(-scores, kind="stable")
    scores, labels, correct, ignored = scores[order], labels[order], correct[order], ignored[order]
    p_curve = np.zeros((nc, len(CONFIDENCES)))
    r_curve, f1_curve = np.zeros_like(p_curve), np.zeros_like(p_curve)
    diagnostic_ap = np.full((nc, 10), np.nan)
    for category in range(nc):
        if gt_counts[category] == 0:
            continue
        for ti in range(10):
            mask = (labels == category) & ~ignored[:, ti]
            values, confidence = correct[mask, ti], scores[mask]
            tp, fp = np.cumsum(values), np.cumsum(~values)
            rec = tp/gt_counts[category]
            prec = tp/np.maximum(tp+fp, 1)
            # Same 101 recall samples as COCO, with one-to-one score-ordered matches.
            envelope = np.maximum.accumulate(prec[::-1])[::-1]
            sampled = [float(envelope[np.searchsorted(rec, q, side="left")]) if np.searchsorted(rec, q, side="left") < len(envelope) else 0.
                       for q in np.linspace(0, 1, 101)]
            diagnostic_ap[category, ti] = float(np.mean(sampled))
            if ti == 0:
                count = np.searchsorted(-confidence, -CONFIDENCES, side="right")
                selected_tp = np.where(count > 0, np.r_[0, tp][count], 0)
                p_curve[category] = selected_tp/np.maximum(count, 1)
                r_curve[category] = selected_tp/gt_counts[category]
                f1_curve[category] = 2*p_curve[category]*r_curve[category]/np.maximum(p_curve[category]+r_curve[category], 1e-12)
    present = gt_counts > 0
    if report_confidence is None:
        index = int(np.argmax(f1_curve[present].mean(0))) if present.any() else 0
    else:
        if not 0 <= report_confidence <= 1:
            raise ValueError("report_confidence 必须在 [0,1]")
        index = int(np.argmin(np.abs(CONFIDENCES-report_confidence)))
    per_class = []
    for category, name in enumerate(CLASS_NAMES):
        exists = bool(present[category])
        per_class.append({"epoch": epoch, "split": split, "class_id": category, "class_name": name,
                          "images": int(images_per_class[category]), "instances": int(gt_counts[category]),
                          "precision": float(p_curve[category, index]) if exists else None,
                          "recall": float(r_curve[category, index]) if exists else None,
                          "f1": float(f1_curve[category, index]) if exists else None,
                          "confidence": float(CONFIDENCES[index]),
                          "AP50": mean_valid(precision[0, :, category, 0, -1]),
                          "AP": mean_valid(precision[:, :, category, 0, -1]),
                          "AP75": mean_valid(precision[5, :, category, 0, -1]),
                          "APsmall": mean_valid(precision[:, :, category, 1, -1]),
                          "APmedium": mean_valid(precision[:, :, category, 2, -1]),
                          "APlarge": mean_valid(precision[:, :, category, 3, -1]),
                          "AR100": mean_valid(recall[:, category, 0, -1]),
                          "ARsmall": mean_valid(recall[:, category, 1, -1]),
                          "ARmedium": mean_valid(recall[:, category, 2, -1]),
                          "ARlarge": mean_valid(recall[:, category, 3, -1]),
                          "diagnostic_AP50": float(diagnostic_ap[category, 0]) if exists else None,
                          "diagnostic_AP": float(diagnostic_ap[category].mean()) if exists else None})
    summary = {"epoch": epoch, "split": split, "images": len(coco.imgs), "instances": int(gt_counts.sum()),
               "precision": float(p_curve[present, index].mean()) if present.any() else None,
               "recall": float(r_curve[present, index].mean()) if present.any() else None,
               "f1": float(f1_curve[present, index].mean()) if present.any() else None,
               "confidence": float(CONFIDENCES[index]), "AP": mean_valid(precision[:, :, :, 0, -1]),
               "confidence_source": "val_max_macro_F1" if report_confidence is None else "fixed_from_validation",
               "AP50": mean_valid(precision[0, :, :, 0, -1]), "AP75": mean_valid(precision[5, :, :, 0, -1]),
               "APsmall": mean_valid(precision[:, :, :, 1, -1]), "APmedium": mean_valid(precision[:, :, :, 2, -1]),
               "APlarge": mean_valid(precision[:, :, :, 3, -1]), "AR1": mean_valid(recall[:, :, 0, 0]),
               "AR10": mean_valid(recall[:, :, 0, 1]), "AR100": mean_valid(recall[:, :, 0, -1]),
               "ARsmall": mean_valid(recall[:, :, 1, -1]), "ARmedium": mean_valid(recall[:, :, 2, -1]), "ARlarge": mean_valid(recall[:, :, 3, -1]),
               "diagnostic_AP50": float(np.nanmean(diagnostic_ap[:, 0])) if present.any() else None,
               "diagnostic_AP": float(np.nanmean(diagnostic_ap)) if present.any() else None,
               "coco_max_dets": cfg.coco_max_dets, "prediction_candidates": cfg.max_det,
               "diagnostic_min_conf": cfg.metric_conf, "confusion_conf": cfg.confusion_conf, "confusion_iou": cfg.confusion_iou}
    atomic_csv(directory/"metrics.csv", [summary], list(summary))
    atomic_csv(directory/"per_class.csv", per_class, list(per_class[0]))
    atomic_csv(directory/"per_image.csv", per_image, list(per_image[0]))
    cm_rows = [{"predicted": CLASS_NAMES[p] if p < nc else "background", "true": CLASS_NAMES[g] if g < nc else "background", "count": int(matrix[p, g])}
               for p in range(N) for g in range(N)]
    atomic_csv(directory/"confusion_matrix.csv", cm_rows, ["predicted", "true", "count"])
    confidence_rows = ({"class_id": c, "class_name": CLASS_NAMES[c], "confidence": float(conf),
                        "precision": float(p_curve[c, i]) if present[c] else None,
                        "recall": float(r_curve[c, i]) if present[c] else None,
                        "f1": float(f1_curve[c, i]) if present[c] else None}
                       for c in range(nc) for i, conf in enumerate(CONFIDENCES))
    atomic_csv(directory/"confidence_curves.csv", confidence_rows, ["class_id", "class_name", "confidence", "precision", "recall", "f1"])
    pr_rows = ({"class_id": c, "class_name": CLASS_NAMES[c], "iou": float(iou), "recall": float(rec),
                "precision": float(precision[t, r, c, 0, -1]) if precision[t, r, c, 0, -1] >= 0 else None}
               for c in range(nc) for t, iou in enumerate(evaluator.params.iouThrs) for r, rec in enumerate(evaluator.params.recThrs))
    atomic_csv(directory/"pr_curves.csv", pr_rows, ["class_id", "class_name", "iou", "recall", "precision"])
    iou_rows = [{"class_id": c, "class_name": CLASS_NAMES[c], "iou": float(iou),
                 "AP": mean_valid(precision[t, :, c, 0, -1])} for c in range(nc) for t, iou in enumerate(evaluator.params.iouThrs)]
    atomic_csv(directory/"ap_by_iou.csv", iou_rows, ["class_id", "class_name", "iou", "AP"])
    write_json(directory/"metric_definitions.json", {
        "headline_AP": "COCOeval bbox, IoU .50:.95, maxDets=100; original annotation area and original pixels",
        "APsmall": "COCO small area range [0,32^2]; no GT gives null, not zero",
        "precision_recall_f1": "macro mean over GT-present classes; same-class IoU=.5 matching; validation chooses max macro-F1 confidence and test uses that frozen confidence",
        "confidence_selection": "validation chooses descriptive max macro-F1 confidence; labeled test uses this frozen validation confidence; never optimizes a test threshold",
        "diagnostic_AP": "101-point interpolated AP, score-ordered same-class matching, min_conf=config.metric_conf, max_det candidates",
        "confusion_matrix": "rows predicted, columns true; background last; fixed confidence/IoU from config; class-agnostic greedy IoU assignment",
        "crowd": "native training excludes crowd; COCOeval applies native ignore rules; diagnostics ignore same-class crowd overlaps",
    })
    if cfg.save_epoch_predictions or directory.parent.name == "final":
        write_json(directory/"predictions.json", converted)
    return {"summary": summary, "per_class": per_class, "per_image": per_image,
            "matrix": matrix, "precision_tensor": precision, "p_curve": p_curve, "r_curve": r_curve,
            "f1_curve": f1_curve, "present": present, "directory": str(directory)}
