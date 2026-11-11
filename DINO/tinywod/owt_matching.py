# Ultralytics YOLO, AGPL-3.0 license.
# OWT(4).zip: box_iou, match_predictions, _process_batch and
# _process_small_batch copied verbatim; only package imports/class container changed.
import numpy as np
import torch

def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

class OWTMatcher:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.iouv = torch.linspace(0.5, 0.95, 10, device=self.device)
        self.niou = 10

    def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):
            """
            Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

            Args:
                pred_classes (torch.Tensor): Predicted class indices of shape(N,).
                true_classes (torch.Tensor): Target class indices of shape(M,).
                iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
                use_scipy (bool): Whether to use scipy for matching (more precise).

            Returns:
                (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
            """
            # Dx10 matrix, where D - detections, 10 - IoU thresholds
            correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
            # LxD matrix where L - labels (rows), D - detections (columns)
            correct_class = true_classes[:, None] == pred_classes
            iou = iou * correct_class  # zero out the wrong classes
            iou = iou.cpu().numpy()
            for i, threshold in enumerate(self.iouv.cpu().tolist()):
                if use_scipy:
                    # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                    import scipy  # scope import to avoid importing for all commands
                    cost_matrix = iou * (iou >= threshold)
                    if cost_matrix.any():
                        labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                        valid = cost_matrix[labels_idx, detections_idx] > 0
                        if valid.any():
                            correct[detections_idx[valid], i] = True
                else:
                    matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                    matches = np.array(matches).T
                    if matches.shape[0]:
                        if matches.shape[0] > 1:
                            matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                            # matches = matches[matches[:, 2].argsort()[::-1]]
                            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                        correct[matches[:, 1].astype(int), i] = True
            return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def _process_batch(self, detections, labels):
            """
            Return correct prediction matrix.

            Args:
                detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
                    Each detection is of the format: x1, y1, x2, y2, conf, class.
                labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
                    Each label is of the format: class, x1, y1, x2, y2.

            Returns:
                (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
            """
            iou = box_iou(labels[:, 1:], detections[:, :4])
            return self.match_predictions(detections[:, 5], labels[:, 0], iou)

    def _process_small_batch(self, detections, labels):
            """Return TP and COCO-style ignore matrices for objects with area < 32^2 pixels."""
            npr = detections.shape[0]
            correct_small = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)
            ignored_match = torch.zeros_like(correct_small)
            if labels.numel():
                wh = (labels[:, 3:5] - labels[:, 1:3]).clamp_min(0)
                small_gt = wh.prod(1) < 32 ** 2
                if small_gt.any():
                    correct_small = self._process_batch(detections, labels[small_gt])
                if (~small_gt).any():
                    ignored_match = self._process_batch(detections, labels[~small_gt])
                small_classes = labels[small_gt, 0]
            else:
                small_classes = labels.new_zeros((0,))

            det_wh = (detections[:, 2:4] - detections[:, :2]).clamp_min(0)
            outside_small_range = det_wh.prod(1) >= 32 ** 2
            ignore = ~correct_small & (ignored_match | outside_small_range.unsqueeze(1))
            return correct_small, ignore, small_classes
