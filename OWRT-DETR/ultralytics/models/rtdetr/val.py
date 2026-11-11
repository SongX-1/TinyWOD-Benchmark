# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

import numpy as np
import torch

from ultralytics.data import YOLODataset
from ultralytics.data.augment import Compose, Format, v8_transforms
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, colorstr, ops
from ultralytics.utils.metrics import DetMetrics, Metric, ap_per_class
from ultralytics.utils.torch_utils import get_flops, get_num_params

__all__ = 'RTDETRValidator',  # tuple or list


class RTDETRMetrics(DetMetrics):
    """Detection metrics extended with COCO-style APsmall (<32^2 pixels)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.small = Metric()

    @property
    def keys(self):
        return super().keys + ['metrics/mAP50-95-small(B)']

    def mean_results(self):
        return super().mean_results() + [self.small.map]

    def class_result(self, i):
        class_id = int(self.box.ap_class_index[i])
        small_ap = 0.0
        matches = np.flatnonzero(np.asarray(self.small.ap_class_index) == class_id)
        if len(matches):
            small_ap = float(self.small.ap[matches[0]])
        return super().class_result(i) + (small_ap,)

    def process_small(self, tp, ignore, conf, pred_cls, target_cls):
        results = ap_per_class(tp, conf, pred_cls, target_cls, names=self.names, ignore=ignore)[2:]
        self.small.nc = len(self.names)
        self.small.update(results)


class RTDETRDataset(YOLODataset):
    """
    Real-Time DEtection and TRacking (RT-DETR) dataset class extending the base YOLODataset class.

    This specialized dataset class is designed for use with the RT-DETR object detection model and is optimized for
    real-time detection and tracking tasks.
    """

    def __init__(self, *args, data=None, **kwargs):
        """Initialize the RTDETRDataset class by inheriting from the YOLODataset class."""
        super().__init__(*args, data=data, use_segments=False, use_keypoints=False, **kwargs)

    # NOTE: add stretch version load_image for RTDETR mosaic
    def load_image(self, i, rect_mode=False):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        return super().load_image(i=i, rect_mode=rect_mode)

    def build_transforms(self, hyp=None):
        """Temporary, only for evaluation."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp, stretch=True)
        else:
            # transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), auto=False, scaleFill=True)])
            transforms = Compose([])
        transforms.append(
            Format(bbox_format='xywh',
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask))
        return transforms


class RTDETRValidator(DetectionValidator):
    """
    RTDETRValidator extends the DetectionValidator class to provide validation capabilities specifically tailored for
    the RT-DETR (Real-Time DETR) object detection model.

    The class allows building of an RTDETR-specific dataset for validation, applies Non-maximum suppression for
    post-processing, and updates evaluation metrics accordingly.

    Note:
        For further details on the attributes and methods, refer to the parent DetectionValidator class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = RTDETRMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.small_stats = []
        self._dense_gflops = None
        self._model_parameters = None
        self._last_stats = None

    def init_metrics(self, model):
        super().init_metrics(model)
        self.small_stats = []
        self.metrics.small = Metric()
        self.metrics.small.nc = len(self.names)
        if self._model_parameters is None:
            self._model_parameters = get_num_params(model)
        if self._dense_gflops is None:
            self._dense_gflops = float(get_flops(model, self.args.imgsz))
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

    def get_desc(self):
        return ('%22s' + '%11s' * 7) % (
            'Class', 'Images', 'Instances', 'Box(P', 'R', 'AP50', 'AP', 'APsmall')

    def build_dataset(self, img_path, mode='val', batch=None):
        """
        Build an RTDETR Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,  # no augmentation
            hyp=self.args,
            rect=False,  # no rect
            cache=self.args.cache or None,
            prefix=colorstr(f'{mode}: '),
            data=self.data)

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        bs, _, nd = preds[0].shape
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
        bboxes *= self.args.imgsz
        outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs
        for i, bbox in enumerate(bboxes):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            score, cls = scores[i].max(-1)  # (300, )
            # Do not need threshold for evaluation as only got 300 boxes here
            # idx = score > self.args.conf
            pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)  # filter
            # Sort by confidence to correctly get internal metrics
            pred = pred[score.argsort(descending=True)]
            outputs[i] = pred  # [idx]

        return outputs

    def update_metrics(self, preds, batch):
        """Update standard AP and COCO small-object AP statistics."""
        for si, pred in enumerate(preds):
            idx = batch['batch_idx'] == si
            cls = batch['cls'][idx]
            bbox = batch['bboxes'][idx]
            nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
            shape = batch['ori_shape'][si]
            correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
            self.seen += 1

            if npr == 0:
                small = (bbox[:, 2] * shape[1]) * (bbox[:, 3] * shape[0]) < 32 ** 2
                empty_ignore = torch.zeros_like(correct_bboxes)
                self.small_stats.append((correct_bboxes, empty_ignore,
                                         *torch.zeros((2, 0), device=self.device), cls[small].squeeze(-1)))
                if nl:
                    self.stats.append((correct_bboxes, *torch.zeros((2, 0), device=self.device), cls.squeeze(-1)))
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            predn[..., [0, 2]] *= shape[1] / self.args.imgsz  # native-space pred
            predn[..., [1, 3]] *= shape[0] / self.args.imgsz  # native-space pred

            # Evaluate
            if nl:
                tbox = ops.xywh2xyxy(bbox)  # target boxes
                tbox[..., [0, 2]] *= shape[1]  # native-space pred
                tbox[..., [1, 3]] *= shape[0]  # native-space pred
                labelsn = torch.cat((cls, tbox), 1)  # native-space labels
                # NOTE: To get correct metrics, the inputs of `_process_batch` should always be float32 type.
                correct_bboxes = self._process_batch(predn.float(), labelsn)
                # TODO: maybe remove these `self.` arguments as they already are member variable
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, labelsn)
            else:
                labelsn = predn.new_zeros((0, 5))
            self.stats.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls)

            correct_small, ignore_small, small_classes = self._process_small_batch(predn.float(), labelsn.float())
            self.small_stats.append(
                (correct_small, ignore_small, pred[:, 4], pred[:, 5], small_classes))

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch['im_file'][si])
            if self.args.save_txt:
                file = self.save_dir / 'labels' / f'{Path(batch["im_file"][si]).stem}.txt'
                self.save_one_txt(predn, self.args.save_conf, shape, file)

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

    def get_stats(self):
        stats = super().get_stats()
        if self.small_stats:
            small = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.small_stats)]
            if len(small) and small[-1].size:
                self.metrics.process_small(small[0], small[1], small[2], small[3], small[4])
        stats['metrics/mAP50-95-small(B)'] = self.metrics.small.map
        self._last_stats = stats
        return stats

    def finalize_metrics(self, *args, **kwargs):
        super().finalize_metrics(*args, **kwargs)
        if self._last_stats is None:
            return
        latency = float(self.speed.get('inference', 0.0))
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            peak_gb = torch.cuda.max_memory_allocated(self.device) / (1 << 30)
        else:
            peak_gb = 0.0
        # This OWRT-DETR is a dense RT-DETR (no token sparsity), so every memory
        # token stays active -> active/compact ratio is always 1.0 (100%).
        active_ratio = compact_ratio = 1.0
        self._last_stats.update({
            'complexity/Parameters': int(self._model_parameters or 0),
            'complexity/GFLOPs-dense-upper': float(self._dense_gflops or 0.0),
            'speed/val-latency-ms': latency,
            'speed/val-FPS': 1000.0 / latency if latency > 0 else 0.0,
            'memory/PeakAllocated-GB': peak_gb,
            'sparsity/active-token-ratio': active_ratio,
            'sparsity/compact-token-ratio': compact_ratio,
        })

    def print_results(self):
        super().print_results()
        if self._last_stats is not None:
            LOGGER.info(
                'Complexity: Params=%d, dense-equivalent upper bound=%.3f GFLOPs, '
                'val inference=%.3f ms/image (%.2f FPS), PeakAllocated=%.3f GB, active tokens=%.2f%%' % (
                    self._last_stats['complexity/Parameters'],
                    self._last_stats['complexity/GFLOPs-dense-upper'],
                    self._last_stats['speed/val-latency-ms'],
                    self._last_stats['speed/val-FPS'],
                    self._last_stats['memory/PeakAllocated-GB'],
                    self._last_stats['sparsity/active-token-ratio'] * 100))
