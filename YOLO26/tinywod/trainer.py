"""官方 YOLO26 训练器的实验记录适配层；保留原生网络、ProgLoss/STAL 和优化流程。"""
import copy
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from ultralytics import __version__
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import DetMetrics, Metric, ap_per_class, box_iou
from ultralytics.utils.torch_utils import unwrap_model

from .common import append_jsonl, experiment_signature, f1_from_pr, now, write_csv, write_json


class SmallDetMetrics(DetMetrics):
    """在标准检测指标(P/R/mAP50/mAP50-95)基础上追加 COCO 风格小目标 AP(<32² 像素)。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.small = Metric()

    @property
    def keys(self):
        return super().keys + ["metrics/mAP50-95-small(B)"]

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


class ComparisonValidator(DetectionValidator):
    @staticmethod
    def _check_max_det(args, datasets):
        """上游会把默认 max_det=300 按数据自动增大；本实验始终保持配置值。"""
        maximum = max((len(label["cls"]) for dataset in datasets.values()
                       for label in getattr(dataset, "labels", []) if isinstance(label, dict)), default=0)
        if maximum > args.max_det:
            raise ValueError(f"图像最多有 {maximum} 个目标，超过固定 max_det={args.max_det}。"
                             "请在 config.py 为两个模型统一增大 max_det 后建立新实验。")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = SmallDetMetrics()
        self.small_stats = []
        self._model_parameters = None
        self._gflops = None
        self._last_stats = None

    def init_metrics(self, model):
        super().init_metrics(model)
        self.small_stats = []
        self.metrics.small = Metric()
        self.metrics.small.nc = self.nc
        self._cache_complexity(model)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _cache_complexity(self, model):
        # 参数量与 GFLOPs 只计算一次并缓存，随后每轮验证写入 CSV 并打印。
        if self._model_parameters is None:
            params = sum(p.numel() for p in model.parameters())
            if not params:  # 最终评估用 AutoBackend 加载，真实模型在 .model 属性
                params = sum(p.numel() for p in getattr(model, "model", model).parameters())
            self._model_parameters = params
        if self._gflops is None:
            from .reporting import compute_gflops
            self._gflops = compute_gflops(model, self.args.imgsz)

    def get_desc(self):
        return ("%22s" + "%11s" * 7) % ("Class", "Images", "Instances", "Box(P", "R", "AP50", "AP", "APsmall")

    def build_dataset(self, img_path, mode="val", batch=None):
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=self.stride)

    def get_dataloader(self, dataset_path, batch_size):
        dataset = self.build_dataset(dataset_path, batch=batch_size)
        return build_dataloader(dataset, batch=batch_size, workers=self.args.workers,
                                shuffle=False, rank=-1, device=self.device)

    def update_metrics(self, preds, batch):
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            cls = pbatch["cls"].cpu().numpy()
            predn = self._prepare_pred(pred)
            no_pred = predn["cls"].shape[0] == 0

            self.metrics.update_stats({
                **self._process_batch(predn, pbatch),
                "target_cls": cls,
                "target_img": np.unique(cls),
                "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                "im_name": Path(pbatch["im_file"]).name,
            })
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)

            # 小目标 AP 统计（<32² imgsz 像素），与 YOLOv10 对齐。
            gt_bboxes = pbatch["bboxes"]
            gt_cls = pbatch["cls"]
            small_gt = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) < 32 ** 2
            if no_pred:
                if gt_cls.numel():
                    self.small_stats.append((
                        torch.zeros(0, self.niou, dtype=torch.bool, device=self.device),
                        torch.zeros(0, self.niou, dtype=torch.bool, device=self.device),
                        torch.zeros(0, device=self.device),
                        torch.zeros(0, device=self.device),
                        gt_cls[small_gt],
                    ))
                continue

            correct_small, ignore, small_classes = self._process_small_batch(predn, gt_bboxes, gt_cls)
            self.small_stats.append((correct_small, ignore, predn["conf"], predn["cls"], small_classes))

            if self.args.save_json or self.args.save_txt:
                predn_scaled = self.scale_preds(predn, pbatch)
            if self.args.save_json:
                self.pred_to_json(predn_scaled, pbatch)
            if self.args.save_txt:
                name = "_".join(Path(pbatch["im_file"]).relative_to(self.label_root).with_suffix("").parts)
                self.save_one_txt(predn_scaled, self.args.save_conf, pbatch["ori_shape"],
                                  self.save_dir / "labels" / f"{name}.txt")

    def _process_small_batch(self, detections, gt_bboxes, gt_cls):
        # 返回小目标(<32²)的 TP 与 COCO 风格 ignore 矩阵。
        npr = detections["cls"].shape[0]
        correct_small = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)
        ignored_match = torch.zeros_like(correct_small)
        if gt_cls.numel():
            wh = (gt_bboxes[:, 2:4] - gt_bboxes[:, :2]).clamp_min(0)
            small_gt = wh.prod(1) < 32 ** 2
            if small_gt.any():
                iou = box_iou(gt_bboxes[small_gt], detections["bboxes"])
                correct_small = self.match_predictions(detections["cls"], gt_cls[small_gt], iou).to(self.device)
            if (~small_gt).any():
                iou = box_iou(gt_bboxes[~small_gt], detections["bboxes"])
                ignored_match = self.match_predictions(detections["cls"], gt_cls[~small_gt], iou).to(self.device)
            small_classes = gt_cls[small_gt]
        else:
            small_classes = gt_cls.new_zeros((0,))
        det_wh = (detections["bboxes"][:, 2:4] - detections["bboxes"][:, :2]).clamp_min(0)
        outside_small_range = det_wh.prod(1) >= 32 ** 2
        ignore = ~correct_small & (ignored_match | outside_small_range.unsqueeze(1))
        return correct_small, ignore, small_classes

    def get_stats(self):
        super().get_stats()  # 处理标准指标并清空 stats（副作用）
        if self.small_stats:
            small = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.small_stats)]
            if len(small) and small[-1].size:
                self.metrics.process_small(small[0], small[1], small[2], small[3], small[4])
        self._last_stats = self.metrics.results_dict
        return self._last_stats

    def finalize_metrics(self):
        super().finalize_metrics()
        if self._last_stats is None:
            return
        latency = float(self.speed.get("inference", 0.0))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_gb = torch.cuda.max_memory_allocated(self.device) / (1 << 30)
        else:
            peak_gb = 0.0
        self._last_stats.update({
            "complexity/Parameters": int(self._model_parameters or 0),
            "complexity/GFLOPs": float(self._gflops or 0.0),
            "speed/val-latency-ms": latency,
            "speed/val-FPS": 1000.0 / latency if latency > 0 else 0.0,
            "memory/PeakAllocated-GB": peak_gb,
        })

    def print_results(self):
        super().print_results()
        if self._last_stats is not None:
            LOGGER.info(
                "Complexity: Params=%d, GFLOPs=%.3f, val inference=%.3f ms/image (%.2f FPS), PeakAllocated=%.3f GB" % (
                    self._last_stats["complexity/Parameters"],
                    self._last_stats["complexity/GFLOPs"],
                    self._last_stats["speed/val-latency-ms"],
                    self._last_stats["speed/val-FPS"],
                    self._last_stats["memory/PeakAllocated-GB"],
                )
            )


class ComparisonTrainer(DetectionTrainer):
    def __init__(self, *args, experiment_cfg, dataset_info, variant, **kwargs):
        self.experiment_cfg, self.dataset_info, self.variant = experiment_cfg, dataset_info, variant
        self.resume_payload = None
        self.history = []
        self.last_epoch_start = None
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_epoch_start", self.epoch_start)
        self.add_callback("on_train_start", self.training_start)
        self.add_callback("on_fit_epoch_end", self.epoch_end)
        if self.resume_payload:
            self.history = self.resume_payload["sds"]["epochs"]
            old_csv = self.resume_payload.get("train_results", {})
            if old_csv:
                rows = [dict(zip(old_csv, values)) for values in zip(*old_csv.values())]
                write_csv(self.csv, rows)
            write_csv(self.save_dir / "metrics" / "epochs.csv", self.history)
            jsonl = self.save_dir / "metrics" / "epochs.jsonl"
            jsonl.write_text("", encoding="utf-8")
            for row in self.history:
                append_jsonl(jsonl, row)
            source = Path(self.args.resume)
            best = source.parent / "best.pt"
            if not best.is_file():
                raise FileNotFoundError("恢复时须保留同目录 best.pt，以免丢失此前最佳模型")
            best_ckpt = torch.load(best, map_location="cpu", weights_only=False)
            if best_ckpt.get("epoch", -1) > self.resume_payload["epoch"]:
                raise ValueError("同目录 best.pt 比所选断点更新，无法正确回退，请使用匹配的断点及 best.pt")
            for key in ("variant", "dataset_fingerprint", "experiment_signature"):
                if best_ckpt.get("sds", {}).get(key) != self.resume_payload["sds"][key]:
                    raise ValueError(f"恢复的 best.pt {key} 与断点不匹配")
            import shutil
            shutil.copy2(best, self.best)
            del best_ckpt

    def check_resume(self, overrides):
        self.resume = bool(self.args.resume)
        if not self.resume:
            return
        source = Path(str(self.args.resume)).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"恢复文件不存在: {source}")
        ckpt = torch.load(source, map_location="cpu", weights_only=False)
        meta = ckpt.get("sds", {})
        expected = {"variant": self.variant, "dataset_fingerprint": self.dataset_info["fingerprint"],
                    "experiment_signature": experiment_signature(self.experiment_cfg)}
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(f"断点 {key} 与当前实验不一致；不允许把不同模型/数据/设置接到同一训练")
        if ckpt.get("optimizer") is None or ckpt.get("model") is None or ckpt.get("scheduler") is None:
            raise ValueError("仅完整 last.pt/epochXXX.pt 支持恢复；best.pt 为推理模型")
        if not 0 < ckpt.get("epoch", -1)+1 < self.experiment_cfg.epochs:
            raise ValueError("断点已完成总轮数，或不是可恢复的训练断点")
        if len(meta.get("epochs", [])) != ckpt["epoch"]+1:
            raise ValueError("断点逐轮 CSV 历史不完整")
        self.args.model = self.args.resume = str(source)
        self.resume_payload = ckpt

    def setup_model(self):
        ckpt = super().setup_model()
        if self.resume_payload:
            # 原生 load_checkpoint 选择 EMA 用于推理；继续训练必须恢复真实训练权重。
            self.model.load_state_dict(self.resume_payload["model"].float().state_dict())
            ckpt = self.resume_payload
        return ckpt

    def _setup_train(self):
        # 原生 AMP 自检下载另一种 YOLO 权重；本工程离线检查当前模型有限值，失败直接报错。
        import ultralytics.engine.trainer as native_trainer
        original = native_trainer.check_amp
        native_trainer.check_amp = self.check_current_model_amp
        try:
            super()._setup_train()
        finally:
            native_trainer.check_amp = original
        if self.resume_payload:
            saved = self.resume_payload
            self.scheduler.load_state_dict(saved["scheduler"])
            rng = saved["sds"]["rng"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("cuda"):
                torch.cuda.set_rng_state_all(rng["cuda"])
            if rng.get("loader") is not None:
                self.train_loader.generator.set_state(rng["loader"])
            self.resume_payload = None  # 释放 CPU 模型与优化器副本
        actual = {**vars(self.args), "effective_workers": self.args.workers,
                  "train_batch": self.train_loader.batch_size, "val_batch": self.test_loader.batch_size,
                  "train_rect": self.train_loader.dataset.rect, "val_rect": self.test_loader.dataset.rect,
                  "loss": "native YOLO26 E2ELoss / ProgLoss / STAL", "nms": False}
        write_json(self.save_dir / "inputs" / "effective_train_args.json", actual)
        write_json(self.save_dir / "inputs" / "optimizer.json", {"type": type(self.optimizer).__name__,
                   "defaults": self.optimizer.defaults,
                   "param_groups": [{k: v for k, v in group.items() if k != "params"} for group in self.optimizer.param_groups],
                   "accumulate": self.accumulate, "scheduler": self.scheduler.state_dict()})
        print(f"[训练设备] {self.device}；实际 workers={self.args.workers}；train/val batch="
              f"{self.train_loader.batch_size}/{self.test_loader.batch_size}；原生 YOLO26 损失，NMS-free", flush=True)

    def check_current_model_amp(self, model):
        training = model.training
        try:
            model.eval()
            image = torch.zeros((1, 3, self.experiment_cfg.imgsz, self.experiment_cfg.imgsz), device=self.device)
            def tensors(value):
                if torch.is_tensor(value):
                    yield value
                elif isinstance(value, (tuple, list)):
                    for item in value:
                        yield from tensors(item)
                elif isinstance(value, dict):
                    for item in value.values():
                        yield from tensors(item)
            with torch.no_grad():
                for enabled in (False, True):
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled):
                        outputs = list(tensors(model(image)))
                    if not outputs or not all(torch.isfinite(v).all() for v in outputs):
                        raise FloatingPointError("当前 YOLO26 模型 AMP/FP32 初始化自检出现非有限输出")
            print("[AMP] 当前模型 FP32/FP16 有限值检查通过（不是精度等价性保证）", flush=True)
            return True
        finally:
            model.train(training)

    def build_dataset(self, img_path, mode="train", batch=None):
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=stride)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        model.end2end = True
        return model

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        # 避免原生 val 自动 batch*2 / workers*2；两模型及 train/val 都使用集中配置。
        batch_size = self.experiment_cfg.batch
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        return build_dataloader(dataset, batch=batch_size, workers=self.args.workers, shuffle=mode == "train",
                                rank=rank, device=self.device, drop_last=False, pin_memory=self.experiment_cfg.pin_memory)

    def _build_train_pipeline(self):
        BaseTrainer._build_train_pipeline(self)
        ComparisonValidator._check_max_det(self.args, {"train": self.train_loader.dataset, "val": self.test_loader.dataset})
        unwrap_model(self.model).set_head_attr(max_det=self.args.max_det)

    def get_validator(self):
        return ComparisonValidator(self.test_loader, save_dir=self.save_dir / "results" / "epoch_validation",
                                   args=copy.copy(self.args), _callbacks=self.callbacks)

    def _handle_nan_recovery(self, epoch):
        if self.loss is not None and not torch.isfinite(self.loss).all():
            raise FloatingPointError(f"Epoch {epoch+1} 出现非有限损失，拒绝自动回退或修补权重")
        if self.fitness is not None and not math.isfinite(float(self.fitness)):
            raise FloatingPointError(f"Epoch {epoch+1} 出现非有限验证 fitness")
        return False

    def save_metrics(self, metrics):
        super().save_metrics(metrics)
        p, r = float(metrics.get("metrics/precision(B)", 0)), float(metrics.get("metrics/recall(B)", 0))
        row = {"epoch": self.epoch+1, "time": time.time()-self.train_time_start,
               "wall_time": now(), "epoch_time_s": time.monotonic()-self.last_epoch_start,
               **{k: float(v) for k, v in metrics.items()}, "F1_from_mean_PR": f1_from_pr(p, r)}
        if len(self.history) != self.epoch:
            raise RuntimeError(f"逐轮记录顺序异常: 历史{len(self.history)}，当前epoch={self.epoch+1}")
        self.history.append(row)
        write_csv(self.save_dir / "metrics" / "epochs.csv", self.history)
        append_jsonl(self.save_dir / "metrics" / "epochs.jsonl", row)

    def epoch_start(self, trainer):
        self.last_epoch_start = time.monotonic()
        print(f"[{self.variant.upper()}] Epoch {self.epoch+1}/{self.epochs}", flush=True)

    def training_start(self, trainer):
        if self.history:
            self.train_time_start -= float(self.history[-1]["time"])
            print(f"[恢复] 接续 {len(self.history)} 轮历史；累计训练时间从 {self.history[-1]['time']:.2f}s 继续", flush=True)

    def epoch_end(self, trainer):
        row = self.history[-1]
        p = row.get("metrics/precision(B)", 0)
        r = row.get("metrics/recall(B)", 0)
        ap50 = row.get("metrics/mAP50(B)", 0)
        ap = row.get("metrics/mAP50-95(B)", 0)
        apsmall = row.get("metrics/mAP50-95-small(B)", 0.0)
        lr = {k: v for k, v in row.items() if k.startswith("lr/")}
        losses = {k: v for k, v in row.items() if k.startswith(("train/", "val/"))}
        print(f"[{self.variant.upper()}] Epoch {self.epoch+1}/{self.epochs} 完成 | "
              f"Box(P={p:.5f} R={r:.5f} AP50={ap50:.5f} AP={ap:.5f} APsmall={apsmall:.5f}) | "
              f"lr={lr} | loss={losses}", flush=True)
        write_json(self.save_dir / "progress.json", {"status": "training", "completed_epochs": self.epoch+1,
                                                    "epochs": self.epochs, "last_epoch": row})

    def save_model(self):
        """原子保存 FP32 训练权重、EMA、优化器、scheduler、随机状态及 CSV 历史。"""
        model = unwrap_model(self.model)
        for label, module in (("model", model), ("ema", self.ema.ema)):
            if not all(torch.isfinite(v).all() for v in module.state_dict().values() if torch.is_tensor(v)):
                raise FloatingPointError(f"{label} 出现 NaN/Inf，拒绝用修补过的权重冒充成功")
        def cpu_copy(module):
            # 原始 criterion 引用 CUDA/训练缓存；不随推理权重序列化。
            criterion = getattr(module, "criterion", None)
            module.criterion = None
            try:
                return copy.deepcopy(module).cpu().float().to(memory_format=torch.contiguous_format)
            finally:
                module.criterion = criterion
        state = {"variant": self.variant, "dataset_fingerprint": self.dataset_info["fingerprint"],
                 "experiment_signature": experiment_signature(self.experiment_cfg), "epochs": self.history,
                 "criterion_updates": getattr(model.criterion, "updates", None),
                 "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                         "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                         "loader": self.train_loader.generator.get_state()}}
        checkpoint = {"epoch": self.epoch, "model": cpu_copy(model), "ema": cpu_copy(self.ema.ema),
                      "optimizer": self.optimizer.state_dict(), "scaler": self.scaler.state_dict(),
                      "scheduler": self.scheduler.state_dict(), "updates": self.ema.updates,
                      "best_fitness": self.best_fitness, "train_args": vars(self.args),
                      "train_metrics": {**self.metrics, "fitness": self.fitness},
                      "train_results": self.read_results_csv(), "version": __version__, "date": now(), "sds": state,  # "sds" 为检查点内部字段名，保持旧值以兼容既有模型
                      "stopper": {"best_fitness": self.stopper.best_fitness, "best_epoch": self.stopper.best_epoch},
                      "license": "AGPL-3.0", "docs": "https://docs.ultralytics.com/models/yolo26/"}
        def atomic_save(payload, path):
            temporary = Path(path).with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            os.replace(temporary, path)
        atomic_save(checkpoint, self.last)
        if self.best_fitness == self.fitness:
            best = {**checkpoint, "model": None, "optimizer": None, "scheduler": None, "scaler": None}
            atomic_save(best, self.best)
        if self.save_period > 0 and (self.epoch+1) % self.save_period == 0:
            import shutil
            shutil.copy2(self.last, self.wdir / f"epoch{self.epoch+1:03d}.pt")
        return True

    def final_eval(self):
        # 不剥除 last.pt 的优化器，也不生成第201行记录；worker 统一评价 best.pt 的 val/test。
        print(f"[权重] best={self.best}\n[权重] last={self.last}（保留完整恢复状态）", flush=True)

    def plot_metrics(self):
        super().plot_metrics()  # 保留框架原生 results.png
        from .reporting import plot_history
        plot_history(self.history, self.save_dir / "results")
