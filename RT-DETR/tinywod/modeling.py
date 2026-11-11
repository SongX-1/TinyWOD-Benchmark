"""Build the provided v1 RT-DETR-R50vd classes and load local weights only."""
from copy import deepcopy
import math
from pathlib import Path
import sys

from config import CLASS_NAMES, PROJECT_ROOT, resolve
from .common import atomic_csv, sha256, write_json
from .environment import BACKBONE_URL, DETECTOR_URL, weight_preflight


def native_imports():
    native = str(PROJECT_ROOT/"rtdetr_pytorch")
    if native not in sys.path:
        sys.path.insert(0, native)
    import src
    if not Path(src.__file__).resolve().is_relative_to((PROJECT_ROOT/"rtdetr_pytorch").resolve()):
        raise RuntimeError(f"导入了其他项目的 src：{src.__file__}，请使用本项目独立运行配置。")
    from src.core import YAMLConfig
    from src.core.yaml_utils import load_config
    return YAMLConfig, load_config


def loss_keys(model):
    keys = ["loss_bbox", "loss_vfl", "loss_giou"]
    if model.decoder.aux_loss:
        keys += [f"{name}_aux_{i}" for i in range(model.decoder.num_decoder_layers) for name in keys[:3]]
        if model.decoder.num_denoising > 0:
            keys += [f"{name}_dn_{i}" for i in range(model.decoder.num_decoder_layers) for name in keys[:3]]
    return keys


def validate_config(cfg):
    if cfg.epochs < 1 or min(cfg.batch_size, cfg.val_batch_size) < 1 or cfg.num_workers < 0:
        raise ValueError("Epoch / batch 必须为正数，num_workers 不能为负数")
    sizes = [cfg.image_size, *cfg.train_scales]
    if any(int(size) != size or size % 32 or size < 160 for size in sizes):
        raise ValueError("输入尺寸必须为不小于 160 的 32 倍数，以容纳原生 300 queries 并匹配多尺度特征。")
    if cfg.max_det < 1 or cfg.max_det > 300 * len(CLASS_NAMES):
        raise ValueError(f"max_det 超出原生 300 query、{len(CLASS_NAMES)} 类的候选数量")
    if cfg.coco_max_dets != 100:
        raise ValueError("本项目的标准 COCO AP 固定 maxDets=100；候选数量由 max_det 配置。")
    if cfg.dataset_mode not in ("auto", "yolo", "coco") or cfg.test_policy not in ("auto", "unlabeled", "required"):
        raise ValueError("dataset_mode / test_policy 非法")
    if cfg.print_freq < 1 or cfg.save_period < 0 or cfg.seed < 0:
        raise ValueError("print_freq 必须为正，save_period / seed 不能为负")
    if cfg.benchmark_images < 1 or cfg.benchmark_repeats < 1 or cfg.benchmark_warmup < 0:
        raise ValueError("速度测试的图像数/重复次数必须为正，预热次数不能为负")
    if min(cfg.learning_rate, cfg.backbone_learning_rate) <= 0 or min(cfg.weight_decay, cfg.clip_max_norm) < 0:
        raise ValueError("学习率必须为正，weight_decay / clip_max_norm 不能为负")
    if any(not 0 <= value <= 1 for value in (cfg.metric_conf, cfg.confusion_conf, cfg.confusion_iou, cfg.prediction_conf)):
        raise ValueError("置信度 / IoU 阈值必须在 [0,1]")


def build_model(cfg, output, device):
    import torch
    import yaml
    validate_config(cfg)
    output = Path(output)
    YAMLConfig, load_config = native_imports()
    active = deepcopy(load_config(str(resolve(cfg.model_config))))
    if active.get("PResNet", {}).get("depth") != 50 or active["PResNet"].get("variant") != "d":
        raise ValueError("本次要求 R50vd，模型配置必须实际指定 PResNet depth=50, variant=d。")
    active.pop("__include__", None)
    # This YAML instantiates model/criterion/postprocessor only. The adapter
    # records its actual loaders separately, never leaves stale COCO paths here.
    active.pop("train_dataloader", None)
    active.pop("val_dataloader", None)
    active.update(num_classes=len(CLASS_NAMES), remap_mscoco_category=False, epoches=cfg.epochs,
                  output_dir=str(output), use_amp=cfg.mixed_precision != "no", use_ema=cfg.use_ema,
                  sync_bn=False, clip_max_norm=cfg.clip_max_norm)
    # Model initialization always starts offline; loading is explicit below.
    active["PResNet"]["pretrained"] = False
    active["RTDETR"]["multi_scale"] = list(cfg.train_scales) if cfg.train_scales else None
    for section in ("HybridEncoder", "RTDETRTransformer"):
        active[section]["eval_spatial_size"] = [cfg.image_size, cfg.image_size]
    active["RTDETRPostProcessor"]["num_top_queries"] = cfg.max_det
    active["num_top_queries"] = cfg.max_det
    active["optimizer"].update(lr=cfg.learning_rate, betas=list(cfg.betas), weight_decay=cfg.weight_decay)
    for group in active["optimizer"]["params"]:
        if group["params"] == "backbone":
            group["lr"] = cfg.backbone_learning_rate
    active["lr_scheduler"].update(milestones=list(cfg.lr_milestones), gamma=cfg.lr_gamma)
    active["ema"].update(decay=cfg.ema_decay, warmups=cfg.ema_warmups)
    path = output/"inputs"/"native_model_active.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(active, allow_unicode=True, sort_keys=False), encoding="utf-8")
    native_cfg = YAMLConfig(str(path))
    model, criterion, postprocessor = native_cfg.model, native_cfg.criterion, native_cfg.postprocessor
    record = {"initialization": cfg.initialization, "source": None, "path": None, "sha256": None,
              "model_class": str(type(model)), "backbone": "PResNet50 variant=d", "decoder_layers": model.decoder.num_decoder_layers,
              "queries": model.decoder.num_queries, "num_classes": model.decoder.num_classes,
              "model_source": str(PROJECT_ROOT/"rtdetr_pytorch"), "model_config": str(resolve(cfg.model_config))}
    if cfg.resume_checkpoint:
        record.update(initialization="resume", path=str(resolve(cfg.resume_checkpoint)), sha256=sha256(resolve(cfg.resume_checkpoint)))
    elif cfg.initialization != "scratch":
        weight = weight_preflight(cfg)
        actual = sha256(weight)
        if cfg.expected_weight_sha256 and actual.lower() != cfg.expected_weight_sha256.lower():
            raise ValueError(f"本地预训练权重 SHA256 不匹配：{weight}")
        print(f"[本地权重] {weight}\n[权重 SHA256] {actual}", flush=True)
        raw = torch.load(weight, map_location="cpu", weights_only=True)
        state = raw
        if isinstance(raw, dict):
            if isinstance(raw.get("ema"), dict) and isinstance(raw["ema"].get("module"), dict):
                state = raw["ema"]["module"]
            elif isinstance(raw.get("model"), dict):
                state = raw["model"]
            elif isinstance(raw.get("state_dict"), dict):
                state = raw["state_dict"]
        if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ValueError("权重不是可识别的 tensor state_dict；请使用 README 中对应 R50vd 的官方文件。")
        state = {k.removeprefix("module."): v for k, v in state.items()}
        if cfg.initialization == "backbone_pretrained":
            if any(k.startswith("backbone.") for k in state):
                state = {k.removeprefix("backbone."): v for k, v in state.items() if k.startswith("backbone.")}
            # Strict key/shape check catches torchvision ResNet50 or other variants.
            model.backbone.load_state_dict(state, strict=True)
            record.update(source=BACKBONE_URL, matched_tensors=len(model.backbone.state_dict()), reinitialized_class_tensors=[])
        else:
            own, matched, skipped, unexpected = model.state_dict(), {}, [], []
            for key, tensor in state.items():
                if key not in own:
                    unexpected.append(key)
                elif tensor.shape == own[key].shape:
                    matched[key] = tensor
                elif any(part in key for part in ("enc_score_head", "dec_score_head", "denoising_class_embed")):
                    skipped.append(key)
                else:
                    raise ValueError(f"检测预训练结构不匹配：{key}: {tuple(tensor.shape)} != {tuple(own[key].shape)}")
            allowed = {key for key in own if any(part in key for part in ("enc_score_head", "dec_score_head", "denoising_class_embed"))}
            missing = set(own) - set(matched)
            if missing - allowed or unexpected:
                raise ValueError(f"检测预训练文件不完整/结构不符：missing={sorted(missing - allowed)}, unexpected={unexpected}")
            model.load_state_dict(matched, strict=False)
            record.update(source=DETECTOR_URL, matched_tensors=len(matched), reinitialized_class_tensors=sorted(missing))
            print(f"[类别适配] 检测预训练迁移 {len(matched)} 个张量，{len(CLASS_NAMES)} 类输出/去噪嵌入重新初始化 {len(missing)} 个张量。", flush=True)
        record.update(path=str(weight), sha256=actual)
    else:
        print("[初始化] 明确配置 scratch：随机初始化，实验记录将标记为 scratch。", flush=True)
    if model.decoder.num_classes != len(CLASS_NAMES) or postprocessor.num_classes != len(CLASS_NAMES):
        raise AssertionError("模型或后处理的类别数没有正确适配为 5")
    model.to(device)
    criterion.to(device)
    postprocessor.to(device)
    write_json(output/"inputs"/"model_initialization.json", record)
    (output/"inputs"/"model_structure.txt").write_text(str(model), encoding="utf-8")
    groups = YAMLConfig.get_optim_params(active["optimizer"], model)
    optimizer = torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=cfg.betas, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(cfg.lr_milestones), gamma=cfg.lr_gamma)
    from src.optim.ema import ModelEMA
    ema = ModelEMA(model, decay=cfg.ema_decay, warmups=cfg.ema_warmups) if cfg.use_ema else None
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg.mixed_precision == "fp16")
    rows = [{"group": i, "parameter_tensors": len(g["params"]), "parameters": sum(p.numel() for p in g["params"]),
             "lr": g["lr"], "weight_decay": g["weight_decay"]} for i, g in enumerate(optimizer.param_groups)]
    atomic_csv(output/"inputs"/"optimizer_groups.csv", rows, list(rows[0]))
    complexity = {"parameters": sum(p.numel() for p in model.parameters()),
                  "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                  "gflops": None, "profile_input": [1, 3, cfg.image_size, cfg.image_size],
                  "flops_convention": "fvcore: one fused multiply-add counts as one FLOP; unsupported ops are listed"}
    if cfg.profile_flops:
        try:
            from fvcore.nn import FlopCountAnalysis
            class InferenceWrapper(torch.nn.Module):
                def __init__(self, inner):
                    super().__init__()
                    self.inner = inner
                def forward(self, value):
                    out = self.inner(value)
                    return out["pred_logits"], out["pred_boxes"]
            model.eval()
            with torch.no_grad():
                profile = FlopCountAnalysis(InferenceWrapper(model), torch.zeros(1, 3, cfg.image_size, cfg.image_size, device=device))
                profile.unsupported_ops_warnings(False).uncalled_modules_warnings(False).tracer_warnings("none")
                complexity["gflops"] = profile.total()/1e9
                complexity["unsupported_ops"] = dict(profile.unsupported_ops())
                complexity["uncalled_modules"] = sorted(profile.uncalled_modules())
        except Exception as exc:
            complexity["profile_error"] = f"{type(exc).__name__}: {exc}"
            print(f"[复杂度统计] 无法完整统计 FLOPs：{exc}；GFLOPs 记为空，不影响训练。", flush=True)
    write_json(output/"metrics"/"model_complexity.json", complexity)
    print(f"[模型] {cfg.model_name} | PResNet depth=50 variant=d | classes={len(CLASS_NAMES)} | decoder={model.decoder.num_decoder_layers} | queries={model.decoder.num_queries}", flush=True)
    print(f"[复杂度] Params={complexity['parameters']} Trainable={complexity['trainable_parameters']} GFLOPs={complexity['gflops']}", flush=True)
    return model, criterion, postprocessor, optimizer, scheduler, ema, scaler, record, complexity
