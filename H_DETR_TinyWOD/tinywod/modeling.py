"""Native H-DETR model/optimizer, local initialization and explicit operator backend."""
from pathlib import Path
import torch

from config import CLASS_NAMES, ROOT, resolve
from .common import digest, sha256, write_csv, write_json

RECIPE = "configs/two_stage/deformable-detr-hybrid-branch/36eps/r50_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh"


def validate_config(cfg):
    for key in ("epochs", "batch_size", "val_batch_size", "image_size", "max_size", "cpu_threads",
                "lr_drop", "checkpoint_interval", "enc_layers", "dec_layers", "dim_feedforward",
                "num_queries_one2one", "num_queries_one2many", "k_one2many"):
        if not isinstance(getattr(cfg, key), int) or getattr(cfg, key) < 1:
            raise ValueError(f"config.py: {key} 必须为正整数")
    if cfg.workers < 0 or cfg.ops_query_chunk_size < 0:
        raise ValueError("workers/ops_query_chunk_size 不能为负数")
    if cfg.augmentation not in ("small_object", "native"):
        raise ValueError("augmentation 必须为 small_object 或 native")
    if cfg.ops_backend not in ("auto", "native", "pytorch"):
        raise ValueError("ops_backend 必须为 auto/native/pytorch")
    if not 0 <= cfg.horizontal_flip <= 1 or cfg.lambda_one2many <= 0:
        raise ValueError("horizontal_flip 或 lambda_one2many 非法")
    if not 1 <= cfg.max_detections <= cfg.num_queries_one2one * len(CLASS_NAMES):
        raise ValueError("max_detections 超过一对一分支可输出的类别查询数")
    if cfg.max_size < cfg.image_size or min(cfg.lr, cfg.lr_backbone, cfg.lr_gamma) <= 0:
        raise ValueError("max_size 必须 >= image_size；学习率相关值必须为正数")
    if cfg.amp and cfg.device == "cpu":
        raise ValueError("本项目 AMP 只支持 CUDA；CPU 请使用 amp=False")
    if not 0 <= cfg.gpu_reserve_fraction < 1:
        raise ValueError("gpu_reserve_fraction 应位于 [0, 1)")


def native_args(cfg, device):
    # Reuse the actual supplied project's defaults rather than a new detector implementation.
    from main import get_args_parser
    args = get_args_parser().parse_args([])
    args.device = str(device); args.dataset_file = "tinywod"; args.num_classes = len(CLASS_NAMES)
    args.backbone = "resnet50"; args.pretrained_backbone = False
    args.with_box_refine = True; args.two_stage = True
    args.mixed_selection = True; args.look_forward_twice = True
    args.dropout = 0.0; args.aux_loss = True; args.use_wandb = False
    for key in ("epochs", "batch_size", "seed", "lr", "lr_backbone", "lr_linear_proj_mult",
                "weight_decay", "lr_drop", "clip_max_norm", "num_queries_one2one",
                "num_queries_one2many", "k_one2many", "lambda_one2many", "enc_layers",
                "dec_layers", "dim_feedforward", "use_checkpoint"):
        setattr(args, key, getattr(cfg, key))
    args.num_workers = cfg.workers; args.topk = cfg.max_detections; args.use_fp16 = cfg.amp
    return args


def configure_ops(model, cfg, device):
    from models.ops.modules.ms_deform_attn import MSDeformAttn
    from models.ops.functions.ms_deform_attn_func import MSDA, EXTENSION_IMPORT_ERROR, MSDeformAttnFunction
    native = device.type == "cuda" and MSDA is not None
    probe_error = ""
    if native and cfg.ops_backend != "pytorch":
        try:
            # Verify the loaded binary on the actual device (including its backward kernel).
            # Constant tensors avoid consuming experiment RNG. No compiler is ever invoked.
            value=torch.ones(1,4,1,4,device=device,requires_grad=True)
            shapes=torch.tensor([[2,2]],device=device,dtype=torch.long)
            starts=torch.tensor([0],device=device,dtype=torch.long)
            locations=torch.full((1,1,1,1,1,2),.45,device=device,requires_grad=True)
            attention=torch.ones(1,1,1,1,1,device=device,requires_grad=True)
            probe=MSDeformAttnFunction.apply(value,shapes,starts,locations,attention,64)
            probe.sum().backward(); torch.cuda.synchronize(device)
            if not torch.isfinite(probe).all(): raise RuntimeError("native probe produced nonfinite values")
        except Exception as exc:
            native=False; probe_error=f"{type(exc).__name__}: {exc}"
            print(f"[算子探测失败] {probe_error}",flush=True)
    if cfg.ops_backend == "native" and not native:
        raise RuntimeError("指定 native 但没有可用的已安装 CUDA 算子；可在 config.py 设置 ops_backend='pytorch'。"
                           f" 不会临时编译。原始错误：{probe_error or EXTENSION_IMPORT_ERROR}")
    chosen = "native" if native and cfg.ops_backend != "pytorch" else "pytorch"
    count = 0
    for layer in model.modules():
        if isinstance(layer, MSDeformAttn):
            layer.backend = chosen; layer.query_chunk_size = cfg.ops_query_chunk_size; count += 1
    info = dict(requested=cfg.ops_backend, actual=chosen, layers=count,
                query_chunk_size=cfg.ops_query_chunk_size, cuda_extension_import_error=EXTENSION_IMPORT_ERROR,
                cuda_extension_probe_error=probe_error,
                implementation="original MSDeformAttn CUDA extension" if chosen == "native" else
                "original ms_deform_attn_core_pytorch, query-chunked, regular autograd",
                jit_compile=False, cuda_grid_sample_backward_deterministic=False if chosen == "pytorch" else None)
    print(f"[算子] {chosen} | MSDeformAttn={count} 层 | query_chunk={cfg.ops_query_chunk_size} | 不临时编译", flush=True)
    if chosen == "pytorch":
        print("[算子说明] 使用原项目自带 PyTorch 参考计算；保留采样和梯度逻辑，速度/显存与 CUDA 扩展不同。", flush=True)
    return info


def preflight_weights(cfg):
    if cfg.resume_from:
        if not resolve(cfg.resume_from).is_file(): raise FileNotFoundError(resolve(cfg.resume_from))
        return
    if cfg.detector_weights:
        path = resolve(cfg.detector_weights)
        if not path.is_file(): raise FileNotFoundError(f"完整检测器权重不存在：{path}")
    elif cfg.pretrained:
        path = resolve(cfg.backbone_weights)
        if not path.is_file():
            raise FileNotFoundError(
                f"缺少本地 ResNet50 ImageNet V1 权重：{path}\n"
                "用浏览器下载 https://download.pytorch.org/models/resnet50-0676ba61.pth，"
                "放入项目 weights 文件夹。训练代码不会访问 HTTPS。详情见 README.md。")
        # The official name embeds its SHA256 prefix. Reject partial/corrupted downloads early.
        if path.name == "resnet50-0676ba61.pth" and not sha256(path).startswith("0676ba61"):
            raise ValueError(f"本地骨干权重校验失败（SHA256 应以 0676ba61 开头）：{path}。请重新完整下载。")


def build_native(cfg, device, output):
    from models import build_model
    args = native_args(cfg, device)
    print(f"[模型配置] {ROOT/RECIPE}",flush=True)
    model, criterion, post = build_model(args)
    initialization = dict(mode="random", local_only=True, file=None, sha256=None)
    if cfg.resume_from:
        initialization.update(mode="resume", file=str(resolve(cfg.resume_from)), sha256=sha256(resolve(cfg.resume_from)))
    elif cfg.detector_weights:
        path = resolve(cfg.detector_weights)
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("model", state.get("state_dict", state))
        model.load_state_dict(state, strict=True)
        initialization.update(mode="full_detector_strict", file=str(path), sha256=sha256(path))
    elif cfg.pretrained:
        path = resolve(cfg.backbone_weights)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or "conv1.weight" not in state:
            raise ValueError(f"不是 torchvision ResNet50 state_dict：{path}")
        fc = {k for k in state if k.startswith("fc.")}
        filtered = {k: v for k, v in state.items() if k not in fc}
        # FrozenBatchNorm removes num_batches_tracked itself. All remaining backbone tensors must match.
        model.backbone[0].body.load_state_dict(filtered, strict=True)
        value=sha256(path)
        mode="ImageNet1K_V1_backbone" if value.startswith("0676ba61") else "custom_local_backbone"
        initialization.update(mode=mode, file=str(path), sha256=value,
                              ignored_classifier_keys=sorted(fc), loaded_tensor_count=len(filtered))
    if initialization["mode"] == "random":
        print("[初始化] 显式随机初始化；本次结果不得与预训练实验混称。", flush=True)
    else:
        print(f"[初始化] {initialization['mode']} | {initialization['file']}", flush=True)
    model.to(device); criterion.to(device)
    ops = configure_ops(model, cfg, device)
    spec = dict(algorithm="H-DETR", backbone="resnet50", num_classes=len(CLASS_NAMES), names=CLASS_NAMES,
                source_recipe=RECIPE, source_recipe_sha256=sha256(ROOT/RECIPE),
                training_queries=cfg.num_queries_one2one+cfg.num_queries_one2many,
                inference_queries=cfg.num_queries_one2one,
                with_box_refine=True, two_stage=True, mixed_selection=True, look_forward_twice=True,
                args=vars(args), ops=ops, smoke_test=cfg.smoke_test)
    spec["architecture_fingerprint"] = digest({k: vars(args)[k] for k in (
        "backbone", "num_classes", "enc_layers", "dec_layers", "dim_feedforward", "hidden_dim",
        "nheads", "num_feature_levels", "num_queries_one2one", "num_queries_one2many",
        "with_box_refine", "two_stage", "mixed_selection", "look_forward_twice", "dropout")})
    write_json(output/"inputs"/"model_specification.json", spec)
    write_json(output/"inputs"/"initialization.json", initialization)
    write_json(output/"inputs"/"native_args.json", vars(args))
    return model, criterion, post, args, spec, initialization


def make_optimizer(model, args, cfg, output):
    def match(name, words): return any(word in name for word in words)
    grouped = [[], [], []]; records = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        group = 1 if match(name, args.lr_backbone_names) else 2 if match(name, args.lr_linear_proj_names) else 0
        grouped[group].append(parameter)
        records.append(dict(parameter=name, group=group, numel=parameter.numel(), shape=str(tuple(parameter.shape))))
    lrs = (cfg.lr, cfg.lr_backbone, cfg.lr*cfg.lr_linear_proj_mult)
    optimizer = torch.optim.AdamW([dict(params=p, lr=lr) for p, lr in zip(grouped, lrs)],
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_drop, gamma=cfg.lr_gamma)
    write_csv(output/"inputs"/"optimizer_groups.csv", records, ["parameter", "group", "numel", "shape"])
    return optimizer, scheduler
