"""Source-faithful VMINet factories, strict local initialization and detector specification."""
from dataclasses import asdict
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from config import CLASS_NAMES,ROOT,resolve
from .common import digest,sha256,write_csv,write_json

WEIGHT_URLS={
 'ti':'https://github.com/yws-wxs/VMINet/releases/download/output/VMINet2.0-Ti-checkpoint.pth.tar',
 'xs':'https://github.com/yws-wxs/VMINet/releases/download/output/VMINet2.0-XS-checkpoint.pth.tar',
 's':'https://github.com/yws-wxs/VMINet/releases/download/output/VMINet2.0-S-checkpoint.pth.tar',
 'b':'https://github.com/yws-wxs/VMINet/releases/download/4.22/B-ckpt.pth.tar',
}


def validate_config(cfg):
    from models_detection.vminet_detector import VARIANTS
    if cfg.variant not in VARIANTS: raise ValueError('variant 必须为 ti/xs/s/b；原项目没有 L。')
    for name in ('epochs','batch_size','val_batch_size','image_size','cpu_threads','pyramid_channels','roi_hidden_dim',
                 'roi_pool_size','rpn_pre_nms_train','rpn_post_nms_train','rpn_pre_nms_test','rpn_post_nms_test',
                 'rpn_batch_size','roi_batch_size','max_detections','checkpoint_interval'):
        if not isinstance(getattr(cfg,name),int) or getattr(cfg,name)<1: raise ValueError(f'{name} 必须为正整数')
    if cfg.image_size%32: raise ValueError('image_size 必须为 32 的倍数，保证固定空间聚合与 P2..P6 尺寸一致。')
    if len(cfg.anchor_sizes)!=5 or any(size<=0 for size in cfg.anchor_sizes): raise ValueError('anchor_sizes 必须为五个正数，对应 P2..P6。')
    if cfg.workers<0 or not 0<=cfg.horizontal_flip<=1: raise ValueError('workers/horizontal_flip 非法')
    if cfg.lr<=0 or cfg.weight_decay<0 or cfg.lr_gamma<=0: raise ValueError('学习率/weight_decay 非法')
    if list(cfg.lr_milestones)!=sorted(set(cfg.lr_milestones)) or any(m<1 for m in cfg.lr_milestones):
        raise ValueError('lr_milestones 必须为升序、不重复的正 Epoch')
    if not 0<=cfg.score_threshold<1 or not 0<cfg.nms_iou<=1: raise ValueError('score_threshold/nms_iou 非法')
    if not cfg.smoke_test and cfg.batch_size<2:
        print('[BatchNorm] batch_size=1 时 VMINet 的 BN 统计可能不稳定；本次配置会照实记录。',flush=True)


def preflight_weights(cfg):
    path=None
    if cfg.resume_from: path=resolve(cfg.resume_from)
    elif cfg.detector_weights: path=resolve(cfg.detector_weights)
    elif cfg.pretrained: path=resolve(cfg.backbone_weights[cfg.variant])
    if path is not None and not path.is_file():
        raise FileNotFoundError(f'本地权重不存在：{path}\n只需将 README 指定的 {cfg.variant.upper()} 预训练文件放入 weights/，'
                                '或在 config.py 修改路径。训练代码不自动下载。')


def interpolate_sum(weight,target_shape):
    """Resize only the learned spatial linear vector, once at initialization.

    The area ratio preserves the sum for uniform coefficients/inputs. At the
    original resolution this is an identity; Block.forward is never rewritten.
    """
    if weight.ndim!=2 or weight.shape[0]!=1 or len(target_shape)!=2 or target_shape[0]!=1:
        raise ValueError('空间 sum 权重必须为 (1,H*W)')
    source=math.isqrt(weight.shape[1]); target=math.isqrt(target_shape[1])
    if source**2!=weight.shape[1] or target**2!=target_shape[1]: raise ValueError('sum 权重长度不是平方网格')
    if tuple(weight.shape)==tuple(target_shape): return weight
    value=F.interpolate(weight.float().reshape(1,1,source,source),size=(target,target),mode='bilinear',align_corners=False)
    return (value.reshape(target_shape)*(source*source)/(target*target)).to(weight.dtype)


def load_local_backbone(backbone,path,state_key):
    # Original train_imagenet.py stores argparse.Namespace/optimizer in its checkpoints.
    # These are user-selected LOCAL checkpoints, loaded with the original checkpoint format.
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if not isinstance(payload,dict): raise ValueError('预训练文件不是字典 checkpoint')
    chosen='raw_state_dict'
    if state_key in payload:
        state=payload[state_key]; chosen=state_key
    elif all(isinstance(v,torch.Tensor) for v in payload.values()): state=payload
    else: raise ValueError(f'预训练文件缺少 {state_key}，可用字段：{list(payload)[:12]}。请在 config.py 明确选择。')
    if not isinstance(state,dict): raise ValueError('state_dict 不是参数字典')
    state=dict(state)
    for prefix in ('module.','model.'):
        if state and all(k.startswith(prefix) for k in state): state={k[len(prefix):]:v for k,v in state.items()}
    expected=backbone.state_dict(); loaded={}; rows=[]
    for name,value in state.items():
        if name in ('head.weight','head.bias'):
            rows.append(dict(parameter=name,action='discard_classifier',source_shape=str(tuple(value.shape)),target_shape=''))
            continue
        if name not in expected: raise ValueError(f'非该 VMINet 骨干参数：{name}；不会静默跳过。')
        if not isinstance(value,torch.Tensor): raise ValueError(f'参数不是 tensor：{name}')
        action='copy'
        shape=tuple(value.shape); target=tuple(expected[name].shape)
        if shape!=target:
            if name.endswith('.sum.weight'):
                value=interpolate_sum(value,target); action='spatial_linear_bilinear_area_normalized'
            else: raise ValueError(f'VMINet 结构不匹配：{name}，checkpoint={shape} model={target}。请核对变体。')
        if value.is_floating_point() and not torch.isfinite(value).all(): raise ValueError(f'权重包含 NaN/Inf：{name}')
        loaded[name]=value
        rows.append(dict(parameter=name,action=action,source_shape=str(shape),target_shape=str(target)))
    missing=set(expected)-set(loaded)
    if missing: raise ValueError(f'骨干缺少必要参数：{sorted(missing)[:15]}')
    backbone.load_state_dict(loaded,strict=True)
    return rows,chosen


def build_native(cfg,device,output):
    from models_detection.vminet_detector import VMINetDetector,VARIANTS
    model=VMINetDetector(cfg)
    initialization=dict(mode='random',file=None,sha256=None,local_only=True)
    if cfg.resume_from:
        initialization.update(mode='resume',file=str(resolve(cfg.resume_from)),sha256=sha256(resolve(cfg.resume_from)))
    elif cfg.detector_weights:
        path=resolve(cfg.detector_weights); payload=torch.load(path,map_location='cpu',weights_only=False)
        state=payload.get('model',payload.get('state_dict',payload))
        model.load_state_dict(state,strict=True)
        initialization.update(mode='full_detector_strict',file=str(path),sha256=sha256(path))
    elif cfg.pretrained:
        path=resolve(cfg.backbone_weights[cfg.variant])
        rows,key=load_local_backbone(model.detector.backbone.vminet,path,cfg.pretrained_state_key)
        write_csv(output/'inputs'/'pretrained_loading.csv',rows,['parameter','action','source_shape','target_shape'])
        initialization.update(mode='local_VMINet_classifier_backbone',file=str(path),sha256=sha256(path),
            checkpoint_key=key,source_url_documented=WEIGHT_URLS[cfg.variant],
            copied=sum(r['action']=='copy' for r in rows),interpolated=sum(r['action'].startswith('spatial_') for r in rows),
            interpolation='bilinear spatial grid, align_corners=False, area source_N/target_N; once at load',
            backbone_block_forward='unchanged from uploaded vminet.py')
    print(f"[初始化] {initialization['mode']} | {initialization['file'] or '显式随机初始化'}",flush=True)
    model.to(device)
    architecture=dict(variant=cfg.variant,**VARIANTS[cfg.variant],image_size=cfg.image_size,
        stage_sizes=list(model.detector.backbone.vminet.stage_sizes),last_map_only=True,
        pyramid='simple_conv_deconv_P2_P6',pyramid_channels=cfg.pyramid_channels,
        roi_hidden_dim=cfg.roi_hidden_dim,roi_pool_size=cfg.roi_pool_size,roi_sampling_ratio=cfg.roi_sampling_ratio,
        anchor_sizes=cfg.anchor_sizes,anchor_ratios=cfg.anchor_ratios,detector='torchvision FasterRCNN',
        foreground_classes=len(CLASS_NAMES),predictor_classes=len(CLASS_NAMES)+1,background_label=0,detector_label_offset=1,
        backbone_source_sha256=sha256(ROOT/'vminet.py'),detector_source_sha256=sha256(ROOT/'models_detection/vminet_detector.py'))
    spec=dict(algorithm=f'VMINet-{cfg.variant.upper()} + Simple Feature Pyramid + Faster R-CNN (bbox)',
        adaptation='bbox-only detector added to classification-only archive; not a claimed exact Mask R-CNN reproduction',
        source_file=str(ROOT/'vminet.py'),source_file_sha256=sha256(ROOT/'vminet.py'),
        original_source_unmodified=True,architecture=architecture,architecture_fingerprint=digest(architecture),names=CLASS_NAMES,
        ops=dict(actual='torchvision_native',jit_compile=False),loss_weights=model.loss_weights,
        input_policy='fixed square letterbox, ImageNet normalization inside detector transform',
        validation_loss_policy='shared eval backbone; native RPN/ROI sampled losses; BN remains eval; isolated RNG',
        smoke_test=cfg.smoke_test)
    write_json(output/'inputs'/'model_specification.json',spec)
    write_json(output/'inputs'/'initialization.json',initialization)
    return model,spec,initialization


def make_optimizer(model,cfg,output):
    decay=[]; no_decay=[]; rows=[]
    for name,parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        no_wd=parameter.ndim==1 or name.endswith('.bias')
        (no_decay if no_wd else decay).append(parameter)
        rows.append(dict(parameter=name,group=1 if no_wd else 0,weight_decay=0. if no_wd else cfg.weight_decay,
                         numel=parameter.numel(),shape=str(tuple(parameter.shape))))
    optimizer=torch.optim.AdamW([dict(params=decay,weight_decay=cfg.weight_decay),dict(params=no_decay,weight_decay=0.)],lr=cfg.lr)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=list(cfg.lr_milestones),gamma=cfg.lr_gamma)
    write_csv(output/'inputs'/'optimizer_groups.csv',rows,['parameter','group','weight_decay','numel','shape'])
    return optimizer,scheduler
