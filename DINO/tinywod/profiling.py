# -*- coding: utf-8 -*-
"""在实际输入尺寸上统计模型参数与算量；不可测时留NaN并记录原因。"""
import math
import torch


@torch.no_grad()
def profile_model(model, imgsz, device, enabled=True):
    result = {'complexity/Parameters':sum(p.numel() for p in model.parameters()),
              'complexity/TrainableParameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
              'complexity/GFLOPs':float('nan'),'complexity/GFLOPs-dense-upper':float('nan'),
              'profile/input-height':imgsz,'profile/input-width':imgsz,'profile/batch':1,
              'profile/status':'disabled','profile/method':'PyTorch FlopCounter + MSDA analytical estimate'}
    if not enabled:
        return result
    was_training = model.training
    hooks = []
    dense_extra = sampled_extra = 0
    def count_msda(module, inputs, output):
        nonlocal dense_extra, sampled_extra
        q, _, memory = inputs[:3]
        b,lq,c = q.shape
        ls = memory.shape[1]
        # 2 FLOPs/MAC; dense-equivalent attention QK + AV + softmax.
        dense_extra += b*(4*lq*ls*c+5*module.n_heads*lq*ls)
        # bilinear sampling 7 ops + weighted aggregation 2 ops, all points/heads/channels.
        sampled_extra += 9*b*lq*module.n_levels*module.n_points*c
    try:
        from torch.utils.flop_counter import FlopCounterMode
        from models.dino.ops.modules import MSDeformAttn
        for m in model.modules():
            if isinstance(m,MSDeformAttn):
                hooks.append(m.register_forward_hook(count_msda))
        model.eval()
        dummy = torch.zeros(1,3,imgsz,imgsz,device=device)
        with FlopCounterMode(display=False) as counter:
            model(dummy)
        # FlopCounterMode 默认按 1 MAC = 1 FLOP 计数；统一到 2 FLOPs/MAC 口径，与 YOLO-World-L / H-DETR 横向可比。
        native = 2 * counter.get_total_flops()
        result.update({'complexity/GFLOPs':(native+sampled_extra)/1e9,
                       'complexity/GFLOPs-dense-upper':(native+dense_extra)/1e9,
                       'profile/status':'estimated',
                       'profile/convention':'one multiply-add counts as 2 FLOPs (aligned with YOLO-World-L / H-DETR)',
                       'profile/note':'Counted ATen conv/mm/attention operations (2 FLOPs/MAC) plus analytical MSDA. '
                                      'Dense column replaces sampled attention with full attention; '
                                      'both are conventional estimates, not an exact count of all elementwise ops.'})
    except Exception as e:
        result.update({'profile/status':'unavailable','profile/error':f'{type(e).__name__}: {e}'})
        print(f'[profile] 算量统计未完成，CSV保留NaN及原因: {e}',flush=True)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return result
