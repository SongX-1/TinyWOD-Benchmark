"""Single-device native H-DETR loops with the supplied hybrid objective unchanged."""
import json
import math
from pathlib import Path
import time
import torch
from tqdm import tqdm

from .hybrid import train_hybrid, one2one_inference
from .recording import loss_summary
from .common import write_csv, write_json


def synchronize(device):
    if device.type == "cuda": torch.cuda.synchronize(device)


def move(samples, targets, device):
    return samples.to(device), [{k:v.to(device) for k,v in target.items()} for target in targets]


def numeric(loss_dict, weights):
    values={k:float(v.detach().item()) for k,v in loss_dict.items()}
    if not all(math.isfinite(v) for v in values.values()):
        raise FloatingPointError(f"非有限损失：{values}")
    values["weighted_total"]=sum(v*weights[k] for k,v in values.items() if k in weights)
    return values


def average(sums, count):
    return {k:v/count for k,v in sums.items()}


def train_epoch(model, criterion, optimizer, scaler, loader, recorder, cfg, device, epoch, step):
    model.train(); criterion.train(); sums={}; previous=time.perf_counter()
    print(f"\n[H-DETR-R50] Epoch {epoch}/{cfg.epochs}",flush=True)
    print(f"{'Epoch':>11}{'GPU_mem':>11}{'box_loss':>11}{'cls_loss':>11}{'l1_loss':>11}{'Instances':>11}{'Size':>14}",flush=True)
    progress=tqdm(loader,total=len(loader),dynamic_ncols=True,mininterval=0.5)
    for batch,(samples,targets) in enumerate(progress,1):
        begin=time.perf_counter(); data_seconds=begin-previous
        samples,targets=move(samples,targets,device)
        optimizer.zero_grad(set_to_none=True)
        synchronize(device); t=time.perf_counter()
        with torch.autocast(device_type=device.type,enabled=cfg.amp):
            outputs=model(samples)
            synchronize(device); forward_seconds=time.perf_counter()-t; t=time.perf_counter()
            loss_dict=train_hybrid(outputs,targets,cfg.k_one2many,criterion,cfg.lambda_one2many)
            loss=sum(value*criterion.weight_dict[key] for key,value in loss_dict.items() if key in criterion.weight_dict)
        values=numeric(loss_dict,criterion.weight_dict)
        synchronize(device); loss_seconds=time.perf_counter()-t; t=time.perf_counter()
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.clip_max_norm,
                                                 error_if_nonfinite=not cfg.amp)
        synchronize(device); backward_seconds=time.perf_counter()-t; t=time.perf_counter()
        old_scale=scaler.get_scale(); scaler.step(optimizer); scaler.update()
        skipped=scaler.get_scale()<old_scale
        synchronize(device); optimizer_seconds=time.perf_counter()-t
        step+=1
        for key,value in values.items(): sums[key]=sums.get(key,0.)+value
        mean=average(sums,batch); view=loss_summary(mean,"train")
        h,w=map(int,samples.tensors.shape[-2:]); instances=sum(len(t["labels"]) for t in targets)
        gpu=f"{torch.cuda.memory_reserved(device)/2**30:.3g}G" if device.type=="cuda" else "CPU"
        progress.set_description(f"{epoch:>5}/{cfg.epochs:<5}{gpu:>10}{view['train/box_loss']:>11.4g}"
            f"{view['train/cls_loss']:>11.4g}{view['train/l1_loss']:>11.4g}{instances:>11}{h:>6}x{w:<6}")
        recorder.batch(epoch,"train",batch,step,loss_dict,criterion.weight_dict,
            dict(images=len(targets),instances=instances,image_ids=json.dumps([int(t['image_id'].item()) for t in targets]),
                 input_height=h,input_width=w,data_seconds=data_seconds,forward_seconds=forward_seconds,
                 loss_seconds=loss_seconds,backward_seconds=backward_seconds,optimizer_seconds=optimizer_seconds,
                 batch_seconds=time.perf_counter()-begin+data_seconds,loss=values['weighted_total'],
                 grad_norm=float(grad_norm.item()) if torch.isfinite(grad_norm) else None,
                 optimizer_step_skipped=skipped),optimizer)
        del outputs,loss_dict,loss,samples,targets
        previous=time.perf_counter()
    means=average(sums,len(loader))
    recorder.epoch_loss(epoch,"train",{k:v for k,v in means.items() if k!="weighted_total"},criterion.weight_dict,len(loader))
    return means,step


@torch.no_grad()
def infer_loader(model, criterion, post, loader, recorder, cfg, device, epoch, step, phase, labeled):
    criterion.eval(); predictions=[]; sums={}; total_forward=0.; n_images=0
    print(f"{'Class':>22}{'Images':>11}{'Instances':>11}      Box(P          R       AP50         AP    APsmall",flush=True)
    previous=time.perf_counter()
    with one2one_inference(model):
        progress=tqdm(loader,total=len(loader),dynamic_ncols=True,mininterval=.5,
                      desc=f"{phase:>12} | Class Images Instances Box(P R AP50 AP APsmall")
        for batch,(samples,targets) in enumerate(progress,1):
            begin=time.perf_counter(); data_seconds=begin-previous
            samples,targets=move(samples,targets,device); synchronize(device); t=time.perf_counter()
            with torch.autocast(device_type=device.type,enabled=cfg.amp):
                outputs=model(samples)
            synchronize(device); forward=time.perf_counter()-t; t=time.perf_counter()
            with torch.autocast(device_type=device.type,enabled=cfg.amp):
                loss_dict=criterion(outputs,targets) if labeled else {}
            values=numeric(loss_dict,criterion.weight_dict)
            synchronize(device); loss_seconds=time.perf_counter()-t
            results=post['bbox'](outputs,torch.stack([target['orig_size'] for target in targets]))
            for target,result in zip(targets,results):
                if not torch.isfinite(result['scores']).all() or not torch.isfinite(result['boxes']).all():
                    raise FloatingPointError("验证预测中出现 NaN/Inf")
                image_id=int(target['image_id'].item())
                for box,score,label in zip(result['boxes'].cpu().tolist(),result['scores'].cpu().tolist(),result['labels'].cpu().tolist()):
                    x1,y1,x2,y2=box
                    predictions.append(dict(image_id=image_id,category_id=int(label),
                                            bbox=[x1,y1,x2-x1,y2-y1],score=float(score)))
            for key,value in values.items(): sums[key]=sums.get(key,0.)+value
            n_images+=len(targets); total_forward+=forward
            recorder.batch(epoch,phase,batch,step,loss_dict,criterion.weight_dict,
                dict(images=len(targets),instances=sum(len(t['labels']) for t in targets),
                    image_ids=json.dumps([int(t['image_id'].item()) for t in targets]),
                    input_height=int(samples.tensors.shape[-2]),input_width=int(samples.tensors.shape[-1]),
                    data_seconds=data_seconds,forward_seconds=forward,loss_seconds=loss_seconds,
                    backward_seconds=None,optimizer_seconds=None,batch_seconds=time.perf_counter()-begin+data_seconds,
                    loss=values['weighted_total'] if labeled else None,grad_norm=None,optimizer_step_skipped=None))
            del outputs,loss_dict,samples,targets
            previous=time.perf_counter()
    means=average(sums,len(loader)) if len(loader) else {}
    if labeled:
        recorder.epoch_loss(epoch,phase,{k:v for k,v in means.items() if k!="weighted_total"},criterion.weight_dict,len(loader))
    inference_ms=1000*total_forward/n_images if n_images else None
    return predictions,means,inference_ms


def print_metrics(headline,classes=None):
    def f(value): return f"{value:11.5g}" if value is not None else f"{'N/A':>11}"
    print(f"{'all':>22}{headline['images']:>11}{headline['instances']:>11}"+
          ''.join(f(headline[k]) for k in ('precision','recall','ap50','ap','apsmall')),flush=True)
    if classes:
        for row in classes:
            print(f"{row['class_name']:>22}{row['images']:>11}{row['instances']:>11}"+
                  ''.join(f(row[k]) for k in ('precision','recall','ap50','ap','apsmall')),flush=True)


def draw_predictions(annotation, predictions, images_root, folder, confidence, count):
    if count<=0: return
    from PIL import Image,ImageDraw
    from config import CLASS_NAMES
    from .common import read_json
    payload=read_json(annotation); grouped={image['id']:[] for image in payload['images']}
    for p in predictions:
        if p['score']>=confidence: grouped[p['image_id']].append(p)
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    for image in payload['images'][:count]:
        with Image.open(Path(images_root)/image['file_name']) as source: canvas=source.convert('RGB')
        draw=ImageDraw.Draw(canvas)
        for p in grouped[image['id']]:
            x,y,w,h=p['bbox']; draw.rectangle((x,y,x+w,y+h),outline='red',width=2)
            draw.text((max(0,x),max(0,y-11)),f"{CLASS_NAMES[p['category_id']]} {p['score']:.3f}",fill='red')
        canvas.save(folder/f"{image['id']:08d}.jpg")


@torch.no_grad()
def complexity(model, sample, device, output):
    params=sum(parameter.numel() for parameter in model.parameters())
    trainable=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    info=dict(params=params,trainable_params=trainable,gflops=None,gflops_partial=None,unsupported_ops={},
              input_shape=list(sample.shape),batch_size=1,inference_queries=model.num_queries_one2one,
              convention="fvcore FlopCountAnalysis: one multiply-add counts as 2 FLOPs (aligned with YOLO-World-L); unsupported ops are never zero-cost claims")
    class Forward(torch.nn.Module):
        def __init__(self,detector): super().__init__(); self.detector=detector
        def forward(self,x):
            out=self.detector(x)
            leaves=[out['pred_logits'],out['pred_boxes']]
            for a in out.get('aux_outputs',[]): leaves.extend((a['pred_logits'],a['pred_boxes']))
            if 'enc_outputs' in out: leaves.extend((out['enc_outputs']['pred_logits'],out['enc_outputs']['pred_boxes']))
            return tuple(leaves)
    try:
        from fvcore.nn import FlopCountAnalysis
        with one2one_inference(model):
            analysis=FlopCountAnalysis(Forward(model),(sample,)).unsupported_ops_warnings(False).uncalled_modules_warnings(False).tracer_warnings('none')
            value=float(2*analysis.total()/1e9); unsupported=dict(analysis.unsupported_ops())
        info['unsupported_ops']=unsupported
        if unsupported:
            info['gflops_partial']=value
            info['note']='N/A total: unsupported operations exist; partial counted operations are saved separately.'
        else: info['gflops']=value
    except Exception as exc:
        info['note']=f"FLOP tracing unavailable: {type(exc).__name__}: {exc}"
    write_json(output/'metrics'/'complexity.json',info)
    write_csv(output/'metrics'/'complexity.csv',[{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in info.items()}],info.keys())
    return info


def print_complexity(info,inference_ms,peak):
    gflops=f"{info['gflops']:.3f}" if info.get('gflops') is not None else 'N/A'
    if info.get('gflops_partial') is not None: gflops+=f" (partial={info['gflops_partial']:.3f})"
    speed=f"{inference_ms:.3f} ms/image ({1000/inference_ms:.2f} FPS)" if inference_ms else 'N/A'
    memory=f"{peak:.3f} GB" if peak is not None else 'N/A (CPU)'
    print(f"Complexity: Params={info['params']}, GFLOPs={gflops}, val inference={speed}, PeakAllocated={memory}",flush=True)


@torch.no_grad()
def benchmark(model,post,dataset,cfg,device,output):
    from util.misc import nested_tensor_from_tensor_list
    import numpy as np
    rows=[]
    if cfg.benchmark_images<=0 or cfg.benchmark_repeats<=0:
        write_json(output/'metrics'/'inference_speed.json',{'status':'disabled_explicitly_in_config'})
        return {}
    with one2one_inference(model):
        for index in range(min(cfg.benchmark_images,len(dataset))):
            t=time.perf_counter(); image,target=dataset[index]; preprocessing=time.perf_counter()-t
            for repeat in range(-cfg.benchmark_warmup,cfg.benchmark_repeats):
                synchronize(device); t=time.perf_counter()
                samples=nested_tensor_from_tensor_list([image]).to(device)
                sizes=target['orig_size'].unsqueeze(0).to(device)
                synchronize(device); transfer=time.perf_counter()-t; t=time.perf_counter()
                with torch.autocast(device_type=device.type,enabled=cfg.amp): result=model(samples)
                synchronize(device); forward=time.perf_counter()-t; t=time.perf_counter()
                detections=post['bbox'](result,sizes)
                # Include device -> host in postprocessing latency, as actual prediction export does.
                _=[(d['boxes'].cpu(),d['scores'].cpu(),d['labels'].cpu()) for d in detections]
                synchronize(device); post_time=time.perf_counter()-t
                if repeat>=0:
                    rows.append(dict(image_id=int(target['image_id'].item()),repeat=repeat,
                        input_height=int(image.shape[-2]),input_width=int(image.shape[-1]),
                        preprocessing_ms=preprocessing*1000,transfer_ms=transfer*1000,
                        inference_ms=forward*1000,postprocess_ms=post_time*1000,
                        total_ms=(preprocessing+transfer+forward+post_time)*1000))
                del result,detections,samples
    values=np.asarray([r['inference_ms'] for r in rows])
    summary=dict(status='completed',batch_size=1,images=min(cfg.benchmark_images,len(dataset)),
        repeats=cfg.benchmark_repeats,warmup_per_image=cfg.benchmark_warmup,
        inference_ms_mean=float(values.mean()),inference_ms_median=float(np.median(values)),
        inference_ms_p95=float(np.percentile(values,95)),fps=1000/float(values.mean()),
        total_ms_mean=float(np.mean([r['total_ms'] for r in rows])),
        note='synchronized wall-clock; batch=1; image decoding+transforms measured once/image; inference excludes preprocessing/postprocessing')
    write_csv(output/'metrics'/'inference_speed.csv',rows,rows[0].keys())
    write_json(output/'metrics'/'inference_speed.json',summary)
    print(f"[单图速度] batch=1 | {summary['inference_ms_mean']:.3f} ms/image | {summary['fps']:.2f} FPS | "
          f"P95={summary['inference_ms_p95']:.3f} ms | {output/'metrics'/'inference_speed.csv'}",flush=True)
    return summary
