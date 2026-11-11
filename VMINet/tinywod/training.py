"""Native detector training; deterministic validation sampling isolated from training RNG."""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import time
import torch
from tqdm import tqdm
from .recording import loss_summary
from .loading import inverse_letterbox
from .common import write_csv,write_json
from config import CLASS_NAMES

INFERENCE_SCOPE='detector forward includes native transform, backbone, pyramid, RPN/ROI, decode and NMS; excludes validation diagnostic losses'


def synchronize(device):
    if device.type=='cuda': torch.cuda.synchronize(device)


def move(images,targets,device):
    return [image.to(device) for image in images],[{k:v.to(device) for k,v in target.items()} for target in targets]


@contextmanager
def validation_randomness(device,seed):
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


def numeric(losses,weights):
    values={k:float(v.detach().item()) for k,v in losses.items()}
    if not all(math.isfinite(v) for v in values.values()): raise FloatingPointError(f'非有限损失：{values}')
    values['weighted_total']=sum(v*weights[k] for k,v in values.items() if k in weights)
    return values


def average(values,count): return {k:v/count for k,v in values.items()}


def train_epoch(model,optimizer,scaler,loader,recorder,cfg,device,epoch,step):
    model.train();sums={};previous=time.perf_counter();label=f'VMINet-{cfg.variant.upper()}'
    print(f'\n[{label}] Epoch {epoch}/{cfg.epochs}',flush=True)
    print(f"{'Epoch':>11}{'GPU_mem':>11}{'box_loss':>11}{'cls_loss':>11}{'rpn_obj':>11}{'rpn_box':>11}{'Instances':>11}{'Size':>10}",flush=True)
    progress=tqdm(loader,total=len(loader),dynamic_ncols=True,mininterval=.5)
    for batch,(images,targets) in enumerate(progress,1):
        begin=time.perf_counter();data_seconds=begin-previous
        images,targets=move(images,targets,device);optimizer.zero_grad(set_to_none=True)
        synchronize(device);start=time.perf_counter()
        with torch.autocast(device_type=device.type,enabled=cfg.amp):
            losses=model(images,targets)
            loss=sum(value*model.loss_weights[key] for key,value in losses.items())
        synchronize(device);forward_seconds=time.perf_counter()-start;start=time.perf_counter()
        values=numeric(losses,model.loss_weights)
        loss_seconds=time.perf_counter()-start;start=time.perf_counter()
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.clip_max_norm,error_if_nonfinite=not cfg.amp)
        synchronize(device);backward_seconds=time.perf_counter()-start;start=time.perf_counter()
        old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
        skipped=scaler.get_scale()<old_scale
        synchronize(device);optimizer_seconds=time.perf_counter()-start;step+=1
        for key,value in values.items():sums[key]=sums.get(key,0.)+value
        view=loss_summary(average(sums,batch),'train');instances=sum(len(t['labels']) for t in targets)
        gpu=f'{torch.cuda.memory_reserved(device)/2**30:.3g}G' if device.type=='cuda' else 'CPU'
        progress.set_description(f'{epoch:>5}/{cfg.epochs:<5}{gpu:>10}'+''.join(f'{view["train/"+key]:>11.4g}'
            for key in ('box_loss','cls_loss','rpn_obj','rpn_box'))+f'{instances:>11}{cfg.image_size:>10}')
        recorder.batch(epoch,'train',batch,step,losses,model.loss_weights,
            dict(images=len(images),instances=instances,image_ids=json.dumps([int(t['image_id'].item()) for t in targets]),
                input_height=cfg.image_size,input_width=cfg.image_size,data_seconds=data_seconds,forward_seconds=forward_seconds,
                loss_seconds=loss_seconds,backward_seconds=backward_seconds,optimizer_seconds=optimizer_seconds,
                batch_seconds=time.perf_counter()-begin+data_seconds,loss=values['weighted_total'],
                grad_norm=float(grad_norm.item()) if torch.isfinite(grad_norm) else None,optimizer_step_skipped=skipped),optimizer)
        del losses,loss,images,targets;previous=time.perf_counter()
    means=average(sums,len(loader))
    recorder.epoch_loss(epoch,'train',{k:v for k,v in means.items() if k!='weighted_total'},model.loss_weights,len(loader))
    return means,step


@torch.no_grad()
def infer_loader(model,loader,recorder,cfg,device,epoch,step,phase,labeled):
    original_training=model.training;model.eval();predictions=[];sums={};forward_seconds=0.;n_images=0
    previous=time.perf_counter()
    print(f"{'Class':>22}{'Images':>11}{'Instances':>11}      Box(P          R       AP50         AP    APsmall",flush=True)
    try:
        with validation_randomness(device,cfg.val_loss_seed):
            progress=tqdm(loader,total=len(loader),dynamic_ncols=True,mininterval=.5,
                desc=f'{phase:>12} | Class Images Instances Box(P R AP50 AP APsmall')
            for batch,(images,targets) in enumerate(progress,1):
                begin=time.perf_counter();data_seconds=begin-previous
                images,targets=move(images,targets,device)
                with torch.autocast(device_type=device.type,enabled=cfg.amp):
                    detections,losses,timing=model.evaluate_with_losses(images,targets if labeled else None)
                values=numeric(losses,model.loss_weights)
                for target,result in zip(targets,detections):
                    if not torch.isfinite(result['boxes']).all() or not torch.isfinite(result['scores']).all():
                        raise FloatingPointError('预测出现 NaN/Inf')
                    boxes=inverse_letterbox(result['boxes'],target)
                    for box,score,label in zip(boxes.tolist(),result['scores'].cpu().tolist(),result['labels'].cpu().tolist()):
                        if not 1<=int(label)<=len(CLASS_NAMES):raise ValueError(f'检测器输出非法前景类别：{label}')
                        x1,y1,x2,y2=box
                        if x2<=x1 or y2<=y1:continue  # Prediction entirely in padding, not a GT filter.
                        predictions.append(dict(image_id=int(target['image_id'].item()),category_id=int(label)-1,
                            bbox=[x1,y1,x2-x1,y2-y1],score=float(score)))
                for key,value in values.items():sums[key]=sums.get(key,0.)+value
                n_images+=len(images);forward_seconds+=timing['inference_seconds']
                recorder.batch(epoch,phase,batch,step,losses,model.loss_weights,
                    dict(images=len(images),instances=sum(len(t['labels']) for t in targets),
                        image_ids=json.dumps([int(t['image_id'].item()) for t in targets]),input_height=cfg.image_size,input_width=cfg.image_size,
                        data_seconds=data_seconds,forward_seconds=timing['inference_seconds'],loss_seconds=timing['loss_seconds'],
                        backward_seconds=None,optimizer_seconds=None,batch_seconds=time.perf_counter()-begin+data_seconds,
                        loss=values['weighted_total'] if labeled else None,grad_norm=None,optimizer_step_skipped=None))
                del detections,losses,images,targets;previous=time.perf_counter()
    finally:model.train(original_training)
    means=average(sums,len(loader)) if len(loader) else {}
    if labeled:recorder.epoch_loss(epoch,phase,{k:v for k,v in means.items() if k!='weighted_total'},model.loss_weights,len(loader))
    return predictions,means,1000*forward_seconds/n_images if n_images else None


def print_metrics(headline,classes=None):
    def f(value):return f'{value:11.5g}' if value is not None else f"{'N/A':>11}"
    for row,name in [(headline,'all'),*[(c,c['class_name']) for c in classes or []]]:
        print(f"{name:>22}{row['images']:>11}{row['instances']:>11}"+''.join(f(row[k]) for k in
            ('precision','recall','ap50','ap','apsmall')),flush=True)


def draw_predictions(annotation,predictions,images_root,folder,confidence,count):
    if count<=0:return
    from PIL import Image,ImageDraw
    from config import CLASS_NAMES
    from .common import read_json
    payload=read_json(annotation);grouped={item['id']:[] for item in payload['images']}
    for p in predictions:
        if p['score']>=confidence:grouped[p['image_id']].append(p)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    for item in payload['images'][:count]:
        with Image.open(Path(images_root)/item['file_name']) as source:canvas=source.convert('RGB')
        draw=ImageDraw.Draw(canvas)
        for p in grouped[item['id']]:
            x,y,w,h=p['bbox'];draw.rectangle((x,y,x+w,y+h),outline='red',width=2)
            draw.text((max(0,x),max(0,y-11)),f"{CLASS_NAMES[p['category_id']]} {p['score']:.3f}",fill='red')
        canvas.save(folder/f"{item['id']:08d}.jpg")


@torch.no_grad()
def complexity(model,sample,device,output):
    original_training=model.training;model.eval()
    info=dict(params=sum(p.numel() for p in model.parameters()),trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        gflops=None,gflops_partial=None,unsupported_ops={},input_shape=list(sample.shape),batch_size=1,
        convention='fvcore one fused multiply-add counts as one FLOP; data-dependent native detector trace; unsupported total left blank')
    class Forward(torch.nn.Module):
        def __init__(self,detector):super().__init__();self.detector=detector
        def forward(self,x):
            result=self.detector([x[0]])[0]
            return result['boxes'],result['scores'],result['labels']
    try:
        from fvcore.nn import FlopCountAnalysis
        analysis=FlopCountAnalysis(Forward(model),(sample,)).unsupported_ops_warnings(False).uncalled_modules_warnings(False).tracer_warnings('none')
        value=float(analysis.total()/1e9);unsupported=dict(analysis.unsupported_ops());info['unsupported_ops']=unsupported
        if unsupported:info['gflops_partial']=value;info['note']='Total N/A: unsupported operations, including potentially ROIAlign/NMS.'
        else:info['gflops']=value
    except Exception as exc:info['note']=f'FLOP trace unavailable: {type(exc).__name__}: {exc}'
    finally:model.train(original_training)
    write_json(output/'metrics'/'complexity.json',info)
    write_csv(output/'metrics'/'complexity.csv',[{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in info.items()}],info.keys())
    return info


def print_complexity(info,inference_ms,peak):
    flops=f"{info['gflops']:.3f}" if info.get('gflops') is not None else 'N/A'
    if info.get('gflops_partial') is not None:flops+=f" (partial={info['gflops_partial']:.3f})"
    speed=f'{inference_ms:.3f} ms/image ({1000/inference_ms:.2f} FPS)' if inference_ms else 'N/A'
    memory=f'{peak:.3f} GB' if peak is not None else 'N/A (CPU)'
    print(f"Complexity: Params={info['params']}, GFLOPs={flops}, val inference={speed}, PeakAllocated={memory}",flush=True)


@torch.no_grad()
def benchmark(model,dataset,cfg,device,output):
    import numpy as np
    if cfg.benchmark_images<=0 or cfg.benchmark_repeats<=0:
        write_json(output/'metrics'/'inference_speed.json',dict(status='disabled_in_config'));return {}
    training=model.training;model.eval();rows=[]
    try:
        for index in range(min(cfg.benchmark_images,len(dataset))):
            start=time.perf_counter();image,target=dataset[index];preprocess=time.perf_counter()-start
            for repeat in range(-cfg.benchmark_warmup,cfg.benchmark_repeats):
                synchronize(device);start=time.perf_counter();images=[image.to(device)]
                synchronize(device);transfer=time.perf_counter()-start;start=time.perf_counter()
                with torch.autocast(device_type=device.type,enabled=cfg.amp):result=model(images)[0]
                synchronize(device);inference=time.perf_counter()-start;start=time.perf_counter()
                _=inverse_letterbox(result['boxes'],target);_=result['scores'].cpu()
                synchronize(device);export=time.perf_counter()-start
                if repeat>=0:rows.append(dict(image_id=int(target['image_id'].item()),repeat=repeat,input_height=cfg.image_size,input_width=cfg.image_size,
                    preprocessing_ms=preprocess*1000,transfer_ms=transfer*1000,inference_ms=inference*1000,export_ms=export*1000,
                    total_ms=(preprocess+transfer+inference+export)*1000))
                del result,images
    finally:model.train(training)
    values=np.asarray([row['inference_ms'] for row in rows])
    summary=dict(status='completed',batch_size=1,images=min(cfg.benchmark_images,len(dataset)),repeats=cfg.benchmark_repeats,
        warmup_per_image=cfg.benchmark_warmup,inference_ms_mean=float(values.mean()),inference_ms_median=float(np.median(values)),
        inference_ms_p95=float(np.percentile(values,95)),fps=1000/float(values.mean()),
        total_ms_mean=float(np.mean([row['total_ms'] for row in rows])),inference_scope=INFERENCE_SCOPE,
        note='synchronized wall time; decoding+letterbox measured once/image; export includes CPU transfer and inverse letterbox')
    write_csv(output/'metrics'/'inference_speed.csv',rows,rows[0].keys());write_json(output/'metrics'/'inference_speed.json',summary)
    print(f"[单图速度] batch=1 | {summary['inference_ms_mean']:.3f} ms/image | {summary['fps']:.2f} FPS | P95={summary['inference_ms_p95']:.3f} ms",flush=True)
    return summary
