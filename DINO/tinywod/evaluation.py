# -*- coding: utf-8 -*-
import json
from pathlib import Path
import time
import numpy as np
import torch

from datasets.coco_eval import CocoEvaluator
from experiment_config import CLASS_NAMES
from .metrics import DetectionAccumulator

COCO_COLUMNS = ['AP','AP50','AP75','APsmall','APmedium','APlarge',
                'AR1','AR10','AR100','ARsmall','ARmedium','ARlarge']


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def preview(image_path, anns, prediction, destination):
    from PIL import Image, ImageDraw
    with Image.open(image_path) as im:
        image = im.convert('RGB')
    draw = ImageDraw.Draw(image)
    for a in anns:
        x,y,w,h = a['bbox']
        draw.rectangle((x,y,x+w,y+h),outline=(0,255,0),width=3)
    for box,score,cls in zip(prediction['boxes'].tolist(),prediction['scores'].tolist(),prediction['labels'].tolist()):
        if score < .25:
            continue
        x1,y1,x2,y2 = box
        if x2 <= x1 or y2 <= y1:
            continue
        draw.rectangle(box,outline=(255,60,180),width=3)
        draw.text((max(0,x1),max(0,y1)),f'{CLASS_NAMES[int(cls)]} {score:.2f}',fill=(255,60,180))
    image.thumbnail((1280,1280))
    destination = Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    image.save(destination)


@torch.no_grad()
def evaluate(model,criterion,postprocessor,loader,device,epoch,config,recorder,complexity):
    model.eval();criterion.eval()
    coco = loader.dataset.coco
    native = CocoEvaluator(coco,('bbox',))
    owt = DetectionAccumulator(coco)
    totals = {}; total_batches=0; images=0
    prep_time=infer_time=loss_time=post_time=0.
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    folder = recorder.run_dir/'results'/f'epoch{epoch:04d}'
    folder.mkdir(parents=True,exist_ok=True)
    json_path = recorder.run_dir/'predictions'/f'val_epoch{epoch:04d}.json'
    handle = None
    first = True
    if config.save_predictions:
        json_path.parent.mkdir(parents=True,exist_ok=True)
        handle = json_path.with_suffix('.json.tmp').open('w',encoding='utf-8')
        handle.write('[')
    print(f"[val] Epoch {epoch}: {'Iter':>11} {'Images':>8} {'Instances':>10} {'Infer(ms/img)':>14} {'Loss':>12}",flush=True)
    try:
        for batch,(samples,targets) in enumerate(loader,1):
            synchronize(device);started=time.perf_counter()
            samples = samples.to(device)
            targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
            synchronize(device);t1=time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config.amp and device.type=='cuda'):
                outputs = model(samples,targets) if getattr(model,'dn_number',0) else model(samples)
            synchronize(device);t2=time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config.amp and device.type=='cuda'):
                losses = criterion(outputs,targets)
            weights = criterion.weight_dict
            weighted = {k:float(v)*weights[k] for k,v in losses.items() if k in weights}
            raw = {k:float(v) for k,v in losses.items()}
            total = sum(weighted.values())
            if not np.isfinite(total):
                raise FloatingPointError(f'验证loss非有限数: {raw}')
            synchronize(device);t3=time.perf_counter()
            orig_sizes = torch.stack([t['orig_size'] for t in targets])
            results = postprocessor(outputs,orig_sizes)
            results = [{k:v.detach().cpu() for k,v in r.items()} for r in results]
            synchronize(device);t4=time.perf_counter()
            ids = [int(t['image_id']) for t in targets]
            native.update(dict(zip(ids,results)))
            owt.update(results,ids)
            n=len(targets);images+=n;total_batches+=1
            prep_time+=t1-started;infer_time+=t2-t1;loss_time+=t3-t2;post_time+=t4-t3
            for key,value in {'loss':total,**weighted,**{k+'_unscaled':v for k,v in raw.items()}}.items():
                totals[key]=totals.get(key,0.)+value
            recorder.csv('metrics/val_iterations.csv',{'epoch':epoch,'batch':batch,'batches':len(loader),
                'images':n,'instances':sum(len(t['labels']) for t in targets),
                'preprocess_seconds':t1-started,'inference_seconds':t2-t1,
                'loss_seconds':t3-t2,'postprocess_seconds':t4-t3,'val/loss':total,
                **{f'unscaled/{k}':v for k,v in raw.items()},**{f'weighted/{k}':v for k,v in weighted.items()}})
            print(f"\r[val] {batch:>5}/{len(loader):<5} {n:>8} {sum(len(t['labels']) for t in targets):>10} "
                  f"{(t2-t1)*1000/n:>14.3f} {total:>12.5g}",
                  end="\n" if batch == len(loader) else "", flush=True)
            for iid,result in zip(ids,results):
                if handle:
                    for box,score,cls in zip(result['boxes'].tolist(),result['scores'].tolist(),result['labels'].tolist()):
                        x1,y1,x2,y2=box
                        payload={'image_id':iid,'file_name':coco.imgs[iid]['file_name'],
                                 'category_id':int(cls)+1,'class_id':int(cls),
                                 'bbox':[x1,y1,x2-x1,y2-y1],'score':score}
                        if not first:handle.write(',\n')
                        json.dump(payload,handle,ensure_ascii=False,allow_nan=False)
                        first=False
                if batch<=3 and iid==ids[0]:
                    image_path=Path(loader.dataset.root)/coco.imgs[iid]['file_name']
                    preview(image_path,coco.imgToAnns.get(iid,[]),result,folder/f'val_batch{batch-1}_pred.jpg')
        if not total_batches:
            raise ValueError('验证DataLoader为空，拒绝输出虚假指标。')
        native.synchronize_between_processes();native.accumulate();native.summarize()
        metrics=owt.finalize(epoch,recorder,folder)
        metrics.update(complexity)
        latency=infer_time*1000/images
        metrics.update({'speed/val-latency-ms':latency,'speed/val-FPS':1000/latency if latency>0 else float('nan'),
                        'speed/preprocess-ms':prep_time*1000/images,'speed/loss-ms':loss_time*1000/images,
                        'speed/postprocess-ms':post_time*1000/images,
                        'memory/PeakAllocated-GB':torch.cuda.max_memory_allocated(device)/(1<<30) if device.type=='cuda' else 0.,
                        'sparsity/active-token-ratio':1.,'sparsity/compact-token-ratio':1.,
                        'sparsity/note':'no token pruning; dense model; deformable sampling is not token pruning'})
        stats=native.coco_eval['bbox'].stats.tolist()
        native_row={'epoch':epoch,**{f'coco/{k}':v for k,v in zip(COCO_COLUMNS,stats)}}
        recorder.csv('metrics/coco_metrics.csv',native_row)
        metrics.update({k:v for k,v in native_row.items() if k!='epoch'})
        for c,name in enumerate(CLASS_NAMES):
            ce=native.coco_eval['bbox']
            cat_index=list(ce.params.catIds).index(c)
            row={'epoch':epoch,'class_id':c,'class_name':name}
            for key,area in [('AP',0),('APsmall',1),('APmedium',2),('APlarge',3)]:
                values=ce.eval['precision'][:,:,cat_index,area,-1]
                values=values[values>=0]
                row[key]=float(values.mean()) if values.size else -1.
            recorder.csv('metrics/coco_per_class.csv',row)
        recorder.csv('metrics/speed.csv',{'epoch':epoch,'images':images,**{k:v for k,v in metrics.items() if k.startswith(('speed/','memory/','sparsity/'))}})
        if config.save_eval_arrays:
            dest=recorder.run_dir/'eval'/f'epoch{epoch:04d}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
            ce=native.coco_eval['bbox']
            np.savez_compressed(dest,precision=ce.eval['precision'],recall=ce.eval['recall'],scores=ce.eval['scores'],
                                iou_thresholds=ce.params.iouThrs,recall_thresholds=ce.params.recThrs,
                                category_ids=ce.params.catIds,stats=stats)
        print('Complexity: Params=%d, dense-equivalent upper estimate=%.3f GFLOPs, '
              'val inference=%.3f ms/image (%.2f FPS), PeakAllocated=%.3f GB, active tokens=%.2f%%' %
              (metrics['complexity/Parameters'],metrics['complexity/GFLOPs-dense-upper'],latency,
               metrics['speed/val-FPS'],metrics['memory/PeakAllocated-GB'],100.),flush=True)
        if handle:
            handle.write(']\n');handle.close();handle=None
            json_path.with_suffix('.json.tmp').replace(json_path)
        return {k:v/total_batches for k,v in totals.items()},metrics
    finally:
        if handle is not None:handle.close()
