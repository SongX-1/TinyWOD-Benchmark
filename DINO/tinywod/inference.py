# -*- coding: utf-8 -*-
"""TinyWOD官方无标签test只导出预测，不计算或伪造mAP。"""
import json
from pathlib import Path
import time
import torch
from PIL import Image

from experiment_config import CLASS_NAMES
from .data import make_transforms
from .evaluation import synchronize, preview
from .recording import append_csv


@torch.no_grad()
def predict_test(model,postprocess,device,config,run):
    model.eval()
    source=Path(config.dataset_root)/'images'/'test'
    files=sorted(p for p in source.rglob('*') if p.is_file() and
                 p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'})
    out=Path(run)/'test_predictions'
    number=1
    candidate=out
    while candidate.exists():
        number+=1;candidate=out.with_name(f'test_predictions_{number}')
    out=candidate;out.mkdir(parents=True)
    transform=make_transforms('val',config.imgsz)
    columns=['image_id','file_name','class_id','category_id','class_name','confidence',
             'x1','y1','x2','y2','width','height','image_width','image_height']
    import csv
    with (out/'detections.csv').open('w',encoding='utf-8-sig',newline='') as f:
        csv.writer(f).writerow(columns)
    with (out/'predictions.json').open('w',encoding='utf-8') as f:
        f.write('[');first=True
        for index,path in enumerate(files,1):
            with Image.open(path) as im:
                image=im.convert('RGB');w,h=image.size
                target={'boxes':torch.zeros((0,4)),'area':torch.zeros(0),'size':torch.tensor([h,w])}
                tensor,_=transform(image,target)
            tensor=tensor[None].to(device)
            synchronize(device);start=time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config.amp and device.type=='cuda'):
                outputs=model(tensor)
            synchronize(device);latency=(time.perf_counter()-start)*1000
            pred=postprocess(outputs,torch.tensor([[h,w]],device=device))[0]
            pred={k:v.detach().cpu() for k,v in pred.items()}
            selected=pred['scores']>=config.test_confidence
            pred={k:v[selected] for k,v in pred.items()}
            name=path.relative_to(source).as_posix()
            count=len(pred['scores'])
            append_csv(out/'images.csv',{'image_id':index,'file_name':name,'image_width':w,'image_height':h,
                'detections':count,'inference_ms':latency,'has_ground_truth':False})
            for box,score,cls in zip(pred['boxes'].tolist(),pred['scores'].tolist(),pred['labels'].tolist()):
                x1,y1,x2,y2=box;cls=int(cls)
                append_csv(out/'detections.csv',{'image_id':index,'file_name':name,'class_id':cls,'category_id':cls+1,
                    'class_name':CLASS_NAMES[cls],'confidence':score,'x1':x1,'y1':y1,'x2':x2,'y2':y2,
                    'width':x2-x1,'height':y2-y1,'image_width':w,'image_height':h},columns)
                if not first:f.write(',\n')
                json.dump({'image_id':index,'file_name':name,'category_id':cls+1,
                           'bbox':[x1,y1,x2-x1,y2-y1],'score':score},f,ensure_ascii=False,allow_nan=False)
                first=False
            if index<=8:preview(path,[],pred,out/f'preview_{index:03d}.jpg')
            print(f'[test] {index}/{len(files)} {name}: {count} detections, {latency:.3f} ms/image（无标签，仅推理）',flush=True)
        f.write(']\n')
    print(f'[test] 预测结果: {out}',flush=True)
    return out
