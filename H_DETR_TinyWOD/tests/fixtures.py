"""Synthetic-only fixture. Its scores are never TinyWOD experiment results."""
from dataclasses import replace
from pathlib import Path
import numpy as np
from PIL import Image
from config import Experiment,CLASS_NAMES
from tinywod.common import write_json


def create_fixture(root):
    root=Path(root); data=root/'dataset'; rng=np.random.default_rng(1234)
    for split,count in [('train',6),('val',5),('test',2)]:
        images=data/'images'/split; images.mkdir(parents=True,exist_ok=True)
        if split!='test': (data/'labels'/split).mkdir(parents=True,exist_ok=True)
        for i in range(count):
            Image.fromarray(rng.integers(0,256,(96,128,3),dtype=np.uint8)).save(images/f'{i:04d}.png')
            if split!='test':
                label=f'{i%5} .5 .5 .08 .10\n'
                if split=='train' and i==0: label+=label
                (data/'labels'/split/f'{i:04d}.txt').write_text(label)
    write_json(data/'manifest.json',dict(names=CLASS_NAMES))
    return Experiment(epochs=2,yolo_root=str(data),coco_root=str(root/'absent'),dataset_mode='yolo',
        output_root=str(root/'runs'),image_size=96,max_size=128,batch_size=2,val_batch_size=2,
        workers=0,device='cpu',cpu_threads=2,pretrained=False,smoke_test=True,ops_backend='pytorch',
        enc_layers=1,dec_layers=2,dim_feedforward=256,num_queries_one2one=10,num_queries_one2many=20,
        k_one2many=2,max_detections=20,lr_drop=1,checkpoint_interval=1,prediction_images=2,
        benchmark_images=1,benchmark_warmup=0,benchmark_repeats=1,make_plots=True)
