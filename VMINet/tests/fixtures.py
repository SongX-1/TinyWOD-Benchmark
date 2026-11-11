"""Synthetic tests only; no real TinyWOD accuracy claim."""
from pathlib import Path
import numpy as np
from PIL import Image
from config import Experiment,CLASS_NAMES
from tinywod.common import write_json


def create_fixture(root):
    root=Path(root);data=root/'dataset';rng=np.random.default_rng(1234)
    for split,count in [('train',6),('val',5),('test',2)]:
        images=data/'images'/split;images.mkdir(parents=True,exist_ok=True)
        if split!='test':(data/'labels'/split).mkdir(parents=True,exist_ok=True)
        for i in range(count):
            Image.fromarray(rng.integers(0,256,(96,128,3),dtype=np.uint8)).save(images/f'{i:04d}.png')
            if split!='test':
                value=f'{i%5} .5 .5 .08 .10\n'
                if split=='train' and i==0:value+=value
                (data/'labels'/split/f'{i:04d}.txt').write_text(value)
    write_json(data/'manifest.json',dict(names=CLASS_NAMES))
    return Experiment(variant='ti',epochs=2,yolo_root=str(data),coco_root=str(root/'missing'),dataset_mode='yolo',
        output_root=str(root/'runs'),device='cpu',cpu_threads=2,workers=0,pretrained=False,smoke_test=True,
        image_size=64,batch_size=2,val_batch_size=2,pyramid_channels=32,roi_hidden_dim=64,roi_pool_size=3,
        rpn_pre_nms_train=50,rpn_post_nms_train=30,rpn_pre_nms_test=30,rpn_post_nms_test=20,
        rpn_batch_size=16,roi_batch_size=16,max_detections=20,lr_milestones=(1,),checkpoint_interval=1,
        prediction_images=2,benchmark_images=1,benchmark_warmup=0,benchmark_repeats=1,make_plots=True)
