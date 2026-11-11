"""Readonly COCO view with exact square letterbox and 0-based -> 1-based detector labels."""
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader,RandomSampler,BatchSampler,SequentialSampler
from torchvision.transforms.functional import to_tensor
from .common import read_json


def letterbox(image,boxes,size,value=114,flip=False):
    width,height=image.size
    ratio=min(size/width,size/height)
    new_width=max(1,round(width*ratio)); new_height=max(1,round(height*ratio))
    left=(size-new_width)//2; top=(size-new_height)//2
    sx,sy=new_width/width,new_height/height
    canvas=Image.new('RGB',(size,size),(value,value,value))
    canvas.paste(image.resize((new_width,new_height),Image.Resampling.BILINEAR),(left,top))
    boxes=boxes.clone()
    boxes[:,0::2]=boxes[:,0::2]*sx+left
    boxes[:,1::2]=boxes[:,1::2]*sy+top
    if flip:
        canvas=canvas.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        boxes[:,[0,2]]=size-boxes[:,[2,0]]
    return to_tensor(canvas),boxes,torch.tensor([sx,sy],dtype=torch.float64),torch.tensor([left,top],dtype=torch.float64)


def inverse_letterbox(boxes,target):
    # Use actual per-axis rounded resize ratios, not the nominal pre-round ratio.
    values=boxes.detach().cpu().double().clone()
    scale=target['resize_scale'].detach().cpu(); pad=target['pad'].detach().cpu()
    values[:,0::2]=(values[:,0::2]-pad[0])/scale[0]
    values[:,1::2]=(values[:,1::2]-pad[1])/scale[1]
    height,width=map(int,target['orig_size'].detach().cpu().tolist())
    values[:,0::2].clamp_(0,width); values[:,1::2].clamp_(0,height)
    return values


class TinyWODDataset(Dataset):
    def __init__(self,cfg,info,split,training=False):
        self.cfg=cfg; self.training=training
        part=info['splits'][split]; self.root=Path(part['images_path']); self.annotation=part['annotation']
        payload=read_json(self.annotation); self.items=payload['images']
        self.annotations={int(image['id']):[] for image in self.items}
        for ann in payload['annotations']:
            if not ann.get('iscrowd',0): self.annotations[int(ann['image_id'])].append(ann)
    def __len__(self): return len(self.items)
    def __getitem__(self,index):
        item=self.items[index]; anns=self.annotations[int(item['id'])]
        boxes=torch.tensor([[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]]
                            for a in anns],dtype=torch.float32).reshape(-1,4)
        labels=torch.tensor([a['category_id']+1 for a in anns],dtype=torch.int64)
        with Image.open(self.root/item['file_name']) as source: image=source.convert('RGB')
        flip=self.training and random.random()<self.cfg.horizontal_flip
        tensor,boxes,scale,pad=letterbox(image,boxes,self.cfg.image_size,self.cfg.letterbox_value,flip)
        target=dict(boxes=boxes,labels=labels,image_id=torch.tensor(int(item['id'])),
            orig_size=torch.tensor([item['height'],item['width']]),resize_scale=scale,pad=pad,
            horizontal_flip=torch.tensor(flip),area=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]))
        return tensor,target


def collate_fn(batch):
    images,targets=zip(*batch)
    return list(images),list(targets)


def seed_worker(_):
    seed=torch.initial_seed()%2**32
    random.seed(seed); np.random.seed(seed)


def make_dataset(cfg,split,info,training=False):
    return TinyWODDataset(cfg,info,split,training)


def make_loaders(cfg,info):
    train=make_dataset(cfg,'train',info,True); val=make_dataset(cfg,'val',info)
    if len(train)<cfg.batch_size:
        raise ValueError(f'训练图片 {len(train)} 少于 batch_size={cfg.batch_size}，drop_last=True 将无有效 batch。')
    generator=torch.Generator().manual_seed(cfg.seed)
    worker_generator=torch.Generator().manual_seed(cfg.seed+1)
    common=dict(collate_fn=collate_fn,num_workers=cfg.workers,pin_memory=cfg.pin_memory,
                worker_init_fn=seed_worker,generator=worker_generator,persistent_workers=False)
    train_loader=DataLoader(train,batch_sampler=BatchSampler(RandomSampler(train,generator=generator),cfg.batch_size,drop_last=True),**common)
    val_loader=DataLoader(val,batch_size=cfg.val_batch_size,sampler=SequentialSampler(val),drop_last=False,**common)
    return train_loader,val_loader,generator,worker_generator
