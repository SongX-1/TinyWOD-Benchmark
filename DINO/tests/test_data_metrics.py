from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

from experiment_config import CONFIG
from tinywod.data import prepare_dataset,LetterBox,OriginalSpacePostProcess
from tinywod.metrics import DetectionAccumulator
from tinywod.owt_matching import OWTMatcher
from tinywod.recording import Recorder,append_csv,trim_for_resume,OWT_COLUMNS
from tests.fixtures import make_dataset


def test_original_images_readonly_and_category_mapping(tmp_path):
    data=make_dataset(tmp_path/'source')
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in data.rglob('*') if p.is_file()}
    cfg=replace(CONFIG,dataset_root=data,cache_root=tmp_path/'cache')
    manifest=prepare_dataset(cfg)
    normalized=json.loads((cfg.cache_root/'annotations'/'instances_train.json').read_text())
    assert {a['category_id'] for a in normalized['annotations']}==set(range(5))
    assert len(normalized['images'])==3 and manifest['splits']['train']['boxes']==10
    after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in data.rglob('*') if p.is_file()}
    assert before==after
    (data/'images'/'val'/'样本_2.png').unlink()
    with pytest.raises(FileNotFoundError,match='找不到'):
        prepare_dataset(cfg)


@pytest.mark.parametrize('size,shape',[(640,(3840,2160)),(256,(311,173)),(640,(173,311))])
def test_letterbox_boxes_roundtrip(size,shape):
    w,h=shape
    image=Image.new('RGB',shape)
    boxes=torch.tensor([[5.,8.,35.,28.],[w/3,h/3,w*.9,h*.8]])
    _,target=LetterBox(size)(image,{'boxes':boxes.clone(),'area':torch.tensor([600.,1000.])})
    class Dummy:
        def __call__(self,outputs,sizes):
            assert sizes.tolist()==[[size,size]]
            return [{'boxes':target['boxes'].clone(),'labels':torch.tensor([0,1]),'scores':torch.ones(2)}]
    restored=OriginalSpacePostProcess(Dummy(),size)(None,torch.tensor([[h,w]]))[0]['boxes']
    assert torch.allclose(restored,boxes,atol=.001)


def test_perfect_predictions_and_small_ignore(tmp_path):
    from pycocotools.coco import COCO
    coco=COCO()
    coco.dataset={'images':[{'id':1,'file_name':'one.png','width':200,'height':200}],
                  'categories':[{'id':i,'name':str(i)} for i in range(5)],
                  'annotations':[{'id':1,'image_id':1,'category_id':0,'bbox':[10,10,10,10],'area':100,'iscrowd':0},
                                 {'id':2,'image_id':1,'category_id':0,'bbox':[50,50,80,80],'area':6400,'iscrowd':0}]}
    coco.createIndex()
    pred={'boxes':torch.tensor([[10.,10.,20.,20.],[50.,50.,130.,130.]]),
          'scores':torch.tensor([.99,.98]),'labels':torch.tensor([0,0])}
    accumulator=DetectionAccumulator(coco);accumulator.update([pred],[1])
    result=accumulator.finalize(1,Recorder(tmp_path))
    assert result['metrics/mAP50-95(B)']>.99
    assert result['metrics/mAP50-95-small(B)']>.99
    # 中大框的正确检测在APsmall中必须被忽略，不能成为小目标假阳性。
    _,ignore,_=accumulator.matcher._process_small_batch(
        torch.tensor([[50.,50.,130.,130.,.99,0.],[10.,10.,20.,20.,.98,0.]]),
        torch.tensor([[0.,10.,10.,20.,20.],[0.,50.,50.,130.,130.]]))
    assert ignore[0].all() and not ignore[1].any()
    rows=list(csv.DictReader((tmp_path/'metrics'/'per_class.csv').open(encoding='utf-8-sig')))
    assert len(rows)==5


def test_no_predictions_no_labels_and_false_positives(tmp_path):
    from pycocotools.coco import COCO
    coco=COCO();coco.dataset={'images':[{'id':1}],'annotations':[],
                             'categories':[{'id':i} for i in range(5)]};coco.createIndex()
    empty={'boxes':torch.zeros(0,4),'scores':torch.zeros(0),'labels':torch.zeros(0,dtype=torch.long)}
    a=DetectionAccumulator(coco);a.update([empty],[1]);metrics=a.finalize(1,Recorder(tmp_path))
    assert not metrics['dataset/val-has-ground-truth']
    b=DetectionAccumulator(coco)
    b.update([{'boxes':torch.tensor([[1.,1.,10.,10.]]),'scores':torch.tensor([.9]),'labels':torch.tensor([2])}],[1])
    assert b.confusion[2,5]==1


def test_csv_schema_growth_and_checkpoint_rewind(tmp_path):
    path=tmp_path/'metrics'/'train_iterations.csv'
    append_csv(path,{'epoch':1,'loss':3})
    append_csv(path,{'epoch':2,'loss':2,'new_aux':.5})
    append_csv(path,{'epoch':3,'loss':1,'new_aux':.25})
    trim_for_resume(tmp_path,1)
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);rows=list(reader)
        assert 'new_aux' in reader.fieldnames
    assert len(rows)==1 and rows[0]['loss']=='3'
    assert list((tmp_path/'resume_backups').rglob('train_iterations.csv'))


def test_native_attention_fallback_has_gradients():
    from models.dino.ops.modules import MSDeformAttn
    torch.manual_seed(1)
    module=MSDeformAttn(d_model=32,n_levels=2,n_heads=4,n_points=2)
    # 上游初始化将这两组weight置零；施加非零weight以验证query梯度路径。
    torch.nn.init.normal_(module.sampling_offsets.weight,std=.01)
    torch.nn.init.normal_(module.attention_weights.weight,std=.01)
    memory=torch.randn(1,20,32,requires_grad=True)
    query=torch.randn(1,3,32,requires_grad=True)
    output=module(query,torch.rand(1,3,2,2),memory,torch.tensor([[4,4],[2,2]]),torch.tensor([0,16]))
    output.square().mean().backward()
    assert torch.isfinite(output).all() and memory.grad.abs().sum()>0 and query.grad.abs().sum()>0
