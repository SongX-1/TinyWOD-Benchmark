"""可选 PyCharm Run：真实两轮训练、精确续训、无 GT/带 GT 测试流程与文件完整性核对。"""
from dataclasses import replace
from pathlib import Path
import csv
import json
import uuid
import numpy as np
import torch
from config import ROOT,Experiment
from tests.fixtures import create_fixture
from tinywod.common import read_json,write_json
from tinywod.runner import launch


def same(a,b,path='root'):
    if isinstance(a,torch.Tensor):
        if not torch.equal(a,b):raise AssertionError('tensor mismatch: '+path)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b,err_msg=path)
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:same(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):same(x,y,path+'/'+str(i))
    else:assert a==b,(path,a,b)


def finished(cfg):
    folder=Path(cfg.output_root)/f'VMINet-{cfg.variant.upper()}-FasterRCNN-SMOKE'
    before=set(folder.iterdir()) if folder.is_dir() else set()
    code=launch(cfg);assert code==0,code
    run,=set(folder.iterdir())-before
    assert read_json(run/'experiment_summary.json')['status']=='completed'
    assert read_json(run/'run_state.json')['status']=='completed'
    text=(run/'logs/console.log').read_text()
    assert '[训练完成]' in text and '[实验汇总] status=completed' in text,'console is incomplete'
    return run


def main():
    root=ROOT/'verification'/('runtime_'+uuid.uuid4().hex[:8])
    cfg=create_fixture(root/'continuous')
    parent=finished(cfg)
    child=finished(replace(cfg,resume_from=str(parent/'weights/epoch_0001.pth'),make_plots=False))
    a=torch.load(parent/'weights/epoch_0002.pth',map_location='cpu',weights_only=False)
    b=torch.load(child/'weights/epoch_0002.pth',map_location='cpu',weights_only=False)
    for key in ('model','optimizer','scheduler','scaler','best_model','best_epoch','best_ap','best_confidence','global_step','rng'):
        same(a[key],b[key],key)
    def losses(run):
        with (run/'metrics/epoch_losses.csv').open(encoding='utf-8-sig') as f:
            return [r for r in csv.DictReader(f) if r['phase'] in ('train','val')]
    assert losses(parent)==losses(child),'train/val loss mismatch'
    for run in (parent,child):
        for rel,keys in [('results.csv',['epoch']),('summary.csv',['epoch']),('metrics/losses.csv',['epoch','phase','batch','name']),
            ('metrics/batches.csv',['epoch','phase','batch']),('metrics/learning_rates.csv',['epoch','batch','group']),
            ('metrics/metrics.csv',['epoch','stage','split'])]:
            with (run/rel).open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
            assert len(rows)==len({tuple(r[k] for k in keys) for r in rows}),rel
        with (run/'summary.csv').open(encoding='utf-8-sig') as file:assert [r['epoch'] for r in csv.DictReader(file)]==['1','2']
    labeled_cfg=create_fixture(root/'labeled')
    for i in range(2):
        path=Path(labeled_cfg.yolo_root)/'labels'/'test'/f'{i:04d}.txt'
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(f'{i} .5 .5 .08 .10\n')
    labeled=finished(replace(labeled_cfg,epochs=1,make_plots=False))
    state=read_json(labeled/'run_state.json')
    assert state['metric_split']=='test'
    assert state['final_metrics']['test']['confidence']==state['best_confidence']
    assert read_json(labeled/'metrics/final/test/metric_definitions.json')['confidence_source']=='frozen_from_best_validation'
    report=dict(status='passed',continuous=str(parent),resumed=str(child),labeled_test=str(labeled),
        continuous_epochs=2,resume_after_epoch=1,resumed_to_epoch=2,exact_model_optimizer_scheduler_scaler_best_model_rng=True,
        train_validation_losses_exact=True,duplicate_csv_keys=0,complete_console_and_summary=True,
        unlabeled_test='predictions only; val metrics',labeled_test_threshold='frozen from best val; test only after training',
        architecture='full original Ti backbone with smaller detector head/input/proposal budgets',device='CPU',
        real_TinyWOD_training=False)
    write_json(ROOT/'verification/runtime_report.json',report)
    print('[运行核对通过] '+json.dumps(report,ensure_ascii=False),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
