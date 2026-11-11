# -*- coding: utf-8 -*-
"""直接Run：合成数据，全结构DINO-R50/6+6层/900queries，1轮训练+恢复至2轮。

仅此检查使用256输入和batch=1，不改experiment_config.py的640/12轮正式配置。
测试输出到_smoke_tests独立目录，不进入正式runs。CPU也可以运行，需要完整依赖。
"""
from dataclasses import replace
from pathlib import Path
import csv
import datetime
import json
import math
import tempfile


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        return reader.fieldnames,list(reader)


def main():
    import multiprocessing
    multiprocessing.freeze_support()
    from experiment_config import CONFIG,PROJECT_ROOT
    from tests.fixtures import make_dataset
    from tinywod.runner import run_experiment,load_trusted_checkpoint
    from tinywod.recording import OWT_COLUMNS,append_csv
    base=PROJECT_ROOT/'_smoke_tests'
    base.mkdir(exist_ok=True)
    root=Path(tempfile.mkdtemp(prefix='smoke_',dir=base))
    source=make_dataset(root/'synthetic_source')
    cfg=replace(CONFIG,dataset_root=source,cache_root=root/'cache',output_root=root/'runs',
                epochs=1,imgsz=256,batch_size=1,val_batch_size=1,workers=0,device='cpu',amp=False,
                save_period=1,profile_flops=True,predict_test_after_training=True)
    assert not cfg.model_overrides
    run=run_experiment(cfg)
    args=json.loads((run/'model_args.json').read_text())
    assert (args['backbone'],args['enc_layers'],args['dec_layers'],args['num_queries'])==('resnet50',6,6,900)
    # 模拟第2轮中途退出留下的CSV，再从第1轮last.pth恢复，必须备份并消除重复行。
    append_csv(run/'metrics'/'train_iterations.csv',{'epoch':2,'batch':1,'train/loss':999})
    run2=run_experiment(replace(cfg,epochs=2,resume=str(run/'weights'/'last.pth')))
    assert run==run2
    fields,rows=read_csv(run/'results.csv')
    assert set(OWT_COLUMNS)<=set(fields) and [int(r['epoch']) for r in rows]==[1,2]
    _,train=read_csv(run/'metrics'/'train_iterations.csv')
    _,val=read_csv(run/'metrics'/'val_iterations.csv')
    _,per_class=read_csv(run/'metrics'/'per_class.csv')
    assert len(train)==6 and len(val)==4 and len(per_class)==10
    assert all(math.isfinite(float(r['train/loss'])) for r in train)
    assert any(int(r['instances'])==0 for r in train)
    assert (run/'test_predictions'/'detections.csv').is_file()
    assert (run/'test_predictions_2'/'detections.csv').is_file()
    assert (run/'results.png').is_file() and (run/'weights'/'best.pth').is_file()
    ckpt=load_trusted_checkpoint(run/'weights'/'last.pth')
    import torch
    steps=[int(v['step']) for v in ckpt['optimizer']['state'].values() if 'step' in v]
    assert max(steps)==6
    bias=ckpt['model']['class_embed.0.bias']
    assert not torch.allclose(bias,torch.full_like(bias,-math.log(99)),atol=1e-6,rtol=0)
    status=json.loads((run/'status.json').read_text())
    assert status['status']=='completed'
    report={'status':'passed','real_dataset_used':False,'epochs':2,'imgsz':256,'batch_size':1,
            'architecture':'DINO-R50, 4 levels, enc=6, dec=6, queries=900',
            'optimizer_steps':max(steps),'train_iteration_rows':len(train),'val_iteration_rows':len(val),
            'per_class_rows':len(per_class),'owt_columns_missing':list(set(OWT_COLUMNS)-set(fields)),
            'run_dir':str(run),'device':'cpu','resume_tested':True}
    (root/'smoke_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('[smoke] 全部通过: '+str(root/'smoke_report.json'),flush=True)
    return root


if __name__=='__main__':
    main()
