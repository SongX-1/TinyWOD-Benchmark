# -*- coding: utf-8 -*-
"""PyCharm直接Run：最新实验最佳权重的val复验 + 官方无标签test推理。"""
from dataclasses import replace
from pathlib import Path

# 不填时自动选择CONFIG.output_root下最新编号实验的best.pth。
CHECKPOINT = ""


def main():
    import multiprocessing
    multiprocessing.freeze_support()
    from experiment_config import CONFIG
    from tinywod.runner import (configure_threads,load_trusted_checkpoint,select_device,
                                    build_args,build_loaders)
    configure_threads(CONFIG)
    import torch
    from models import build_model
    from tinywod.data import prepare_dataset,OriginalSpacePostProcess
    from tinywod.recording import Recorder,ConsoleLog,reserve_run
    from tinywod.evaluation import evaluate
    from tinywod.profiling import profile_model
    from tinywod.inference import predict_test
    if CHECKPOINT:
        path=Path(CHECKPOINT).expanduser().resolve()
    else:
        candidates=[p for p in Path(CONFIG.output_root).glob('*run/weights/best.pth')
                    if p.parent.parent.name[:-3].isdigit()]
        if not candidates:raise FileNotFoundError('没有best.pth；请先运行train_dino_r50.py，或设置CHECKPOINT。')
        path=max(candidates,key=lambda p:int(p.parent.parent.name[:-3]))
    ckpt=load_trusted_checkpoint(path)
    saved=ckpt.get('run_config',{})
    cfg=replace(CONFIG,model_overrides=saved.get('model_overrides',CONFIG.model_overrides),
                imgsz=saved.get('imgsz',CONFIG.imgsz))
    out=reserve_run(path.parent.parent/'evaluations')
    with ConsoleLog(out/'logs'/'console.log'):
        print(f'[test] 使用权重: {path}',flush=True)
        device=select_device(cfg);manifest=prepare_dataset(cfg);args=build_args(cfg,device,manifest)
        model,criterion,post=build_model(args);model.to(device);model.load_state_dict(ckpt['model'])
        del ckpt
        processor=OriginalSpacePostProcess(post['bbox'],cfg.imgsz)
        _,val,_=build_loaders(args,cfg)
        complexity=profile_model(model,cfg.imgsz,device,cfg.profile_flops)
        recorder=Recorder(out)
        _,metrics=evaluate(model,criterion,processor,val,device,1,cfg,recorder,complexity)
        recorder.csv('summary.csv',{'checkpoint':str(path),'split':'val',**metrics})
        if manifest['splits']['test']['images']:predict_test(model,processor,device,cfg,out)
        recorder.save_schema()
    return out


if __name__=='__main__':
    main()
