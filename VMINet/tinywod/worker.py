"""VMINet TinyWOD training, final evaluation and exact epoch-boundary recovery."""
from dataclasses import asdict
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

from config import CLASS_NAMES,ROOT,resolve
from .common import now,read_json,write_csv,write_json,resource_snapshot


def setup_environment(cfg,output):
    # Ordinary imports only: no dependency probes, package scanning, version gates or installation.
    import numpy as np
    import torch,torchvision
    torch.set_num_threads(cfg.cpu_threads)
    if hasattr(torch.backends,'nnpack'):torch.backends.nnpack.set_flags(False)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if cfg.device=='auto' else torch.device(cfg.device)
    if device.type not in ('cpu','cuda'):raise ValueError('device 使用 cpu/cuda:0/auto')
    if device.type=='cuda':torch.cuda.set_device(device)
    if cfg.amp and device.type!='cuda':raise ValueError('CPU 运行请使用 amp=False')
    random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=cfg.cudnn_deterministic
    # Native CUDA ROIAlign backward is not bitwise deterministic. Global strict mode
    # also selects torchvision's lazy-compiled fallback; retain native wheel operators.
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.allow_tf32=False
    if hasattr(torch.backends.cuda.matmul,'allow_tf32'):torch.backends.cuda.matmul.allow_tf32=False
    env=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),torch=torch.__version__,torchvision=torchvision.__version__,
        cuda_build=torch.version.cuda,device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor() or 'CPU',
        seed=cfg.seed,val_loss_seed=cfg.val_loss_seed,amp=cfg.amp,workers=cfg.workers,pin_memory=cfg.pin_memory,
        cpu_threads=cfg.cpu_threads,cudnn_deterministic=cfg.cudnn_deterministic,global_strict_determinism=False,
        determinism_note='seeded sampling and cuDNN; native CUDA ROIAlign backward may be nondeterministic; no claim of bitwise GPU reproducibility',
        extension_compile=False,torch_compile=False)
    write_json(output/'logs'/'environment.json',env)
    write_csv(output/'logs'/'software_versions.csv',[dict(name=k,value=env[k]) for k in ('python','torch','torchvision','cuda_build')],['name','value'])
    print(f"[设备] {device} | {env['device_name']} | AMP={cfg.amp}",flush=True)
    print(f"[随机设置] seed={cfg.seed} | val_loss_seed={cfg.val_loss_seed} | workers={cfg.workers} | cuDNN deterministic={cfg.cudnn_deterministic}",flush=True)
    return device,env


def execute(cfg,output,state):
    import torch
    from torch.utils.data import DataLoader
    from .modeling import validate_config,preflight_weights,build_native,make_optimizer
    from .data import prepare_dataset
    from .loading import make_loaders,make_dataset,collate_fn,seed_worker
    from .recording import Recorder,loss_summary,loss_console,summary_row
    from .training import train_epoch,infer_loader,print_metrics,complexity,print_complexity,benchmark,draw_predictions,INFERENCE_SCOPE
    from .evaluation import evaluate,plot_training
    from .checkpoint import load_resume,restore_rng,rng_state,atomic_save,cpu_state,training_signature

    state['phase']='initialization';validate_config(cfg);preflight_weights(cfg)
    device,env=setup_environment(cfg,output);state.update(device=str(device),device_name=env['device_name'])
    state['phase']='dataset_preparation';info=prepare_dataset(cfg,output/'inputs')
    state['dataset_fingerprint']=info['fingerprint']
    print(f"[数据配置] {output/'inputs'/'dataset_info.json'}",flush=True)
    for split,item in info['splits'].items():
        print(f"[数据划分/{split}] images={item['images_path']} | annotations={item['annotation']} | GT={item['labeled']}",flush=True)
    train_loader,val_loader,generator,worker_generator=make_loaders(cfg,info)
    state['phase']='model_initialization';model,spec,initialization=build_native(cfg,device,output)
    state.update(ops_backend='torchvision_native',initialization=initialization['mode'])
    optimizer,scheduler=make_optimizer(model,cfg,output);scaler=torch.amp.GradScaler('cuda',enabled=cfg.amp)
    checkpoint=load_resume(cfg,info,spec,output,device)
    completed=0;step=0;best_ap=-1.;best_epoch=0;best_confidence=None;best_state=None
    if checkpoint:
        model.load_state_dict(checkpoint['model'],strict=True);optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler']);scaler.load_state_dict(checkpoint['scaler'])
        completed=checkpoint['completed_epoch'];step=checkpoint['global_step'];best_ap=checkpoint['best_ap'];best_epoch=checkpoint['best_epoch']
        best_confidence=checkpoint['best_confidence'];best_state=checkpoint['best_model']
        parent_initialization=checkpoint.get('initialization')
        write_json(output/'inputs'/'resume_initialization.json',dict(current=initialization,original_initialization=parent_initialization))
        if parent_initialization is not None:initialization=parent_initialization
        restore_rng(checkpoint['rng'],generator,worker_generator,device);state['completed_epochs']=completed
        atomic_save(output/'weights'/'best.pth',dict(format='VMINET_SDS_INFERENCE_V1',model=best_state,epoch=best_epoch,ap=best_ap,
            confidence=best_confidence,names=CLASS_NAMES,architecture_fingerprint=spec['architecture_fingerprint'],config=asdict(cfg),initialization=initialization))
    recorder=Recorder(output,cfg,device)
    path=output/'metrics'/'complexity.json'
    if path.is_file():comp=read_json(path)
    else:
        saved_rng=rng_state(generator,worker_generator);image,_=val_loader.dataset[0]
        state['phase']='complexity';comp=complexity(model,image.unsqueeze(0).to(device),device,output)
        restore_rng(saved_rng,generator,worker_generator,device)
    state.update(params=comp['params'],gflops=comp['gflops'],gflops_partial=comp.get('gflops_partial'))
    print(f"[模型结构] VMINet-{cfg.variant.upper()} | 原 Block/四阶段保持 | last-map Simple Pyramid P2..P6 | Faster R-CNN bbox | Params={comp['params']}",flush=True)
    print(f"[模型配置] {ROOT/'config.py'} | 检测适配={ROOT/'models_detection'/'vminet_detector.py'}",flush=True)
    print(f"[输入] {cfg.image_size}×{cfg.image_size} letterbox | 原图坐标还原 | {len(CLASS_NAMES)} 类 0..{len(CLASS_NAMES)-1} -> 检测器前景 1..{len(CLASS_NAMES)}，0 为背景",flush=True)
    print('[损失列] box_loss=ROI SmoothL1，cls_loss=ROI CE，rpn_obj=RPN BCE，rpn_box=RPN SmoothL1；完整四项 train/val 同步保存。',flush=True)
    print(f"[保存位置] weights={output/'weights'} | logs={output/'logs'} | CSV={output/'metrics'} | results={output/'results'}",flush=True)

    for epoch in range(completed+1,cfg.epochs+1):
        epoch_started=now();epoch_start=time.perf_counter()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        state['phase']='training';start=time.perf_counter()
        train_means,step=train_epoch(model,optimizer,scaler,train_loader,recorder,cfg,device,epoch,step)
        train_seconds=time.perf_counter()-start
        state['phase']='validation';start=time.perf_counter()
        predictions,val_means,inference_ms=infer_loader(model,val_loader,recorder,cfg,device,epoch,step,'val',True)
        val_seconds=time.perf_counter()-start
        state['phase']='epoch_evaluation';start=time.perf_counter()
        headline,classes,images=evaluate(info['splits']['val']['annotation'],predictions,output/'metrics'/f'epoch_{epoch:04d}',make_plots=cfg.make_plots)
        evaluation_seconds=time.perf_counter()-start;recorder.evaluation(epoch,'epoch','val',headline,classes,images);print_metrics(headline)
        if headline['ap'] is None:raise RuntimeError('验证 AP 不可定义；请核对有效非 crowd GT。')
        improved=headline['ap']>best_ap
        if improved:best_ap=headline['ap'];best_epoch=epoch;best_confidence=headline['confidence'];best_state=cpu_state(model)
        lrs=[group['lr'] for group in optimizer.param_groups];resource=resource_snapshot(device)
        if resource['peak_allocated_gb'] is not None:state['peak_allocated_gb']=max(state.get('peak_allocated_gb') or 0.,resource['peak_allocated_gb'])
        row=dict(epoch=epoch,start_time=epoch_started,end_time=now(),**loss_summary(train_means,'train'),**loss_summary(val_means,'val'),**headline,
            **{f'lr/pg{i}':lr for i,lr in enumerate(lrs)},lr_min=min(lrs),lr_max=max(lrs),train_seconds=train_seconds,val_seconds=val_seconds,
            evaluation_seconds=evaluation_seconds,epoch_seconds=time.perf_counter()-epoch_start,inference_ms=inference_ms,fps=1000/inference_ms if inference_ms else None,
            peak_allocated_gb=resource['peak_allocated_gb'],params=comp['params'],gflops=comp['gflops'],gflops_partial=comp.get('gflops_partial'),
            best_epoch=best_epoch,is_best=improved,input_height=cfg.image_size,input_width=cfg.image_size,variant=cfg.variant,inference_scope=INFERENCE_SCOPE)
        recorder.results.append(row)
        recorder.summary.append(summary_row(row))
        recorder.timing.append(dict(epoch=epoch,start_time=epoch_started,end_time=row['end_time'],train_seconds=train_seconds,val_seconds=val_seconds,
            evaluation_seconds=evaluation_seconds,epoch_seconds=row['epoch_seconds'],train_batches=len(train_loader),val_batches=len(val_loader)))
        with (output/'log.txt').open('a',encoding='utf-8') as file:
            file.write(json.dumps(dict(epoch=epoch,train=train_means,val=val_means,metrics=headline,lr=lrs),ensure_ascii=False)+'\n');file.flush();os.fsync(file.fileno())
        if cfg.make_plots:plot_training(output)
        scheduler.step();state['phase']='checkpoint'
        full=dict(format='VMINET_SDS_FULL_V1',model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
            completed_epoch=epoch,global_step=step,best_ap=best_ap,best_epoch=best_epoch,best_confidence=best_confidence,best_model=best_state,
            rng=rng_state(generator,worker_generator),dataset_fingerprint=info['fingerprint'],architecture_fingerprint=spec['architecture_fingerprint'],
            training_signature=training_signature(cfg),ops_backend='torchvision_native',device_type=device.type,names=CLASS_NAMES,config=asdict(cfg),initialization=initialization)
        targets=[('last',output/'weights'/'last.pth',full)]
        if improved:targets.insert(0,('best',output/'weights'/'best.pth',dict(format='VMINET_SDS_INFERENCE_V1',model=best_state,
            epoch=best_epoch,ap=best_ap,confidence=best_confidence,names=CLASS_NAMES,architecture_fingerprint=spec['architecture_fingerprint'],config=asdict(cfg),initialization=initialization)))
        if epoch%cfg.checkpoint_interval==0 or epoch==cfg.epochs:targets.append(('periodic',output/'weights'/f'epoch_{epoch:04d}.pth',full))
        for kind,path,payload in targets:
            start=time.perf_counter();atomic_save(path,payload)
            recorder.checkpoints.append(dict(epoch=epoch,kind=kind,path=str(path),seconds=time.perf_counter()-start,bytes=path.stat().st_size))
        row['end_time']=now();row['epoch_seconds']=time.perf_counter()-epoch_start;recorder.results.update(row)
        recorder.summary.update(summary_row(row))
        recorder.timing.update(dict(epoch=epoch,start_time=epoch_started,end_time=row['end_time'],train_seconds=train_seconds,val_seconds=val_seconds,
            evaluation_seconds=evaluation_seconds,epoch_seconds=row['epoch_seconds'],train_batches=len(train_loader),val_batches=len(val_loader)))
        state.update(completed_epochs=epoch,best_epoch=best_epoch,best_ap=best_ap,best_confidence=best_confidence,last_epoch_metrics=headline,global_step=step)
        write_json(output/'run_state.json',state);print_complexity(comp,inference_ms,resource['peak_allocated_gb'])
        text=' '.join(f'{name}={headline[key]:.5f}' if headline[key] is not None else f'{name}=N/A' for name,key in
            [('P','precision'),('R','recall'),('AP50','ap50'),('AP','ap'),('APsmall','apsmall')])
        print(f"[VMINet-{cfg.variant.upper()}] Epoch {epoch}/{cfg.epochs} 完成 | Box({text}) | "
            f"lr={dict((f'lr/pg{i}',lr) for i,lr in enumerate(lrs))} | loss={dict(**loss_console(train_means,'train'),**loss_console(val_means,'val'))}",flush=True)
        print(f'[耗时] train={train_seconds:.2f}s val={val_seconds:.2f}s evaluation={evaluation_seconds:.2f}s | epoch含保存={time.perf_counter()-epoch_start:.2f}s | 结束={now()}',flush=True)
        if improved:print(f"[最佳模型更新] Epoch={best_epoch} AP={best_ap:.6f} | {output/'weights'/'best.pth'}",flush=True)
        print(f"[已保存] last={output/'weights'/'last.pth'} | CSV={output/'results.csv'}",flush=True)
        del full,targets,predictions

    state['phase']='final_evaluation';model.load_state_dict(best_state,strict=True)
    print(f'\n[最终评价] 验证集选出的 best.pth | Epoch={best_epoch} | 固定 confidence={best_confidence:.3f}',flush=True)
    final={}
    predictions,_,val_ms=infer_loader(model,val_loader,recorder,cfg,device,best_epoch,step,'final_val',True)
    val_metrics,classes,images=evaluate(info['splits']['val']['annotation'],predictions,output/'metrics'/'final'/'val',confidence=best_confidence,make_plots=cfg.make_plots)
    recorder.evaluation(best_epoch,'final','val',val_metrics,classes,images);final['val']=val_metrics;print_metrics(val_metrics,classes)
    draw_predictions(info['splits']['val']['annotation'],predictions,info['splits']['val']['images_path'],output/'results'/'val_predictions',best_confidence,cfg.prediction_images)
    test=info['splits']['test']
    if test['images']:
        dataset=make_dataset(cfg,'test',info)
        loader=DataLoader(dataset,batch_size=cfg.val_batch_size,shuffle=False,num_workers=cfg.workers,collate_fn=collate_fn,
            pin_memory=cfg.pin_memory,worker_init_fn=seed_worker,generator=worker_generator)
        phase='test' if test['labeled'] else 'test_unlabeled'
        predictions,_,_=infer_loader(model,loader,recorder,cfg,device,best_epoch,step,phase,test['labeled'])
        folder=output/'metrics'/'final'/phase;folder.mkdir(parents=True,exist_ok=True)
        if test['labeled']:
            metrics,classes,images=evaluate(test['annotation'],predictions,folder,confidence=best_confidence,make_plots=cfg.make_plots)
            recorder.evaluation(best_epoch,'final','test',metrics,classes,images);final['test']=metrics;state['test_status']='evaluated_with_GT';print_metrics(metrics,classes)
        else:
            write_json(folder/'predictions.json',predictions)
            write_csv(folder/'predictions.csv',[dict(image_id=p['image_id'],class_id=p['category_id'],score=p['score'],x=p['bbox'][0],y=p['bbox'][1],width=p['bbox'][2],height=p['bbox'][3])
                for p in predictions],['image_id','class_id','score','x','y','width','height'])
            write_json(folder/'status.json',dict(status='predictions_only',images=test['images'],GT=False,metrics=None,names=CLASS_NAMES,category_ids='internal 0..4; not an official submission conversion'))
            state['test_status']='predictions_only_no_GT';print(f"[测试集] {test['images']} 张无 GT，仅保存预测；最终指标来自 val。",flush=True)
        draw_predictions(test['annotation'],predictions,test['images_path'],output/'results'/'test_predictions',best_confidence,cfg.prediction_images)
    else:state['test_status']='no_test_images'
    state['phase']='benchmark';speed=benchmark(model,val_loader.dataset,cfg,device,output)
    state.update(final_metrics=final,metric_split='test' if 'test' in final else 'val',inference_ms=speed.get('inference_ms_mean',val_ms),
        fps=speed.get('fps',1000/val_ms if val_ms else None),inference_scope=INFERENCE_SCOPE,
        peak_allocated_gb=max(state.get('peak_allocated_gb') or 0.,resource_snapshot(device)['peak_allocated_gb'] or 0.) if device.type=='cuda' else None)
    write_json(output/'metrics'/'final'/'summary.json',dict(best_epoch=best_epoch,best_confidence=best_confidence,metric_split=state['metric_split'],
        test_status=state['test_status'],metrics=final,inference_speed=speed))
    result=final[state['metric_split']]
    core=' '.join(f'{name}={result[key]:.6f}' if result[key] is not None else f'{name}=N/A' for name,key in
        [('P','precision'),('R','recall'),('F1','f1'),('AP50','ap50'),('AP','ap'),('APsmall','apsmall')])
    print(f"[训练完成] {state['completed_epochs']}/{cfg.epochs} Epoch | best Epoch={best_epoch} | 最终指标划分={state['metric_split']}",flush=True)
    print(f"[最终核心指标/{state['metric_split']}] {core}",flush=True)
