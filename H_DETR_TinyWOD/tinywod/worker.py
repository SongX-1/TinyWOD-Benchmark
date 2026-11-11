"""Experiment orchestration using the supplied H-DETR model and hybrid loss."""
import csv
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from dataclasses import asdict

from config import CLASS_NAMES, ROOT, resolve
from .common import now, read_json, write_csv, write_json, resource_snapshot


def setup_environment(cfg, output):
    from .dependencies import check_dependencies
    packages=check_dependencies()
    import numpy as np
    import torch,torchvision
    torch.set_num_threads(cfg.cpu_threads)
    if hasattr(torch.backends,'nnpack'): torch.backends.nnpack.set_flags(False)
    if cfg.device == 'auto': device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else: device=torch.device(cfg.device)
    if device.type not in ('cpu','cuda'): raise ValueError('本项目支持 cpu 或 cuda 设备')
    if device.type=='cuda':
        if not torch.cuda.is_available(): raise RuntimeError('指定 CUDA 但当前解释器无法使用 GPU；不会静默改成 CPU。')
        torch.cuda.set_device(device)
        # 显存保护：为同时运行的其它训练进程预留总显存的 gpu_reserve_fraction 作为安全余量，
        # 用 per-process 上限封顶本进程缓存分配器，避免本进程把整卡吃满导致其它进程 OOM。
        total=torch.cuda.get_device_properties(device).total_memory
        free,_=torch.cuda.mem_get_info(device)
        reserve=cfg.gpu_reserve_fraction*total
        usable=max(0.0,free-reserve)
        if usable<=0:
            raise RuntimeError(f'空闲显存不足：总 {total/2**30:.1f}G 空闲 {free/2**30:.1f}G，'
                               f'预留 {reserve/2**30:.1f}G 后无可分配空间；请先释放其它训练进程或降低 gpu_reserve_fraction。')
        try:
            torch.cuda.set_per_process_memory_fraction(usable/total,device)
            print(f'[显存保护] 总 {total/2**30:.2f}G 空闲 {free/2**30:.2f}G 预留 {reserve/2**30:.2f}G'
                  f'({cfg.gpu_reserve_fraction:.0%}) 本进程上限 {usable/2**30:.2f}G',flush=True)
        except Exception as exc:
            print(f'[显存保护] 设置失败，按系统默认分配：{type(exc).__name__}: {exc}',flush=True)
    if cfg.amp and device.type!='cuda': raise ValueError('AMP 只适用于 CUDA；CPU 请在 config.py 设 amp=False。')
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=cfg.deterministic
    torch.use_deterministic_algorithms(cfg.deterministic,warn_only=not cfg.strict_determinism)
    if hasattr(torch.backends.cuda.matmul,'allow_tf32'): torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if device.type=='cuda':
        try:
            probe=torch.ones(8,8,device=device,requires_grad=True)
            (probe@probe).sum().backward(); torch.cuda.synchronize(device); del probe
        except Exception as exc:
            raise RuntimeError(f'GPU 实际运算失败（torch={torch.__version__}, CUDA={torch.version.cuda}）。'
                '请检查驱动和当前解释器 GPU wheel；RTX 5090 可在新环境运行安装脚本安装 CUDA 12.8 版本。'
                '本次未修改环境或转用 CPU。') from exc
    env=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
        torch=torch.__version__,torchvision=torchvision.__version__,cuda_build=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),device=str(device),
        device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor() or 'CPU',
        device_capability=torch.cuda.get_device_capability(device) if device.type=='cuda' else None,
        cuda_arch_list=torch.cuda.get_arch_list() if device.type=='cuda' else [],
        seed=cfg.seed,amp=cfg.amp,workers=cfg.workers,pin_memory=cfg.pin_memory,cpu_threads=cfg.cpu_threads,
        deterministic_requested=cfg.deterministic,strict_determinism=cfg.strict_determinism,
        determinism_note='CUDA grid_sample backward may be nondeterministic in the reference operator; strict mode raises instead of hiding this.',
        pythonhashseed=os.environ.get('PYTHONHASHSEED','not fixed at interpreter startup; no claim of hash seed control'),
        cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),packages=packages)
    write_json(output/'logs'/'environment.json',env)
    write_csv(output/'logs'/'package_versions.csv',packages,packages[0].keys())
    print(f"[设备] {device} | {env['device_name']} | AMP={cfg.amp}",flush=True)
    print(f"[复现设置] seed={cfg.seed} | deterministic={cfg.deterministic} | strict={cfg.strict_determinism} | workers={cfg.workers}",flush=True)
    return device,env


def execute(cfg,output,state):
    import torch
    from torch.utils.data import DataLoader
    from .modeling import validate_config,preflight_weights,build_native,make_optimizer
    from .data import prepare_dataset
    from .loading import make_loaders,make_dataset,seed_worker
    from .recording import Recorder,loss_summary,loss_console
    from .training import train_epoch,infer_loader,print_metrics,complexity,print_complexity,benchmark,draw_predictions
    from .evaluation import evaluate,plot_training
    from .checkpoint import load_resume,restore_rng,rng_state,atomic_save,cpu_state,training_signature
    from util.misc import collate_fn,nested_tensor_from_tensor_list

    state['phase']='initialization'; validate_config(cfg); preflight_weights(cfg)
    device,env=setup_environment(cfg,output)
    state['device']=str(device); state['device_name']=env['device_name']
    state['phase']='dataset_preparation'
    info=prepare_dataset(cfg,output/'inputs'); state['dataset_fingerprint']=info['fingerprint']
    print(f"[数据配置] {output/'inputs'/'dataset_info.json'}",flush=True)
    for split,item in info['splits'].items():
        print(f"[数据划分/{split}] images={item['images_path']} | annotations={item['annotation']} | GT={item['labeled']}",flush=True)
    train_loader,val_loader,generator,worker_generator=make_loaders(cfg,info)
    state['phase']='model_initialization'
    model,criterion,post,args,spec,initialization=build_native(cfg,device,output)
    state['ops_backend']=spec['ops']['actual']; state['initialization']=initialization['mode']
    optimizer,scheduler=make_optimizer(model,args,cfg,output)
    scaler=torch.amp.GradScaler('cuda',enabled=cfg.amp)
    checkpoint=load_resume(cfg,info,spec,output,device)
    completed=0; step=0; best_ap=-1.; best_epoch=0; best_confidence=None; best_state=None
    if checkpoint:
        model.load_state_dict(checkpoint['model'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer']); scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        completed=checkpoint['completed_epoch']; step=checkpoint['global_step']
        best_ap=checkpoint['best_ap']; best_epoch=checkpoint['best_epoch']
        best_confidence=checkpoint['best_confidence']; best_state=checkpoint['best_model']
        parent_initialization=checkpoint.get('initialization')
        if parent_initialization is None:
            parent_file=resolve(cfg.resume_from).parent.parent/'inputs'/'initialization.json'
            if parent_file.is_file(): parent_initialization=read_json(parent_file)
        write_json(output/'inputs'/'resume_initialization.json',dict(
            current=initialization,original_initialization=parent_initialization))
        if parent_initialization is not None: initialization=parent_initialization
        restore_rng(checkpoint['rng'],generator,worker_generator,device)
        state['completed_epochs']=completed
        atomic_save(output/'weights'/'best.pth',dict(format='HDETR_SDS_INFERENCE_V1',model=best_state,
            epoch=best_epoch,ap=best_ap,confidence=best_confidence,names=CLASS_NAMES,
            architecture_fingerprint=spec['architecture_fingerprint'],args=vars(args),initialization=initialization))
    recorder=Recorder(output,cfg,device)
    complexity_path=output/'metrics'/'complexity.json'
    if complexity_path.is_file(): comp=read_json(complexity_path)
    else:
        # Validation's RandomResize([one_size]) still consumes Python RNG. Restore it exactly.
        preserved_rng=rng_state(generator,worker_generator)
        image,_=val_loader.dataset[0]; sample=image.unsqueeze(0).to(device)
        state['phase']='complexity'
        comp=complexity(model,sample,device,output); del sample
        restore_rng(preserved_rng,generator,worker_generator,device)
    state['params']=comp['params']; state['gflops']=comp['gflops']; state['gflops_partial']=comp.get('gflops_partial')
    print(f"[模型结构] R50 | train queries={spec['training_queries']} | inference queries={spec['inference_queries']} | "
          f"classes={len(CLASS_NAMES)} | Params={comp['params']}",flush=True)
    print(f"[预处理] 等比例短边={cfg.image_size} 长边上限={cfg.max_size} | augmentation={cfg.augmentation} | ImageNet Normalize | NestedTensor padding",flush=True)
    print('[损失列] box_loss=两分支主层 GIoU 之和；cls_loss=分类；l1_loss=L1。辅助层/编码器和两分支所有项另存 losses.csv。',flush=True)
    print('[验证损失] 沿用原算法的一对一 + aux + encoder；不计算/伪造一对多验证损失。',flush=True)
    print(f"[保存位置] 权重={output/'weights'} | 日志={output/'logs'} | CSV={output/'metrics'} | 曲线={output/'results'}",flush=True)

    for epoch in range(completed+1,cfg.epochs+1):
        epoch_started=now(); epoch_start=time.perf_counter()
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        state['phase']='training'; t=time.perf_counter()
        train_means,step=train_epoch(model,criterion,optimizer,scaler,train_loader,recorder,cfg,device,epoch,step)
        train_seconds=time.perf_counter()-t
        state['phase']='validation'; t=time.perf_counter()
        predictions,val_means,inference_ms=infer_loader(model,criterion,post,val_loader,recorder,cfg,device,epoch,step,'val',True)
        val_seconds=time.perf_counter()-t
        state['phase']='epoch_evaluation'; t=time.perf_counter()
        headline,classes,images=evaluate(info['splits']['val']['annotation'],predictions,
                                       output/'metrics'/f'epoch_{epoch:04d}',make_plots=cfg.make_plots)
        evaluation_seconds=time.perf_counter()-t
        recorder.evaluation(epoch,'epoch','val',headline,classes,images)
        print_metrics(headline)
        if headline['ap'] is None: raise RuntimeError('验证集 AP 不可定义，不能选择最佳模型。请核查非 crowd GT。')
        improved=headline['ap']>best_ap
        if improved:
            best_ap=headline['ap']; best_epoch=epoch; best_confidence=headline['confidence']; best_state=cpu_state(model)
        lrs=[float(group['lr']) for group in optimizer.param_groups]
        resource=resource_snapshot(device)
        if resource['peak_allocated_gb'] is not None:
            state['peak_allocated_gb']=max(state.get('peak_allocated_gb') or 0.,resource['peak_allocated_gb'])
        end_time=now(); elapsed=time.perf_counter()-epoch_start
        row=dict(epoch=epoch,start_time=epoch_started,end_time=end_time,
            **loss_summary(train_means,'train'),**loss_summary(val_means,'val'),**headline,
            **{f'lr/pg{i}':lr for i,lr in enumerate(lrs)},lr_min=min(lrs),lr_max=max(lrs),
            train_seconds=train_seconds,val_seconds=val_seconds,evaluation_seconds=evaluation_seconds,epoch_seconds=elapsed,
            inference_ms=inference_ms,fps=1000/inference_ms if inference_ms else None,
            peak_allocated_gb=resource['peak_allocated_gb'],params=comp['params'],gflops=comp['gflops'],
            gflops_partial=comp.get('gflops_partial'),best_epoch=best_epoch,is_best=improved,
            ops_backend=spec['ops']['actual'],input_height=comp['input_shape'][-2],input_width=comp['input_shape'][-1])
        recorder.results.append(row)
        recorder.timing.append(dict(epoch=epoch,start_time=epoch_started,end_time=end_time,train_seconds=train_seconds,
            val_seconds=val_seconds,evaluation_seconds=evaluation_seconds,epoch_seconds=elapsed,
            train_batches=len(train_loader),val_batches=len(val_loader)))
        # Keep the native JSON-per-epoch log.txt convention alongside the complete CSV records.
        with (output/'log.txt').open('a',encoding='utf-8') as file:
            file.write(json.dumps(dict(epoch=epoch,train=train_means,val=val_means,metrics=headline,lr=lrs),ensure_ascii=False)+'\n')
            file.flush(); os.fsync(file.fileno())
        if cfg.make_plots: plot_training(output)
        scheduler.step()  # Save next-epoch LR/scheduler state exactly; no resume LR override.
        state['phase']='checkpoint'
        full=dict(format='HDETR_SDS_FULL_V1',model=model.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),completed_epoch=epoch,global_step=step,
            best_ap=best_ap,best_epoch=best_epoch,best_confidence=best_confidence,best_model=best_state,
            rng=rng_state(generator,worker_generator),dataset_fingerprint=info['fingerprint'],
            architecture_fingerprint=spec['architecture_fingerprint'],training_signature=training_signature(cfg),
            ops_backend=spec['ops']['actual'],device_type=device.type,names=CLASS_NAMES,config=asdict(cfg),args=vars(args),
            initialization=initialization)
        targets=[('last',output/'weights'/'last.pth',full)]
        if improved:
            targets.insert(0,('best',output/'weights'/'best.pth',dict(format='HDETR_SDS_INFERENCE_V1',model=best_state,
                epoch=best_epoch,ap=best_ap,confidence=best_confidence,names=CLASS_NAMES,
                architecture_fingerprint=spec['architecture_fingerprint'],args=vars(args),initialization=initialization)))
        if epoch%cfg.checkpoint_interval==0 or epoch==cfg.epochs:
            targets.append(('periodic',output/'weights'/f'epoch_{epoch:04d}.pth',full))
        for kind,path,payload in targets:
            t=time.perf_counter(); atomic_save(path,payload); seconds=time.perf_counter()-t
            recorder.checkpoints.append(dict(epoch=epoch,kind=kind,path=str(path),seconds=seconds,bytes=path.stat().st_size))
        # Include serialization/plots in actual epoch wall time; atomically update, never duplicate.
        row['end_time']=now(); row['epoch_seconds']=time.perf_counter()-epoch_start
        recorder.results.update(row)
        recorder.timing.update(dict(epoch=epoch,start_time=epoch_started,end_time=row['end_time'],train_seconds=train_seconds,
            val_seconds=val_seconds,evaluation_seconds=evaluation_seconds,epoch_seconds=row['epoch_seconds'],
            train_batches=len(train_loader),val_batches=len(val_loader)))
        state.update(completed_epochs=epoch,best_epoch=best_epoch,best_ap=best_ap,best_confidence=best_confidence,
                     last_epoch_metrics=headline,global_step=step)
        write_json(output/'run_state.json',state)
        print_complexity(comp,inference_ms,resource['peak_allocated_gb'])
        metrics_text=' '.join(f"{label}={headline[key]:.5f}" for label,key in
            [('P','precision'),('R','recall'),('AP50','ap50'),('AP','ap'),('APsmall','apsmall')] if headline[key] is not None)
        print(f"[H-DETR-R50] Epoch {epoch}/{cfg.epochs} 完成 | Box({metrics_text}) | "
              f"lr={dict((f'lr/pg{i}',lr) for i,lr in enumerate(lrs))} | "
              f"loss={dict(**loss_console(train_means,'train'),**loss_console(val_means,'val'))}",flush=True)
        print(f"[耗时] train={train_seconds:.2f}s val={val_seconds:.2f}s evaluation={evaluation_seconds:.2f}s | "
              f"epoch含保存={time.perf_counter()-epoch_start:.2f}s | 结束={now()}",flush=True)
        if improved:
            print(f"[最佳模型更新] Epoch={best_epoch} AP={best_ap:.6f} | {output/'weights'/'best.pth'}",flush=True)
        print(f"[已保存] last={output/'weights'/'last.pth'} | CSV={output/'results.csv'}",flush=True)
        del full,targets,predictions

    state['phase']='final_evaluation'
    model.load_state_dict(best_state,strict=True)
    print(f"\n[最终评价] 使用验证集选出的 best.pth | Epoch={best_epoch} | 固定 confidence={best_confidence:.3f}",flush=True)
    final={}
    predictions,_,val_ms=infer_loader(model,criterion,post,val_loader,recorder,cfg,device,best_epoch,step,'final_val',True)
    val_metrics,classes,images=evaluate(info['splits']['val']['annotation'],predictions,
        output/'metrics'/'final'/'val',confidence=best_confidence,make_plots=cfg.make_plots)
    recorder.evaluation(best_epoch,'final','val',val_metrics,classes,images); final['val']=val_metrics
    print_metrics(val_metrics,classes)
    draw_predictions(info['splits']['val']['annotation'],predictions,info['splits']['val']['images_path'],
                     output/'results'/'val_predictions',best_confidence,cfg.prediction_images)
    test=info['splits']['test']
    if test['images']:
        test_dataset=make_dataset(cfg,'test',info)
        test_loader=DataLoader(test_dataset,batch_size=cfg.val_batch_size,shuffle=False,num_workers=cfg.workers,
            collate_fn=collate_fn,pin_memory=cfg.pin_memory,worker_init_fn=seed_worker,generator=worker_generator)
        phase='test' if test['labeled'] else 'test_unlabeled'
        predictions,_,_=infer_loader(model,criterion,post,test_loader,recorder,cfg,device,best_epoch,step,phase,test['labeled'])
        folder=output/'metrics'/'final'/phase; folder.mkdir(parents=True,exist_ok=True)
        if test['labeled']:
            metrics,classes,images=evaluate(test['annotation'],predictions,folder,confidence=best_confidence,make_plots=cfg.make_plots)
            recorder.evaluation(best_epoch,'final','test',metrics,classes,images); final['test']=metrics
            state['test_status']='evaluated_with_GT'; print_metrics(metrics,classes)
        else:
            write_json(folder/'predictions.json',predictions)
            write_csv(folder/'predictions.csv',[dict(image_id=p['image_id'],class_id=p['category_id'],score=p['score'],
                x=p['bbox'][0],y=p['bbox'][1],width=p['bbox'][2],height=p['bbox'][3]) for p in predictions],
                ['image_id','class_id','score','x','y','width','height'])
            write_json(folder/'status.json',dict(status='predictions_only',images=test['images'],GT=False,
                metrics=None,names=CLASS_NAMES,category_ids=f'internal contiguous IDs 0..{len(CLASS_NAMES)-1}, not an official submission conversion'))
            state['test_status']='predictions_only_no_GT'
            print(f"[测试集] {test['images']} 张图无 GT；仅保存预测，最终指标来自 val。",flush=True)
        draw_predictions(test['annotation'],predictions,test['images_path'],output/'results'/'test_predictions',best_confidence,cfg.prediction_images)
    else: state['test_status']='no_test_images'
    state['phase']='benchmark'
    speed=benchmark(model,post,val_loader.dataset,cfg,device,output)
    state.update(final_metrics=final,metric_split='test' if 'test' in final else 'val',
        inference_ms=speed.get('inference_ms_mean',val_ms),fps=speed.get('fps',1000/val_ms if val_ms else None),
        peak_allocated_gb=max(state.get('peak_allocated_gb') or 0.,resource_snapshot(device)['peak_allocated_gb'] or 0.)
                          if device.type=='cuda' else None)
    write_json(output/'metrics'/'final'/'summary.json',dict(best_epoch=best_epoch,best_confidence=best_confidence,
        metric_split=state['metric_split'],test_status=state['test_status'],metrics=final,inference_speed=speed))
    print(f"[训练完成] 已完成 {state['completed_epochs']}/{cfg.epochs} Epoch | best Epoch={best_epoch} | "
          f"最终指标划分={state['metric_split']} | APsmall={final[state['metric_split']]['apsmall']}",flush=True)
    result=final[state['metric_split']]
    core=' '.join(f"{name}={result[key]:.6f}" if result[key] is not None else f"{name}=N/A"
        for name,key in [('P','precision'),('R','recall'),('F1','f1'),('AP50','ap50'),('AP','ap'),('APsmall','apsmall')])
    print(f"[最终核心指标/{state['metric_split']}] {core}",flush=True)
