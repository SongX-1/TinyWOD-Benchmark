# -*- coding: utf-8 -*-
"""保留原始DINO模型/匹配器/损失/优化流程，整合OWT数据与完整实验记录。"""
from dataclasses import asdict
import datetime
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

from experiment_config import CLASS_NAMES
from .data import atomic_json, prepare_dataset
from .recording import ConsoleLog, Recorder, reserve_run, safe_json, trim_for_resume, plot_training


def configure_threads(config):
    n=os.cpu_count() or 1
    threads=max(1,min(8,round(n*(1-config.reserve_fraction))))
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(key,str(threads))
    print(f'[train] 性能调度: CPU {n} 核，torch线程上限 {threads}；workers={config.workers}',flush=True)
    return threads


def build_args(config,device,manifest):
    from argparse import Namespace
    from util.slconfig import SLConfig
    cfg=SLConfig.fromfile(str(config.model_config))._cfg_dict.to_dict()
    reserved={'epochs','batch_size','lr','lr_backbone','weight_decay','lr_drop','clip_max_norm',
              'dataset_file','device','imgsz','amp','seed'}
    if reserved.intersection(config.model_overrides):
        raise ValueError('训练参数请在experiment_config.py对应字段配置，不要放入model_overrides: '
                         +str(sorted(reserved.intersection(config.model_overrides))))
    cfg.update(num_classes=len(CLASS_NAMES),dn_labelbook_size=len(CLASS_NAMES),epochs=config.epochs,
               batch_size=config.batch_size,lr=config.lr,lr_backbone=config.lr_backbone,
               weight_decay=config.weight_decay,lr_drop=config.lr_drop,
               clip_max_norm=config.clip_max_norm,backbone_pretrained=config.backbone_pretrained)
    cfg.update(config.model_overrides)
    # 这两个值必须与数据映射一致，不允许通过自由覆盖制造类别错位。
    if cfg['num_classes']!=len(CLASS_NAMES) or cfg['dn_labelbook_size']<len(CLASS_NAMES):
        raise ValueError(f'当前数据集使用0..{len(CLASS_NAMES)-1}共{len(CLASS_NAMES)}类；num_classes必须为{len(CLASS_NAMES)}，dn_labelbook_size至少为{len(CLASS_NAMES)}。')
    cfg.update(device=str(device),seed=config.seed,dataset_file='tinywod',
               coco_path=str(config.dataset_root),num_workers=config.workers,
               tinywod_images_root=str(Path(config.dataset_root).resolve()/'images'),
               tinywod_annotations_root=str(Path(config.cache_root).resolve()/'annotations'),
               imgsz=config.imgsz,amp=config.amp and device.type=='cuda',
               distributed=False,rank=0,world_size=1,local_rank=0,debug=False,
               save_results=False,save_log=True,fix_size=False,config_file=str(config.model_config))
    if cfg.get('modelname')!='dino' or cfg.get('backbone')!='resnet50':
        raise ValueError('此入口明确训练DINO-R50，禁止用改名替换真实网络结构。')
    if cfg.get('use_ema'):
        raise ValueError('当前对齐入口固定使用普通模型权重，与OWT记录对齐；需要EMA请另建明确实验配置。')
    if cfg.get('onecyclelr') or cfg.get('multi_step_lr'):
        raise ValueError('当前入口使用上传R50默认StepLR，请勿在model_overrides中开启其他调度器。')
    return Namespace(**cfg)


def select_device(config):
    import torch
    value=str(config.device)
    if value=='auto':
        value='cuda:0' if torch.cuda.is_available() else 'cpu'
    if value.isdigit():
        value='cuda:'+value
    device=torch.device(value)
    if device.type=='cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('配置要求CUDA，但当前PyTorch无法使用CUDA；请安装对应CUDA版PyTorch或将device改为auto/cpu。')
        torch.cuda.set_device(device)
    elif device.type!='cpu':
        raise ValueError('本项目已适配CPU/CUDA，device请使用auto、cpu或cuda:0。')
    return device


def seed_worker(worker_id):
    import numpy as np
    import torch
    worker_seed=torch.initial_seed()%(2**32)
    np.random.seed(worker_seed);random.seed(worker_seed)


def build_loaders(args,config):
    import torch
    from datasets import build_dataset
    from util.misc import collate_fn
    train=build_dataset('train',args);val=build_dataset('val',args)
    generator=torch.Generator().manual_seed(config.seed)
    loader_train=torch.utils.data.DataLoader(train,batch_size=config.batch_size,shuffle=True,
        num_workers=config.workers,collate_fn=collate_fn,drop_last=False,pin_memory=False,
        worker_init_fn=seed_worker,generator=generator)
    loader_val=torch.utils.data.DataLoader(val,batch_size=config.val_batch_size,shuffle=False,
        num_workers=config.workers,collate_fn=collate_fn,drop_last=False,pin_memory=False,worker_init_fn=seed_worker)
    return loader_train,loader_val,generator


def save_checkpoint(path,payload):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.pth.tmp')
    torch.save(payload,tmp)
    tmp.replace(path)


def load_trusted_checkpoint(path):
    import torch
    path=Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'找不到本地checkpoint: {path}')
    # 自己的断点含优化器、随机数状态；明确关闭2.6+的weights_only默认限制。
    # 此入口不接收或下载远程checkpoint。
    return torch.load(path,map_location='cpu',weights_only=False)


def validate_config(config):
    for key in ('epochs','imgsz','batch_size','val_batch_size','lr_drop'):
        if int(getattr(config,key))<=0:
            raise ValueError(f'experiment_config.py: {key} 必须大于0')
    if config.imgsz%32:
        raise ValueError('imgsz必须为32的整数倍，以保证DINO四尺度特征与padding一致。')
    if config.workers<0 or not 0<=config.test_confidence<=1:
        raise ValueError('workers必须非负，test_confidence必须在0..1。')
    if config.resume and config.pretrain_model_path:
        raise ValueError('resume与pretrain_model_path不能同时指定。')
    if config.backbone_pretrained:
        raise ValueError('此交付包不自动下载骨干权重。需要预训练请使用可信本地pretrain_model_path，backbone_pretrained保持False。')


def resume_directory(config):
    if config.resume:
        p=Path(config.resume).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f'resume不存在: {p}')
        if p.parent.name=='weights' and (p.parent.parent/'run_config.json').is_file():
            return p.parent.parent
    return reserve_run(config.output_root)


def run_experiment(config):
    validate_config(config)
    run=resume_directory(config)
    recorder=Recorder(run)
    with ConsoleLog(run/'logs'/'console.log'):
        started=time.perf_counter()
        try:
            print('='*78,flush=True)
            print('[train] DINO-R50 TinyWOD，按OWT范例记录；原图输入，不过滤背景',flush=True)
            print(f'[train] 模型配置: {config.model_config}',flush=True)
            print(f'[train] 数据源: {config.dataset_root}',flush=True)
            print(f'[train] epochs={config.epochs} imgsz={config.imgsz} batch={config.batch_size} '
                  f'seed={config.seed} workers={config.workers}',flush=True)
            print(f'[train] 结果目录: {run}',flush=True)
            threads=configure_threads(config)
            import numpy as np
            import torch
            import torchvision
            from engine import train_one_epoch
            from models import build_model
            from util.get_param_dicts import get_param_dict
            from .data import OriginalSpacePostProcess
            from .evaluation import evaluate
            from .profiling import profile_model
            torch.set_num_threads(threads)
            device=select_device(config)
            torch.manual_seed(config.seed);np.random.seed(config.seed);random.seed(config.seed)
            if device.type=='cuda':torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.benchmark=False
            torch.backends.cudnn.deterministic=config.deterministic
            torch.use_deterministic_algorithms(config.deterministic)
            manifest=prepare_dataset(config)
            atomic_json(run/'dataset_manifest.json',manifest)
            args=build_args(config,device,manifest)
            config_dump=safe_json(asdict(config))
            if not (run/'run_config.json').exists():
                atomic_json(run/'run_config.json',config_dump)
            else:
                stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                atomic_json(run/f'resume_config_{stamp}.json',config_dump)
            atomic_json(run/'model_args.json',vars(args))
            atomic_json(run/'environment.json',{'python':sys.version,'platform':sys.platform,
                'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__,
                'device':str(device),'cuda_runtime':torch.version.cuda,'amp_enabled':args.amp,
                'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                'threads':threads,'workers':config.workers})
            model,criterion,post=build_model(args)
            model.to(device);criterion.to(device)
            if config.backbone_pretrained_path:
                _bp=Path(config.backbone_pretrained_path).expanduser().resolve()
                if not _bp.is_file():
                    raise FileNotFoundError(f'找不到本地ImageNet骨干权重: {_bp}')
                _bstate=torch.load(str(_bp),map_location='cpu')
                if isinstance(_bstate,dict) and 'model' in _bstate: _bstate=_bstate['model']
                _bstate={k.removeprefix('module.'):v for k,v in _bstate.items()}
                _breport=model.backbone[0].body.load_state_dict(_bstate,strict=False)
                print(f'[train] 从本地加载ImageNet骨干权重 {len(_bstate)-len(_breport.unexpected_keys)} 项（fc等忽略）',flush=True)
            postprocess=OriginalSpacePostProcess(post['bbox'],config.imgsz)
            from models.dino.ops.functions.ms_deform_attn_func import MSDA
            print(f"[train] 网络: resnet50 / {args.num_feature_levels}尺度 / encoder={args.enc_layers} "
                  f"decoder={args.dec_layers} / queries={args.num_queries} / classes={len(CLASS_NAMES)}",flush=True)
            print(f"[train] device={device} AMP={args.amp}；可变形注意力="
                  f"{'CUDA扩展' if MSDA is not None and device.type=='cuda' else 'PyTorch等价实现（无需编译，较慢）'}",flush=True)
            optimizer=torch.optim.AdamW(get_param_dict(args,model),lr=args.lr,weight_decay=args.weight_decay)
            scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=args.lr_drop)
            scaler=torch.amp.GradScaler('cuda',enabled=args.amp)
            train_loader,val_loader,generator=build_loaders(args,config)
            start_epoch=0;best_fitness=-float('inf');best_epoch=0;elapsed_before=0.;best_metrics={};last_metrics={}
            checkpoint=None
            if config.resume:
                checkpoint=load_trusted_checkpoint(config.resume)
                if checkpoint.get('format')!='dino-sds-owt-v1':  # 保持旧值以兼容既有检查点
                    raise ValueError('resume需要本项目生成的完整last.pth；原版DINO权重请填pretrain_model_path。')
                old=checkpoint['run_config']
                for key in ('imgsz','batch_size','seed','model_overrides','lr_drop','lr','lr_backbone','weight_decay'):
                    if old.get(key)!=config_dump.get(key):
                        raise ValueError(f'断点恢复实验条件不一致: {key}；需要改变条件请开启新实验。')
                if checkpoint['source_hashes']!={s:manifest['splits'][s]['source_sha256'] for s in ('train','val')}:
                    raise ValueError('训练/验证标注与checkpoint不同，不允许混合恢复。')
                model.load_state_dict(checkpoint['model'],strict=True)
                optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['lr_scheduler'])
                scaler.load_state_dict(checkpoint['scaler'])
                start_epoch=checkpoint['epoch']+1
                best_fitness=checkpoint['best_fitness'];best_epoch=checkpoint['best_epoch']
                best_metrics=checkpoint['best_metrics'];last_metrics=checkpoint['last_metrics']
                elapsed_before=checkpoint['elapsed_seconds']
                torch.set_rng_state(checkpoint['rng']['torch'])
                np.random.set_state(checkpoint['rng']['numpy']);random.setstate(checkpoint['rng']['python'])
                generator.set_state(checkpoint['rng']['loader'])
                if device.type=='cuda' and checkpoint['rng']['cuda'] is not None:
                    torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])
                trim_for_resume(run,start_epoch)
                print(f'[resume] 已完成{start_epoch}轮，从第{start_epoch+1}轮恢复至总共{config.epochs}轮；最佳轮={best_epoch}',flush=True)
                if start_epoch>config.epochs:
                    raise ValueError('checkpoint已完成的轮数大于配置的总epochs。')
            elif config.pretrain_model_path:
                weight=load_trusted_checkpoint(config.pretrain_model_path)
                state=weight.get('model',weight)
                own=model.state_dict()
                matching={k.removeprefix('module.'):v for k,v in state.items()
                          if k.removeprefix('module.') in own and own[k.removeprefix('module.')].shape==v.shape}
                if not matching:
                    raise ValueError('pretrain_model_path没有与DINO-R50匹配的权重。')
                report=model.load_state_dict(matching,strict=False)
                atomic_json(run/'pretrain_loading.json',{'loaded_keys':list(matching),
                    'missing_keys':report.missing_keys,'unexpected_keys':report.unexpected_keys})
                print(f'[train] 从本地预训练加载{len(matching)}项，形状不匹配的分类头保持随机初始化。',flush=True)
                del weight,state,matching
            # checkpoint包含AdamW状态，不在整个训练期间保留第二份CPU副本。
            checkpoint=None
            # Profile可能影响随机数；统计前后恢复，保证是否开启统计不改变训练初始随机序列。
            rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None
            complexity=profile_model(model,config.imgsz,device,config.profile_flops)
            torch.set_rng_state(rng)
            if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
            recorder.csv('metrics/complexity.csv',complexity)
            print(f"[train] 模型参数={complexity['complexity/Parameters']:,}；"
                  f"GFLOPs={complexity['complexity/GFLOPs']:.3f}；"
                  f"dense-upper={complexity['complexity/GFLOPs-dense-upper']:.3f}",flush=True)
            args._recorder=recorder;args._grad_scaler=scaler
            atomic_json(run/'status.json',{'status':'running','start_epoch':start_epoch,'target_epochs':config.epochs})
            for epoch in range(start_epoch,config.epochs):
                ep_start=time.perf_counter()
                if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
                train_stats=train_one_epoch(model,criterion,train_loader,optimizer,device,epoch,
                    max_norm=args.clip_max_norm,args=args,lr_scheduler=scheduler)
                train_peak=torch.cuda.max_memory_allocated(device)/(1<<30) if device.type=='cuda' else 0.
                # 与原DINO保持epoch结束后StepLR；CSV另留本轮实际训练学习率。
                lr_used=[g['lr'] for g in optimizer.param_groups]
                scheduler.step()
                val_stats,metrics=evaluate(model,criterion,postprocess,val_loader,device,epoch+1,config,recorder,complexity)
                metrics['memory/train-PeakAllocated-GB']=train_peak
                for i,value in enumerate(lr_used):metrics[f'lr_used/pg{i}']=value
                fitness=.1*metrics['metrics/mAP50(B)']+.9*metrics['metrics/mAP50-95(B)']
                metrics['fitness']=fitness
                is_best=fitness>best_fitness
                if is_best:best_fitness=fitness;best_epoch=epoch+1;best_metrics=dict(metrics)
                last_metrics=dict(metrics)
                elapsed=elapsed_before+time.perf_counter()-started
                recorder.record_epoch(epoch+1,train_stats,val_stats,metrics,optimizer,
                                      time.perf_counter()-ep_start,elapsed,best_epoch)
                payload={'format':'dino-sds-owt-v1','model':model.state_dict(),'optimizer':optimizer.state_dict(),
                    'lr_scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,
                    'best_fitness':best_fitness,'best_epoch':best_epoch,'best_metrics':best_metrics,
                    'last_metrics':last_metrics,'elapsed_seconds':elapsed,'run_config':config_dump,
                    'args':{k:v for k,v in vars(args).items() if not k.startswith('_')},
                    'source_hashes':{s:manifest['splits'][s]['source_sha256'] for s in ('train','val')},
                    'rng':{'torch':torch.get_rng_state(),'numpy':np.random.get_state(),'python':random.getstate(),
                           'loader':generator.get_state(),
                           'cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else None}}
                save_checkpoint(run/'weights'/'last.pth',payload)
                if is_best:save_checkpoint(run/'weights'/'best.pth',payload)
                if config.save_period>0 and (epoch+1)%config.save_period==0:
                    save_checkpoint(run/'weights'/f'epoch{epoch+1:04d}.pth',payload)
                plot_training(run);recorder.save_schema()
                atomic_json(run/'status.json',{'status':'running','completed_epochs':epoch+1,
                                              'target_epochs':config.epochs,'best_epoch':best_epoch,'best_fitness':best_fitness})
            best_path=run/'weights'/'best.pth'
            if not best_path.is_file():
                raise FileNotFoundError('最佳权重缺失，不能进行最终评估；请从同一实验目录的last.pth恢复。')
            best_checkpoint=load_trusted_checkpoint(best_path)
            model.load_state_dict(best_checkpoint['model'])
            del best_checkpoint,checkpoint
            print(f'[train] 最终复验最佳权重: {best_path}（第{best_epoch}轮）',flush=True)
            final_recorder=Recorder(run/'final_best')
            _,best_metrics=evaluate(model,criterion,postprocess,val_loader,device,best_epoch,config,final_recorder,complexity)
            best_metrics['fitness']=best_fitness
            final_recorder.save_schema()
            if config.predict_test_after_training and manifest['splits']['test']['images']:
                from .inference import predict_test
                predict_test(model,postprocess,device,config,run)
            summary=run/'summary.csv'
            if summary.exists():summary.unlink()
            for variant,ep,data in [('best',best_epoch,best_metrics),('last',config.epochs,last_metrics)]:
                recorder.csv('summary.csv',{'model':'DINO-R50','variant':variant,'epoch':ep,
                    'target_epochs':config.epochs,'status':'completed','dataset_root':str(config.dataset_root),
                    'imgsz':config.imgsz,'batch_size':config.batch_size,'seed':config.seed,
                    'total_seconds':elapsed_before+time.perf_counter()-started,**data})
            from .comparison import build_comparison
            build_comparison(config)
            recorder.save_schema()
            atomic_json(run/'status.json',{'status':'completed','completed_epochs':config.epochs,
                'best_epoch':best_epoch,'best_fitness':best_fitness,'total_seconds':elapsed_before+time.perf_counter()-started})
            print(f'[train] 训练及最佳权重复验完成，结果: {run}',flush=True)
            return run
        except BaseException as exc:
            error=traceback.format_exc()
            (run/'logs'/'error.log').write_text(error,encoding='utf-8')
            atomic_json(run/'status.json',{'status':'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                                          'error':f'{type(exc).__name__}: {exc}'})
            print(error,file=sys.stderr,flush=True)
            recorder.save_schema()
            raise
