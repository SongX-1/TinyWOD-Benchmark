"""Stdlib-first entry, complete console/error logs and truthful experiment summaries."""
import contextlib
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import uuid

from config import CONFIG,ROOT,resolve
from .common import CSV,Tee,now,write_csv,write_json

SUMMARY_FIELDS=['run_id','model','status','completed_epochs','requested_epochs','start_time','end_time','seconds',
    'best_epoch','best_ap','metric_split','precision','recall','f1','ap50','ap','apsmall','confidence',
    'params','gflops','gflops_partial','inference_ms','fps','peak_allocated_gb','device','device_name',
    'ops_backend','initialization','dataset_fingerprint','test_status','smoke_test','phase','error','output']


@contextlib.contextmanager
def comparison_lock(path):
    # OS locks are released on crashes; no stale existence-only lock files.
    with path.open('a+b') as file:
        if os.name=='nt':
            import msvcrt
            if file.tell()==0: file.write(b'\0'); file.flush()
            file.seek(0); msvcrt.locking(file.fileno(),msvcrt.LK_LOCK,1)
            try: yield
            finally: file.seek(0); msvcrt.locking(file.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(file.fileno(),fcntl.LOCK_EX)
            try: yield
            finally: fcntl.flock(file.fileno(),fcntl.LOCK_UN)


def launch(cfg=None):
    cfg=cfg or CONFIG
    output_root=resolve(cfg.output_root)
    run_id=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    model='H-DETR-R50-SMOKE' if cfg.smoke_test else 'H-DETR-R50'
    output=output_root/model/run_id
    for name in ('inputs','weights','logs','metrics','results'): (output/name).mkdir(parents=True,exist_ok=False)
    state=dict(run_id=run_id,model=model,status='running',completed_epochs=0,requested_epochs=cfg.epochs,
        start_time=now(),smoke_test=cfg.smoke_test,phase='entry',output=str(output.resolve()))
    started=time.perf_counter(); code=1
    write_json(output/'inputs'/'config.json',asdict(cfg))
    shutil.copy2(ROOT/'config.py',output/'inputs'/'config_source.py')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    # 减少缓存分配器碎片并尽早归还空闲显存，避免本进程长期占用不用的块挤占其它训练进程。
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','garbage_collection_threshold:0.6,max_split_size_mb:128')
    with (output/'logs'/'console.log').open('w',encoding='utf-8',buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
            try:
                print(f"[实验] {'合成数据冒烟测试' if cfg.smoke_test else '正式TinyWOD训练'} | {model} | {run_id}",flush=True)
                print(f"[开始时间] {state['start_time']}",flush=True)
                print(f"[配置] {ROOT/'config.py'}\n[数据路径] YOLO={cfg.yolo_root} COCO={cfg.coco_root}",flush=True)
                print(f"[轮数] {cfg.epochs}\n[模型] R50 two_stage / hybrid_branch / mixed_selection / look_forward_twice",flush=True)
                print(f"[本地骨干权重] {resolve(cfg.backbone_weights)}\n[输出] {output.resolve()}",flush=True)
                if cfg.detector_weights: print(f"[完整检测器权重] {resolve(cfg.detector_weights)}",flush=True)
                if cfg.resume_from: print(f"[断点权重] {resolve(cfg.resume_from)}",flush=True)
                # Check ALL dependencies before importing torch or reading the whole dataset.
                from .dependencies import check_dependencies
                state['phase']='dependency_check'; check_dependencies()
                from .worker import execute
                execute(cfg,output,state)
                state['status']='completed'; state['phase']='completed'; code=0
            except KeyboardInterrupt:
                state.update(status='interrupted',error='KeyboardInterrupt',failed_phase=state['phase'])
                (output/'logs'/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
                traceback.print_exc()
                print('[中断] 已完成 batch 的 CSV 已落盘；使用 last.pth 从最后提交的 Epoch 恢复。',flush=True)
                code=130
            except Exception as exc:
                state.update(status='failed',error=f'{type(exc).__name__}: {exc}',failed_phase=state['phase'])
                (output/'logs'/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
                traceback.print_exc()
                print(f"[失败] phase={state['phase']} | {output/'logs'/'error.log'}",flush=True)
                if 'out of memory' in str(exc).lower():
                    print('[显存不足] 未改变 batch、图像尺寸、精度或查询数。可显式启用 use_checkpoint，或在新实验统一降低 batch_size；所有变更记录在 config.json。',flush=True)
            finally:
                state['end_time']=now(); state['seconds']=time.perf_counter()-started
                headline=state.get('final_metrics',{}).get(state.get('metric_split','val'),{})
                summary={key:state.get(key) for key in SUMMARY_FIELDS}; summary.update({k:headline.get(k) for k in
                    ('precision','recall','f1','ap50','ap','apsmall','confidence')})
                write_json(output/'run_state.json',state)
                write_json(output/'experiment_summary.json',state)
                write_csv(output/'experiment_summary.csv',[summary],SUMMARY_FIELDS)
                with comparison_lock(output_root/'.comparison.lock'):
                    CSV(output_root/'experiment_comparison.csv',SUMMARY_FIELDS,['run_id']).append(summary)
                print(f"[实验汇总] status={state['status']} completed={state['completed_epochs']}/{cfg.epochs} | "
                      f"start={state['start_time']} end={state['end_time']} elapsed={state['seconds']:.2f}s",flush=True)
                print(f"[本次] {output/'experiment_summary.csv'}\n[对比] {output_root/'experiment_comparison.csv'}",flush=True)
    return code
