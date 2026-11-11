"""PyCharm entry, complete stdout/stderr logs, failure records, isolated run directories."""
import contextlib
from dataclasses import asdict
from datetime import datetime
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import uuid
from config import CONFIG,ROOT,resolve
from .common import CSV,Tee,now,write_csv,write_json

SUMMARY_FIELDS=['run_id','model','status','completed_epochs','requested_epochs','start_time','end_time','seconds','best_epoch','best_ap',
    'metric_split','precision','recall','f1','ap50','ap','apsmall','confidence','params','gflops','gflops_partial','inference_ms','fps',
    'peak_allocated_gb','device','device_name','ops_backend','initialization','dataset_fingerprint','test_status','smoke_test','phase','error','output']


@contextlib.contextmanager
def comparison_lock(path):
    with path.open('a+b') as file:
        if os.name=='nt':
            import msvcrt
            if file.tell()==0:file.write(b'\0');file.flush()
            file.seek(0);msvcrt.locking(file.fileno(),msvcrt.LK_LOCK,1)
            try:yield
            finally:file.seek(0);msvcrt.locking(file.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(file.fileno(),fcntl.LOCK_EX)
            try:yield
            finally:fcntl.flock(file.fileno(),fcntl.LOCK_UN)


def launch(cfg=None):
    cfg=cfg or CONFIG;output_root=resolve(cfg.output_root)
    run_id=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    model=f'VMINet-{cfg.variant.upper()}-FasterRCNN'+('-SMOKE' if cfg.smoke_test else '')
    output=output_root/model/run_id
    for folder in ('inputs','weights','logs','metrics','results'):(output/folder).mkdir(parents=True,exist_ok=False)
    state=dict(run_id=run_id,model=model,status='running',completed_epochs=0,requested_epochs=cfg.epochs,start_time=now(),
        smoke_test=cfg.smoke_test,phase='entry',output=str(output.resolve()))
    started=time.perf_counter();code=1
    write_json(output/'inputs'/'config.json',asdict(cfg));shutil.copy2(ROOT/'config.py',output/'inputs'/'config_source.py')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    with (output/'logs'/'console.log').open('w',encoding='utf-8',buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
            try:
                print(f"[实验] {'合成数据冒烟测试' if cfg.smoke_test else '正式TinyWOD训练'} | {model} | {run_id}",flush=True)
                print(f"[开始时间] {state['start_time']}\n[配置] {ROOT/'config.py'}",flush=True)
                print(f'[数据路径] YOLO={cfg.yolo_root} COCO={cfg.coco_root}',flush=True)
                print(f'[轮数] {cfg.epochs}\n[模型] 原始 VMINet 骨干 + 最后特征多尺度金字塔 + Faster R-CNN 边界框检测',flush=True)
                if cfg.pretrained:print(f"[本地骨干权重] {resolve(cfg.backbone_weights[cfg.variant])}",flush=True)
                if cfg.detector_weights:print(f'[完整检测器权重] {resolve(cfg.detector_weights)}',flush=True)
                if cfg.resume_from:print(f'[断点权重] {resolve(cfg.resume_from)}',flush=True)
                print(f'[输出] {output.resolve()}',flush=True)
                # No dependency detection or package installation is called here.
                from .worker import execute
                execute(cfg,output,state)
                state.update(status='completed',phase='completed');code=0
            except KeyboardInterrupt:
                state.update(status='interrupted',error='KeyboardInterrupt',failed_phase=state['phase'])
                (output/'logs'/'error.log').write_text(traceback.format_exc(),encoding='utf-8');traceback.print_exc()
                print('[中断] 已完成 batch CSV 已落盘；使用 last.pth 从最后完整 Epoch 恢复。',flush=True);code=130
            except Exception as exc:
                state.update(status='failed',error=f'{type(exc).__name__}: {exc}',failed_phase=state['phase'])
                (output/'logs'/'error.log').write_text(traceback.format_exc(),encoding='utf-8');traceback.print_exc()
                print(f"[失败] phase={state['phase']} | {output/'logs'/'error.log'}",flush=True)
                if 'out of memory' in str(exc).lower():
                    print('[显存不足] 未改变 batch、图像尺寸、精度或跳过模型。请在 config.py 显式调整，并另开可追溯实验。',flush=True)
            finally:
                state.update(end_time=now(),seconds=time.perf_counter()-started)
                metric=state.get('final_metrics',{}).get(state.get('metric_split','val'),{})
                summary={k:state.get(k) for k in SUMMARY_FIELDS};summary.update({k:metric.get(k) for k in ('precision','recall','f1','ap50','ap','apsmall','confidence')})
                write_json(output/'run_state.json',state);write_json(output/'experiment_summary.json',state)
                write_csv(output/'experiment_summary.csv',[summary],SUMMARY_FIELDS)
                with comparison_lock(output_root/'.comparison.lock'):CSV(output_root/'experiment_comparison.csv',SUMMARY_FIELDS,['run_id']).append(summary)
                print(f"[实验汇总] status={state['status']} completed={state['completed_epochs']}/{cfg.epochs} | start={state['start_time']} end={state['end_time']} elapsed={state['seconds']:.2f}s",flush=True)
                print(f"[本次] {output/'experiment_summary.csv'}\n[对比] {output_root/'experiment_comparison.csv'}",flush=True)
    return code
