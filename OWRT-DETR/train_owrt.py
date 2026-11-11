# -*- coding: utf-8 -*-
"""
OWRT-DETR 训练入口(在 PyCharm 中直接运行本文件)。

首次运行会直接使用 D:/dataset 的 YOLO 数据集 (images/labels 已划分 train/val/test,
单类 target), 无需再从 TinyWOD 源重建。

运行前:
1. 选择任一 Python 3.10–3.12 解释器(建议独立 venv)
2. 工作目录设置为项目根目录(本文件所在目录)
"""
import gc
import os
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 环境变量必须在 import torch / ultralytics 之前设置。
# 注意: 本文件顶层只允许轻量 import。Windows 下 DataLoader 用 spawn 启动 worker,
# 每个 worker 都会重新 import 本文件, 顶层如果有 torch/ultralytics 重导入, 会
# 导致每个子进程都加载一遍完整依赖, 多个 worker 同时加载即耗尽内存 (MemoryError)。
# ---------------------------------------------------------------------------
# 1) PyTorch 2.9 起使用新名称 PYTORCH_ALLOC_CONF (旧 PYTORCH_CUDA_ALLOC_CONF 已
#    deprecated); expandable_segments 修复 RTX 50 系(Blackwell) 在 deepcopy(EMA)
#    等场景下的 CUDA 碎片化 OOM。
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
# 2) Windows 上关闭 pin_memory, 减少 page-locked 内存占用 (训练是计算瓶颈,
#    关闭后数据搬运速度几乎无影响)。
os.environ.setdefault('PIN_MEMORY', 'false')

# ------------------------- 训练配置 -------------------------
MODEL = 'ultralytics/cfg/models/OWRT/OWRT.yaml'
DATA = 'owrt_data.yaml'
EPOCHS = 100
IMGSZ = 640
BATCH = 8
DEVICE = 0        # GPU 编号; 无 GPU 改为 'cpu'
WORKERS = 4       # Windows spawn 稳定优先: 每个 worker 都要重新加载 torch,
                  # 过大(如 8)会多进程同时吃满内存导致 MemoryError
SEED = 0
AMP = True        # 混合精度 (降低显存占用)
DETERMINISTIC = False  # RT-DETR 的 grid_sample 不支持 deterministic=True
DELAY_HOURS = 0        # 延迟启动小时数; 改成 0 = 立即开始训练
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 运行稳定性辅助: 僵尸进程清理 + 每 50 个 batch 定期清理 (不改算法功能)
# ---------------------------------------------------------------------------
_cleanup_state = {'batches': 0}


def _cleanup_zombie_workers(quiet=False):
    """清理历史崩溃遗留的 DataLoader spawn worker 僵尸进程。

    Windows 下训练崩溃/中断后, DataLoader 的 spawn worker 子进程不会被系统回收,
    会持续占用系统 commit 配额, 配额耗尽后 cudaMalloc 失败并被 PyTorch 包装成
    假的 "CUDA out of memory"。只杀「命令行含 spawn_main 且父进程已不存在」的孤儿,
    绝不影响当前进程与任何仍在运行的训练。返回被杀掉的 PID 列表。
    """
    import psutil

    me = os.getpid()
    killed = []
    try:
        for p in psutil.process_iter(['pid', 'ppid', 'name', 'cmdline']):
            try:
                pid, ppid = p.info['pid'], p.info['ppid']
                name = (p.info['name'] or '').lower()
                cmd = ' '.join(p.info['cmdline'] or [])
                if pid == me or ('python' not in name and 'python' not in cmd.lower()):
                    continue
                if 'spawn_main' not in cmd:
                    continue
                if not ppid or psutil.pid_exists(ppid):
                    continue  # 父进程存活 => 属于活训练, 不动
                p.kill()
                killed.append(pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except Exception:
        pass
    if not quiet:
        if killed:
            print(f'[train] 僵尸进程清理: 已结束 {len(killed)} 个孤儿 worker (PID: {sorted(killed)})', flush=True)
        else:
            print('[train] 僵尸进程清理: 未发现需要清理的僵尸进程', flush=True)
    return killed


def _periodic_cleanup():
    """定期轻量清理: gc + 释放空闲显存块 + 清扫孤儿 worker。静默执行, 不打印。"""
    try:
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _cleanup_zombie_workers(quiet=True)
    except Exception:
        pass


def _on_train_batch_end(trainer=None):
    """训练每个 batch 结束回调: 每 50 个 batch 触发一次定期清理。"""
    _cleanup_state['batches'] += 1
    if _cleanup_state['batches'] % 50 == 0:
        _periodic_cleanup()


def _next_run_name():
    """返回下一个编号的结果目录名: 1run, 2run, ... (扫描 runs/ 下已存在的 *run 目录)。"""
    runs_dir = PROJECT_ROOT / 'runs'
    max_n = 0
    try:
        if runs_dir.is_dir():
            for d in runs_dir.iterdir():
                if d.is_dir() and d.name.endswith('run'):
                    prefix = d.name[:-3]  # 去掉 'run'
                    if prefix.isdigit():
                        max_n = max(max_n, int(prefix))
    except OSError:
        pass
    return f'{max_n + 1}run'


def main():
    """训练主流程。仅主进程执行, DataLoader 的 spawn worker 不会走到这里。"""
    import multiprocessing
    multiprocessing.freeze_support()  # 普通 Python 脚本为 no-op, 仅冻结可执行文件需要

    if DELAY_HOURS > 0:
        print(f'[train] 延迟启动: 等待 {DELAY_HOURS} 小时后开始训练 ...', flush=True)
        time.sleep(DELAY_HOURS * 3600)

    # 重导入全部放到 main() 内: 主进程独占模型/数据集构建, 避免 worker 重复初始化
    import torch
    from ultralytics import RTDETR

    t_start = time.time()
    print('=' * 60)
    print('[train] OWRT-DETR 训练启动')
    print(f'[train] 模型配置: {MODEL}')
    print(f'[train] 数据配置: {DATA}  (数据源 D:/dataset)')
    print(f'[train] epochs={EPOCHS} imgsz={IMGSZ} batch={BATCH} device={DEVICE} seed={SEED} workers={WORKERS}')
    print('=' * 60)

    # 运行前清理历史崩溃遗留的僵尸 worker (仅主进程执行)
    _cleanup_zombie_workers()

    # 数据集已由 D:/dataset 直接提供 (YOLO 格式, 单类 target), 无需再从 TinyWOD 源重建。
    torch.manual_seed(SEED)
    model = RTDETR(MODEL)
    model.add_callback('on_train_batch_end', _on_train_batch_end)  # 每 50 个 batch 定期清理一次
    print(f'[train] 模型构建完成, 参数 {sum(p.numel() for p in model.model.parameters()) / 1e6:.1f}M')
    run_name = _next_run_name()  # 每次运行递增编号, 结果互不覆盖 (1run, 2run, ...)
    print(f'[train] 开始训练(结果目录 runs/{run_name}) ...')
    model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        seed=SEED,
        amp=AMP,
        deterministic=DETERMINISTIC,
        pretrained=False,   # OWRT.yaml 无官方权重, 从随机初始化训练
        exist_ok=True,
        project='runs',     # 直接存到 runs/ 下, 对齐新方法的目录结构
        name=run_name,      # 每次运行写到独立编号目录
    )
    print(f'[train] 训练结束, 总用时 {time.time()-t_start:.0f}s, 结果目录 {model.trainer.save_dir}')


if __name__ == '__main__':
    main()
