"""启动时清理历史崩溃遗留的 DataLoader spawn worker 僵尸进程，并结束本进程残留的子进程。"""
import os


def cleanup_zombie_workers(tag="run_all"):
    """结束父进程已死的孤儿 DataLoader spawn worker；父进程存活的活任务绝不误杀。"""
    try:
        import psutil
        me = os.getpid()
        killed, orphan_mains = [], []
        for p in psutil.process_iter(['pid', 'ppid', 'name', 'cmdline']):
            try:
                pid, ppid = p.info['pid'], p.info['ppid']
                name = (p.info['name'] or '').lower()
                cmd = ' '.join(p.info['cmdline'] or [])
                if pid == me or ('python' not in name and 'python' not in cmd.lower()):
                    continue
                is_worker = 'spawn_main' in cmd
                is_main = any(a.lower().endswith('.py') for a in (p.info['cmdline'] or [])) or 'tinywod.worker' in cmd
                if not (is_worker or is_main):
                    continue
                if not ppid or psutil.pid_exists(ppid):
                    continue  # 父进程存活 => 属于活任务，不动
                if is_worker:
                    p.kill()
                    killed.append(pid)
                else:
                    orphan_mains.append(pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        if killed:
            print(f"[{tag}] 僵尸进程清理: 已结束 {len(killed)} 个孤儿 DataLoader worker (PID: {sorted(killed)})", flush=True)
        else:
            print(f"[{tag}] 僵尸进程清理: 未发现需要清理的僵尸进程", flush=True)
        if orphan_mains:
            print(f"[{tag}] 提示: 发现 {len(orphan_mains)} 个残留的孤儿训练主进程 (PID: {sorted(orphan_mains)})，未自动结束", flush=True)
    except ImportError:
        print(f"[{tag}] psutil 未安装；训练依赖检查将给出具体错误。", flush=True)
    except (psutil.Error, OSError) as exc:
        print(f"[{tag}] 当前系统无法枚举进程，未清理任何其他任务: {exc}", flush=True)


def terminate_own_children():
    """结束本进程残留的 DataLoader spawn worker，避免僵尸进程拖住 stdout 管道。"""
    try:
        import psutil
        parent = psutil.Process()
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except (psutil.Error, OSError):
                pass
        _, alive = psutil.wait_procs(children, timeout=5)
        for child in alive:
            try:
                child.kill()
            except (psutil.Error, OSError):
                pass
    except (psutil.Error, OSError):
        pass
