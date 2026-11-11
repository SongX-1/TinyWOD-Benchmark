"""PyCharm 一键 Run：检查当前解释器，仅安装缺失/不符合版本的依赖，然后实际导入验证。"""
from pathlib import Path
import subprocess
import sys


def run(arguments):
    message='当前解释器：实际导入依赖和 H-DETR，检查可用设备' if len(arguments)>1 and arguments[1]=='-c' else ' '.join(map(str,arguments))
    print('[执行] '+message,flush=True)
    subprocess.run(arguments,check=True)


def main():
    from tinywod.dependencies import dependency_status
    print(f'[解释器] {sys.executable}',flush=True)
    if not (3,10) <= sys.version_info[:2] <= (3,12):
        raise RuntimeError('推荐使用 Python 3.10–3.12；本包实际检查环境为 Python 3.12。')
    try: import pip  # noqa: F401
    except ImportError: run([sys.executable,'-m','ensurepip','--upgrade'])
    records=dependency_status()
    for row in records:
        print(f"[{'正常' if row['ok'] else '待安装/修复'}] {row['package']}{row['required_version']} | {row['installed'] or '缺失'}",flush=True)
    torch_rows=[r for r in records if r['package'] in ('torch','torchvision')]
    installed=[r for r in torch_rows if r['installed']]
    if installed:
        if len(installed)!=2 or any(not r['ok'] for r in torch_rows):
            raise RuntimeError('现有 torch/torchvision 不完整或不兼容；为保护共享解释器，不自动替换 GPU 栈。'
                               '请在 PyCharm 新建 Python 3.10–3.12 虚拟环境后 Run 本文件。')
        import re
        t,v=[tuple(map(int,re.match(r"(\d+)\.(\d+)",r["installed"]).groups())) for r in torch_rows]
        if v != (0,t[1]+15):
            raise RuntimeError('torch 与 torchvision 版本不配对；请用新的虚拟环境运行本文件。')
        print('[保留] 已有可用的 torch/torchvision，不升级、不换 CUDA。',flush=True)
    else:
        # CUDA 12.8 wheels support recent NVIDIA cards; no nvcc/cl build is used.
        run([sys.executable,'-m','pip','install','torch==2.7.1','torchvision==0.22.1',
             '--index-url','https://download.pytorch.org/whl/cu128'])
    missing=[r['package']+r['required_version'] for r in records if not r['ok'] and r['package'] not in ('torch','torchvision')]
    if missing: run([sys.executable,'-m','pip','install',*missing])
    # A fresh process avoids stale imports after installation and checks actual binary imports.
    probe="""
import sys
from tinywod.dependencies import check_dependencies
check_dependencies()
import torch,torchvision,numpy,PIL,scipy,pycocotools,tqdm,matplotlib,yaml
from models import build_model
from models.ops.functions.ms_deform_attn_func import MSDA
print('[导入通过]',sys.executable,'torch='+torch.__version__,'torchvision='+torchvision.__version__)
print('[CUDA]',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
if torch.cuda.is_available():
    x=torch.ones(8,8,device='cuda',requires_grad=True)
    (x@x).sum().backward()
    torch.cuda.synchronize()
    print('[GPU 实际运算通过]',torch.cuda.get_device_capability(0),torch.cuda.get_arch_list())
print('[MSDeformAttn]', '已安装原生扩展' if MSDA is not None else '使用项目自带 PyTorch 参考实现，无需编译')
"""
    run([sys.executable,'-c',probe])
    print('[完成] 依赖安装与实际导入通过。将本地骨干权重放入 weights 后，Run train_tinywod.py。',flush=True)


if __name__=='__main__':
    import os
    os.chdir(Path(__file__).resolve().parent)
    try: main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
