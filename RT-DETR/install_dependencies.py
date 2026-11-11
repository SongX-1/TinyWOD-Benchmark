"""PyCharm 直接 Run：一次检查附加依赖，补缺失/过旧项，保留已有 PyTorch/CUDA。"""
import importlib
import importlib.util
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback
from datetime import datetime

PACKAGES = (
    ("numpy", "numpy>=1.24", "numpy"), ("Pillow", "Pillow>=9.4", "PIL"),
    ("PyYAML", "PyYAML>=6.0", "yaml"), ("scipy", "scipy>=1.10", "scipy.optimize"),
    ("pycocotools", "pycocotools>=2.0.7", "pycocotools.cocoeval"),
    ("matplotlib", "matplotlib>=3.7", "matplotlib.pyplot"), ("tqdm", "tqdm>=4.66", "tqdm"),
    ("psutil", "psutil>=5.9", "psutil"), ("packaging", "packaging>=23.0", "packaging"),
    ("fvcore", "fvcore>=0.1.5.post20221221", "fvcore.nn"),
    ("tensorboard", "tensorboard>=2.14", "tensorboard"),
)


def normalized(value):
    return re.sub(r"[-_.]+", "-", value).lower()


def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def requirement_ok(installed, requirement):
    if installed is None:
        return False
    try:
        from packaging.requirements import Requirement
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement
    return Requirement(requirement).specifier.contains(installed, prereleases=True)


def command(args, path):
    with path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace",
                                   env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise RuntimeError(f"子进程退出码 {code}：{path}")


def verify_imports():
    problems = []
    for distribution, _, module in PACKAGES:
        try:
            importlib.import_module(module)
            print(f"[通过] {distribution} {version(distribution)}", flush=True)
        except Exception:
            problems.append(module+"\n"+traceback.format_exc())
    try:
        import torch
        import torchvision
        from torch.utils.tensorboard import SummaryWriter
        if not requirement_ok(str(torch.__version__), "torch>=2.7") or not requirement_ok(torchvision.__version__, "torchvision>=0.22"):
            raise RuntimeError("本适配需要配套的 torch>=2.7、torchvision>=0.22；本脚本不自动替换现有 CUDA 环境。")
        torchvision.ops.nms(torch.empty(0, 4), torch.empty(0), 0.5)
        print(f"[通过] torch={torch.__version__} torchvision={torchvision.__version__} CUDA={torch.version.cuda}", flush=True)
    except Exception:
        problems.append("torch / torchvision / TensorBoard\n"+traceback.format_exc()+
                        "\n请在当前解释器安装配套的 GPU 版本：https://pytorch.org/get-started/locally/")
    if problems:
        raise RuntimeError("仍未通过的依赖一次列出：\n"+"\n".join(problems))


def main():
    root = Path(__file__).resolve().parent
    output = root/"dependency_install_logs"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    try:
        print(f"[当前解释器] {sys.executable}", flush=True)
        print("[检查] RT-DETR 训练附加依赖；不自动安装/替换 torch、torchvision 或 CUDA。", flush=True)
        if importlib.util.find_spec("pip") is None:
            command([sys.executable, "-m", "ensurepip"], output/"pip.log")
        planned = []
        for name, requirement, _ in PACKAGES:
            value = version(name)
            print(f"[依赖] {name}: {value or '缺失'}", flush=True)
            if not requirement_ok(value, requirement):
                planned.append((name, requirement))
        excluded = {normalized(name) for name, _ in planned}
        installed = {}
        for dist in metadata.distributions():
            name = dist.metadata.get("Name", "")
            if name and normalized(name) not in excluded:
                if re.fullmatch(r"[A-Za-z0-9_.-]+", name) and re.fullmatch(r"[A-Za-z0-9_.+!-]+", dist.version):
                    key = normalized(name)
                    # A venv can legitimately shadow a system-site distribution.
                    # Keep the version resolved by this interpreter's metadata path.
                    installed[key] = version(name)
        constraints = output/"keep_existing_versions.txt"
        constraints.write_text("\n".join(f"{name}=={value}" for name, value in sorted(installed.items()))+"\n", encoding="utf-8")
        if planned:
            command([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
                     "--constraint", str(constraints), *[requirement for _, requirement in planned]], output/"pip.log")
        changed = {name: (value, version(name)) for name, value in installed.items() if version(name) != value}
        if changed:
            raise RuntimeError(f"已有包版本发生非预期变化：{changed}")
        child = "import importlib.util,sys;s=importlib.util.spec_from_file_location('_rtdetr_dep_check',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.verify_imports()"
        command([sys.executable, "-u", "-c", child, str(Path(__file__).resolve())], output/"verify.log")
        (output/"result.json").write_text(json.dumps({"status": "success", "python": sys.executable, "installed_or_updated": planned}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[完成] 依赖实际导入通过。运行 check_environment.py 检查 GPU 与本地文件。日志：{output}", flush=True)
        return 0
    except Exception:
        detail = traceback.format_exc()
        (output/"error.log").write_text(detail, encoding="utf-8")
        (output/"result.json").write_text(json.dumps({"status": "failed", "python": sys.executable}, indent=2), encoding="utf-8")
        print(detail, flush=True)
        print(f"[失败] 未标记安装成功；详见 {output/'error.log'}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
