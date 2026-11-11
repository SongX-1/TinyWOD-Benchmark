"""PyCharm Run：语法、依赖、真实网络/算子/数据/评价/记录的离线回归检查。"""
from pathlib import Path
import sys
import unittest
from config import ROOT


def main():
    skipped={'runs','verification','.venv','venv','__pycache__'}
    count=0
    for path in ROOT.rglob('*.py'):
        if set(path.relative_to(ROOT).parts) & skipped: continue
        compile(path.read_text(encoding='utf-8-sig'),str(path),'exec'); count+=1
    print(f'[语法检查] {count} 个 Python 文件通过',flush=True)
    from tinywod.dependencies import check_dependencies
    check_dependencies()
    print(f'[依赖检查] 当前解释器 {sys.executable}',flush=True)
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_*.py',top_level_dir=str(ROOT))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    print('[数据/权重说明] 回归测试用合成数据；正式路径、真实权重在 train_tinywod.py 初始化时完整检查。',flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__': raise SystemExit(main())
