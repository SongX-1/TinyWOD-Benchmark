"""PyCharm Run：语法与算法/数据/记录回归测试，不包含依赖检测或安装。"""
import unittest
from config import ROOT


def main():
    ignored={'verification','runs','upstream','__pycache__','.venv','venv'};count=0
    for path in ROOT.rglob('*.py'):
        if set(path.relative_to(ROOT).parts)&ignored:continue
        compile(path.read_text(encoding='utf-8-sig'),str(path),'exec');count+=1
    print(f'[语法检查] {count} 个 Python 文件通过',flush=True)
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_*.py',top_level_dir=str(ROOT))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
