"""可选 PyCharm Run：合成数据两轮训练。使用真实完整 Ti 骨干、小输入和较小检测头。"""
from datetime import datetime
from multiprocessing import freeze_support
import uuid

if __name__=='__main__':
    freeze_support()
    from config import ROOT
    from tests.fixtures import create_fixture
    from tinywod.runner import launch
    folder=ROOT/'verification'/'local_smoke'/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    cfg=create_fixture(folder)
    print('[合成测试] 6 张 train、5 张 val、2 张无 GT test；随机初始化；不代表正式数据集实验精度。')
    raise SystemExit(launch(cfg))
