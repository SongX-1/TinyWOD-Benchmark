"""可选 PyCharm Run：合成图片、缩小层数/查询数的两轮真实前后向测试，不访问正式数据/权重。"""
from datetime import datetime
from multiprocessing import freeze_support
from pathlib import Path
import uuid

if __name__=='__main__':
    freeze_support()
    from config import ROOT
    from tests.fixtures import create_fixture
    from tinywod.runner import launch
    folder=ROOT/'verification'/'local_smoke'/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    cfg=create_fixture(folder)
    print('[仅冒烟测试] 6 张训练 / 5 张验证 / 2 张无标注测试图；随机初始化；不代表正式 H-DETR/TinyWOD 精度。')
    raise SystemExit(launch(cfg))
