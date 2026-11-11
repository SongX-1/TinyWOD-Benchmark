# TinyWOD R50专用配置（12轮）：继承上传的R50四尺度网络。
# epochs、imgsz、数据路径、优化参数统一由experiment_config.py传入。
_base_ = ['DINO_4scale.py']
num_classes = 5
dn_labelbook_size = 5
backbone = 'resnet50'
