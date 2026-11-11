# 数据说明

数据不随代码分发。唯一运行配置为根目录 config.py，无需修改本目录。

默认 YOLO 根目录 `D:/dataset`；读取 images/{train,val,test} 与 labels/{train,val,test}，不重新划分。训练/验证必须有完整标签，背景图为对应空 TXT。每行五列 class cx cy width height，归一化坐标。类别顺序见 classes.json 与 config.py（单类 target），源 YAML 顺序不一致会报错。

COCO 根目录 `D:/dataset`；auto 优先存在的 YOLO 根目录。特殊命名或自动定位有歧义时，只在 config.py 的 coco_annotations / coco_image_dirs 明确指定各 split。

测试无标签时只预测，不计算虚假 AP。每次实验独立生成标注视图和清单，不改源图像/TXT/JSON；去重审计、几何还原、crowd 和 area 口径见根 README。
