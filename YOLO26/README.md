# YOLO26：TinyWOD S / X 对比实验

**正式训练每个模型200 Epoch；`run_all.py` 按 YOLO26-S → YOLO26-X 顺序执行。** 三个入口都能在 PyCharm 直接点击 Run，训练不需要命令行参数。

本次上传的 `yolo26-main.zip` 实际是官方说明页，没有网络源码；工程按它指向的官方实现补齐 **Ultralytics 8.4.182 完整运行源码**。你改写的 RAR 只有19个根文件，缺少被入口引用的 `tinywod/`，无法逐行核对未上传的打印/CSV代码。本次保留可见配置、入口调用、延迟功能和 README 规定的主要记录文件，补齐实际运行实现，不声称看过缺失实现。

附件不含真实 TinyWOD 数据、预训练权重或真实实验结果。已做的检查和合成测试不可用于论文。

## 1. PyCharm 直接运行

1. 解压，使用 PyCharm **Open** 打开整个 `YOLO26_TinyWOD` 文件夹。
2. 设置独立 **Python 3.10–3.12** 解释器，推荐3.11；先安装本项目 `requirements.txt` 的依赖，不要安装其他 ultralytics 覆盖本地源码。
3. 只打开 **`config.py`** 核对数据路径。路径在你的机器上仍有效时不需要修改。
4. 右键目标文件 → **Run**，或点击绿色三角。Script parameters 留空；不需要填写模型名、Epoch、数据YAML或结果目录。路径相对本工程，不依赖Working directory。

| 任务 | 运行文件 |
| --- | --- |
| 只训练 YOLO26-S | `train_yolo26_s.py` |
| 只训练 YOLO26-X | `train_yolo26_x.py` |
| 连续训练 S → X | **`run_all.py`** |
| 检查语法、真实网络、当前数据；不训练 | `check_project.py` |
| 合成数据CPU、128px、各1 Epoch功能测试 | `smoke_test.py` |
| 合成数据S，中断/恢复至3轮并评价有标签test | `smoke_resume_test.py` |

正式入口强制200 Epoch、关闭提前停止、不设置时限。合成测试输出到 `verification/`，不会改正式配置或覆盖 `runs/`。

上传入口的延迟功能集中到 `startup_delay_hours={"s":0,"x":0}`，默认立即运行。原S实际等待4小时、X实际2小时（旧注释写10小时）；需要时只在此配置恢复，保留“延迟启动/延迟结束”打印用语。

### 环境安装（一次即可）

可用 PyCharm Python Packages/requirements 界面安装。以下仅为等效的首次配置方式，训练不需终端：

```bash
# NVIDIA示例，按显卡/驱动选择官方wheel；RTX50系列可使用CUDA12.8 wheel
python -m pip install numpy==1.26.4 Pillow==11.1.0
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

CPU将第一条 `cu128` 换成 `cpu`。本工程实测Linux Python3.12、PyTorch2.7.1 CPU，未实测Windows GPU。根入口优先导入本地源码，不要求可编辑安装；如要在其他工程导入本分支，可安装依赖后 `python -m pip install -e . --no-deps`。不要 `pip install -U ultralytics`。

支持单张CUDA GPU或CPU；`device="auto"` 优先CUDA:0，否则CPU，显式指定不可用GPU会报错。两个模型各自独立子进程，前一个结束后再启动下一个。无需Gradio、W&B、ONNX Runtime；数据和训练默认离线，不上传日志。

## 2. 数据路径只改 config.py

以下默认值沿用你本次上传的config.py，当前环境没有这些Windows目录，无法核验你机器上的实际数据：

| 配置 | 默认值 |
| --- | --- |
| `dataset_mode` | `auto`，已有YOLO目录存在则核验并复用，否则转换COCO |
| `existing_yolo_root` | `D:/dataset` |
| `raw_dataset_root` | `D:/dataset` |
| `prepared_dataset_root` | 本工程 `datasets/tinywod` |
| `output_root` | 本工程 `runs` |

已有YOLO数据设 `dataset_mode="yolo"` 和 `existing_yolo_root`；原始COCO设 `"coco"` 和 `raw_dataset_root`。auto发现已有目录不完整会报错，不静默更换数据。`data/tinywod.yaml` 仅为阅读模板，真实YAML按config自动生成，不需要重复改路径。

COCO目录：`images/train/`、`images/val/`、可选 `images/test/`，以及 `annotations/instances_train.json`、`instances_val.json`、可选有真实GT的 `instances_test.json`。JSON file_name相对于对应images/split。

已有YOLO目录：`images/train|val|test/` 和 `labels/train|val|test/`；标签每行 **`cls cx cy w h`**，原图归一化0–1。每张已标注背景图也须有空txt，不把缺标签当背景。无标签test只保留图片。

| YOLO cls | 类别（沿用上传配置） | 常用COCO id |
| --- | --- | --- |
| 0 | target | 1 |
| 1 | target | 2 |
| 2 | target | 3 |
| 3 | target | 4 |
| 4 | target | 5 |

COCO按categories名称核对映射，未知非背景类别拒绝；category0仅在不属于目标类时按背景跳过并记录。保留crowd，越界框按边界裁剪，无交集框剔除，数量留档。不重新划分、不做背景过滤、不使用OWT网络。

已有YOLO语义由manifest.json或数据YAML的 `names/class_names/classes` 顺序核验。没有可识别声明时，须人工确认cls0–4与上表一致，再设 `accept_unverified_yolo_names=True`，确认留档，不能凭数字猜类别。

检查缺图、部分缺标签、未知/越界类别、NaN/Inf、非法/重复/越界框、尺寸不符、同名标签覆盖、跨split同一物理图像；整个train/val无目标拒绝。默认图像指纹使用路径/大小/mtime，标签始终SHA256；`hash_images=True`还检查复制后跨split字节重复。跨算法严格比较应复用同一份已核验标签；本地AP协议不等同于官方服务器可能涉及ignore-region的评分。

每次在 `runs/comparisons/<编号>/inputs/dataset_view/` 建独立视图，图像优先硬链接、跨盘复制，标签独立复制，S/X共用同一视图/指纹，不继承旧缓存。请保留视图以便复核/恢复，跨盘准备大数据需要对应磁盘空间。

### 独立测试集

`test_policy="auto"`：有完整test标签则训练结束、按val选定best后才测试，汇总用test；无test标签则最终指标明确用val，test只保存预测，不伪造test指标。`"required"`强制完整test，`"unlabeled"`显式只预测。部分标签报错。test不参与训练或选择best。

## 3. 两种真实网络和初始化

调用官方 `YOLO`、`DetectionModel`、`Detect` 和YOLO26训练损失，不调用YOLOv10/OWT网络。

| 实验名 | 官方名 | 实际配置 | depth / width / max_channels |
| --- | --- | --- | --- |
| YOLO26-S | yolo26s | `models/yolo26s.yaml` | 0.50 / 0.50 / 1024 |
| YOLO26-X | yolo26x | `models/yolo26x.yaml` | 1.00 / 1.50 / 512 |

两份配置从同版 `ultralytics/cfg/models/26/yolo26.yaml`生成，物理存在、显式scale不同，实际重复数/通道/参数形状不同。运行保存模块清单和结构指纹，不靠改名证明模型。YAML nc80是官方COCO定义，训练自动按数据建立 **单类(target)检测头**；实际参数量见验证报告和 `inputs/trained_structure.json`。

YOLO26 reg_max=1，公开损失分量为 **box_loss/cls_loss/l1_loss**，没有DFL；保留dfl字段是上游兼容配置，不能当作计算了DFL。CSV按上游公开的one-to-one分量记录，实际保留双分支加权优化和ProgLoss/STAL，不套用旧YOLOv10六项损失。显式 `nms=False`，逐轮验证/最终评价/预测/测速统一one-to-one NMS-free，区别于当前上游默认的外部NMS流程。

默认 `initialization="scratch"` 沿用上传配置的随机初始化，不需要权重。预训练比较只在config改 `"pretrained"`，准备 `weights/yolo26s.pt`、`weights/yolo26x.pt`，或显式开启 `download_pretrained=True`：

- https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt
- https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt

训练前两模型都先构建和检查，核对真实模块/形状，拒绝重命名权重冒充；缺失/下载失败不回退随机。来源声明、文件SHA256留档，不声称手动文件经过官方签名认证。只加载可信官方/本工程pt。**本次未实际下载验证官方预训练权重，已测试默认scratch通路。**

## 4. 公平性与可复现

| 条件 | S/X共同设置 |
| --- | --- |
| Epoch / seed | **200** / 0；deterministic=True；关闭提前停止 |
| 图像 | 640×640方形letterbox保持比例；train/val/test均rect=False |
| Batch / workers | train/val均8 / 4；CPU实际workers=0留档 |
| 优化器 | **SGD**，lr0=.01，momentum=.937，weight_decay=.0005 |
| LR / 累积 | 原生线性衰减lrf=.01、预热3 Epoch、nbs64和原生累积 |
| 增强 | HSV=.015/.7/.4，translate=.1，scale=.5，fliplr=.5 |
| 其他增强 | mosaic/mixup/copy_paste/flipud=0；禁用可选Albumentations |
| 精度 | CUDA AMP（按配置），CPU关闭；最终评价/测速FP32 |
| 评价 | conf=.001、max_det300、AP IoU=.50:.05:.95、NMS-free |
| best选择 | 固定8.4.182原生val fitness，即 **mAP50-95** |
| 留档 | 配置、环境、代码SHA256、数据指纹、初始化、优化器及实际参数 |

SGD沿用上传配置，是本对比配方；没有静默换成YOLO26可用的MuSGD或auto。OWT原生损失/优化机制不同；本次也未收到完整OWT，不宣称再次核验或完全统一其条件。新版fitness只用mAP50-95，已更正旧README的0.1×mAP50+0.9×mAP50-95定义。

AMP对当前模型做FP32/FP16有限值检查，不下载其他模型；这不是精度等价性保证。NaN/Inf直接失败，不自动修补权重。OOM不会自动减batch/imgsz或记作成功。显存不足先释放其他任务，必要时为**两个模型统一**减batch后重开新组，勿混合旧S与新X。密集目标超过max_det也会报错，须统一提高。跨硬件/版本或多worker预取不保证逐位相同，论文应固定环境并按需做多种子重复。

## 5. 输出、窗口打印和CSV

每次Run生成独立编号，输出互不覆盖：

```text
runs/
  YOLO26-S/<编号>/
  YOLO26-X/<编号>/
  comparisons/<编号>/summary.csv
  comparisons/<编号>/summary.json
  comparisons/<编号>/console.log
  comparisons/<编号>/inputs/...
  summary.csv
```

每模型目录保存：

| 路径 | 内容 |
| --- | --- |
| `weights/best.pt` | 按val选择的最佳EMA模型，FP32 |
| `weights/last.pt` | 完整断点：训练权重、EMA、优化器、scheduler/scaler、RNG、CSV历史 |
| `weights/epoch010.pt`等 | 每10轮完整断点，config的save_period可统一修改 |
| `logs/console.log` | 完整控制台、进度、指标、时间和异常 |
| `logs/error.log` | 失败时完整traceback，成功无此文件 |
| `results.csv` | 框架原生逐轮记录 |
| `results.png` | 框架原生训练/验证损失和指标组合图 |
| `metrics/epochs.csv`、`epochs.jsonl` | 全部真实训练/验证损失、P/R/mAP、各组LR、F1、耗时；CSV含UTF-8 BOM |
| `metrics/val/metrics.json`、`metrics.csv` | best的val指标、协议、速度 |
| `metrics/val/per_class.csv`、`per_image.csv` | 每类别/每图原生统计，无GT类别AP留空 |
| `metrics/val/ap_and_curves.npz` | 各类AP、PR/P/R/F1原始数组，供论文重绘 |
| `metrics/val/confusion_matrix.csv`、`confusion_matrix_axes.json` | 行预测、列真实、含background；匹配IoU=.45，与AP协议分开 |
| `metrics/test/` | 仅有完整test GT时保存上述同类文件 |
| `metrics/complexity.json` | 实际参数、GFLOPs和范围，统计失败留空说明 |
| `metrics/benchmark_samples.csv` | val固定排序前32张、batch1/FP32/预热5次/每张3次逐样本测速 |
| `metrics/benchmark_summary.csv`、`benchmark.json` | 推理均值/中位数、调用管线耗时；不含读盘解码 |
| `metrics/final_metrics.json`、`final_metrics.csv` | 最终split、指标、复杂度、测速、权重路径 |
| `results/loss_curves.png`、`metric_curves.png`、`learning_rate.png` | 损失/指标/LR曲线 |
| `results/val_best/` | 原生BoxPR/P/R/F1、混淆矩阵、验证图和预测labels |
| `results/test_best/` | 有test GT时同类图表/labels |
| `results/test_predict/` | 无test GT时含置信度txt及predictions.jsonl，可选渲染图 |
| `inputs/` | config.py/json、自动YAML、manifest、环境、代码/结构指纹、初始化、实际参数、optimizer |
| `status.json`、`progress.json` | 状态、实际完成轮数、开始/结束时间和耗时 |

框架原生训练示例、labels.jpg、args.yaml仍保留。相对子目录的预测标签保持路径，避免basename覆盖。

每轮保留原生进度，再打印 `Epoch 当前/总数`、全部实际训练/验证损失、LR、P/R/mAP/F1、用时和CSV位置；完整输出同步日志。失败不会标成功，默认继续后续模型，最终有失败返回非零；`continue_on_failure=False`明确将后续标skipped。

比较以**同一次** `runs/comparisons/<编号>/summary.csv`为准；`runs/summary.csv`是累计便捷索引，并发会话可能重建此索引，独立表不丢失。失败/未计算用空值，不以0冒充。汇总包含状态、split、P/R、mAP50/50-95、两种F1、参数、GFLOPs、推理/管线耗时、指纹、种子和所有输出路径。

`F1_from_mean_PR`=2×均值P×均值R/(均值P+均值R)；`mean_per_class_F1`是原生逐类F1均值，两者不可混称。速度比较需固定设备/输入/预热。参数与GFLOPs按**完整未融合双头训练模型**统计，不能直接与去除one-to-many后的部署单头参数表混用。

### 断点恢复

只在config的 `resume_checkpoints={"s":".../weights/last.pt","x":""}`分别指定。总轮数仍200，不额外加200；必须保留同目录匹配的best.pt，模型、数据指纹和配置一致。恢复另建目录，接续CSV及训练/EMA/optimizer/scheduler/RNG，不覆盖原结果。best.pt仅推理评价；已完成200轮拒绝恢复。

FP32完整断点较大，X每个last/周期断点含两套权重和优化器，正式训练应预留数十GB磁盘或统一调整save_period。真实数据和大权重不在代码ZIP中；合成测试权重不作为预训练权重发放。

## 6. 检查报告与原始参考

`delivery/`（`VALIDATION.md`、`INPUT_AUDIT.md`、`PROVENANCE.json` 等）在本仓库分发前已移除；相关检查说明随实际运行输出。合成功能证据非 TinyWOD 结果，不得用于论文。

实际入口在根目录，官方完整模型/配置在 `ultralytics/`，许可为AGPL-3.0。
