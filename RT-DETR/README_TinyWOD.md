# RT-DETR-R50vd · TinyWOD · 72 Epoch

## 直接运行

1. 解压完整 ZIP，在 PyCharm 打开 `RTDETR_TinyWOD72` 目录。
2. 选择你已有的 CUDA Python 解释器(建议 Python 3.10–3.12)。本适配使用 `torch>=2.7`、`torchvision>=0.22` 的配套版本；验证环境为 Python 3.12、torch 2.7.1 / torchvision 0.22.1 CPU。
3. 第一次可直接 Run `install_dependencies.py`。它检查全部附加依赖，只补缺失/过旧项，保留已有 torch/torchvision/CUDA，不需要命令行参数。已安装但导入失败的包会一次列出，不会假报成功。需要修复 torch 配对时，通过 PyCharm 的包管理或 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择适合 GPU 的版本。
4. 核对 `config.py` 中数据根目录。沿用了你之前日志里的路径：

   ```python
   yolo_root = "D:/dataset"
   coco_root = "D:/dataset"
   epochs = 72
   ```

   路径与本机一致就不用修改。`dataset_mode="auto"` 优先使用已存在的 YOLO 根目录；不会重划分数据。所有正式实验参数只在 `config.py` 修改。
5. **直接 Run `train_tinywod.py` 开始正式训练。** `train_rtdetr_r50.py`、`run_all.py` 是同一个 R50vd 实验的清晰别名，任选一个入口即可。本次只有 R50，不额外训练其他型号。

**压缩包已经包含正确的本地 R50vd 骨干权重，不需要训练时联网下载。** 不包含 TinyWOD 图像/标签。正式运行不调用自动安装或自动下载。

可先 Run `check_environment.py`，一次检查依赖实际导入、GPU 基础运算、本地权重、模型配置及数据根目录。完整类别/图像/标签检查在训练入口执行。`device="auto"` 会优先用可用 CUDA 并打印设备；如只允许 GPU，可在 `config.py` 明确设为 `"cuda:0"`。

## 这次使用的原始算法

上传包同时包含 Paddle、PyTorch RT-DETR v1 和 RT-DETRv2。本实验调用 **原版 v1 的 `rtdetr_pytorch`**：

- 原网络：`src/nn/backbone/presnet.py` 的 PResNet，`depth=50, variant=d`，三层 3×3 stem，stage block 数为 3/4/6/3。
- 原 HybridEncoder，stride 8/16/32 的三尺度特征，hidden_dim=256。
- 原 RTDETRTransformer，6 层 decoder、300 queries、100 denoising 配置。
- 类别、后处理、分类头和去噪嵌入统一为单类(target)。
- 原 Hungarian matching / VFL / L1 bbox / GIoU / auxiliary / denoising 损失保持不变。
- 默认原 AdamW 参数分组、骨干学习率 1e-5、其余 1e-4、weight_decay=1e-4、梯度裁剪 0.1、EMA。
- 原配置 `milestones=[1000]`，所以 72 Epoch 内学习率不衰减；没有悄悄改成另一套调度。

已适配旧 torchvision `datapoints`、旧 v2 transform 名称，修复 YAML 的 Windows 绝对路径/UTF-8 读取与共享默认字典问题；RegNet 的 transformers 依赖改为仅在实例化 RegNet 时导入。原始 Paddle/v2/导出/benchmark 文件均保留；它们有各自依赖，不是本次 PyCharm 训练入口。

没有混入 OWT、DQ-DETR 或 YOLO 的网络模块，也没有添加未经验证的小目标增强结构。原版 RT-DETR 可训练单类目标；小目标表现通过真实 APsmall 等指标评价，不承诺未经训练的精度提升。

## 数据输入与公平性

YOLO 目录：`images/train`、`images/val`、`images/test`，相应标注为 `labels/<split>/<image_stem>.txt`。每行 `class cx cy w h`，归一化坐标。

类别顺序固定为（单类）：

| 编号 | 类别 |
|---|---|
| 0 | target |

YOLO 根目录的 manifest/YAML 需声明同一顺序。项目 `data/tinywod.yaml` 提供模板；根目录已有正确声明时无需复制。若没有声明，只有在你确实核对过编号后才可在集中配置设置 `accept_unverified_yolo_names=True`。

COCO 备用格式：`images/<split>` 和 `annotations/instances_<split>.json`。按 category name 映射单类，允许 id=0 的 ignored/ignore/background 类并明确统计忽略数量，保留 crowd 给 COCOeval，训练排除 crowd。不会将 COCO 稀疏 category_id 直接当类别索引。

- train/val 必须完整有标签；背景图使用空 TXT。缺失标签不会被当成背景。
- test 允许整体无标签；部分有标签会报错。COCO test 文件只有图像清单而没有 annotations 时标为无 GT。
- 完全相同的同类框保留首个，逐框审计；不按 IoU 合并邻近小目标，不修改源标签。
- 直接引用原图，只在本次 `inputs/dataset_view/annotations` 保存转换后的标注，不复制整套图像。
- 保存划分、类别证明、逐文件路径/尺寸/标签哈希、转换标注、去重统计和数据指纹。默认检查跨划分物理文件别名；`hash_images=True` 可额外检查图像字节级跨集重复，但会增加读取时间。
- 不把 test 用于模型选择或训练调参。有 GT 的 test 仅在选定 val 最佳权重后执行最终评价；无 GT 只输出预测。

验证固定 640×640。默认训练保留原 SSD 风格增强和原生 480～800 多尺度（其中三个 640 的采样权重也保留），控制台/CSV 记录实际尺寸。预处理是原生 RGB float [0,1]、方形 resize；不是 YOLO letterbox。如要所有算法训练固定 640，可显式设置 `train_scales=()`，并在所有对比实验中一致记录这一选择。

默认 train batch=4、val batch=8、workers=4、seed=0、FP32。不会在 OOM 后自动缩 batch/尺寸、换精度或跳过失败。需要调整时只改集中配置，论文中明确实验条件。相同 seed 不代表 GPU 所有算子逐位确定；严格 deterministic 选项可能被算子拒绝，程序不会静默绕过。

## 本地预训练权重

默认 `initialization="backbone_pretrained"`，包内文件：

`weights/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth`

- 大小：94,419,322 字节。
- SHA256：`641dc59835d5228e6013ac45acb633592ec825144fb3d78055bc07699f404353`。
- 来源：[原项目 PResNet50 官方权重](https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth)。清单另存 `weights/manifest.json`。
- 这是 ImageNet/SSLD 骨干预训练，检测器其余部分按原代码初始化，不是已训练的 TinyWOD 检测器。
- **不能替换成此前 DQ-DETR 使用的 `resnet50-0676ba61.pth`**，它们的 stem/键名不同。

可选完整 COCO 检测预训练：浏览器下载 [rtdetr_r50vd_6x_coco_from_paddle.pth](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth)，放入 `weights`，只在 `config.py` 将 `initialization` 改为 `"detector_pretrained"`。此可选文件不在包内；已验证其迁移到单类时只重新初始化类别相关张量，并记录全部迁移信息。

`scratch` 只能由你明确配置，缺文件不会自动切换。训练前会检查本地文件、计算 SHA256、严格检查结构并记录初始化来源。所有对比算法的初始化来源应在论文中列明，不能假定骨干预训练与其他算法的 COCO 检测预训练等价。

## 窗口输出与 CSV

运行前打印模型、配置、数据/权重位置、SHA256、设备、种子、batch、轮数和保存目录；每轮显示：

```text
[RT-DETR-R50vd] Epoch <当前轮>/72
      Epoch    GPU_mem    box_loss    cls_loss   giou_loss   total_loss   Instances      Size
    <进度、实际损失、显存、框数、实际输入尺寸、batch 进度/速度/ETA>
                 Class     Images  Instances      Box(P          R       AP50         AP     APsmall: <验证进度>
                   all     <图像数>    <实例数>    <实际指标>
Complexity: Params=..., GFLOPs=..., val inference=... ms/image (... FPS), PeakAllocated=... GB
[最佳模型更新] <仅在真实更新且保存成功时打印>
[RT-DETR-R50vd] Epoch ... 完成 | Box(...) | lr={每个参数组} | loss={全部实际训练损失和验证损失}
[耗时] train=... val=... metrics=... epoch=... | 最佳 Epoch=... AP=...
```

RT-DETR 的真实主损失是 bbox L1、VFL、GIoU；不会伪装成 YOLO 的 DFL/one-to-many/one-to-one。进度行是加权主头损失；总损失包括原生 auxiliary/DN。完整异常栈、最终每类指标、起止时间、实际完成轮数和汇总路径也会打印。

当前附件未包含早期 OWT 改写范例全文，因此不能声称已逐行复刻不可访问的代码。当前实现依据本轮基础代码、对话中的明确打印示例和完整记录要求；可访问的精确去重/审计模块已沿用。实际 CSV 字段清单随每次运行写入 `inputs/csv_schema.json`。

## 结果目录和保存时机

每次启动创建新的：`runs/RT-DETR-R50vd/<时间_随机编号>/`，不会覆盖以前实验。

| 路径 | 内容 / 时机 |
|---|---|
| `weights/best.pth` | 每轮 val COCO AP 提升后保存；含原模型、EMA、优化器等完整恢复状态 |
| `weights/last.pth` | 每轮训练、验证及逐轮 CSV 写出后原子保存；失败不会冒充完成 |
| `weights/epoch_0012.pth` 等 | 默认每 12 Epoch 和最后一轮的恢复文件，集中配置可改 |
| `logs/console.log` | Python stdout/stderr、进度、配置、训练与汇总输出，同步刷新 |
| `logs/error.log` | 出错时完整异常栈 |
| `logs/environment.json`、`package_versions.csv` | 解释器、设备、库版本/模块位置、随机设置 |
| `logs/tensorboard/` | 实际数值的 TensorBoard 事件 |
| `results.csv` | 每轮一行，**216 列**：主损失、39 个原生损失的 train/val 加权/去系数值、各组 LR、COCO AP/AR、P/R/F1、速度、耗时、资源、复杂度及最佳状态 |
| `log.txt` | 保留原 DetSolver 的逐轮 JSON 风格和 12 项 COCO 数组；原生 epoch 为 0 起，另有 epoch_one_based；这里的 test_coco_eval_bbox 明确指 val |
| `metrics/train_batches.csv`、`val_batches.csv` | 每个 batch 立即追加/关闭刷新：实际损失、LR、尺寸、框数、时间、梯度范数（训练）、资源；phase 区分逐轮/最终评价 |
| `metrics/losses.csv` | 每轮每个原生损失的加权值、去 weight_dict 系数值、系数、观测 batch 数和是否可用 |
| `metrics/learning_rates.csv` | 每轮每组本轮实际 LR、调度后下一轮 LR、weight decay |
| `metrics/metrics.csv` | 每轮/最终各划分的 COCO 全 12 项、P/R/F1 和诊断 AP |
| `metrics/per_class_metrics.csv` | 每轮/最终每类图像数、实例数、P/R/F1、AP50/AP/APsmall 等 |
| `metrics/per_image_metrics.csv` | 每轮/最终每图 GT、小目标数、预测数、TP/FP/FN、P/R/F1、crowd 忽略数、匹配 IoU |
| `metrics/epoch_timing.csv`、`resources.csv` | 完整轮次耗时和资源记录；资源读取受系统限制时留空并写 resource_note，不阻断训练 |
| `metrics/epoch_0001/` 等 | 逐轮指标/每类/每图/PR/置信度曲线/AP-by-IoU/混淆矩阵 CSV，COCO precision/recall/scores NPZ、指标定义、评价文本和 PNG 图 |
| `results/training_curves.png` | 每轮更新的损失、指标、LR、时间曲线；原始历史始终保存在 CSV |
| `metrics/final/val`、`final/test` | 最佳权重最终评价；test 仅在有 GT 时生成评价结果，含最终预测 JSON |
| `results/test_unlabeled/` | test 无 GT 时的预测 CSV/JSON；可选预测图片，指标留空 |
| `metrics/inference_speed*.csv/json` | 训练后 batch=1、预热、同步计时的逐样本/汇总速度 |
| `inputs/` | 实际配置/原配置源、实际原生 YAML、结构、初始化和优化器分组、数据标注快照、指纹、去重逐框/逐图/划分 CSV |
| `experiment_summary.csv/json` | 本次状态、实际完成轮数、最佳轮、最终指标、初始化/设备/指纹及路径 |
| `runs/experiment_comparison.csv` | 各次成功/失败/中断实验汇总，每个 run_id 仅一行，写入时加锁 |

逐轮/逐 batch CSV 都校验唯一键和表头，不重复追加同一条记录。每行关闭刷新，关键逐轮表另做 fsync；权重采用临时文件 + 原子替换。结果 CSV 的 `epoch_seconds` 是写该行时已耗时间；`epoch_timing.csv` 和控制台的 epoch 耗时包含随后绘图/权重保存。图表会更新，历史数值和每轮图表不丢弃。

原 criterion 在训练时可产生 39 项损失；原推理模式只返回末层，因此验证只计算 3 个主损失，验证 auxiliary/DN 列为空。背景 batch 没有 DN 时也留空，真正计算得到的零保留为零。train/total_loss 与 val/total_loss 范围不同，不应直接相减判断过拟合；可比较对应主头损失。去系数值仅除 weight_dict 系数，不移除 VFL 内部的 IoU/alpha 权重。

## 论文比较时的指标口径

- headline `AP/AP50/APsmall` 使用 COCOeval，原图像素和原注释 area、IoU=.50:.95、标准 maxDets=100。不按 640 缩放后的面积定义小目标。缺少该尺度/类别 GT 时留空，不假报零。
- P/R/F1 是同类一对一、按置信度顺序匹配、IoU=.5；使用验证集宏平均 F1 最大处的统一置信度，仅用于描述指标，不用于模型选择/调参。F1 为有 GT 类别的 F1 均值。
- 有 GT 的 test 沿用最佳 val 确定的 P/R/F1 阈值，不在 test 上重新挑选阈值。
- `diagnostic_AP*` 是另行注明的 score-matching 诊断 AP，候选上限默认 300；不要与 COCO AP 混用。
- 每图统计/混淆矩阵采用集中配置的固定 conf=.25、IoU=.5，口径与上面的最优 F1 阈值不同。
- `val inference` 是验证批量前向耗时 / 图像数，包含按配置的精度，不包含损失/后处理。`inference_speed.csv` 则是 batch=1、预加载输入的模型前向基准；分开保存预处理、后处理时间，不与 TensorRT FPS 混用。
- Params 是实际模型参数数，另有 trainable 参数数。GFLOPs 是 fvcore 支持算子的估计，**FMA=1**，未统计算子逐项保存在 `model_complexity.json`；不同工具/FMA=2/融合部署的数值不可直接混比。
- PeakAllocated 是训练进程在验证阶段的峰值，含仍驻留的模型、优化器、EMA 等，不是独立部署的显存测试。
- 先核对汇总中的数据指纹、初始化来源、尺寸/增强、seed、batch/优化器和指标口径，再比较 P/R、AP50、AP、APsmall、速度和参数量。模型选择只用 val AP。

## 断点恢复与检查

只在 `config.py` 填 `resume_checkpoint="原运行目录/weights/last.pth"`（也支持匹配的 best.pth），再 Run 正式入口。恢复原模型/EMA、优化器、scheduler、scaler、Python/NumPy/torch/loader 随机状态，校验类别、数据指纹和关键配置。仍创建新目录，继承已提交轮次 CSV；原中断目录及未提交记录保留作审计。

**保留原运行目录及 CSV，不要只复制一个权重文件。** 从较早的周期权重恢复而其来源 best 已属于未来轮次时会拒绝，避免错误继承未来结果。完整恢复文件含优化器/EMA，比纯推理权重大。

可直接 Run：

- `smoke_test.py`：合成数据、CPU、小图、1 Epoch 的真实 R50vd 前后向/保存测试，不读取实际 TinyWOD，不改变正式配置，结果放 `verification_runs`。
- `tests/test_invariants.py`：数据/去重/指标数值/本地权重/CSV 等回归检查。
- `tests/check_resume.py`：合成数据 3 Epoch 连续训练 vs 中断后恢复，校验权重/EMA/优化器/RNG 逐位相等及 CSV 不重复；会生成较大临时恢复文件。
- `tests/check_spawn.py`：本机验证 2-worker spawn 数据加载；当前交付环境禁止本地 Unix socket，实际跨进程张量传输检查受限，并未宣称通过 Windows 实测。

已完成的具体检查与边界见 `verification/VALIDATION.md`。真实数据未附带，未在你的 Windows / RTX 5090 上完成初始化或 72 Epoch 实训；随包示例均明确标记为合成软件测试，不是 TinyWOD 论文指标。
