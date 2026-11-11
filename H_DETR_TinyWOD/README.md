# H-Deformable-DETR：TinyWOD 小目标检测

本项目在本次提供的 `H-Deformable-DETR-master(2).zip` 上修改。原始模型、匹配器、Transformer、算子源文件、配置脚本和许可证均保留。新增 TinyWOD 实验适配层，使用原生 PyTorch H-DETR，不需要 MMEngine/MMCV，也没有混入 OWT 或 YOLO 网络。

本次采用原仓库 **R50、two-stage、hybrid branch、36 Epoch、lr_drop=30** 配方。新的请求没有另指定轮数，因此使用这一原生配方；实际参数集中在 `config.py`。

## 1. 在 PyCharm 中直接运行

1. 解压后打开包含 `config.py` 的项目根目录（Windows 建议短路径，例如 `D:/H_DETR_TinyWOD`，便于原生长文件名配置脚本解压），选择 Python **3.10–3.12** 解释器。已实际验证 Python 3.12 + torch 2.7.1 + torchvision 0.22.1 的 CPU 环境。
2. 右键 `install_dependencies.py` → **Run**。它检查当前 `sys.executable`，一次列出所有依赖，只安装缺失/不兼容的普通依赖；保留匹配的现有 torch/torchvision。新环境默认安装 torch 2.7.1 / torchvision 0.22.1 的 CUDA 12.8 wheels。已有 GPU 栈不匹配时会明确停止，建议新建解释器，避免破坏其他基线环境。
3. 把本地骨干权重按下一节放好。数据路径若与下文相同，可以保留默认配置。
4. 右键 **`train_tinywod.py`** → 点击绿色 **Run**。**`run_all.py`** 使用完全相同的单模型流程。本包只运行 H-DETR-R50，不包含串行训练其他模型的入口。

无需命令行参数，不依赖 PyCharm 工作目录。`check_project.py` 可直接 Run 做离线回归检查；`smoke_test.py` 可直接 Run 做合成数据两轮训练测试，二者不使用正式数据。

## 2. 本地权重：训练过程中不联网下载

默认 `pretrained=True`，使用 torchvision **ResNet50 ImageNet-1K V1** 骨干初始化，检测头按原项目随机初始化。

用浏览器下载官方文件：

<https://download.pytorch.org/models/resnet50-0676ba61.pth>

保存为项目下 **`weights/resnet50-0676ba61.pth`**。也可以复制自己已经完整下载的同名文件。启动时校验 SHA256 前缀 `0676ba61`，截断/损坏的下载会在扫描数据集前报错。加载时去掉原分类器 `fc.*`，其余骨干严格匹配，FrozenBatchNorm 按原项目处理。

**压缩包不包含数据集和预训练权重。** 训练入口只读本地文件，没有 HTTPS 自动下载、临时编译或静默改成随机初始化。文件位置改变，只改 `config.py` 的 `backbone_weights`。若明确要从零训练，在同一文件设 `pretrained=False`；日志会记录随机初始化，不能把它与预训练实验混称。自定义骨干权重按实际 SHA256 标记为 `custom_local_backbone`，不冒称 ImageNet 权重。

可选 `detector_weights` 必须是与本项目类别数和网络结构严格匹配的本地 state_dict；不会静默忽略检测头形状错误。恢复训练使用 `resume_from`，不使用此项。

## 3. 数据和唯一配置位置

`config.py` 已预填此前提供的路径：

| 配置项 | 默认值 |
| --- | --- |
| `yolo_root` | `D:/dataset` |
| `coco_root` | `D:/dataset` |
| `dataset_mode` | `auto`，优先已存在的 YOLO 根目录，否则 COCO |
| `epochs` | **36** |
| `output_root` | 项目下 `runs` |

若实际路径不同，**只改 `config.py`**，三个运行文件均不需要传参。可显式设 `dataset_mode='yolo'` 或 `'coco'`；不会混合不同来源的 train/val 划分。

YOLO 布局为 `images/train`、`images/val`、可选 `images/test` 及对应 `labels/<split>/*.txt`。一行是 `class cx cy w h`，坐标归一化到 `[0,1]`。训练/验证背景图片应有空 txt；缺标签不是背景。测试集有部分标签会报错，避免把未知标签误当负样本。

固定类别编号如下；源 YAML/manifest 若声明类别，必须与此顺序一致：

| ID | 类别 |
| --- | --- |
| 0 | target |
| 1 | target |
| 2 | target |
| 3 | target |
| 4 | target |

源目录没有类别声明时，默认使用上述显式声明并打印“未通过源 YAML 验证”的限制；若希望强制源声明，把 `accept_unverified_yolo_names=False`。

COCO 自动识别 `images/<split>` 或 `<split>` 图片目录，以及 `annotations/instances_<split>.json`、`annotations/<split>.json`、`<split>.json`、`<split>/annotations.json`。有多个候选时停止，使用同一配置中的 `coco_annotations`、`coco_image_dirs` 按 split 指定。COCO 类别按名称映射到上述连续编号，不把原 COCO 91 类沿用进来；已命名的 ignore/background 类可明确排除。

程序只生成本次实验的 COCO 标注视图，直接读原图片，**不改原图/标签，不复制整个数据集，不重划分训练验证集**。保留正面积小框，只有无效/非正面积框按记录的规则处理。类别、NaN、越界、尺寸及跨划分同一物理图像都会检查。

完全相同的“类别 + 裁剪后像素坐标 + area + iscrowd”框保留第一条，其余只在实验视图中移除，输出去重明细/汇总/逐图统计。不会按 IoU 合并邻近小目标。

默认指纹包含标注 SHA256、图像路径/尺寸/字节数/mtime 和处理策略。`hash_images=True` 会额外计算图像内容 SHA256、检查跨划分相同图像内容，启动更慢。默认不声称已经做了所有图像内容哈希。

## 4. 真实模型、预处理与公平性

| 项目 | 本包默认 |
| --- | --- |
| 骨干 / 特征 | ResNet50 + FrozenBatchNorm，4 个特征尺度 |
| 编码器 / 解码器 | 6 / 6 层，hidden_dim=256，FFN=2048 |
| 训练查询 | 300 one-to-one + **1500 one-to-many** |
| 推理查询 | **300 one-to-one**，原生 top-k=100 |
| 混合训练 | GT 重复 6 次，lambda_one2many=1 |
| 原生机制 | two_stage、with_box_refine、mixed_selection、look_forward_twice、aux 和 encoder loss |
| 轮数 / 种子 | **36 / 42** |
| Batch / workers | train=2、val=2、workers=0，原配方 `drop_last=True` |
| 优化器 | 原分组 AdamW，主干组 LR=2e-5，其他组=2e-4，采样/参考点组=2e-5，WD=1e-4 |
| 学习率 / 梯度 | StepLR，第 30 轮结束后乘 0.1；clip_max_norm=0.1 |
| 精度 | FP32；`amp=False` |
| 输入 | **等比例短边 640、长边上限 1067**，ImageNet Normalize，原生 NestedTensor padding |
| 数据增强 | 水平翻转 0.5，默认不随机裁剪以减少裁掉小目标；无额外尺寸过滤 |

这里的 `image_size=640` 是 DETR 短边策略，实际张量常为 `600×1067` 等，并非固定 640×640；控制台和 batch CSV 打印实际 H×W。保持原生坐标/掩码逻辑。若论文需要严格固定某种输入策略，应显式修改配置并另开可追溯实验。`augmentation='native'` 可启用原项目多尺度/随机裁剪；具体训练尺度和验证尺寸都在配置中记录。

验证依照原项目禁用一对多查询，只计算一对一、aux、encoder 的真实验证损失。一对多验证损失写空，不写 0 冒充。训练的两分支、所有辅助层/编码器的损失完整保存。

MSDeformAttn 优先使用**已安装且在实际 GPU 上前后向探测通过**的原生 CUDA 扩展；没有时使用原仓库提供的 `ms_deform_attn_core_pytorch`，按查询分块计算，数学和自动求导路径已测试。不会调用 `cl`/`nvcc`/Ninja 临时编译。实际 backend 单独记录，比较实验应保持一致。PyTorch 参考实现较慢、显存行为不同；`ops_backend='native'` 时扩展不可用会明确失败。

CUDA `grid_sample` 反向存在非确定性；固定随机种子和 cuDNN 设置不能承诺 GPU 位级相同。默认记录此限制并使用 deterministic warn-only；`strict_determinism=True` 时遇不支持的算子报错。CPU 精确续训已核对。

显存不足会完整记录失败，**不会自动改 batch、图像尺寸、精度、查询数或跳过训练**。可在配置中明确启用原生 `use_checkpoint=True`，或另开实验统一降低 batch；改动留在配置快照中。

## 5. 控制台和日志

运行前显示模型/原生配方路径、数据路径及 COCO 视图、实际初始化文件、设备型号、种子、参数、开始时间、轮数和输出位置。每轮使用 tqdm 训练/验证进度条，格式为：

```text
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances          Size
       E/36      ...       ...        ...        ...        ...       HxW: 100%|...|
                 Class     Images  Instances      Box(P          R       AP50         AP    APsmall
                   all       ...        ...       ...        ...        ...        ...        ...
Complexity: Params=..., GFLOPs=..., val inference=... ms/image (... FPS), PeakAllocated=... GB
[H-DETR-R50] Epoch E/36 完成 | Box(P=... R=... AP50=... AP=... APsmall=...) | lr={所有组} | loss={所有训练/验证原生项}
[耗时] train=... val=... evaluation=... epoch含保存=... | 结束=...
[最佳模型更新] Epoch=... AP=... | .../weights/best.pth
```

这段是**格式示意，不是实验数据**。`box_loss` 表示两分支主层 GIoU 之和，`cls_loss` 为分类、`l1_loss` 为 L1；完整字典保留 `_one2many`、`_0`…`_4`、`_enc` 等真实项，没有伪造 YOLO DFL。CPU 时 GPU_mem 显示 CPU，显存留空/N/A。

GFLOPs 采用 fvcore 的“一次 fused multiply-add 计一次 FLOP”约定；若有不支持的算子，完整 GFLOPs 写空，控制台标 `N/A (partial=...)`，只把已统计部分存为 `gflops_partial`，并保存 unsupported_ops。不能把部分计算量当总 GFLOPs 写进论文。参数量为实际模型统计；验证速度含批处理条件，另有 batch=1 同步测速结果。

标准输出、警告、进度条及异常同步到 `logs/console.log`；完整异常另存 `logs/error.log`。最佳模型更新、逐轮耗时、文件位置、结束时间和真实运行状态都会打印。失败/中断不标成功。

当前可访问的输入是本次 H-DETR 源码，以及工作区已有的实验记录模块；最初 OWT 原始/修改压缩包未在本次附件中提供，因此不声称逐行复用了其中未展示的私有字段。本包已按对话明确要求保留完整原生损失和下面列出的记录逻辑，未缩减为单一结果 CSV。

## 6. 输出位置与文件

每次创建 `runs/H-DETR-R50/时间_随机ID/`，重复 Run 不覆盖已有实验。合成冒烟测试放在 `verification/local_smoke/.../runs/H-DETR-R50-SMOKE/...`，不与正式结果混用。

| 目录/文件 | 内容与写入时机 |
| --- | --- |
| `inputs/` | 有效 config.json、配置源码、模型/原生参数、初始化来源+SHA256、优化器分组、数据指纹/清单/统计、只读 COCO 视图、精确去重审计；初始化时写入 |
| `weights/best.pth` | 每次 val AP@.5:.95 提升后原子替换，模型选择只用 val；包含模型、最佳 Epoch、工作点和结构信息，**用于推理** |
| `weights/last.pth` | 每轮模型/优化器/调度器/scaler/Python+NumPy+Torch+CUDA RNG/采样器与 worker RNG/最佳模型快照/数据和配置签名；**可恢复训练** |
| `weights/epoch_XXXX.pth` | 每 5 轮及末轮的完整状态检查点，可恢复 |
| `logs/console.log`、`error.log` | 全程日志/完整异常；日志实时 flush |
| `logs/environment.json`、`package_versions.csv` | 解释器、设备、软件包、可复现设置；未知可选包不会因读取 metadata 崩溃 |
| `metrics/losses.csv` | 每个 batch 的所有原生损失/诊断项、分支、层、lambda 前 raw / 原生返回 native / coefficient / weighted，完整 batch 写完 flush+fsync |
| `metrics/batches.csv` | batch/step/图像ID/实例/实际尺寸、读数/前向/损失/反向/优化器/总耗时、梯度范数、AMP 跳步、CPU/GPU 资源；每 batch 写入 |
| `metrics/learning_rates.csv` | 每个训练 batch 的全部优化器组 LR、weight_decay |
| `metrics/epoch_losses.csv` | 每轮 train/val 的全部原生项、raw/native/weighted 均值；与原 MetricLogger 一致按 batch 平均 |
| `metrics/metrics.csv` | 每轮 val、最终 best-val/有 GT test 的 P/R/F1、AP50、AP、APsmall/medium/large、AR100、阈值和 TP/FP/FN |
| `metrics/per_class_metrics.csv`、`per_image_metrics.csv` | 每轮/最终逐类指标及逐图 P/R/F1/TP/FP/FN |
| `metrics/epoch_timing.csv`、`resources.csv`、`checkpoint_events.csv` | 轮级时间/资源、每次权重保存耗时/大小/路径 |
| `metrics/epoch_XXXX/` | 每轮 COCO 文本、预测 JSON/CSV、metrics/per_class/per_image CSV、10 IoU×101 recall PR 表、AP-by-IoU、逐类/全局置信度曲线、混淆矩阵 CSV/PNG、PR/置信度 PNG、COCO 原始张量 NPZ |
| `metrics/final/val`、`metrics/final/test` | 最佳模型的统一最终评价，同样保留完整表格/图/原始张量 |
| `metrics/final/test_unlabeled/` | 无 GT 测试集仅预测 CSV/JSON 和明确状态，不计算伪造指标；预测类别是内部 0–4，未冒充官方提交格式 |
| `metrics/complexity.json/csv` | 参数量、计算量覆盖情况、实际输入与查询数 |
| `metrics/inference_speed.csv/json` | 最佳模型 batch=1，逐图预处理/传输/前向/后处理、均值/中位数/P95/FPS；CUDA 同步后计时 |
| `results.csv` | 逐轮便于论文绘图的宽表：主损失/分支损失、所有主要指标、各组 LR、速度/资源、best 更新标志等 |
| `results/training_curves.png` | 训练/验证主损失、P/R/F1/AP/APsmall、LR、分项时间、速度曲线 |
| `results/val_predictions`、`test_predictions` | 最佳模型的部分原图尺度预测可视化 |
| `log.txt` | 兼容原项目的逐 Epoch JSON 日志 |
| `experiment_summary.csv/json`、`run_state.json` | 开始/结束/总耗时、已完成轮数、最佳轮、评价划分、完整状态及失败原因；成功/失败/中断都会保存 |
| 上一级 `runs/experiment_comparison.csv` | 追加每次实验，不互相覆盖；OS 文件锁防并发写入；合成测试汇总在自己的测试目录 |

CSV 为 UTF-8 BOM，可用 Excel 打开。唯一键严格检查；不重复追加同一 epoch/phase/batch/loss。逐 batch 一次写全损失后 fsync，按轮更新宽表也不产生重复行。`metrics/CSV_SCHEMA.md` 说明列含义与评价口径。

## 7. 评价与最终比较

AP 使用 pycocotools 标准 bbox COCOeval：IoU=.50:.05:.95，maxDets=[1,10,100]，原图坐标；**APsmall 根据原始标注 area < 32² 像素**，不按缩放后的训练框面积计算。某尺寸/类别没有 GT 时对应 AP 导出空/null，不填 0。

P/R/F1 使用单类 micro 聚合、IoU=.5、COCO 原生匹配及 crowd ignore。在 val 的 1001 个置信度点上选择最大 micro-F1 工作点；最佳模型选定后固定该 val 阈值评估 test，不在 test 重新调阈值。AP 本身使用完整预测排序，与这个 P/R 工作点分开。

有独立带 GT 测试集：训练和 best 选择结束后才做 test，最终汇总列 `metric_split=test`。测试无 GT/不存在：最终指标明确来自 val，测试图片只导出预测。`test_policy='required'` 可以强制要求完整测试 GT。

论文比较查看 `runs/experiment_comparison.csv` 的 P/R/F1、AP50、AP、**APsmall**、参数、可靠计算量、batch=1 延时/FPS、峰值显存，并核对 `metric_split`、初始化、输入策略、backend、batch 和种子。失败行的最终指标留空，不能当成功实验参与比较。

## 8. 断点恢复与异常

仅在 `config.py` 设置 `resume_from='原实验/weights/last.pth'`（或某轮完整检查点）后 Run。恢复**另建目录**：复制父实验截至检查点的已提交 CSV/轮级结果，过滤未提交和 final/test 记录；恢复模型、最佳快照、优化器、调度器、scaler、所有 RNG 及数据采样状态。从下一 Epoch 开始，不重复写第一轮，不覆盖父实验。

数据指纹、网络、训练设置、backend、设备类型必须一致。不能在续训时悄悄改 Epoch 总数、batch、LR 或增强；本实现不包含原 CLI 中强行重设恢复后 LR 的逻辑。已完成全部轮数会提示无需再次训练。中断在 batch 中间时，保留已落盘记录；恢复以最后完整 Epoch 的检查点为准，未提交半轮不拼成成功轮。

`best.pth` 是推理快照，`last.pth`/周期文件才是完整续训文件。完整检查点还保存最佳模型快照，文件通常较大；这是为了在父目录 best 指针更新或恢复旧周期文件时仍保留当时正确的最佳模型。

## 9. 实际验证范围

- `check_project.py`：新增/原始 Python 语法检查，**38 项回归测试**，包括单类/划分/无 GT test、精确去重、原图标注不改、crowd/空预测/无 small GT、固定 test 阈值、CSV 唯一键、全部依赖缺失报告、损坏本地权重、严格骨干加载、双分支梯度、激活检查点、原生算子 gradcheck 与分块梯度一致性。
- 完整生产结构：**47,791,359 参数**，R50、6+6 层、1800 训练查询、300 推理查询、1 类(target)；CPU 320×320 完成前向、原生混合损失（54 项）和一对一推理检查，详见 `verification/production_model_smoke.json`。
- 合成 6 张训练、5 张验证、2 张无 GT test，缩小到 1 encoder / 2 decoder、10+20 查询、96×128，实际完成 **2 Epoch 前后向训练**、COCO 评价、权重保存、全部 CSV/图/日志和单图测速。
- 从合成实验第 1 轮完整检查点恢复至第 2 轮，与连续训练核对：模型、最佳快照、优化器、scheduler、scaler 和每轮 train/val 损失精确一致；CSV 无重复键。报告在 `verification/resume_report.json`。
- 安装脚本在已有完整环境直接 Run，检查并实际导入通过，没有无故重装包。
- 最终 ZIP 使用标准 Deflate，CRC 检查并完整解压逐文件 SHA256 对比。

本环境**没有用户完整 TinyWOD 数据、真实预训练权重或可用 GPU**，没有运行正式 36 Epoch，没有验证 Windows/RTX 5090 实训，也不提供虚构精度。`verification/synthetic_smoke/` 和 `synthetic_resumed/` 中是明确标注的合成测试记录，权重/临时大文件不随包提供；运行 `smoke_test.py` 可自行重建。
