# VMINet-B：TinyWOD 小目标边界框检测，36 Epoch

本项目在本次提供的 `VMINet-main.zip` 上修改。原包是 ImageNet 分类代码，未提供检测头或 TinyWOD 加载器。交付保留原始 `vminet.py`，增加 **VMINet-B 骨干 → 最后特征构建 Simple Feature Pyramid（P2～P6）→ torchvision 原生 Faster R-CNN 边界框检测器**。这是明确标注的检测适配版本，不冒充原论文完整 Mask R-CNN 实现，不训练掩码，也没有混入 YOLO、OWT 或 DETR 网络。

原仓库只有 Ti、XS、S、B，**没有 L**。本次默认使用实际最大的 **B，训练 36 Epoch**，不自动轮训其他变体。640 输入的完整检测器有 **49,140,669 个参数**，含金字塔和检测头。

按用户最新要求，**没有依赖检测、版本拦截或自动安装脚本**。训练仅正常导入模块并留档实际使用的 Python / torch / torchvision 和设备，不扫描安装包，不调用 pip。数据、类别、模型配置及本地权重的有效性检查保留。

## 1. PyCharm 直接运行

1. 完整解压，用 PyCharm 打开 `VMINet_TinyWOD` 文件夹，选择准备好依赖的解释器。
2. 核对 `config.py` 的数据目录，已沿用历史日志中的路径，见下一节。
3. 手动将 B 分类预训练文件放到 **`weights/B-ckpt.pth.tar`**；地址和格式见 `weights/README.md`。
4. 打开 **`train_vminet_b.py`**，点击绿色 **Run**，无需命令行参数。

| 文件 | 作用 |
|---|---|
| `train_vminet_b.py` | 固定真实 B，其余参数来自 config.py；推荐入口 |
| `train_tinywod.py` | 运行 config.py 指定的一个模型，默认 B |
| `run_all.py` | 本次也只运行配置指定的一个模型，默认 B，不额外训练其他变体 |
| `check_project.py` | 可选语法/算法回归检查；训练不会自动调用，不是依赖检测 |
| `smoke_test.py` / `verify_runtime.py` | 可选合成数据验证，输出到 verification，不替代真实实验 |

相对路径均从项目根目录解析，不受 PyCharm 工作目录影响。入口含 Windows freeze_support，默认 workers=0。

依赖见 `requirements.txt`，可用 PyCharm 解释器包管理界面准备。已验证 **Python 3.12、torch 2.7.1 CPU、torchvision 0.22.1 CPU、timm 0.6.13**。GPU 请使用与显卡/驱动兼容且相互配对的 torch / torchvision。原生 NMS/RoIAlign 来自已安装 wheel；不构建 CUDA 扩展，不调用 cl/nvcc/Ninja，不启用 torch.compile。

## 2. 配置只改 config.py

| 配置项 | 默认值 / 说明 |
|---|---|
| yolo_root | `D:/dataset`；本机相同则无需修改 |
| coco_root | `D:/dataset` |
| dataset_mode | auto：优先存在的 YOLO 根目录，否则 COCO；可明确 yolo/coco，不合并两份数据 |
| coco_annotations / coco_image_dirs | 常见 COCO 布局自动定位，有歧义/特殊命名时只在这里指定各 split |
| epochs / variant | **36 / b**；train_vminet_b.py 固定 B |
| image_size / batch_size / val_batch_size | **640 / 2 / 2**；尺寸须为 32 倍数，不因 OOM 自动改变 |
| device / seed / workers / amp | auto / 42 / 0 / False |
| pretrained | True 必须有本地对应骨干权重；False 明确从头初始化并留档 |
| backbone_weights['b'] | `weights/B-ckpt.pth.tar`，可改本地绝对路径 |
| pretrained_state_key | state_dict；选择 EMA 时必须显式改为真实存在的 state_dict_ema |
| detector_weights | 可选完整检测器参数初始化，不恢复训练进度 |
| resume_from | 本项目 last.pth / epoch_XXXX.pth，完整续训，见第 7 节 |
| output_root | runs |

数据和正式预训练权重未附在代码包。上述 Windows 路径来自用户日志，当前验证环境不能访问，不能声称核验过真实文件。

## 3. 数据、类别、小目标

YOLO 读取 `images/{train,val,test}` 和对应 `labels/{train,val,test}`，不重新划分。每行 `class cx cy width height`，坐标归一化。训练/验证背景图片必须有对应空 TXT，缺标签不当成背景；test 只允许全部有标签或全部无标签。

| 源及导出编号 | 类别 | Faster R-CNN 内部编号 |
|---|---|---|
| 0 | target | 1 |
| 1 | target | 2 |
| 2 | target | 3 |
| 3 | target | 4 |
| 4 | target | 5 |

检测器内部 0 为背景，避免把 target 误当背景。类别优先核对源根目录 manifest/YAML；若缺少声明，默认采用 config.py 的上述历史顺序，打印并记录缺少独立类别证明，需用户确认真实编号顺序。严格模式可设 accept_unverified_yolo_names=False。`data/classes.json` 只是说明，不会重写标签。

COCO 按类别名称映射，保留源 JSON 的精确图像子集。常见布局为 images/train + annotations/instances_train.json；存在多个候选会报出歧义，仅在 config.py 明确指定，不猜测划分。

640 正方形 letterbox 保持纵横比，填充值 114；框按实际四舍五入后的横纵缩放比和填充量还原原图，保留亚像素。训练仅 0.5 概率水平翻转，不直接沿用分类 MixUp/CutMix。ImageNet 均值方差只在检测器 transform 做一次。P2 高分辨率特征和 8/16/32/64/128 anchor 兼顾小目标；不删除小于 2 像素的有效 GT。真实效果必须以 APsmall 验证，不能仅由结构保证。

训练排除 crowd 正例，COCOeval 保留 crowd ignore。只对“同类别、裁剪后坐标、area、iscrowd 全部一致”的精确重复框保留首条并审计，不按 IoU 合并相邻目标，不改源数据。每次实验的 inputs 保存标注视图、数据清单/指纹、去重和各类统计。

## 4. 对原结构的适配

`vminet.py` 与上传文件逐字节一致；原 stem、四阶段、24 个 Block、乘法分支、tril 注意力、残差/下采样均保留。B 实际调用原始 vminet_B，base_dim=96、depths=[2,2,18,2]、mlp_ratio=2，没有靠重命名冒充。

分类模型的空间 sum 线性层绑定 224 输入的 56/28/14/7 网格。适配 640 时构建 160/80/40/20 网格；初始化加载分类权重时，仅对空间 sum 向量作二维双线性插值并按面积比缩放，保持常量输入的求和尺度。原 Block.forward 不改，原分辨率插值为恒等。每个加载/插值/忽略分类 head 的动作存入 inputs/pretrained_loading.csv；不能称所有层尺寸完全不变。

论文检测采用最后骨干特征经卷积/反卷积变成多尺度。本交付沿用该方向，自行实现 last-map Simple Feature Pyramid，接原生 Faster R-CNN bbox 分支；原包没有官方检测配置，不声称精确复现 Mask R-CNN。规格保存到 inputs/model_specification.json。参考：[作者仓库](https://github.com/yws-wxs/VMINet)、[论文 4.3 节](https://arxiv.org/html/2501.02040v2#S4.SS3)。

原分类训练、工具、许可证仍在原路径。本项目只保留 TinyWOD 检测适配所需代码，分类训练入口与上游源码压缩包已移除。

## 5. 实验设置与指标口径

默认 AdamW，lr=1e-4，weight_decay=0.05；bias/一维参数不衰减；27、33 Epoch 后 lr 乘 0.1（论文 12 轮 9/11 节点等比例延长）；梯度裁剪 0.1，验证采样独立 seed=12387。所有真实优化器分组和参数留档。原分类 300 轮、Top1/Top5、分类增强不直接套用。

四项损失均为原生 Faster R-CNN，权重均 1：ROI CE、ROI 框 SmoothL1、RPN BCE、RPN 框 SmoothL1；不虚构 DFL、OM/OO 或额外 L1。验证保持骨干 BatchNorm 为 eval，预测路径不带 GT；额外诊断损失只切换原生 RPN/ROI 的采样标志，不改 BN，不让 GT 参与预测，额外损失时间不计入推理。

AP 用 pycocotools：IoU .50:.95、maxDets=1/10/100，APsmall 按原图 area 的 COCO 定义。P/R/F1 在 IoU=.5 下按 COCO 匹配/ignore 进行全局 micro 汇总；验证 1001 个置信度点取最大 F1，平局选较高阈值。best 按验证 AP；最终 test P/R/F1/混淆矩阵冻结 best val 阈值，不用 test 调参。AP 按完整分数排序，不限于一个展示阈值。

test 有 GT 时只在模型选择后评估 best；无 GT 只预测，最终指标明确来自 val；没有 test 明确记状态。未定义指标留空/N/A，不伪造 0。详细口径和阈值来源见各评价目录 metric_definitions.json。

固定种子和 cuDNN 设置；CPU 连续训练/续训已验证参数、状态和损失完全一致。CUDA 原生 RoIAlign backward 可能非逐位确定，不能保证 GPU 逐位重现。为保留原生 wheel 算子并避免 torchvision lazy compile，未启用全局 strict deterministic algorithms，实际设置存入环境 JSON。

## 6. 窗口输出和保存位置

保留以下结构，损失列使用本算法真实项；下面省略号仅为说明，不是伪造训练数值：

```text
      Epoch    GPU_mem   box_loss   cls_loss    rpn_obj    rpn_box  Instances      Size
       1/36        ...        ...        ...        ...        ...        ...       640: tqdm 进度
                 Class     Images  Instances      Box(P          R       AP50         AP    APsmall
                   all        ...        ...        ...        ...        ...        ...        ...
Complexity: Params=..., GFLOPs=..., val inference=... ms/image (... FPS), PeakAllocated=... GB
[VMINet-B] Epoch 1/36 完成 | Box(P=... R=... AP50=... AP=... APsmall=...) | lr={所有分组} | loss={全部四项train/val}
[耗时] ...
[最佳模型更新] ...
```

训练前打印模型/配置/数据/本地权重/设备/轮数/输出；逐轮打印训练和验证进度、all 指标、复杂度、全部损失和学习率、耗时、best 更新及位置；结束打印最佳轮次、核心指标和划分、实际轮数/状态、始末和总耗时。stdout/stderr 同步到 console.log，失败保留完整 traceback 和 error.log。OOM 不静默改 batch、尺寸、精度或跳过；如需调整，只在 config.py 显式改并另开实验。

每次输出新时间戳+UUID，互不覆盖，根目录为 `runs/VMINet-B-FasterRCNN/<本次编号>/`：

| 本次目录内路径 | 内容 / 时机 |
|---|---|
| weights/best.pth | 验证 AP 严格改善时保存参数、架构、初始化、类别和阈值；供评估/初始化 |
| weights/last.pth | 每完整 Epoch 的模型/优化器/调度器/scaler/best快照/全部 RNG；供续训 |
| weights/epoch_XXXX.pth | 默认每 5 轮及最后轮保存完整状态 |
| logs/console.log / error.log | 完整窗口与异常；实际环境另存 JSON/CSV |
| results.csv | 每 Epoch 宽表：所有 train/val 损失、P/R/F1/各 AP、学习率、时间/速度/显存、参数/计算量、best 等 |
| summary.csv | 保留原项目同名表和 train_/eval_ 风格，改记检测损失与 P/R/F1/AP50/AP/APsmall 等，不保留假 Top1/Top5 |
| log.txt | 每轮 JSON 行：全部损失、指标、学习率 |
| metrics/*.csv | 每 batch、每损失、每参数组学习率、资源、时间、逐类/逐图指标、检查点、测速/复杂度 |
| metrics/epoch_XXXX/ | 每轮预测、总体/逐类/逐图指标、COCO 数组、PR/置信度/AP-IoU 明细、混淆矩阵及图 |
| metrics/final/val 或 test | best 最终评价；无 GT 为 test_unlabeled，仅预测/status |
| results/training_curves.png | 每轮更新损失、指标、学习率、时间/速度曲线 |
| results/val_predictions 和 test_predictions | best 预测图示例，默认各最多 12 张 |
| inputs/ | 配置/模型/优化器/加载审计、数据清单/指纹/标注视图、去重与类别统计 |
| experiment_summary.csv / .json / run_state.json | 成功/失败/中断、实际轮数、原因、指标与位置 |

另有上级 **runs/experiment_comparison.csv**，带文件锁追加各次实验对比。比较前核对状态、数据指纹、初始化、配置、metric_split 和设备，再比较 APsmall、AP、AP50、P/R/F1、参数、速度和显存。

CSV 为 UTF-8 BOM，Excel 可打开；完整 batch 写入后 flush/fsync/关闭，唯一键拒绝重复，Epoch 保存结束后原子更新同一行耗时。每轮曲线/预测会保留，需充足磁盘。强制杀死进程时不保证正在写的 batch 完整，续训以最后完整 Epoch 为准。

参数量按实际模型统计。FVCore 遇到 RoIAlign/NMS 等未支持算子时，完整 GFLOPs 为 N/A/空值，partial 和算子列表另存，不能把部分计数当总量；一次乘加计一次 FLOP。val inference 含归一化/RPN/ROI/解码/NMS，排除额外验证损失；最终基准 batch=1，预热+重复、CUDA 同步，分别记预处理/拷贝/完整推理/导出/总耗时。跨算法比较必须核对计时范围。

**历史范例可访问性：** 本轮有 VMINet 原包、当前工作区记录和对话中的具体格式要求；最早 OWT 修改范例未作为本轮附件提供，不能声称逐行复刻不可访问代码。这里保留可核实的完整记录功能和原 summary.csv/log.txt 习惯，没有以简易单表替代。复用的数据/日志/评价工具不改变 VMINet 网络。

## 7. 恢复训练

仅在 config.py 设置 resume_from 为旧实验 last.pth / epoch_XXXX.pth；保留父目录 results.csv、summary.csv、log.txt 和 metrics。best.pth 不是完整续训文件。

总轮数仍是 36，例如第 10 轮从第 11 轮继续。恢复创建新目录，校验数据指纹/结构/训练超参/CPU-CUDA 类型，复制已提交 Epoch 记录，排除旧 final/test 和未提交未来轮次，恢复完整状态，不覆盖父实验。优先级：resume > detector_weights > 本地骨干 pretrained > 明确从头初始化。续训无需另备已在检查点中的骨干权重。改关键超参或迁移 CPU/GPU 应另开初始化实验，不能称精确恢复。

## 8. 已做验证和未完成事项

- **39 项回归通过**：原文件哈希、四个真实变体、原 224 特征等价、空间插值/严格加载、标签映射、几何还原、四项损失/原注意力梯度、空背景、验证 BN/RNG 不变、无下载/compile、CSV 唯一键/更新等。
- **端到端合成数据**：Ti 完整 24 Block，缩小输入/检测头/proposal，连续两轮真实反向/AdamW，验证/CSV/图表/best/last/周期/最终评价/测速全部执行。
- **精确续训**：第一轮恢复至第二轮，模型、优化器、调度器、scaler、best、全部 RNG、train/val 损失与连续运行一致，CSV（含 summary.csv）无重复。
- **独立 test 分支**：有 GT 的 test 只在训练后评估且冻结 val 阈值；无 GT 只预测，两种分支均执行。
- **完整默认 B**：640 CPU 前向通过；同完整骨干/金字塔/ROI 头宽度，在 128、batch=2 下真实反向/AdamW 更新，原注意力参数梯度有限非零且确实更新。
- **ZIP**：标准 Deflate、CRC、完整解压和逐文件 SHA-256 比对，见交付检查报告。

材料在 verification，合成实验明确为 SMOKE，不能当 TinyWOD 论文结果；生成的测试权重不随包附带，可运行验证入口重新生成。

**未验证**：没有真实 TinyWOD、官方预训练文件或 CUDA GPU，未做真实 36 Epoch、完整 B 的 640 训练显存/收敛、Windows PyCharm GUI 实机测试。官方权重只验证原保存格式的加载逻辑，未核验未提供文件的实际张量；若结构不符会明确报差异，不静默丢层或随机兜底。正式精度、速度、显存必须以本机实验为准。
