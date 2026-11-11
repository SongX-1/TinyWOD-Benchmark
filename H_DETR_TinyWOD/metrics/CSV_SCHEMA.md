# CSV 与指标口径

本目录是说明；实际表格在每次 `runs/H-DETR-R50/<run_id>/metrics/`。统一 UTF-8 BOM，未知/不可定义的数值写空，JSON 写 null；确实测得的 0 才写 0。

| 文件 | 唯一键 | 主要内容 |
| --- | --- | --- |
| losses.csv | epoch, phase, batch, name | 全部主层/aux/encoder 和 one2many 的每批损失/诊断 |
| batches.csv | epoch, phase, batch | step、图像ID、实例、H/W、各阶段秒数、总优化损失、梯度/跳步和资源 |
| learning_rates.csv | epoch, batch, group | 每批全部分组 LR、WD |
| epoch_losses.csv | epoch, phase, name | 上述每项按 batch 平均、批次数 |
| metrics.csv | epoch, stage, split | 每轮/最终全部主要检测指标 |
| per_class_metrics.csv | epoch, stage, split, class_id | 单类分别 P/R/F1/AP50/AP/各尺寸AP/TP/FP/FN |
| per_image_metrics.csv | epoch, stage, split, image_id | 原图名称、P/R/F1/TP/FP/FN；不虚构没有定义的逐图 mAP |
| epoch_timing.csv | epoch | ISO 开始结束、train/val/evaluation/完整 epoch 秒数、batch 数 |
| resources.csv | epoch, phase | RSS、当前/保留/峰值显存、遥测失败原因 |
| checkpoint_events.csv | epoch, kind | best/last/periodic 文件、保存耗时与大小 |
| results.csv（上一级） | epoch | 主层及分支损失宽表、主要指标、三组 LR、时间/速度/资源、best 标记 |

`phase` 区分 train、val、final_val、test、test_unlabeled。`stage` 区分 epoch/final；即使最佳模型来自早期 Epoch，最终重评也不会与当轮记录混为一条。

## 原生损失

- `loss_ce`：focal 分类损失，系数 2。
- `loss_bbox`：L1，系数 5；控制台别名 `l1_loss`。
- `loss_giou`：GIoU，系数 2；控制台别名 `box_loss`。
- `_0`…`_4`：原生解码器辅助层；`_enc`：two-stage encoder；`_one2many`：一对多分支。
- `class_error`、`cardinality_error` 是诊断，`optimized=False`，weight/weighted_loss 留空，不加进优化目标。
- `raw_loss` 是 criterion 计算值；一对多的 `native_loss=raw_loss*lambda_one2many`，`weighted_loss=native_loss*weight`。原生 train_hybrid 在返回前就乘 lambda，本适配不会乘两次。
- `batches.csv.loss`、`results.csv train/loss` 为所有实际优化项加权总和，包含 aux/encoder。主表 `box_loss/cls_loss/l1_loss` 是主层两分支 native 值之和，不能当成完整总目标。
- val/final/test 的损失是原项目一对一 + aux + encoder，一对多验证栏留空。没有 DFL 项。每轮 loss 按 batch 均值，与原 engine 的 MetricLogger 口径一致；不把最后小 batch 按图像重新加权。

## 检测与曲线

`precision/recall/f1` 是单类 micro、IoU=.5、COCO crowd-ignore 匹配；最大 F1 置信度仅在 val 选择。AP50、AP(.5:.95)、APsmall/medium/large、AR100 是标准 COCOeval，maxDets=[1,10,100]。APsmall 使用源标注原图 area<1024。无对应 GT 则空。

每轮 `epoch_XXXX/` 与 `final/<split>/` 同时写预测 JSON/CSV、per_class/per_image、metrics、PR 曲线（每类/10 IoU/101 recall）、AP-by-IoU、每类 1001 点置信度 P/R/F1/TP/FP/FN、全局工作点曲线、混淆矩阵及原始 COCO precision/recall/scores NPZ。诊断混淆矩阵按 IoU 贪心匹配、行为真列为预测，有 background 行列；其规则与 COCO AP 匹配分开标注。

## 速度、资源与恢复

`inference_ms`：按 CUDA 同步后的网络前向墙钟时间/图像数；逐 Epoch 值来自实际 val batch，最终独立 `inference_speed.csv` 使用 batch=1、固定图片/重复次数/预热次数。后者另记解码+变换、传输、后处理、总时延，中位数/P95 和 FPS。不能把 batch=2 val 速度与另一模型 batch=1 的单图速度直接当同一口径。

每轮 GPU 峰值从该轮开始 reset 后统计，包含该轮 train/val；最终汇总取所有轮及最终过程的已知峰值最大值。CPU 的 GPU 列为空；无法读 RSS 只记录原因，不让可选遥测阻止训练。计算量含 unsupported_ops 时总 GFLOPs 为空，partial 单列。

完整 batch 先校验所有唯一键，再一次 writerows/flush/fsync；轮级宽表根据同一键原子更新时间，不追加重复行。续训新建目录，只继承检查点前已提交的 train/val 记录，排除父实验未来轮数及 final/test。配置、数据、网络、backend 不匹配则拒绝恢复。
