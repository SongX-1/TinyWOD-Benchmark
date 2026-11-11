# TinyWOD Object Detection Benchmark

Benchmark code for 7 detectors on the TinyWOD small-object detection dataset, unified as **single-class detection** (the only foreground class is `target`).

| Directory | Method | Training entry |
|---|---|---|
| `DINO/` | DINO-R50 | `train_dino_r50.py` |
| `H_DETR_TinyWOD/` | H-Deformable-DETR | `train_tinywod.py` |
| `OWRT-DETR/` | OWRT-DETR | `train_owrt.py` |
| `RT-DETR/` | RT-DETR-R50vd | `train_tinywod.py` |
| `VMINet/` | VMINet-B + Faster R-CNN | `train_vminet_b.py` |
| `YOLO26/` | YOLO26-S / YOLO26-X | `train_yolo26_s.py` / `train_yolo26_x.py` |

## Data path

The dataset is not distributed with the code. It is placed at `D:/dataset` by default. To change the path, edit only each project's root config file:

- `DINO/experiment_config.py` — `dataset_root`
- `H_DETR_TinyWOD/config.py` — `yolo_root` / `coco_root`
- `OWRT-DETR/train_owrt.py` — `DATA` at the top
- `RT-DETR/config.py` — `yolo_root` / `coco_root`
- `VMINet/config.py` — `yolo_root` / `coco_root`
- `YOLO26/config.py` — `existing_yolo_root` / `raw_dataset_root`

Both YOLO format (`images/` + `labels/`) and COCO format (`images/` + `annotations/`) are supported and detected automatically; DINO supports COCO only, and OWRT-DETR supports YOLO only.

## Run

```bash
cd DINO            && python train_dino_r50.py
cd H_DETR_TinyWOD  && python train_tinywod.py
cd OWRT-DETR       && python train_owrt.py
cd RT-DETR         && python train_tinywod.py
cd VMINet          && python train_vminet_b.py
cd YOLO26          && python train_yolo26_s.py   # or train_yolo26_x.py
```

Training results are written to each project's `runs/` directory.
