# -*- coding: utf-8 -*-
"""
从 TinyWOD 数据集构建 YOLO 格式检测数据集 (原始 RGB + 标签)。

数据筛选与划分严格对齐 D:/dataset 的方法:
  - 数据源: D:/dataset/train_v2 (原始 RGB) + train_target_segmentations_v2.csv (官方 RLE GT)
  - 标签: TinyWOD 官方 GT CSV 的实例 RLE, 按转置顺序解码为 bbox
  - 只保留有目标图像: EncodedPixels 非空的图 (41149 张)
  - 有目标图像按 SEED=0 打乱后 70%/20%/10% 划分 train/val/test
  - 无目标背景图不进入数据集
  - 三个集合严格不重叠; test 不参与训练与 best.pt 选择
  - 标签类别固定为 0 (target), 归一化到 768x768

输出 (D:/dataset/owrt):
  images/{train,val,test}  原始 RGB 的硬链接/副本
  labels/{train,val,test}  YOLO bbox 标签
  manifest.json
"""
import json
import os
import random
import shutil
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = Path(os.getenv('OWRT_DATASET_ROOT', 'D:/dataset'))
ORIGINAL_IMAGES = Path(os.getenv('OWRT_ORIGINAL_IMAGES', str(DATASET_ROOT / 'train_v2')))
GT_CSV = Path(os.getenv('OWRT_GT_CSV', str(DATASET_ROOT / 'train_target_segmentations_v2.csv')))
OUT = Path(os.getenv('OWRT_DATASET_OUT', str(DATASET_ROOT / 'owrt')))

SEED = 0
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
IMG_H = 768           # TinyWOD 图像高度
IMG_W = 768           # TinyWOD 图像宽度
DATASET_VERSION = 'tinywod-positive-only-70-20-10-v1'
IMAGE_SUFFIXES = {'.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp'}


def log(msg):
    print(f'[prepare] {msg}', flush=True)


def _rle_runs(rle, width=IMG_W, height=IMG_H):
    """Parse TinyWOD 1-based ``start length`` pairs into validated 0-based runs."""
    nums = [int(v) for v in str(rle).split()]
    if not nums or len(nums) % 2:
        raise ValueError(f'Invalid TinyWOD RLE: expected start/length pairs, got {rle!r}')

    total = width * height
    runs = []
    for i in range(0, len(nums), 2):
        start, length = nums[i] - 1, nums[i + 1]
        if start < 0 or length <= 0 or start + length > total:
            raise ValueError(f'TinyWOD RLE run is outside {height}x{width}: {rle!r}')
        runs.append((start, length))
    return runs


def rle_bbox(rle, width=IMG_W, height=IMG_H):
    """Return a TinyWOD RLE bbox as ``[x, y, w, h]`` (transposed coordinate order)."""
    x0, y0, x1, y1 = width, height, -1, -1
    for start, length in _rle_runs(rle, width=width, height=height):
        end = start + length - 1
        run_x0, run_x1 = start // height, end // height
        if run_x0 == run_x1:
            run_y0, run_y1 = start % height, end % height
        else:
            run_y0, run_y1 = 0, height - 1
        x0, x1 = min(x0, run_x0), max(x1, run_x1)
        y0, y1 = min(y0, run_y0), max(y1, run_y1)
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def read_gt(valid_stems):
    """读取真实 TinyWOD GT, 返回 ``{stem: [{rle, bbox}, ...]}``。"""
    log(f'读取真实 GT {GT_CSV.name} ...')
    t0 = time.time()
    df = pd.read_csv(GT_CSV, dtype={'ImageId': str, 'EncodedPixels': str}, keep_default_na=False)
    required_columns = {'ImageId', 'EncodedPixels'}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f'GT CSV 缺少必要列: {sorted(missing_columns)}')
    df = df[df['EncodedPixels'] != '']  # 过滤掉 EncodedPixels 为空的无目标记录
    gt = {}
    n_target = 0
    for _, row in df.iterrows():
        stem = row['ImageId'].rsplit('.', 1)[0]
        if stem not in valid_stems:
            continue
        rle = row['EncodedPixels']
        gt.setdefault(stem, []).append({'rle': rle, 'bbox': rle_bbox(rle)})
        n_target += 1
    log(f'  真实 GT: {n_target} 个目标, 分布在 {len(gt)} 张图, 用时 {time.time()-t0:.1f}s')
    return gt


def hardlink_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def scan_images(directory):
    """Return one image path per stem and fail on ambiguous duplicate stems."""
    paths = sorted(p for p in directory.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    by_stem = {}
    for path in paths:
        if path.stem in by_stem:
            raise ValueError(f'图像文件名不唯一: {by_stem[path.stem]} 与 {path}')
        by_stem[path.stem] = path
    return by_stem


def dataset_layout_matches(manifest):
    """Return whether manifest counts match the generated image/label layout."""
    expected = {}
    for split in ('train', 'val', 'test'):
        expected[OUT / f'images/{split}'] = ('image', manifest.get(f'{split}_images'))
        expected[OUT / f'labels/{split}'] = ('.txt', manifest.get(f'{split}_images'))
    for directory, (kind, count) in expected.items():
        if not directory.is_dir() or isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return False
        suffixes = IMAGE_SUFFIXES if kind == 'image' else {kind}
        if sum(1 for p in directory.iterdir() if p.is_file() and p.suffix.lower() in suffixes) != count:
            return False
    return True


def build():
    if not ORIGINAL_IMAGES.exists():
        raise FileNotFoundError(f'原始 RGB 图像目录不存在: {ORIGINAL_IMAGES}')
    if not GT_CSV.exists():
        raise FileNotFoundError(f'GT CSV 不存在: {GT_CSV}')

    output_path = OUT.resolve()
    protected_paths = [PROJECT_ROOT.resolve(), ORIGINAL_IMAGES.resolve(), GT_CSV.resolve()]
    if any(output_path == path or output_path in path.parents for path in protected_paths):
        raise ValueError(f'OWRT_DATASET_OUT 不能是项目或输入数据的自身/上级目录: {OUT}')

    if OUT.exists():
        shutil.rmtree(OUT)
        log(f'清理旧数据集 {OUT}')

    log('扫描原始 RGB 图像 ...')
    t0 = time.time()
    original_stems = scan_images(ORIGINAL_IMAGES)
    if not original_stems:
        raise FileNotFoundError(f'原始图像目录中没有可用图像: {ORIGINAL_IMAGES}')
    log(f'  共 {len(original_stems)} 张, 用时 {time.time()-t0:.1f}s')

    gt = read_gt(original_stems)
    if not gt:
        raise ValueError('没有任何 TinyWOD GT 与图像文件名匹配, 请检查 GT_CSV 和图像目录')

    # 只取有目标图像, 按 SEED=0 打乱后 70%/20%/10% 划分
    rng = random.Random(SEED)
    positive = sorted(gt.keys())
    rng.shuffle(positive)
    n = len(positive)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train = positive[:n_train]
    val = positive[n_train:n_train + n_val]
    test = positive[n_train + n_val:]
    log(f'  有目标图像 {n} 张 -> train {len(train)} ({TRAIN_RATIO:.0%}), '
        f'val {len(val)} ({VAL_RATIO:.0%}), test {len(test)} ({TEST_RATIO:.0%})')

    splits = {'train': train, 'val': val, 'test': test}

    n_box_total = 0
    for split, stems in splits.items():
        img_dir = OUT / 'images' / split
        lbl_dir = OUT / 'labels' / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        n_box = 0
        for i, stem in enumerate(stems, 1):
            original = original_stems[stem]
            hardlink_or_copy(original, img_dir / original.name)
            boxes = [instance['bbox'] for instance in gt.get(stem, [])]
            n_box += len(boxes)
            n_box_total += len(boxes)
            with open(lbl_dir / f'{stem}.txt', 'w', encoding='utf-8') as f:
                for x, y, w, h in boxes:
                    cx, cy = (x + w / 2) / IMG_W, (y + h / 2) / IMG_H
                    f.write(f'0 {cx:.6f} {cy:.6f} {w / IMG_W:.6f} {h / IMG_H:.6f}\n')
            if i % 2000 == 0 or i == len(stems):
                rate = i / max(time.time() - t0, 1e-6)
                log(f'  {split} {i}/{len(stems)} (速率 {rate:.0f} 张/s), 累计框 {n_box}')

    manifest = {
        'dataset_version': DATASET_VERSION,
        'built_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'seed': SEED,
        'train_ratio': TRAIN_RATIO,
        'val_ratio': VAL_RATIO,
        'test_ratio': TEST_RATIO,
        'positive_images': n,
        'train_images': len(train),
        'val_images': len(val),
        'test_images': len(test),
        'gt_boxes_kept': n_box_total,
        'gt_boxes_dropped': 0,
        'original_images': str(ORIGINAL_IMAGES),
    }
    with open(OUT / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log(f'完成: {manifest}')
    return manifest


def ensure_data():
    """数据集缺失或配置不符时自动重建。"""
    manifest_path = OUT / 'manifest.json'
    need = True
    if manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text(encoding='utf-8'))
            need = (
                m.get('dataset_version') != DATASET_VERSION
                or m.get('seed') != SEED
                or m.get('train_ratio') != TRAIN_RATIO
                or m.get('val_ratio') != VAL_RATIO
                or not dataset_layout_matches(m)
            )
        except Exception:
            need = True
    if need:
        print('=' * 60)
        print('[train] 数据集不存在或配置变化, 开始构建 ...')
        print(f'[train] train/val/test = {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}, seed={SEED} (仅含目标图像)')
        print('=' * 60)
        t0 = time.time()
        m = build()
        print(f'[train] 数据集构建完成, 用时 {time.time()-t0:.1f}s, '
              f'train={m["train_images"]} val={m["val_images"]} test={m["test_images"]}')
    else:
        m = json.loads(manifest_path.read_text(encoding='utf-8'))
        print(f'[train] 数据集已就绪: train={m["train_images"]} val={m["val_images"]} '
              f'test={m["test_images"]} (构建于 {m["built_at"]})')
    return m


if __name__ == '__main__':
    ensure_data()
