# -*- coding: utf-8 -*-
"""汇总真实CSV；没有OWT结果时不伪造OWT指标。"""
import csv
from pathlib import Path
from .recording import append_csv


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return [{k.strip():v.strip() for k,v in row.items()} for row in csv.DictReader(f)]


def build_comparison(config):
    out=Path(config.output_root).parent/'comparisons'/'comparison.csv'
    out.parent.mkdir(parents=True,exist_ok=True)
    temp=out.with_suffix('.csv.tmp')
    if temp.exists():temp.unlink()
    count=0
    for summary in sorted(Path(config.output_root).glob('*run/summary.csv')):
        for row in read_rows(summary):
            append_csv(temp,{'run':summary.parent.name,'source_csv':str(summary),
                             'conditions_verified':'see run_config.json + dataset_manifest.json',**row})
            count+=1
    if config.owt_results_csv:
        path=Path(config.owt_results_csv).expanduser().resolve()
        rows=read_rows(path)
        if not rows:raise ValueError(f'OWT CSV为空: {path}')
        def fitness(row):
            return .1*float(row['metrics/mAP50(B)'])+.9*float(row['metrics/mAP50-95(B)'])
        for variant,row in [('best',max(rows,key=fitness)),('last',rows[-1])]:
            append_csv(temp,{'run':'OWT','model':'OWRT-DETR','variant':variant,'source_csv':str(path),
                             'conditions_verified':'external CSV only; verify seed,split,imgsz,epochs in OWT args',**row})
            count+=1
    if temp.is_file():temp.replace(out)
    print(f'[comparison] 汇总{count}行真实结果 -> {out}；'
          +('包含指定OWT CSV。' if config.owt_results_csv else '未配置OWT结果CSV，未填造OWT指标。'),flush=True)
    return out
