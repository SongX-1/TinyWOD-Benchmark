"""Atomic full-state last/periodic checkpoints and isolated, exact epoch-boundary resume."""
import csv
import os
from pathlib import Path
import random
import shutil
import numpy as np
import torch
from dataclasses import asdict
from .common import digest, write_csv, write_json


def training_signature(cfg):
    skip={"output_root","resume_from","device","cpu_threads","prediction_images","benchmark_images",
          "benchmark_warmup","benchmark_repeats","make_plots","yolo_root","coco_root","dataset_mode",
          "coco_annotations","coco_image_dirs","verify_images","hash_images","accept_unverified_yolo_names",
          "gpu_reserve_fraction"}
    return digest({k:v for k,v in asdict(cfg).items() if k not in skip})


def rng_state(generator, worker_generator):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                sampler=generator.get_state(), workers=worker_generator.get_state())


def restore_rng(state, generator, worker_generator, device):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    generator.set_state(state["sampler"]); worker_generator.set_state(state["workers"])
    if device.type == "cuda" and state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("续训可见 GPU 数量不同，不能声称恢复相同 CUDA RNG。")
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_save(path, state):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+".tmp")
    with temp.open("wb") as file:
        torch.save(state,file); file.flush(); os.fsync(file.fileno())
    os.replace(temp,path)


def cpu_state(model):
    return {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}


def load_resume(cfg, info, spec, output, device):
    if not cfg.resume_from: return None
    from config import resolve
    path=resolve(cfg.resume_from)
    # Only this project's trusted local full-state checkpoints; never download/eval remote code.
    checkpoint=torch.load(path,map_location="cpu",weights_only=False)
    if checkpoint.get("format") != "HDETR_SDS_FULL_V1":  # 保持旧值以兼容既有检查点
        raise ValueError("resume_from 必须为本项目生成的 last.pth 或 epoch_XXXX.pth；best.pth 只用于推理。")
    checks={"dataset_fingerprint":info["fingerprint"],"architecture_fingerprint":spec["architecture_fingerprint"],
            "training_signature":training_signature(cfg)}
    for key,value in checks.items():
        if checkpoint.get(key) != value: raise ValueError(f"续训 {key} 不一致，禁止静默改变数据/网络/超参数。")
    if checkpoint.get("ops_backend") != spec["ops"]["actual"]:
        raise ValueError("续训算子 backend 不一致；请使用与原实验相同的 backend。")
    if checkpoint.get("device_type") != device.type:
        raise ValueError("续训设备类型不同；CPU/GPU 迁移应另开实验，避免声称可复现恢复。")
    completed=int(checkpoint["completed_epoch"])
    if not 0 <= completed < cfg.epochs: raise ValueError(f"检查点已完成 {completed}/{cfg.epochs}，无需再次训练。")
    parent=path.parent.parent
    if not (parent/"results.csv").is_file(): raise FileNotFoundError(f"续训需要同目录训练记录：{parent}/results.csv")
    # Fork a new run. Never append over the old experiment or copy uncommitted epochs.
    for csv_path in [parent/"results.csv",*(parent/"metrics").glob("*.csv")]:
        with csv_path.open(encoding="utf-8-sig",newline="") as file:
            reader=csv.DictReader(file); fields=reader.fieldnames; rows=list(reader)
        if fields and "epoch" in fields:
            rows=[row for row in rows if int(row["epoch"]) <= completed and row.get("stage","epoch") != "final"
                  and row.get("phase","") in ("","train","val")]
            write_csv(output/csv_path.relative_to(parent),rows,fields)
    with (output/"results.csv").open(encoding="utf-8-sig",newline="") as file:
        epochs=[int(row["epoch"]) for row in csv.DictReader(file)]
    if epochs != list(range(1,completed+1)):
        raise ValueError("父实验已提交 Epoch 的 results.csv 不连续或有重复，拒绝伪造/补齐记录。")
    for folder in (parent/"metrics").glob("epoch_*"):
        if folder.is_dir() and int(folder.name.split("_")[-1]) <= completed:
            shutil.copytree(folder,output/"metrics"/folder.name)
    for name in ("complexity.json","complexity.csv"):
        if (parent/"metrics"/name).is_file(): shutil.copy2(parent/"metrics"/name,output/"metrics"/name)
    if (parent/"log.txt").is_file():
        import json
        lines=[]
        for line in (parent/"log.txt").read_text(encoding="utf-8").splitlines():
            item=json.loads(line)
            if int(item["epoch"]) <= completed: lines.append(json.dumps(item,ensure_ascii=False))
        (output/"log.txt").write_text('\n'.join(lines)+'\n',encoding="utf-8")
    write_json(output/"inputs"/"resume_lineage.json",dict(parent=str(parent),checkpoint=str(path),
        committed_epoch=completed,policy="new directory; committed epochs only; complete model/optimizer/scheduler/scaler/RNG"))
    print(f"[断点恢复] 父实验={parent} | 已提交={completed} | 下一轮={completed+1} | 本次独立目录={output}",flush=True)
    return checkpoint
