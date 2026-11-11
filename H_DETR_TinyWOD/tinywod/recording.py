"""Durable, uniquely keyed epoch/batch/loss/metric records; no invented loss columns."""
from pathlib import Path
import json
from .common import CSV, resource_snapshot

METRIC_KEYS = ["precision","recall","f1","ap50","ap","apsmall","apmedium","aplarge","ar100",
               "confidence","tp","fp","fn","instances","images","max_dets"]
LOSS_COLUMNS = [f"{phase}/{name}" for phase in ("train","val") for name in
                ("loss","box_loss","cls_loss","l1_loss","box_oo","cls_oo","l1_oo","box_om","cls_om","l1_om")]
RESULT_FIELDS = ["epoch","start_time","end_time",*LOSS_COLUMNS,*METRIC_KEYS,"lr/pg0","lr/pg1","lr/pg2",
                 "lr_min","lr_max","train_seconds","val_seconds","evaluation_seconds","epoch_seconds",
                 "inference_ms","fps","peak_allocated_gb","params","gflops","gflops_partial",
                 "best_epoch","is_best","ops_backend","input_height","input_width"]
RESOURCE_FIELDS = ["cpu_rss_gb","allocated_gb","reserved_gb","peak_allocated_gb","resource_note"]


class Recorder:
    def __init__(self, output, cfg, device):
        self.output=Path(output); self.cfg=cfg; self.device=device
        m=self.output/"metrics"
        self.losses=CSV(m/"losses.csv",
            ["epoch","phase","batch","global_step","name","branch","layer","raw_loss","lambda_factor",
             "native_loss","weight","weighted_loss","optimized"], ["epoch","phase","batch","name"])
        self.batches=CSV(m/"batches.csv",["epoch","phase","batch","global_step","images","instances","image_ids",
            "input_height","input_width","data_seconds","forward_seconds","loss_seconds","backward_seconds",
            "optimizer_seconds","batch_seconds","loss","grad_norm","optimizer_step_skipped",*RESOURCE_FIELDS],
            ["epoch","phase","batch"])
        self.lrs=CSV(m/"learning_rates.csv",["epoch","batch","global_step","group","lr","weight_decay"],
                     ["epoch","batch","group"])
        self.epoch_losses=CSV(m/"epoch_losses.csv",["epoch","phase","name","raw_loss","lambda_factor",
            "native_loss","weight","weighted_loss","optimized","batches"],["epoch","phase","name"])
        self.metrics=CSV(m/"metrics.csv",["epoch","stage","split",*METRIC_KEYS],["epoch","stage","split"])
        self.classes=CSV(m/"per_class_metrics.csv",["epoch","stage","split","class_id","class_name","images",
            "instances","tp","fp","fn","precision","recall","f1","ap50","ap","apsmall","apmedium","aplarge","confidence"],
            ["epoch","stage","split","class_id"])
        self.images=CSV(m/"per_image_metrics.csv",["epoch","stage","split","image_id","file_name",
            "precision","recall","f1","tp","fp","fn","instances"],["epoch","stage","split","image_id"])
        self.resources=CSV(m/"resources.csv",["epoch","phase",*RESOURCE_FIELDS],["epoch","phase"])
        self.timing=CSV(m/"epoch_timing.csv",["epoch","start_time","end_time","train_seconds","val_seconds",
            "evaluation_seconds","epoch_seconds","train_batches","val_batches"],["epoch"])
        self.checkpoints=CSV(m/"checkpoint_events.csv",["epoch","kind","path","seconds","bytes"],["epoch","kind"])
        self.results=CSV(self.output/"results.csv", RESULT_FIELDS,["epoch"])

    def loss_rows(self, epoch, phase, batch, step, loss_dict, weight_dict):
        rows=[]
        for name,value in sorted(loss_dict.items()):
            native=float(value.detach().item()) if hasattr(value,"detach") else float(value)
            many=name.endswith("_one2many"); factor=self.cfg.lambda_one2many if many else 1.
            base=name.removesuffix("_one2many"); tokens=base.split("_")
            layer="encoder" if base.endswith("_enc") else f"aux_{tokens[-1]}" if tokens[-1].isdigit() else "main"
            weight=weight_dict.get(name)
            rows.append(dict(epoch=epoch,phase=phase,batch=batch,global_step=step,name=name,
                branch="one2many" if many else "one2one",layer=layer,raw_loss=native/factor,
                lambda_factor=factor,native_loss=native,weight=weight,
                weighted_loss=native*weight if weight is not None else None,optimized=weight is not None))
        return rows

    def batch(self, epoch, phase, batch, step, loss_dict, weight_dict, details, optimizer=None):
        rows=self.loss_rows(epoch,phase,batch,step,loss_dict,weight_dict)
        self.losses.extend(rows)
        resource=resource_snapshot(self.device)
        self.batches.append(dict(epoch=epoch,phase=phase,batch=batch,global_step=step,**details,**resource))
        if optimizer is not None:
            self.lrs.extend(dict(epoch=epoch,batch=batch,global_step=step,group=i,lr=g["lr"],weight_decay=g["weight_decay"])
                            for i,g in enumerate(optimizer.param_groups))

    def epoch_loss(self, epoch, phase, means, weights, batches):
        rows=self.loss_rows(epoch,phase,0,0,means,weights)
        self.epoch_losses.extend({**{k:v for k,v in row.items() if k in self.epoch_losses.fields},"batches":batches}
                                 for row in rows)
        self.resources.append(dict(epoch=epoch,phase=phase,**resource_snapshot(self.device)))

    def evaluation(self, epoch, stage, split, headline, classes, images):
        prefix=dict(epoch=epoch,stage=stage,split=split)
        self.metrics.append(dict(**prefix,**headline))
        self.classes.extend(dict(**prefix,**r) for r in classes)
        self.images.extend(dict(**prefix,**r) for r in images)


def loss_summary(values, phase):
    # GIoU is the native box-overlap loss; bbox is L1, not DFL.
    result={f"{phase}/loss":sum(v for k,v in values.items() if k == "weighted_total")}
    for alias,native in (("box","loss_giou"),("cls","loss_ce"),("l1","loss_bbox")):
        one=values.get(native); many=values.get(native+"_one2many")
        result[f"{phase}/{alias}_oo"]=one; result[f"{phase}/{alias}_om"]=many
        result[f"{phase}/{alias}_loss"]=(one or 0)+(many or 0) if one is not None else None
    return result


def loss_console(values, phase):
    return {f"{phase}/{k}":round(v,7) for k,v in sorted(values.items()) if k != "weighted_total"}
