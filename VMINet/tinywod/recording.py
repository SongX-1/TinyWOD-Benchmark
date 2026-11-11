"""Complete native bbox/RPN losses, batch timing, metrics and durable CSV records."""
from pathlib import Path
from .common import CSV,resource_snapshot

METRIC_KEYS=['precision','recall','f1','ap50','ap','apsmall','apmedium','aplarge','ar100',
             'confidence','tp','fp','fn','instances','images','max_dets']
RESOURCE_FIELDS=['cpu_rss_gb','allocated_gb','reserved_gb','peak_allocated_gb','resource_note']
ALIASES={'box_loss':'loss_box_reg','cls_loss':'loss_classifier','rpn_obj':'loss_objectness','rpn_box':'loss_rpn_box_reg'}
RESULT_FIELDS=['epoch','start_time','end_time',*[f'{p}/{k}' for p in ('train','val') for k in ('loss',*ALIASES)],
    *METRIC_KEYS,'lr/pg0','lr/pg1','lr_min','lr_max','train_seconds','val_seconds','evaluation_seconds','epoch_seconds',
    'inference_ms','fps','peak_allocated_gb','params','gflops','gflops_partial','best_epoch','is_best',
    'input_height','input_width','variant','inference_scope']
SUMMARY_FIELDS=['epoch','train_loss','eval_loss','eval_precision','eval_recall','eval_f1','eval_ap50','eval_ap','eval_apsmall',
                'lr_pg0','lr_pg1','train_seconds','val_seconds','epoch_seconds','best_epoch','is_best']


def summary_row(row):
    """Keep the original timm summary.csv train_/eval_ prefix convention for detection."""
    return dict(epoch=row['epoch'],train_loss=row['train/loss'],eval_loss=row['val/loss'],
        **{f'eval_{key}':row[key] for key in ('precision','recall','f1','ap50','ap','apsmall')},
        lr_pg0=row['lr/pg0'],lr_pg1=row['lr/pg1'],
        **{key:row[key] for key in ('train_seconds','val_seconds','epoch_seconds','best_epoch','is_best')})


class Recorder:
    def __init__(self,output,cfg,device):
        self.output=Path(output);self.device=device;m=self.output/'metrics'
        loss_fields=['epoch','phase','batch','global_step','name','branch','raw_loss','native_loss','weight','weighted_loss','optimized']
        self.losses=CSV(m/'losses.csv',loss_fields,['epoch','phase','batch','name'])
        self.batches=CSV(m/'batches.csv',['epoch','phase','batch','global_step','images','instances','image_ids',
            'input_height','input_width','data_seconds','forward_seconds','loss_seconds','backward_seconds',
            'optimizer_seconds','batch_seconds','loss','grad_norm','optimizer_step_skipped',*RESOURCE_FIELDS],['epoch','phase','batch'])
        self.lrs=CSV(m/'learning_rates.csv',['epoch','batch','global_step','group','lr','weight_decay'],['epoch','batch','group'])
        self.epoch_losses=CSV(m/'epoch_losses.csv',['epoch','phase','name','branch','raw_loss','native_loss','weight','weighted_loss','optimized','batches'],['epoch','phase','name'])
        self.metrics=CSV(m/'metrics.csv',['epoch','stage','split',*METRIC_KEYS],['epoch','stage','split'])
        self.classes=CSV(m/'per_class_metrics.csv',['epoch','stage','split','class_id','class_name','images','instances',
            'tp','fp','fn','precision','recall','f1','ap50','ap','apsmall','apmedium','aplarge','confidence'],['epoch','stage','split','class_id'])
        self.images=CSV(m/'per_image_metrics.csv',['epoch','stage','split','image_id','file_name','precision','recall','f1','tp','fp','fn','instances'],
                        ['epoch','stage','split','image_id'])
        self.resources=CSV(m/'resources.csv',['epoch','phase',*RESOURCE_FIELDS],['epoch','phase'])
        self.timing=CSV(m/'epoch_timing.csv',['epoch','start_time','end_time','train_seconds','val_seconds',
            'evaluation_seconds','epoch_seconds','train_batches','val_batches'],['epoch'])
        self.checkpoints=CSV(m/'checkpoint_events.csv',['epoch','kind','path','seconds','bytes'],['epoch','kind'])
        self.results=CSV(self.output/'results.csv',RESULT_FIELDS,['epoch'])
        self.summary=CSV(self.output/'summary.csv',SUMMARY_FIELDS,['epoch'])

    def loss_rows(self,epoch,phase,batch,step,losses,weights):
        rows=[]
        for name,value in sorted(losses.items()):
            raw=float(value.detach().item()) if hasattr(value,'detach') else float(value)
            weight=weights.get(name)
            rows.append(dict(epoch=epoch,phase=phase,batch=batch,global_step=step,name=name,
                branch='rpn' if name in ('loss_objectness','loss_rpn_box_reg') else 'roi',
                raw_loss=raw,native_loss=raw,weight=weight,weighted_loss=raw*weight if weight is not None else None,
                optimized=weight is not None))
        return rows

    def batch(self,epoch,phase,batch,step,losses,weights,details,optimizer=None):
        self.losses.extend(self.loss_rows(epoch,phase,batch,step,losses,weights))
        self.batches.append(dict(epoch=epoch,phase=phase,batch=batch,global_step=step,**details,**resource_snapshot(self.device)))
        if optimizer is not None:
            self.lrs.extend(dict(epoch=epoch,batch=batch,global_step=step,group=i,lr=g['lr'],weight_decay=g['weight_decay'])
                            for i,g in enumerate(optimizer.param_groups))

    def epoch_loss(self,epoch,phase,means,weights,batches):
        rows=self.loss_rows(epoch,phase,0,0,means,weights)
        self.epoch_losses.extend({**{k:v for k,v in row.items() if k in self.epoch_losses.fields},'batches':batches} for row in rows)
        self.resources.append(dict(epoch=epoch,phase=phase,**resource_snapshot(self.device)))

    def evaluation(self,epoch,stage,split,headline,classes,images):
        prefix=dict(epoch=epoch,stage=stage,split=split)
        self.metrics.append(dict(**prefix,**headline))
        self.classes.extend(dict(**prefix,**row) for row in classes)
        self.images.extend(dict(**prefix,**row) for row in images)


def loss_summary(values,phase):
    return {f'{phase}/loss':values.get('weighted_total'),**{f'{phase}/{k}':values.get(v) for k,v in ALIASES.items()}}


def loss_console(values,phase):
    return {f'{phase}/{k}':round(v,7) for k,v in values.items() if k!='weighted_total'}
