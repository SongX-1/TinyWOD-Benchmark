"""标准 COCO AP 与明确置信度工作点的 P/R/F1，保留评价原始张量。"""
import contextlib
import io
from pathlib import Path

import numpy as np
from config import CLASS_NAMES
from .common import read_json, write_csv, write_json


def _mean(values):
    valid = np.asarray(values); valid = valid[valid >= 0]
    return float(valid.mean()) if valid.size else None


def _pr(tp, fp, gt):
    p = tp/(tp+fp) if tp+fp else 0.
    r = tp/gt if gt else 0.
    return p,r,2*p*r/(p+r) if p+r else 0.


def evaluate(annotation, predictions, output, confidence=None, make_plots=True):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    capture=io.StringIO()
    with contextlib.redirect_stdout(capture):
        gt=COCO(str(annotation))
        if predictions:
            dt=gt.loadRes(predictions)
        else:
            dt=COCO(); dt.dataset={'images':list(gt.imgs.values()),'categories':list(gt.cats.values()),
                                   'annotations':[]}; dt.createIndex()
        ev=COCOeval(gt,dt,'bbox'); ev.params.maxDets=[1,10,100]
        ev.params.catIds=list(range(len(CLASS_NAMES)))
        ev.evaluate(); ev.accumulate(); ev.summarize()
    (output/'coco_evaluation.txt').write_text(capture.getvalue(),encoding='utf-8')
    write_json(output/'predictions.json',predictions)
    prediction_rows=[dict(image_id=p['image_id'],file_name=gt.imgs[p['image_id']]['file_name'],
        class_id=p['category_id'],class_name=CLASS_NAMES[p['category_id']],score=p['score'],
        x=p['bbox'][0],y=p['bbox'][1],width=p['bbox'][2],height=p['bbox'][3]) for p in predictions]
    write_csv(output/'predictions.csv',prediction_rows,
              ['image_id','file_name','class_id','class_name','score','x','y','width','height'])
    prec=ev.eval['precision']; rec=ev.eval['recall']; scores=ev.eval['scores']
    np.savez_compressed(output/'coco_precision_recall.npz',precision=prec,recall=rec,scores=scores,
                        iou_thresholds=ev.params.iouThrs,recall_thresholds=ev.params.recThrs)
    # evalImgs 在 maxDets=100 上有原生 crowd/area ignore，诊断 P/R 直接复用其匹配。
    area_all=ev.params.areaRng[0]
    items=[e for e in ev.evalImgs if e is not None and e['aRng']==area_all]
    matches={}; counts=np.zeros(len(CLASS_NAMES),dtype=np.int64)
    for e in items:
        c=e['category_id']; counts[c]+=int(np.count_nonzero(~np.asarray(e['gtIgnore'],dtype=bool)))
        good=~np.asarray(e['dtIgnore'][0],dtype=bool)
        score=np.asarray(e['dtScores'])[good]
        tp=(np.asarray(e['dtMatches'][0])[good]>0).astype(np.int64)
        matches[(e['image_id'],c)]=(score,tp,int(np.count_nonzero(~np.asarray(e['gtIgnore'],dtype=bool))))

    pooled = {}
    def at(threshold, category=None, image_id=None):
        key=(category,image_id)
        if key not in pooled:
            selected=[v for (i,c),v in matches.items() if
                      (category is None or category==c) and (image_id is None or image_id==i)]
            s=np.concatenate([v[0] for v in selected]) if selected else np.empty(0)
            t=np.concatenate([v[1] for v in selected]) if selected else np.empty(0,dtype=int)
            order=np.argsort(s,kind='stable'); s=s[order]; t=t[order]
            pooled[key]=(s,np.r_[0,np.cumsum(t)],sum(v[2] for v in selected))
        s,cumulative,ng=pooled[key]
        index=int(np.searchsorted(s,threshold,side='left'))
        tp=int(cumulative[-1]-cumulative[index]); fp=len(s)-index-tp
        p,r,f1=_pr(tp,fp,ng)
        return dict(precision=p,recall=r,f1=f1,tp=tp,fp=fp,fn=ng-tp,instances=ng)

    thresholds=np.linspace(0.,1.,1001)
    global_curves=[dict(confidence=float(t),**at(float(t))) for t in thresholds]
    if confidence is None:
        # 相同最大 F1 取最高阈值；避免大量无贡献低分 FP。
        index=max(range(len(thresholds)),key=lambda j:(global_curves[j]['f1'],j))
        confidence=float(thresholds[index]); threshold_source='selected_on_this_validation_split'
    else:
        confidence=float(confidence); threshold_source='frozen_from_best_validation'
    headline=at(confidence)
    headline.update(ap50=float(ev.stats[1]) if ev.stats[1]>=0 else None,
        ap=float(ev.stats[0]) if ev.stats[0]>=0 else None,
        apsmall=float(ev.stats[3]) if ev.stats[3]>=0 else None,
        apmedium=float(ev.stats[4]) if ev.stats[4]>=0 else None,
        aplarge=float(ev.stats[5]) if ev.stats[5]>=0 else None,
        ar100=float(ev.stats[8]) if ev.stats[8]>=0 else None,
        confidence=confidence,images=len(gt.imgs),max_dets=100)
    class_rows=[]; curves=[]; prs=[]; ap_iou=[]
    for c,name in enumerate(CLASS_NAMES):
        row=dict(class_id=c,class_name=name,images=sum(any(a['category_id']==c and not a.get('iscrowd',0)
            for a in gt.imgToAnns[i]) for i in gt.imgs),**at(confidence,category=c),
            ap50=_mean(prec[0,:,c,0,-1]),ap=_mean(prec[:,:,c,0,-1]),
            apsmall=_mean(prec[:,:,c,1,-1]),apmedium=_mean(prec[:,:,c,2,-1]),
            aplarge=_mean(prec[:,:,c,3,-1]),confidence=confidence)
        class_rows.append(row)
        for t in thresholds: curves.append(dict(class_id=c,class_name=name,confidence=float(t),**at(float(t),category=c)))
        for j,iou in enumerate(ev.params.iouThrs):
            ap_iou.append(dict(class_id=c,class_name=name,iou=float(iou),ap=_mean(prec[j,:,c,0,-1])))
            for r,value,score in zip(ev.params.recThrs,prec[j,:,c,0,-1],scores[j,:,c,0,-1]):
                prs.append(dict(class_id=c,class_name=name,iou=float(iou),recall=float(r),
                                precision=float(value) if value>=0 else None,
                                confidence=float(score) if score>=0 else None))
    image_rows=[dict(image_id=i,file_name=gt.imgs[i]['file_name'],**at(confidence,image_id=i)) for i in gt.imgs]
    write_csv(output/'metrics.csv',[headline],headline.keys())
    write_csv(output/'per_class.csv',class_rows,class_rows[0].keys())
    write_csv(output/'per_image.csv',image_rows,['image_id','file_name','precision','recall','f1','tp','fp','fn','instances'])
    write_csv(output/'confidence_curves.csv',curves,['class_id','class_name','confidence','precision','recall','f1','tp','fp','fn','instances'])
    write_csv(output/'global_confidence_curve.csv',global_curves,global_curves[0].keys())
    write_csv(output/'pr_curves.csv',prs,['class_id','class_name','iou','recall','precision','confidence'])
    write_csv(output/'ap_by_iou.csv',ap_iou,['class_id','class_name','iou','ap'])
    cm=confusion_matrix(gt,predictions,confidence)
    labels=[*CLASS_NAMES,'background']
    write_csv(output/'confusion_matrix.csv',[{'true_class':name,**dict(zip(labels,map(int,cm[i])))}
        for i,name in enumerate(labels)],['true_class',*labels])
    write_json(output/'metrics.json',headline)
    write_json(output/'metric_definitions.json',{
        'AP':'pycocotools COCO bbox AP; IoU=.50:.05:.95; maxDets=[1,10,100]; original image coordinates',
        'APsmall':'COCO original annotation area < 32^2 pixels; unavailable=-1 exported as blank/null',
        'P_R_F1':'micro across classes; native COCO IoU=.5 dtMatches/dtIgnore; maxDets=100; noncrowd GT',
        'confidence':confidence,'confidence_source':threshold_source,
        'threshold_selection':'maximum global micro F1 on 1001 thresholds [0,1], tie chooses highest threshold',
        'no_positive_predictions':'P/R/F1=0 when GT exists; absent categories AP blank',
        'confusion_matrix':'rows=true, columns=predicted; class-agnostic greedy IoU>=.5; crowd detections ignored; diagnostic, not AP matching',
        'test_policy':'test threshold must be frozen from best validation; no threshold tuning on test'})
    if make_plots: plot_evaluation(output,ev,class_rows,global_curves,cm)
    return headline,class_rows,image_rows


def _iou(a,b):
    a=np.asarray(a,dtype=float).reshape(-1,4); b=np.asarray(b,dtype=float).reshape(-1,4)
    if not len(a) or not len(b): return np.zeros((len(a),len(b)))
    aa=a.copy(); bb=b.copy(); aa[:,2:]+=aa[:,:2]; bb[:,2:]+=bb[:,:2]
    inter=np.maximum(0,np.minimum(aa[:,None,2:],bb[None,:,2:])-np.maximum(aa[:,None,:2],bb[None,:,:2])).prod(2)
    return inter/np.maximum(a[:,2:].prod(1)[:,None]+b[:,2:].prod(1)[None,:]-inter,1e-12)


def confusion_matrix(gt,predictions,threshold):
    matrix=np.zeros((len(CLASS_NAMES)+1, len(CLASS_NAMES)+1),dtype=np.int64); grouped={i:[] for i in gt.imgs}
    for p in predictions:
        if p['score']>=threshold: grouped[p['image_id']].append(p)
    for i,preds in grouped.items():
        preds=sorted(preds,key=lambda p:-p['score'])[:100]
        anns=[a for a in gt.imgToAnns[i] if not a.get('iscrowd',0)]
        crowds=[a for a in gt.imgToAnns[i] if a.get('iscrowd',0)]
        overlap=_iou([a['bbox'] for a in anns],[p['bbox'] for p in preds])
        used_a=set(); used_p=set()
        for ai,pi in sorted(zip(*np.where(overlap>=.5)),key=lambda q:-overlap[q]):
            if ai in used_a or pi in used_p: continue
            matrix[anns[ai]['category_id'],preds[pi]['category_id']]+=1; used_a.add(ai); used_p.add(pi)
        for ai,a in enumerate(anns):
            if ai not in used_a: matrix[a['category_id'],len(CLASS_NAMES)]+=1
        for pi,p in enumerate(preds):
            if pi in used_p: continue
            # crowd 使用交集/检测面积，与 COCO crowd IoU 分母一致。
            ignored=False
            for a in crowds:
                if a['category_id']!=p['category_id']: continue
                x,y,w,h=p['bbox']; u,v,s,t=a['bbox']
                inter=max(0,min(x+w,u+s)-max(x,u))*max(0,min(y+h,v+t)-max(y,v))
                if inter/max(w*h,1e-12)>=.5: ignored=True; break
            if not ignored: matrix[len(CLASS_NAMES),p['category_id']]+=1
    return matrix


def plot_evaluation(output,ev,classes,curves,cm):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,5))
    for c,row in enumerate(classes):
        p=ev.eval['precision'][0,:,c,0,-1]; good=p>=0
        if good.any(): ax.plot(ev.params.recThrs[good],p[good],label=row['class_name'])
    ax.set(xlabel='Recall',ylabel='Precision (IoU=0.5)',xlim=(0,1),ylim=(0,1)); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output/'PR_curve.png',dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    for key in ('precision','recall','f1'): ax.plot([r['confidence'] for r in curves],[r[key] for r in curves],label=key)
    ax.set(xlabel='Confidence',ylabel='Micro metric',xlim=(0,1),ylim=(0,1)); ax.legend()
    fig.tight_layout(); fig.savefig(output/'confidence_curves.png',dpi=150); plt.close(fig)
    for normalized in (False,True):
        data=cm.astype(float)
        if normalized: data=data/np.maximum(data.sum(1,keepdims=True),1)
        fig,ax=plt.subplots(figsize=(9,7)); im=ax.imshow(data,cmap='Blues')
        labels=[*CLASS_NAMES,'background']; ax.set_xticks(range(len(labels)),labels,rotation=40,ha='right')
        ax.set_yticks(range(len(labels)),labels); ax.set(xlabel='Predicted',ylabel='True')
        for i in range(len(labels)):
            for j in range(len(labels)): ax.text(j,i,f'{data[i,j]:.2f}' if normalized else str(int(data[i,j])),ha='center',va='center',fontsize=8,
                                      color='white' if data[i,j]>data.max()/2 else 'black')
        fig.colorbar(im,ax=ax); fig.tight_layout()
        fig.savefig(output/('confusion_matrix_normalized.png' if normalized else 'confusion_matrix.png'),dpi=150); plt.close(fig)


def plot_training(output):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    with (output/'results.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    if not rows: return
    x=[int(r['epoch']) for r in rows]
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    groups=[('Training loss',['train/box_loss','train/cls_loss','train/l1_loss']),
            ('Validation loss',['val/box_loss','val/cls_loss','val/l1_loss']),
            ('Detection metrics',['precision','recall','f1','ap50','ap','apsmall']),
            ('Learning rate',['lr_min','lr_max']),('Duration (seconds)',['train_seconds','val_seconds','evaluation_seconds']),
            ('Inference time (ms/image)',['inference_ms'])]
    for ax,(title,keys) in zip(axes.flat,groups):
        for key in keys:
            valid=[(a,r[key]) for a,r in zip(x,rows) if r.get(key) not in ('',None)]
            if valid: ax.plot([a for a,b in valid],[float(b) for a,b in valid],marker='o',markersize=3,label=key)
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.xaxis.set_major_locator(MaxNLocator(integer=True)); ax.grid(alpha=.2)
        if len(x)<=12: ax.set_xticks(x)
        if len(x)==1: ax.set_xlim(x[0]-.5,x[0]+.5)
        if title=='Detection metrics': ax.set_ylim(0,1)
        if ax.lines: ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output/'results'/'training_curves.png',dpi=160); plt.close(fig)
