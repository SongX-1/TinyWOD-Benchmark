# -*- coding: utf-8 -*-
"""DINO预测接入用户OWT的IoU匹配、AP、APsmall实现，另保留原生COCO评价。"""
from pathlib import Path
import numpy as np
import torch

from experiment_config import CLASS_NAMES
from .owt_matching import OWTMatcher, box_iou
from .owt_metrics import ap_per_class, plot_mc_curve, plot_pr_curve, compute_ap


class DetectionAccumulator:
    def __init__(self, coco_gt):
        self.coco = coco_gt
        self.matcher = OWTMatcher()
        self.stats, self.small_stats = [], []
        self.image_counts = np.zeros(len(CLASS_NAMES), dtype=int)
        self.seen = 0
        self.confusion = np.zeros((len(CLASS_NAMES)+1, len(CLASS_NAMES)+1), dtype=int)

    def update(self, results, image_ids):
        for result, image_id in zip(results, image_ids):
            p = {k:v.detach().cpu() for k,v in result.items()}
            order = p['scores'].argsort(descending=True)
            det = torch.cat([p['boxes'].float(), p['scores'][:,None].float(),
                             p['labels'][:,None].float()],1)[order]
            anns = self.coco.imgToAnns.get(int(image_id), [])
            labels = []
            for a in anns:
                x,y,w,h = a['bbox']
                labels.append([a['category_id'],x,y,x+w,y+h])
            labels = torch.tensor(labels, dtype=torch.float32).reshape(-1,5)
            self.seen += 1
            for cls in labels[:,0].unique().int().tolist():
                self.image_counts[cls] += 1
            correct = self.matcher._process_batch(det, labels)
            small, ignore, small_cls = self.matcher._process_small_batch(det, labels)
            self.stats.append((correct.numpy(), det[:,4].numpy(), det[:,5].numpy(), labels[:,0].numpy()))
            self.small_stats.append((small.numpy(), ignore.numpy(), det[:,4].numpy(), det[:,5].numpy(), small_cls.numpy()))
            self._confusion(det, labels)

    def _confusion(self, det, labels):
        # OWT阈值conf=0.25/IoU=0.45；补齐参考实现无任何匹配时漏记FP的边界。
        det = det[det[:,4] > .25]
        iou = box_iou(labels[:,1:], det[:,:4]).numpy()
        matches = np.array(np.nonzero(iou > .45)).T
        if len(matches):
            score = iou[matches[:,0],matches[:,1]]
            matches = matches[np.argsort(-score)]
            matches = matches[np.unique(matches[:,1],return_index=True)[1]]
            matches = matches[np.argsort(-iou[matches[:,0],matches[:,1]])]
            matches = matches[np.unique(matches[:,0],return_index=True)[1]]
        gt_match = {int(g):int(p) for g,p in matches}
        used = set(gt_match.values())
        for i,c in enumerate(labels[:,0].int().tolist()):
            self.confusion[int(det[gt_match[i],5]) if i in gt_match else len(CLASS_NAMES), c] += 1
        for i,c in enumerate(det[:,5].int().tolist()):
            if i not in used:
                self.confusion[c,len(CLASS_NAMES)] += 1

    @staticmethod
    def _calculate(stats, small=False):
        if not stats:
            return None, None
        data = [np.concatenate(x,axis=0) for x in zip(*stats)]
        if not len(data[-1]):
            return None, data
        if small:
            tp,ignore,conf,pcls,tcls = data
            r = ap_per_class(tp,conf,pcls,tcls,names=dict(enumerate(CLASS_NAMES)),ignore=ignore)
        else:
            r = ap_per_class(*data,names=dict(enumerate(CLASS_NAMES)))
        return r,data

    def finalize(self, epoch, recorder, plot_dir=None):
        result, data = self._calculate(self.stats)
        small, small_data = self._calculate(self.small_stats,True)
        counts = np.bincount(data[-1].astype(int),minlength=len(CLASS_NAMES)) if data is not None else np.zeros(len(CLASS_NAMES),dtype=int)
        small_counts = np.bincount(small_data[-1].astype(int),minlength=len(CLASS_NAMES)) if small_data is not None else np.zeros(len(CLASS_NAMES),dtype=int)
        present = {int(c):i for i,c in enumerate(result[6])} if result is not None else {}
        small_present = {int(c):i for i,c in enumerate(small[6])} if small is not None else {}
        metrics = dict(zip(['metrics/precision(B)','metrics/recall(B)','metrics/mAP50(B)',
                            'metrics/mAP50-95(B)','metrics/mAP50-95-small(B)','metrics/F1(B)'],
                           [float(result[2].mean()) if result is not None else 0.,
                            float(result[3].mean()) if result is not None else 0.,
                            float(result[5][:,0].mean()) if result is not None else 0.,
                            float(result[5].mean()) if result is not None else 0.,
                            float(small[5].mean()) if small is not None else 0.,
                            float(result[4].mean()) if result is not None else 0.]))
        metrics.update({'dataset/val-images':self.seen,'dataset/val-instances':int(counts.sum()),
                        'dataset/val-small-instances':int(small_counts.sum()),
                        'dataset/val-has-ground-truth':bool(counts.sum())})
        print(('%22s'+'%11s'*7) % ('Class','Images','Instances','Box(P','R','AP50','AP','APsmall'),flush=True)
        print(('%22s'+'%11i'*2+'%11.5g'*5) % ('all',self.seen,int(counts.sum()),
              *[metrics[k] for k in ['metrics/precision(B)','metrics/recall(B)','metrics/mAP50(B)',
                                     'metrics/mAP50-95(B)','metrics/mAP50-95-small(B)']]),flush=True)
        for c,name in enumerate(CLASS_NAMES):
            i,j = present.get(c),small_present.get(c)
            p,r,f1 = (float(result[k][i]) for k in (2,3,4)) if i is not None else (0.,0.,0.)
            ap50 = float(result[5][i,0]) if i is not None else 0.
            ap = float(result[5][i].mean()) if i is not None else 0.
            aps = float(small[5][j].mean()) if j is not None else 0.
            tp = float(result[0][i]) if i is not None else 0.
            fp = float(result[1][i]) if i is not None else 0.
            recorder.csv('metrics/per_class.csv',{'epoch':epoch,'class_id':c,'class_name':name,
                'images':int(self.image_counts[c]),'instances':int(counts[c]),'small_instances':int(small_counts[c]),
                'precision':p,'recall':r,'F1':f1,'AP50':ap50,'AP50-95':ap,'APsmall':aps,
                'TP_at_best_F1':tp,'FP_at_best_F1':fp,'has_ground_truth':bool(counts[c])})
            print(('%22s'+'%11i'*2+'%11.5g'*5) % (name,int(self.image_counts[c]),int(counts[c]),p,r,ap50,ap,aps),flush=True)
        for p in range(len(CLASS_NAMES)+1):
            for t in range(len(CLASS_NAMES)+1):
                recorder.csv('metrics/confusion_matrix.csv',{'epoch':epoch,'predicted':CLASS_NAMES[p] if p<len(CLASS_NAMES) else 'background',
                    'true':CLASS_NAMES[t] if t<len(CLASS_NAMES) else 'background','count':int(self.confusion[p,t])})
        if not counts.sum():
            print('[val] 验证集没有真值，OWT兼容列为0；不能据此声称模型精度为0或有效。',flush=True)
        if plot_dir is not None:
            self._save_plots_and_curves(epoch,recorder,Path(plot_dir),result,data)
        return metrics

    def _save_plots_and_curves(self,epoch,recorder,folder,result,data):
        import matplotlib.pyplot as plt
        folder.mkdir(parents=True,exist_ok=True)
        if result is not None:
            names = dict(enumerate(CLASS_NAMES[int(c)] for c in result[6]))
            for idx,label in [(7,'P'),(8,'R'),(9,'F1')]:
                plot_mc_curve(result[10],result[idx],folder/f'{label}_curve.png',names,ylabel=label)
            # 参考OWT的AP积分结果，修正其绘图插值参数次序；不改变评价计算。
            prec = []
            tp,conf,pcls,tcls = data
            order = np.argsort(-conf)
            tp,conf,pcls = tp[order],conf[order],pcls[order]
            for i,c in enumerate(result[6]):
                selected = pcls == c
                n = (tcls==c).sum()
                tpc = tp[selected,0].cumsum()
                fpc = (1-tp[selected,0]).cumsum()
                recall = tpc/(n+1e-16)
                precision = tpc/(tpc+fpc+1e-16)
                if len(recall):
                    _,mpre,mrec = compute_ap(recall,precision)
                    prec.append(np.interp(result[10],mrec,mpre))
                else:
                    prec.append(np.zeros(1000))
                for k,x in enumerate(result[10]):
                    recorder.csv('metrics/curves.csv',{'epoch':epoch,'class_id':int(c),'confidence':float(x),
                        'precision':float(result[7][i,k]),'recall':float(result[8][i,k]),
                        'F1':float(result[9][i,k]),'recall_axis':float(x),'PR_precision':float(prec[-1][k])})
            plot_pr_curve(result[10],prec,result[5],folder/'PR_curve.png',names)
        for normalize in (False,True):
            cm = self.confusion.astype(float)
            if normalize:
                cm = cm/(cm.sum(0,keepdims=True)+1e-9)
            fig,ax = plt.subplots(figsize=(9,8),tight_layout=True)
            im = ax.imshow(cm,cmap='Blues')
            names = CLASS_NAMES+['background']
            ax.set_xticks(range(len(CLASS_NAMES)+1),labels=names,rotation=70)
            ax.set_yticks(range(len(CLASS_NAMES)+1),labels=names)
            ax.set_xlabel('True');ax.set_ylabel('Predicted')
            for y in range(len(CLASS_NAMES)+1):
                for x in range(len(CLASS_NAMES)+1):
                    ax.text(x,y,f'{cm[y,x]:.2f}' if normalize else str(int(cm[y,x])),ha='center',va='center',fontsize=8)
            fig.colorbar(im,ax=ax)
            fig.savefig(folder/('confusion_matrix_normalized.png' if normalize else 'confusion_matrix.png'),dpi=150)
            plt.close(fig)
