"""VMINet source backbone + last-map simple pyramid + native Faster R-CNN bbox detector.

The uploaded vminet.py is imported unchanged. The detector is an explicit bbox
adaptation, not a claim to reproduce unpublished Mask R-CNN training code.
"""
from collections import OrderedDict
from contextlib import contextmanager
import time
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import TwoMLPHead,FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
from config import CLASS_NAMES

VARIANTS={
    'ti':dict(factory='vminet_Ti',base_dim=24,depths=(2,2,18,2),mlp_ratio=2),
    'xs':dict(factory='vminet_XS',base_dim=48,depths=(2,2,18,2),mlp_ratio=2),
    's':dict(factory='vminet_S',base_dim=48,depths=(2,2,18,2),mlp_ratio=4),
    'b':dict(factory='vminet_B',base_dim=96,depths=(2,2,18,2),mlp_ratio=2),
}
LOSS_WEIGHTS={'loss_classifier':1.,'loss_box_reg':1.,'loss_objectness':1.,'loss_rpn_box_reg':1.}


class VMINetFeatures(nn.Module):
    def __init__(self,variant,image_size):
        super().__init__()
        import vminet
        spec=VARIANTS[variant]
        self.image_size=image_size
        self.stage_sizes=tuple(image_size//stride for stride in (4,8,16,32))
        original=getattr(vminet,spec['factory'])(pretrained=False,size=list(self.stage_sizes),num_classes=1000)
        self.stem=original.stem; self.stages=original.stages; self.norm=original.norm
        self.out_channels=spec['base_dim']*8
        # The original classifier/pool is deliberately not registered in a detection model.

    def forward(self,x):
        if tuple(x.shape[-2:])!=(self.image_size,self.image_size):
            raise ValueError(f'VMINet 固定空间聚合需要 {self.image_size}×{self.image_size}，实际 {tuple(x.shape[-2:])}；'
                             '请使用本项目 letterbox 数据加载，不要让框架再次改尺寸。')
        x=self.stem(x)
        for stage in self.stages: x=stage(x)
        return self.norm(x)


class LayerNorm2d(nn.Module):
    def __init__(self,channels,eps=1e-6):
        super().__init__(); self.weight=nn.Parameter(torch.ones(channels)); self.bias=nn.Parameter(torch.zeros(channels)); self.eps=eps
    def forward(self,x):
        return F.layer_norm(x.permute(0,2,3,1),(x.shape[1],),self.weight,self.bias,self.eps).permute(0,3,1,2)


class SimpleFeaturePyramid(nn.Module):
    """Use only the final stride-32 VMINet map; generate P2..P6 through conv/deconv."""
    def __init__(self,in_channels,out_channels):
        super().__init__(); self.branches=nn.ModuleDict()
        for name,steps in [('p2',3),('p3',2),('p4',1),('p5',0)]:
            layers=[]; channels=in_channels
            for step in range(steps):
                next_channels=channels//2
                layers.append(nn.ConvTranspose2d(channels,next_channels,2,stride=2))
                channels=next_channels
                if step<steps-1: layers.extend([LayerNorm2d(channels),nn.GELU()])
            layers.extend([nn.Conv2d(channels,out_channels,1,bias=False),LayerNorm2d(out_channels),
                           nn.Conv2d(out_channels,out_channels,3,padding=1,bias=False),LayerNorm2d(out_channels)])
            self.branches[name]=nn.Sequential(*layers)
    def forward(self,x):
        values=OrderedDict((name,branch(x)) for name,branch in self.branches.items())
        values['p6']=F.max_pool2d(values['p5'],kernel_size=1,stride=2)
        return values


class BackboneWithPyramid(nn.Module):
    def __init__(self,variant,image_size,out_channels):
        super().__init__()
        self.vminet=VMINetFeatures(variant,image_size)
        self.pyramid=SimpleFeaturePyramid(self.vminet.out_channels,out_channels)
        self.out_channels=out_channels
    def forward(self,x): return self.pyramid(self.vminet(x))


@contextmanager
def sampled_loss_mode(detector):
    """Enable native RPN/ROI loss branches without training the backbone or its BN.

    Assignment to these two flags is intentionally non-recursive. The native
    RPN/TwoMLP heads contain no BN/dropout; all original backbone BN stays eval.
    """
    before=(detector.rpn.training,detector.roi_heads.training)
    try:
        detector.rpn.training=True; detector.roi_heads.training=True
        yield
    finally:
        detector.rpn.training,detector.roi_heads.training=before


def _sync(device):
    if device.type=='cuda': torch.cuda.synchronize(device)


class VMINetDetector(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.loss_weights=dict(LOSS_WEIGHTS)
        backbone=BackboneWithPyramid(cfg.variant,cfg.image_size,cfg.pyramid_channels)
        anchors=AnchorGenerator(tuple((size,) for size in cfg.anchor_sizes),
                                tuple(tuple(cfg.anchor_ratios) for _ in cfg.anchor_sizes))
        pool=MultiScaleRoIAlign(featmap_names=['p2','p3','p4','p5'],output_size=cfg.roi_pool_size,
                               sampling_ratio=cfg.roi_sampling_ratio)
        box_head=TwoMLPHead(cfg.pyramid_channels*cfg.roi_pool_size**2,cfg.roi_hidden_dim)
        predictor=FastRCNNPredictor(cfg.roi_hidden_dim,len(CLASS_NAMES)+1)  # 0=background, 1..N=前景
        self.detector=FasterRCNN(backbone,num_classes=None,min_size=cfg.image_size,max_size=cfg.image_size,
            rpn_anchor_generator=anchors,box_roi_pool=pool,box_head=box_head,box_predictor=predictor,
            rpn_pre_nms_top_n_train=cfg.rpn_pre_nms_train,rpn_post_nms_top_n_train=cfg.rpn_post_nms_train,
            rpn_pre_nms_top_n_test=cfg.rpn_pre_nms_test,rpn_post_nms_top_n_test=cfg.rpn_post_nms_test,
            rpn_batch_size_per_image=cfg.rpn_batch_size,box_batch_size_per_image=cfg.roi_batch_size,
            rpn_nms_thresh=cfg.rpn_nms_iou,box_nms_thresh=cfg.nms_iou,
            box_score_thresh=cfg.score_threshold,box_detections_per_img=cfg.max_detections)
        self.detector.rpn.min_size=0.0  # Retain positive-area tiny RPN proposals.

    def forward(self,images,targets=None):
        return self.detector(images,targets)

    @torch.no_grad()
    def evaluate_with_losses(self,images,targets=None):
        if self.training or self.detector.training:
            raise RuntimeError('evaluate_with_losses 必须在 model.eval() 下使用，防止更新 BatchNorm。')
        model=self.detector; device=images[0].device
        original_sizes=[tuple(image.shape[-2:]) for image in images]
        _sync(device); start=time.perf_counter()
        image_list,transformed_targets=model.transform(images,targets)
        features=model.backbone(image_list.tensors)
        proposals,_=model.rpn(image_list,features,None)
        detections,_=model.roi_heads(features,proposals,image_list.image_sizes,None)
        detections=model.transform.postprocess(detections,image_list.image_sizes,original_sizes)
        _sync(device); inference_seconds=time.perf_counter()-start
        start=time.perf_counter(); losses={}
        if transformed_targets is not None:
            with sampled_loss_mode(model):
                sampled_proposals,rpn_losses=model.rpn(image_list,features,transformed_targets)
                _,roi_losses=model.roi_heads(features,sampled_proposals,image_list.image_sizes,transformed_targets)
                losses={**roi_losses,**rpn_losses}
        _sync(device); loss_seconds=time.perf_counter()-start
        return detections,losses,dict(inference_seconds=inference_seconds,loss_seconds=loss_seconds)
