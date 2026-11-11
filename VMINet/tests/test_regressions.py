import contextlib
import copy
import csv
import gc
import io
import json
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from config import CLASS_NAMES
from tinywod.common import CSV,read_json,sha256,write_json
from tinywod.data import prepare_dataset
from tests.fixtures import create_fixture


class DatasetChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(); cls.root=Path(cls.temp.name)
        cls.fixture=create_fixture(cls.root/'original')
    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()
    def setUp(self):
        self.case=self.root/self._testMethodName
        self.data=self.case/'dataset'; shutil.copytree(self.root/'original'/'dataset',self.data)
        self.cfg=replace(self.fixture,yolo_root=str(self.data))
    def prepare(self,cfg=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare_dataset(cfg or self.cfg,self.case/'inputs')
    def test_exact_duplicate_readonly_and_fingerprint(self):
        original=self.data/'labels/train/0000.txt'; before=sha256(original)
        info=self.prepare(); self.assertEqual(info['splits']['train']['duplicate_boxes_removed'],1)
        self.assertEqual(sha256(original),before)
        self.assertEqual(info['splits']['train']['instances'],6)
        self.assertFalse((self.case/'inputs/dataset_view/images').exists())
        with (self.case/'inputs/duplicate_annotations.csv').open(encoding='utf-8-sig') as f:
            self.assertEqual(len(list(csv.DictReader(f))),1)
    def test_adjacent_small_boxes_not_deduplicated(self):
        path=self.data/'labels/train/0000.txt'; path.write_text('0 .5 .5 .08 .1\n0 .5001 .5 .08 .1\n')
        info=self.prepare(); self.assertEqual(info['splits']['train']['duplicate_boxes_removed'],0)
        self.assertEqual(info['splits']['train']['instances'],7)
    def test_missing_train_label_is_error(self):
        (self.data/'labels/train/0001.txt').unlink()
        with self.assertRaises(FileNotFoundError): self.prepare()
    def test_partial_test_labels_is_error(self):
        (self.data/'labels/test').mkdir(); (self.data/'labels/test/0000.txt').write_text('0 .5 .5 .1 .1\n')
        with self.assertRaises(ValueError): self.prepare()
    def test_invalid_class_is_error(self):
        (self.data/'labels/val/0000.txt').write_text('5 .5 .5 .1 .1\n')
        with self.assertRaises(ValueError): self.prepare()
    def test_nonfinite_box_is_error(self):
        (self.data/'labels/val/0000.txt').write_text('0 nan .5 .1 .1\n')
        with self.assertRaises(ValueError): self.prepare()
    def test_yaml_wrong_order_is_error(self):
        write_json(self.data/'manifest.json',{'names':list(reversed(CLASS_NAMES))})
        with self.assertRaises(ValueError): self.prepare()
    def test_cross_split_physical_image_is_error(self):
        target=self.data/'images/val/0000.png'; target.unlink()
        target.hardlink_to(self.data/'images/train/0000.png')
        with self.assertRaises(ValueError): self.prepare()
    def test_require_test_gt(self):
        with self.assertRaises(ValueError): self.prepare(replace(self.cfg,test_policy='required'))
    def test_yolo_unlabeled_test_manifest_written(self):
        info=self.prepare(); test=info['splits']['test']
        self.assertFalse(test['labeled']); self.assertEqual(len(read_json(test['annotation'])['images']),2)
    def test_coco_semantic_mapping_and_test_subset(self):
        info=self.prepare()
        coco=self.case/'coco'; (coco/'annotations').mkdir(parents=True)
        for split in ('train','val','test'):
            shutil.copytree(self.data/'images'/split,coco/'images'/split)
            payload=read_json(info['splits'][split]['annotation'])
            payload['categories']=[{'id':i+1,'name':n} for i,n in enumerate(CLASS_NAMES)]
            for a in payload['annotations']: a['category_id']+=1
            if split=='test': payload['images']=payload['images'][:1]; payload['annotations']=[]
            write_json(coco/'annotations'/f'instances_{split}.json',payload)
        cfg=replace(self.cfg,dataset_mode='coco',coco_root=str(coco))
        with contextlib.redirect_stdout(io.StringIO()): result=prepare_dataset(cfg,self.case/'coco_inputs')
        self.assertEqual(result['splits']['val']['class_counts'],[1]*5)
        self.assertEqual(result['splits']['test']['images'],1)
        self.assertFalse(result['splits']['test']['labeled'])
        self.assertEqual(len(read_json(result['splits']['test']['annotation'])['images']),1)


class EvaluationChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.annotation=self.root/'annotation.json'
        self.payload={'info':{},'images':[{'id':1,'file_name':'a.jpg','width':200,'height':200}],
            'categories':[{'id':i,'name':n} for i,n in enumerate(CLASS_NAMES)],
            'annotations':[{'id':1,'image_id':1,'category_id':0,'bbox':[10,10,8,8],'area':64,'iscrowd':0}]}
        self.pred=[{'image_id':1,'category_id':0,'bbox':[10,10,8,8],'score':.9}]
    def tearDown(self): self.tmp.cleanup()
    def run_eval(self,pred=None,confidence=None):
        from tinywod.evaluation import evaluate
        write_json(self.annotation,self.payload)
        return evaluate(self.annotation,self.pred if pred is None else pred,self.root/'metrics',confidence,False)
    def test_perfect_ap_small(self):
        result,classes,images=self.run_eval()
        for key in ('ap','ap50','apsmall','precision','recall','f1'): self.assertAlmostEqual(result[key],1.)
        self.assertIsNone(classes[1]['ap']); self.assertEqual(images[0]['tp'],1)
        self.assertTrue((self.root/'metrics/coco_precision_recall.npz').is_file())
    def test_crowd_prediction_ignored(self):
        self.payload['annotations'].append({'id':2,'image_id':1,'category_id':0,'bbox':[100,100,30,30],
                                             'area':900,'iscrowd':1})
        pred=self.pred+[{'image_id':1,'category_id':0,'bbox':[105,105,4,4],'score':.99}]
        result,_,_=self.run_eval(pred)
        self.assertAlmostEqual(result['ap'],1.); self.assertEqual(result['fp'],0)
    def test_no_small_gt_is_blank(self):
        self.payload['annotations'][0].update(bbox=[0,0,100,100],area=10000)
        self.pred[0]['bbox']=[0,0,100,100]
        result,_,_=self.run_eval(); self.assertIsNone(result['apsmall'])
    def test_test_threshold_frozen(self):
        self.pred[0]['score']=.8
        result,_,_=self.run_eval(confidence=.9)
        self.assertEqual(result['confidence'],.9); self.assertEqual(result['recall'],0)
        self.assertAlmostEqual(result['ap'],1.)
    def test_empty_predictions_real_zero(self):
        result,_,_=self.run_eval([])
        self.assertEqual(result['ap'],0); self.assertEqual(result['f1'],0)


class PersistenceChecks(unittest.TestCase):
    def test_csv_unique_keys_and_bom(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'a.csv'; writer=CSV(path,['epoch','value'],['epoch'])
            writer.append({'epoch':1,'value':.123})
            with self.assertRaises(ValueError): writer.append({'epoch':1,'value':9})
            reloaded=CSV(path,['epoch','value'],['epoch'])
            with self.assertRaises(ValueError): reloaded.append({'epoch':1,'value':9})
            self.assertTrue(path.read_bytes().startswith(b'\xef\xbb\xbf'))
    def test_csv_incompatible_header_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'a.csv'; CSV(path,['epoch','value'],['epoch'])
            with self.assertRaises(ValueError): CSV(path,['epoch','different'],['epoch'])



class GeometryChecks(unittest.TestCase):
    def test_letterbox_rounding_inverse_and_subpixel_boxes(self):
        import torch
        from PIL import Image
        from tinywod.loading import letterbox,inverse_letterbox
        image=Image.new('RGB',(151,97));boxes=torch.tensor([[15.,10.,15.5,10.75],[0.,0.,151.,97.]])
        tensor,converted,scale,pad=letterbox(image,boxes,64)
        self.assertEqual(tuple(tensor.shape),(3,64,64));self.assertNotEqual(float(scale[0]),float(scale[1]))
        target=dict(resize_scale=scale,pad=pad,orig_size=torch.tensor([97,151]))
        torch.testing.assert_close(inverse_letterbox(converted,target),boxes.double(),atol=1e-4,rtol=1e-5)
        self.assertTrue(((converted[:,2:]-converted[:,:2])>0).all())
    def test_horizontal_flip_keeps_positive_box_geometry(self):
        import torch
        from PIL import Image
        from tinywod.loading import letterbox
        boxes=torch.tensor([[5.,6.,8.,9.]])
        _,a,_,_=letterbox(Image.new('RGB',(128,96)),boxes,64,flip=False)
        _,b,_,_=letterbox(Image.new('RGB',(128,96)),boxes,64,flip=True)
        torch.testing.assert_close(b[:,[0,2]],64-a[:,[2,0]])
        torch.testing.assert_close(b[:,[1,3]],a[:,[1,3]])
    def test_foreground_zero_is_shifted_and_last_class_is_five(self):
        from tinywod.loading import make_dataset
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cfg=create_fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):info=prepare_dataset(cfg,root/'inputs')
            dataset=make_dataset(cfg,'val',info)
            self.assertEqual(dataset[0][1]['labels'].tolist(),[1])
            self.assertEqual(dataset[4][1]['labels'].tolist(),[5])


class BackboneChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2);torch.backends.nnpack.set_flags(False)
    def test_original_vminet_file_is_byte_identical(self):
        original=read_json(ROOT/'verification/original_files_sha256.json')
        self.assertEqual(sha256(ROOT/'vminet.py'),original['vminet.py'])
    def test_true_variant_channels_depths_and_ordered_parameter_counts(self):
        from models_detection.vminet_detector import VMINetFeatures
        counts=[]
        for name,width,ratio in [('ti',24,2),('xs',48,2),('s',48,4),('b',96,2)]:
            model=VMINetFeatures(name,64)
            self.assertEqual([len(stage)-1 for stage in model.stages],[2,2,18,2])
            self.assertEqual([stage[0].conv.out_channels for stage in model.stages],[width,width*2,width*4,width*8])
            self.assertEqual(model.stages[0][1].f1.conv.out_channels,width*ratio)
            self.assertEqual(model.stages[0][1].sum.in_features,16*16)
            self.assertFalse(hasattr(model,'head'))
            counts.append(sum(p.numel() for p in model.parameters()))
            del model;gc.collect()
        self.assertEqual(counts,sorted(set(counts)))
    def test_canonical_features_reproduce_original_classifier_exactly(self):
        import torch,vminet
        from models_detection.vminet_detector import VMINetFeatures
        original=vminet.vminet_Ti(pretrained=False);features=VMINetFeatures('ti',224)
        features.load_state_dict({k:v for k,v in original.state_dict().items() if not k.startswith('head.')},strict=True)
        original.eval();features.eval();image=torch.randn(1,3,224,224)
        with torch.no_grad():
            a=original(image);b=original.head(torch.flatten(original.avgpool(features(image)),1))
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    def test_spatial_sum_identity_and_constant_sum_scale(self):
        import torch
        from tinywod.modeling import interpolate_sum
        values=torch.randn(1,56*56)
        self.assertIs(interpolate_sum(values,values.shape),values)
        resized=interpolate_sum(torch.ones(1,56*56),(1,160*160))
        self.assertAlmostEqual(float(resized.sum()),56*56,places=2)
        with self.assertRaises(ValueError):interpolate_sum(torch.ones(1,20),(1,100))
    def test_original_checkpoint_namespace_and_strict_load_resize_audit(self):
        import argparse,torch,vminet
        from models_detection.vminet_detector import VMINetFeatures
        from tinywod.modeling import load_local_backbone
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'local.pth.tar';source=vminet.vminet_Ti(pretrained=False)
            torch.save(dict(state_dict=source.state_dict(),args=argparse.Namespace(model='vminet_Ti')),path)
            model=VMINetFeatures('ti',64);rows,key=load_local_backbone(model,path,'state_dict')
            self.assertEqual(key,'state_dict')
            self.assertEqual(sum(row['action'].startswith('spatial_') for row in rows),24)
            self.assertEqual(sum(row['action']=='discard_classifier' for row in rows),2)
            torch.testing.assert_close(source.stem[0].conv.weight,model.stem[0].conv.weight,atol=0,rtol=0)
            wrong=VMINetFeatures('xs',64)
            with self.assertRaisesRegex(ValueError,'不匹配'):load_local_backbone(wrong,path,'state_dict')
    def test_missing_backbone_tensor_not_silently_ignored(self):
        import torch,vminet
        from models_detection.vminet_detector import VMINetFeatures
        from tinywod.modeling import load_local_backbone
        with tempfile.TemporaryDirectory() as folder:
            source=vminet.vminet_Ti(pretrained=False).state_dict();source.pop('stages.0.1.stack')
            path=Path(folder)/'bad.pth';torch.save(source,path)
            with self.assertRaisesRegex(ValueError,'缺少必要参数'):load_local_backbone(VMINetFeatures('ti',64),path,'state_dict')


class DetectorChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2);torch.backends.nnpack.set_flags(False)
        cls.tmp=tempfile.TemporaryDirectory();cls.cfg=create_fixture(Path(cls.tmp.name))
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def model(self):
        from models_detection.vminet_detector import VMINetDetector
        return VMINetDetector(self.cfg)
    def inputs(self,empty=False):
        import torch
        images=[torch.rand(3,64,64),torch.rand(3,64,64)]
        targets=[dict(boxes=torch.zeros(0,4) if empty else torch.tensor([[20.,20.,24.,26.]]),
            labels=torch.zeros(0,dtype=torch.long) if empty else torch.tensor([1])) for _ in images]
        return images,targets
    def test_default_36_b_six_class_predictor_and_last_map_pyramid(self):
        import torch
        from config import Experiment
        from models_detection.vminet_detector import VMINetDetector
        cfg=Experiment();model=VMINetDetector(cfg)
        self.assertEqual(cfg.epochs,36);self.assertEqual(cfg.variant,'b')
        self.assertEqual(model.detector.roi_heads.box_predictor.cls_score.out_features,6)
        self.assertEqual(model.detector.roi_heads.box_predictor.bbox_pred.out_features,24)
        self.assertEqual(model.detector.backbone.vminet.out_channels,768)
        self.assertEqual(model.detector.backbone.vminet.stage_sizes,(160,80,40,20))
        self.assertEqual(model.detector.rpn.anchor_generator.sizes,((8,),(16,),(32,),(64,),(128,)))
        self.assertGreater(sum(p.numel() for p in model.parameters()),40_000_000)
        del model;gc.collect()
    def test_p2_to_p6_actual_shapes(self):
        import torch
        model=self.model();model.eval()
        with torch.no_grad():out=model.detector.backbone(torch.zeros(1,3,64,64))
        self.assertEqual(list(out),['p2','p3','p4','p5','p6'])
        self.assertEqual([tuple(x.shape[-2:]) for x in out.values()],[(16,16),(8,8),(4,4),(2,2),(1,1)])
    def test_actual_four_losses_backward_reaches_original_attention(self):
        import torch
        model=self.model();images,targets=self.inputs();losses=model(images,targets)
        self.assertEqual(set(losses),{'loss_classifier','loss_box_reg','loss_objectness','loss_rpn_box_reg'})
        sum(losses.values()).backward()
        gradient=model.detector.backbone.vminet.stages[0][1].sum.weight.grad
        self.assertTrue(torch.isfinite(gradient).all());self.assertGreater(float(gradient.abs().sum()),0)
        self.assertIsNotNone(model.detector.roi_heads.box_predictor.cls_score.weight.grad)
    def test_background_training_is_finite(self):
        import torch
        model=self.model();images,targets=self.inputs(True);losses=model(images,targets)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        self.assertEqual(float(losses['loss_box_reg'].detach()),0.)
    def test_val_losses_keep_bn_and_predictions_unchanged_and_rng_isolated(self):
        import torch
        from tinywod.training import validation_randomness
        model=self.model();model.eval();images,targets=self.inputs()
        buffers={k:v.clone() for k,v in model.named_buffers()}
        with torch.no_grad():expected=model(images)
        rng=torch.get_rng_state().clone()
        with validation_randomness(torch.device('cpu'),12387):
            detections,losses,timing=model.evaluate_with_losses(images,targets)
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        for k,v in model.named_buffers():torch.testing.assert_close(v,buffers[k],rtol=0,atol=0)
        for a,b in zip(expected,detections):
            for name in ('boxes','labels','scores'):torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)
        self.assertEqual(len(losses),4);self.assertFalse(model.detector.rpn.training);self.assertFalse(model.detector.roi_heads.training)
        self.assertFalse(any(v.requires_grad for v in losses.values()))
        self.assertGreater(timing['inference_seconds'],0)
    def test_eval_without_gt_has_no_fabricated_losses(self):
        model=self.model();model.eval();images,_=self.inputs()
        detections,losses,_=model.evaluate_with_losses(images,None)
        self.assertEqual(losses,{})
        self.assertTrue(all(1<=int(label)<=5 for d in detections for label in d['labels']))
    def test_sampled_mode_restores_after_error(self):
        from models_detection.vminet_detector import sampled_loss_mode
        model=self.model();model.eval()
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with sampled_loss_mode(model.detector):
                self.assertFalse(model.detector.backbone.training)
                self.assertTrue(model.detector.rpn.training)
                raise RuntimeError('intentional')
        self.assertFalse(model.detector.rpn.training);self.assertFalse(model.detector.roi_heads.training)
    def test_native_inference_does_not_download_or_compile(self):
        import torch,importlib
        from unittest.mock import patch
        module=importlib.import_module('torchvision.ops.roi_align')
        model=self.model();model.eval();images,_=self.inputs()
        with patch('torch.hub.download_url_to_file',side_effect=AssertionError('download forbidden')),\
             patch('torch.compile',side_effect=AssertionError('compile forbidden')),\
             patch.object(module,'_roi_align',side_effect=AssertionError('lazy fallback forbidden')):
            with torch.no_grad():detections=model(images)
        self.assertEqual(len(detections),2)
    def test_wrong_fixed_input_rejected(self):
        import torch
        model=self.model();model.eval()
        # The backbone, rather than silent positional resizing, checks its actual input.
        with self.assertRaisesRegex(ValueError,'固定空间聚合'):
            model.detector.backbone.vminet(torch.zeros(1,3,96,96))


class RecordingChecks(unittest.TestCase):
    def test_csv_batch_transaction_and_atomic_update(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'a.csv';writer=CSV(path,['epoch','loss'],['epoch'])
            writer.extend([dict(epoch=1,loss=.1),dict(epoch=2,loss=.2)]);before=path.read_bytes()
            with self.assertRaises(ValueError):writer.extend([dict(epoch=3,loss=.3),dict(epoch=3,loss=.4)])
            self.assertEqual(path.read_bytes(),before)
            writer.update(dict(epoch=2,loss=.25))
            with path.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),2);self.assertEqual(rows[-1]['loss'],'0.25')
    def test_all_native_terms_and_mean_weighted_fields(self):
        import torch
        from config import Experiment
        from tinywod.recording import Recorder
        with tempfile.TemporaryDirectory() as folder:
            recorder=Recorder(Path(folder),Experiment(),torch.device('cpu'))
            rows=recorder.loss_rows(1,'train',1,1,{'loss_box_reg':torch.tensor(.5),'loss_objectness':torch.tensor(.25)},
                                    {'loss_box_reg':2.,'loss_objectness':1.})
            self.assertEqual(rows[0]['branch'],'roi');self.assertEqual(rows[0]['weighted_loss'],1.)
            self.assertEqual(rows[1]['branch'],'rpn')
    def test_missing_local_weight_failed_status_before_data_scan(self):
        from tinywod.runner import launch
        with tempfile.TemporaryDirectory() as folder:
            cfg=create_fixture(Path(folder));cfg=replace(cfg,pretrained=True,backbone_weights={'ti':str(Path(folder)/'missing.pth')})
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):code=launch(cfg)
            self.assertEqual(code,1)
            output=next((Path(folder)/'runs'/'VMINet-TI-FasterRCNN-SMOKE').iterdir())
            state=read_json(output/'run_state.json')
            self.assertEqual(state['status'],'failed');self.assertEqual(state['completed_epochs'],0)
            self.assertFalse((output/'inputs/dataset_view').exists())
            self.assertTrue((output/'logs/error.log').is_file())


if __name__=='__main__':unittest.main(verbosity=2)
