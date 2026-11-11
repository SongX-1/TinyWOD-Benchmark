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


class NativeModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)
        torch.backends.nnpack.set_flags(False)
        cls.tmp=tempfile.TemporaryDirectory(); cls.root=Path(cls.tmp.name)
        cls.cfg=create_fixture(cls.root)
    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()
    def build_small(self):
        import torch
        from models import build_model
        from tinywod.modeling import native_args,configure_ops
        cfg=replace(self.cfg,make_plots=False)
        model,criterion,post=build_model(native_args(cfg,torch.device('cpu')))
        configure_ops(model,cfg,torch.device('cpu'))
        return model,criterion,post
    def test_full_official_hybrid_architecture_and_offline_initialization(self):
        import torch
        from unittest.mock import patch
        from config import Experiment
        from models import build_model
        from tinywod.modeling import native_args,RECIPE
        cfg=Experiment()
        args=native_args(cfg,torch.device('cpu'))
        with patch('torch.hub.download_url_to_file',side_effect=AssertionError('network forbidden')):
            model,criterion,post=build_model(args)
        self.assertEqual(cfg.epochs,36); self.assertEqual(cfg.lr_drop,30)
        self.assertTrue((ROOT/RECIPE).is_file())
        self.assertEqual(criterion.num_classes,5)
        self.assertEqual(model.num_queries,1800); self.assertEqual(model.num_queries_one2one,300)
        self.assertEqual(model.transformer.encoder.num_layers,6); self.assertEqual(model.transformer.decoder.num_layers,6)
        self.assertTrue(model.two_stage); self.assertTrue(model.mixed_selection)
        self.assertTrue(model.transformer.decoder.look_forward_twice)
        self.assertEqual(model.num_feature_levels,4)
        self.assertTrue(all(layer.out_features==5 for layer in model.class_embed))
        self.assertFalse(args.pretrained_backbone)
        self.assertTrue(any('one2many' in k for k in criterion.weight_dict))
        self.assertGreater(sum(p.numel() for p in model.parameters()),40_000_000)
        del model; gc.collect()
    def test_local_resnet_state_dict_loads_strictly_without_network(self):
        import torch
        from torchvision.models import resnet50
        from tinywod.modeling import build_native,preflight_weights
        from unittest.mock import patch
        output=self.root/'local_backbone_load'
        for folder in ('inputs','logs','metrics','weights','results'): (output/folder).mkdir(parents=True,exist_ok=True)
        path=output/'generated_backbone.pth'
        source=resnet50(weights=None)
        expected=source.conv1.weight.detach().clone()
        torch.save(source.state_dict(),path); del source
        cfg=replace(self.cfg,pretrained=True,backbone_weights=str(path))
        preflight_weights(cfg)
        with patch('torch.hub.download_url_to_file',side_effect=AssertionError('network forbidden')):
            model,_,_,_,_,initialization=build_native(cfg,torch.device('cpu'),output)
        torch.testing.assert_close(model.backbone[0].body.conv1.weight,expected,rtol=0,atol=0)
        self.assertEqual(initialization['mode'],'custom_local_backbone')
        self.assertTrue(initialization['local_only'])
        del model; path.unlink(); gc.collect()
    def test_official_named_corrupted_weight_rejected(self):
        from tinywod.modeling import preflight_weights
        path=self.root/'resnet50-0676ba61.pth'; path.write_bytes(b'partial download')
        with self.assertRaisesRegex(ValueError,'SHA256'):
            preflight_weights(replace(self.cfg,pretrained=True,backbone_weights=str(path)))
        path.unlink()
    def test_activation_checkpoint_backward(self):
        import torch
        from models import build_model
        from tinywod.modeling import native_args
        from tinywod.hybrid import train_hybrid
        cfg=replace(self.cfg,use_checkpoint=True)
        model,criterion,_=build_model(native_args(cfg,torch.device('cpu')))
        out=model(torch.randn(1,3,96,128))
        losses=train_hybrid(out,[dict(labels=torch.tensor([0]),boxes=torch.tensor([[.5,.5,.1,.1]]))],2,criterion,1.)
        sum(v*criterion.weight_dict[k] for k,v in losses.items() if k in criterion.weight_dict).backward()
        self.assertTrue(torch.isfinite(model.transformer.encoder.layers[0].self_attn.value_proj.weight.grad).all())
    def test_real_hybrid_both_branch_gradients(self):
        import torch
        from tinywod.hybrid import train_hybrid
        model,criterion,post=self.build_small(); model.train()
        output=model(torch.randn(1,3,96,128))
        output['pred_logits'].retain_grad(); output['pred_logits_one2many'].retain_grad()
        target=[{'labels':torch.tensor([0]),'boxes':torch.tensor([[.5,.5,.08,.1]])}]
        losses=train_hybrid(output,target,2,criterion,1.)
        total=sum(v*criterion.weight_dict[k] for k,v in losses.items() if k in criterion.weight_dict)
        total.backward()
        self.assertGreater(float(output['pred_logits'].grad.abs().sum()),0)
        self.assertGreater(float(output['pred_logits_one2many'].grad.abs().sum()),0)
        for key in ('loss_ce_0','loss_ce_enc','loss_bbox_0_one2many','loss_giou_one2many'):
            self.assertIn(key,losses); self.assertTrue(torch.isfinite(losses[key]))
    def test_empty_background_targets_finite(self):
        import torch
        from tinywod.hybrid import train_hybrid
        model,criterion,_=self.build_small()
        out=model(torch.randn(1,3,96,128))
        losses=train_hybrid(out,[{'labels':torch.zeros(0,dtype=torch.long),'boxes':torch.zeros(0,4)}],2,criterion,1.)
        self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
    def test_one2one_inference_restores_even_on_exception(self):
        from tinywod.hybrid import one2one_inference
        model,_,_=self.build_small(); model.train()
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with one2one_inference(model):
                self.assertEqual(model.num_queries,10)
                self.assertEqual(model.transformer.two_stage_num_proposals,10)
                self.assertFalse(model.training)
                raise RuntimeError('intentional')
        self.assertEqual(model.num_queries,30)
        self.assertEqual(model.transformer.two_stage_num_proposals,30)
        self.assertTrue(model.training)
    def test_inference_excludes_one2many_queries(self):
        import torch
        from tinywod.hybrid import one2one_inference
        model,_,post=self.build_small()
        with torch.no_grad(),one2one_inference(model):
            out=model(torch.zeros(1,3,96,128))
            self.assertEqual(tuple(out['pred_logits'].shape),(1,10,5))
            self.assertEqual(out['pred_logits_one2many'].shape[1],0)
            pred=post['bbox'](out,torch.tensor([[96,128]]))
            self.assertEqual(len(pred[0]['scores']),20)
    def test_insufficient_proposals_fail_not_silently_change_network(self):
        import torch
        model,_,_=self.build_small()
        with self.assertRaisesRegex(ValueError,'queries|proposals|encoder positions'):
            model(torch.zeros(1,3,24,24))
    def test_native_unavailable_is_explicit(self):
        import torch
        from tinywod.modeling import configure_ops
        model,_,_=self.build_small()
        with self.assertRaisesRegex(RuntimeError,'native'):
            configure_ops(model,replace(self.cfg,ops_backend='native'),torch.device('cpu'))
    def test_subpixel_boxes_are_not_discarded(self):
        import torch
        from PIL import Image
        from datasets.coco import ConvertCocoPolysToMask
        _,target=ConvertCocoPolysToMask()(Image.new('RGB',(100,100)),dict(image_id=1,annotations=[
            dict(category_id=0,bbox=[10.,10.,.5,.75],area=.375,iscrowd=0)]))
        self.assertEqual(len(target['labels']),1)
        torch.testing.assert_close(target['boxes'],torch.tensor([[10.,10.,10.5,10.75]]))
    def test_original_coordinate_and_normalization_pipeline(self):
        import torch
        from tinywod.loading import make_transforms
        from datasets.coco import ConvertCocoPolysToMask
        from PIL import Image
        image,target=ConvertCocoPolysToMask()(Image.new('RGB',(128,96)),dict(image_id=1,annotations=[
            dict(category_id=0,bbox=[58.88,43.2,10.24,9.6],area=98.304,iscrowd=0)]))
        image,target=make_transforms(self.cfg,False)(image,target)
        torch.testing.assert_close(target['boxes'],torch.tensor([[.5,.5,.08,.1]]),atol=1e-6,rtol=1e-5)
        self.assertEqual(tuple(image.shape),(3,96,128))
    def test_lambda_applied_once_and_repeat_does_not_mutate_source(self):
        import torch
        from tinywod.hybrid import train_hybrid
        calls=[]
        def criterion(out,targets):
            calls.append([len(t['labels']) for t in targets])
            return {'loss_ce':torch.tensor(2.)}
        outputs=dict(pred_logits_one2many=torch.zeros(1),pred_boxes_one2many=torch.zeros(1),aux_outputs_one2many=[])
        targets=[dict(labels=torch.tensor([1,2]),boxes=torch.zeros(2,4))]
        losses=train_hybrid(outputs,targets,6,criterion,.5)
        self.assertEqual(calls,[[2],[12]])
        self.assertEqual(len(targets[0]['labels']),2)
        self.assertEqual(float(losses['loss_ce_one2many']),1.)


class OperatorChecks(unittest.TestCase):
    def test_chunked_forward_and_backward_match_original_reference(self):
        import torch
        from models.ops.modules.ms_deform_attn import MSDeformAttn
        torch.manual_seed(7)
        a=MSDeformAttn(d_model=32,n_levels=2,n_heads=4,n_points=2)
        b=copy.deepcopy(a); a.backend=b.backend='pytorch'; a.query_chunk_size=2; b.query_chunk_size=0
        q=torch.randn(1,5,32,requires_grad=True); x=torch.randn(1,5,32,requires_grad=True)
        q2=q.detach().clone().requires_grad_(); x2=x.detach().clone().requires_grad_()
        refs=torch.rand(1,5,2,2); shapes=torch.tensor([[2,2],[1,1]]); starts=torch.tensor([0,4])
        o=a(q,refs,x,shapes,starts); p=b(q2,refs,x2,shapes,starts)
        torch.testing.assert_close(o,p,atol=1e-6,rtol=1e-5)
        o.square().sum().backward(); p.square().sum().backward()
        torch.testing.assert_close(q.grad,q2.grad,atol=1e-5,rtol=1e-4)
        torch.testing.assert_close(x.grad,x2.grad,atol=1e-5,rtol=1e-4)
        for pa,pb in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(pa.grad,pb.grad,atol=1e-5,rtol=1e-4)
    def test_original_grid_sample_reference_gradcheck(self):
        import torch
        from models.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch
        torch.manual_seed(1)
        value=torch.randn(1,5,1,2,dtype=torch.double,requires_grad=True)
        locations=(torch.rand(1,2,1,2,2,2,dtype=torch.double)*.4+.3).requires_grad_()
        weights=torch.randn(1,2,1,2,2,dtype=torch.double,requires_grad=True)
        def core(v,l,w): return ms_deform_attn_core_pytorch(v,[(2,2),(1,1)],l,w)
        self.assertTrue(torch.autograd.gradcheck(core,(value,locations,weights),eps=1e-6,atol=1e-4))


class RecordingChecks(unittest.TestCase):
    def test_csv_batch_flush_rejects_duplicate_transaction(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'a.csv'; writer=CSV(path,['epoch','loss'],['epoch'])
            writer.extend([dict(epoch=1,loss=1),dict(epoch=2,loss=2)])
            before=path.read_bytes()
            with self.assertRaises(ValueError): writer.extend([dict(epoch=3,loss=3),dict(epoch=3,loss=4)])
            self.assertEqual(path.read_bytes(),before)
            writer.update(dict(epoch=2,loss=4))
            with path.open(encoding='utf-8-sig') as file: rows=list(csv.DictReader(file))
            self.assertEqual(len(rows),2); self.assertEqual(rows[-1]['loss'],'4')
    def test_raw_native_weighted_and_missing_validation_branch(self):
        import torch
        from config import Experiment
        from tinywod.recording import Recorder,loss_summary
        with tempfile.TemporaryDirectory() as folder:
            r=Recorder(Path(folder),Experiment(lambda_one2many=.5),torch.device('cpu'))
            rows=r.loss_rows(1,'train',1,1,{'loss_ce_one2many':torch.tensor(2.)},{'loss_ce_one2many':2.})
            self.assertEqual(rows[0]['raw_loss'],4.); self.assertEqual(rows[0]['native_loss'],2.)
            self.assertEqual(rows[0]['weighted_loss'],4.)
            v=loss_summary({'loss_giou':1.,'loss_ce':2.,'loss_bbox':3.,'weighted_total':9.},'val')
            self.assertIsNone(v['val/box_om']); self.assertEqual(v['val/l1_loss'],3.)
            self.assertFalse(any('dfl' in k for k in v))
    def test_r50_import_and_forward_without_unrelated_frameworks(self):
        import subprocess
        code=r"""
import importlib.abc,sys
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'mmcv','mmdet','mmengine','timm','wandb','transformers','addict','yapf','MultiScaleDeformableAttention'}:
            raise ModuleNotFoundError('Deliberately blocked optional framework: '+fullname)
sys.meta_path.insert(0,Deny())
import torch
from config import Experiment
from tinywod.modeling import native_args
from models import build_model
from unittest.mock import patch
torch.set_num_threads(2);torch.backends.nnpack.set_flags(False)
cfg=Experiment(pretrained=False,enc_layers=1,dec_layers=2,dim_feedforward=256,num_queries_one2one=10,num_queries_one2many=20,max_detections=20)
with patch('torch.hub.download_url_to_file',side_effect=AssertionError('download forbidden')):
    model,criterion,_=build_model(native_args(cfg,torch.device('cpu')))
    with torch.no_grad(): output=model(torch.zeros(1,3,96,128))
assert tuple(output['pred_logits'].shape)==(1,10,5)
print('MINIMAL_R50_FORWARD_PASSED')
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        self.assertIn('MINIMAL_R50_FORWARD_PASSED',result.stdout)
    def test_missing_package_metadata_reported_together(self):
        from unittest.mock import patch
        import importlib.metadata
        from tinywod.dependencies import dependency_status,DEPENDENCIES
        with patch('tinywod.dependencies.metadata.version',side_effect=importlib.metadata.PackageNotFoundError('missing')):
            rows=dependency_status()
        self.assertEqual(len(rows),len(DEPENDENCIES)); self.assertTrue(all(not r['ok'] for r in rows))
    def test_missing_local_weight_fails_before_dataset_and_records_failure(self):
        from tinywod.runner import launch
        with tempfile.TemporaryDirectory() as folder:
            cfg=create_fixture(Path(folder)); cfg=replace(cfg,pretrained=True,backbone_weights=str(Path(folder)/'missing.pth'))
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()): code=launch(cfg)
            self.assertEqual(code,1)
            output=next((Path(folder)/'runs'/'H-DETR-R50-SMOKE').iterdir())
            state=read_json(output/'run_state.json')
            self.assertEqual(state['status'],'failed'); self.assertEqual(state['completed_epochs'],0)
            self.assertFalse((output/'inputs'/'dataset_view').exists())
            self.assertTrue((output/'logs'/'error.log').is_file())
            with (output/'experiment_summary.csv').open(encoding='utf-8-sig') as f: row=next(csv.DictReader(f))
            self.assertEqual(row['ap'],'')


if __name__=='__main__': unittest.main(verbosity=2)
