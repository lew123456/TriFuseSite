from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn
from trifusesite import TriFuseSite, ModelConfig
from trifusesite.config import load_config
from trifusesite.modules.geometry import build_distance_geometry_graph
from trifusesite.modules.graph import BranchCrossAttention, segment_softmax
from trifusesite.data.graphs import radius_graph
from trifusesite.data.features import handcrafted62, AA, BLOSUM
from trifusesite.data.dataset import ProteinDataset, save_jsonl, model_inputs
from trifusesite.losses import weighted_binary_loss
from trifusesite.metrics import residue_metrics, evaluate_rows, validation_mcc_threshold
from trifusesite.engine import train, load_checkpoint, predict
from trifusesite.statistics import paired_summary

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)

def synthetic_inputs(n=9, mask=None):
    # Synthetic numerical fixtures, never experimental data.
    rng = np.random.default_rng(17)
    ca = rng.normal(size=(n, 3)).astype(np.float32)*3
    mask = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    adjacency, edges, edge_features, _ = radius_graph(ca, mask)
    return {'residue_features': torch.from_numpy(rng.normal(size=(n, 62)).astype(np.float32)),
            'esm_embedding': torch.from_numpy(rng.normal(size=(n, 1280)).astype(np.float32)),
            'patch_adjacency': torch.from_numpy(adjacency),
            'global_edge_index': torch.from_numpy(edges),
            'global_edge_features': torch.from_numpy(edge_features),
            'ca': torch.from_numpy(ca), 'structure_mask': torch.from_numpy(mask),
            'confidence': torch.from_numpy(mask.astype(np.float32))}

class ModelTests(unittest.TestCase):
    def test_smpi_training_defaults(self):
        for task in ('ppi','smpi','dpi','rpi'):
            c=load_config(ROOT/'config'/(task+'.yaml'))
            r,t=c['runtime'],c['training']
            self.assertEqual(r['epochs'],20)
            self.assertEqual(r['seed'],2025)
            self.assertEqual(r['class_weights'],'balanced')
            self.assertEqual(r['threshold_rule'],'validation_mcc')
            self.assertEqual(r['auprc_method'],'average_precision')
            self.assertEqual(r['paired_seeds'],list(range(2025,2035)))
            self.assertEqual(t['early_stopping_patience'],5)
            self.assertEqual(t['checkpoint_min_delta'],1e-5)
            self.assertEqual(t['precision'],'cuda_bfloat16')

    def test_bfloat16_autocast_forward_backward(self):
        # CPU autocast checks mixed dtypes; it is not a CUDA performance test.
        model=TriFuseSite(task='smpi')
        with torch.autocast('cpu',dtype=torch.bfloat16):
            logits=model(**synthetic_inputs())
            loss=weighted_binary_loss(logits,torch.arange(9)%2,[1.,2.])
        self.assertEqual(loss.dtype,torch.float32)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_paper_parameters(self):
        c = ModelConfig()
        expected = {'residue_feature_dim':62, 'esm_dim':1280, 'hidden_dim':256,
                    'semantic_projection_dim':256, 'patch_layers':3, 'global_graph_layers':8,
                    'global_attention_heads':4, 'geometry_attention_heads':4, 'geometry_layers':2,
                    'geometry_neighbors':24, 'rbf_min':0.0, 'rbf_max':20.0, 'rbf_width':2.5,
                    'radius_max_neighbors':64, 'classifier_dim':128, 'dropout':0.1}
        for key, value in expected.items():
            self.assertEqual(getattr(c, key), value)
        for task in ('ppi', 'smpi', 'dpi', 'rpi'):
            cfg = load_config(ROOT / 'config' / (task + '.yaml'))
            self.assertEqual(ModelConfig(**cfg['model']), c)
            self.assertEqual(cfg['training']['optimizer'], 'AdamW')
            self.assertEqual(cfg['training']['gradient_clip'], 1)

    def test_all_tasks_forward_backward(self):
        for task in ('ppi', 'smpi', 'dpi', 'rpi'):
            model = TriFuseSite(task=task)
            x = synthetic_inputs()
            x['esm_embedding'].requires_grad_(True)
            logits, state = model(**x, return_intermediates=True)
            self.assertEqual(logits.shape, (9, 2))
            self.assertEqual(state['X0'].shape, (9, 62))
            self.assertEqual(state['H'].shape, (9, 256))
            weighted_binary_loss(logits, torch.arange(9)%2, [1., 2.]).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            self.assertIsNone(x['esm_embedding'].grad)
            self.assertIsNotNone(model.geometry_gate_logit.grad)

    def test_missing_structure_gate(self):
        for task in ('smpi', 'dpi', 'rpi'):
            model = TriFuseSite(task=task).eval()
            x = synthetic_inputs(mask=np.zeros(9, bool))
            with torch.no_grad():
                a, state = model(**x, return_intermediates=True)
                x['ca'] = torch.full_like(x['ca'], float('nan'))
                b = model(**x)
            self.assertEqual(float(state['geometry_gate']), 0.0)
            self.assertTrue(torch.equal(a, b))

    def test_partial_structure_coverage(self):
        model = TriFuseSite(task='dpi').eval()
        _, s = model(**synthetic_inputs(mask=[1,1,1,0,0,0,0,0,0]), return_intermediates=True)
        self.assertAlmostEqual(float(s['geometry_gate']), float(model.geometry_gate_logit.sigmoid())/3, places=6)

    def test_ppi_gate_not_coverage_scaled(self):
        model = TriFuseSite(task='ppi').eval()
        x = synthetic_inputs()
        x['structure_mask'][:] = False
        _, s = model(**x, return_intermediates=True)
        self.assertAlmostEqual(float(s['geometry_gate']), float(model.geometry_gate_logit.sigmoid()), places=6)

    def test_single_residue(self):
        for task in ('ppi', 'smpi'):
            self.assertTrue(torch.isfinite(TriFuseSite(task=task)(**synthetic_inputs(1))).all())

    def test_geometry_rbf_and_mask(self):
        ca = torch.tensor([[0.,0.,0.], [5.,0.,0.], [0.,0.,0.]])
        g = build_distance_geometry_graph(ca, torch.tensor([True,True,False]), torch.ones(3))
        self.assertEqual(g['edge_index'].shape[1], 2)
        self.assertFalse((g['edge_index'] == 2).any())
        expected = torch.exp(-((5-torch.linspace(0,20,8))/2.5)**2)
        torch.testing.assert_close(g['edge_features'][0,:8], expected)
        self.assertTrue((g['edge_features'][:,8:20] == 0).all())

    def test_geometry_nearest_24(self):
        ca = torch.randn(35, 3)
        g = build_distance_geometry_graph(ca, torch.ones(35, dtype=torch.bool), torch.ones(35))
        self.assertTrue(torch.equal(torch.bincount(g['edge_index'][1]), torch.full((35,),24)))

    def test_full_cross_attention_equation(self):
        attention = BranchCrossAttention(256)
        p, g = torch.randn(11,256), torch.randn(11,256)
        q = attention.query(p).relu()
        k = attention.key(p).relu()
        v = attention.value(g).relu()
        expected = attention.output((q @ k.T / 16).softmax(-1) @ v).relu()
        torch.testing.assert_close(attention(p,g), expected)

    def test_segment_softmax_four_heads(self):
        score = torch.randn(7,4)
        dst = torch.tensor([0,0,1,1,1,2,2])
        weights = segment_softmax(score, dst, 4)
        for i in range(3):
            torch.testing.assert_close(weights[dst==i].sum(0), torch.ones(4))

class DataMetricTests(unittest.TestCase):
    def test_mcc_threshold_matches_smpi_source(self):
        from sklearn.metrics import matthews_corrcoef
        y=np.array([0,0,0,1,1,1,0,1,0,0])
        p=np.array([.01,.2,.49,.19,.52,.81,.51,.78,.1,.94])
        grid=np.linspace(.05,.95,181)
        expected=float(grid[np.argmax([matthews_corrcoef(y,p>=t) for t in grid])])
        self.assertEqual(validation_mcc_threshold([{'labels':y,'probabilities':p}]),expected)
        # Every threshold has a tied MCC=0: choose smallest grid value.
        self.assertEqual(validation_mcc_threshold([{'labels':[0,1],'probabilities':[1.,1.]}]),.05)

    def test_balanced_weights_training_only(self):
        from trifusesite.engine import balanced_weights
        dataset=[{'labels':torch.tensor([0,0,0,1,-1])}, {'labels':torch.tensor([1,0,0])}]
        np.testing.assert_allclose(balanced_weights(dataset),[7/10,7/4])

    def test_repeat_plan(self):
        from main_repeat import build_plan
        p=build_plan(ROOT/'config/ppi.yaml',ROOT/'runs/not_executed')
        self.assertEqual(p['seeds'],list(range(2025,2035)))
        self.assertEqual(len(p['commands']),40)
        first=p['commands'][:4]
        self.assertEqual([r['stage'] for r in first],['five_fold_selection','training','bound25_evaluation','unbound25_evaluation'])
        self.assertIn('--cv-selection',first[1]['argv'])
        checkpoints=[cmd['argv'][cmd['argv'].index('--checkpoint')+1] for cmd in first[2:]]
        self.assertEqual(checkpoints[0],checkpoints[1])

    def test_features_and_blosum(self):
        f = handcrafted62(AA)
        self.assertEqual(f.shape, (20,62))
        self.assertEqual(BLOSUM[AA.index('V'), AA.index('V')], 4)
        self.assertEqual(BLOSUM[AA.index('W'), AA.index('W')], 11)
        self.assertEqual(BLOSUM[AA.index('Y'), AA.index('Y')], 7)
        self.assertTrue(np.isfinite(handcrafted62('AX')).all())

    def test_radius_mask_and_cap(self):
        xyz = np.zeros((100,3), np.float32)
        a, edges, edge, _ = radius_graph(xyz, np.ones(100,bool))
        self.assertTrue(np.max(np.bincount(edges[1])) <= 65)  # 64 neighbors plus self
        self.assertTrue(np.allclose(a, a.T))
        xyz[1] = [999,999,999]
        a, edges, edge, fallback = radius_graph(xyz[:3], [1,0,1])
        self.assertFalse(fallback)
        nonself = edges[:,edges[0] != edges[1]]
        self.assertFalse((nonself == 1).any())

    def test_fallback_and_ppi_graph_preservation(self):
        xyz = np.zeros((3,3), np.float32)
        _, edges, edge, fallback = radius_graph(xyz, np.zeros(3,bool))
        self.assertTrue(fallback)
        self.assertTrue((edge == 0).all())
        d = np.ones((3,3),np.float32)
        a, _, _, fallback = radius_graph(xyz, np.ones(3,bool), distance_map=d)
        self.assertFalse(fallback)
        self.assertEqual(np.count_nonzero(a), 9)

    def test_loss_equation11(self):
        logits = torch.tensor([[0.,1.], [1.,0.], [2.,3.]])
        y = torch.tensor([1,0,-1])
        loss = weighted_binary_loss(logits,y,[1.,3.])
        p = logits.softmax(-1)[:,1]
        expected = -(3*torch.log(p[0])+torch.log(1-p[1]))/4
        torch.testing.assert_close(loss,expected)

    def test_metrics_pooled(self):
        rows = [{'id':'a','labels':[0,1],'probabilities':[.1,.9]},
                {'id':'b','labels':[1,1,0,-1],'probabilities':[.4,.8,.2,.99]}]
        r = evaluate_rows(rows,.5,'average_precision')
        self.assertEqual(r['pooled']['residues'],5)
        self.assertAlmostEqual(r['pooled']['Brier'],(.01+.01+.36+.04+.04)/5)
        self.assertEqual(len(r['per_protein']),2)

    def test_run_statistics_protocol(self):
        pairs = [{'bound_id':f'b{i}', 'unbound_id':f'u{i}'} for i in range(25)]
        b,u = [],[]
        metric = {k:.5 for k in ('ACC','Precision','Recall','F1','MCC','AUROC','AUPRC','Brier')}
        for seed in range(10):
            b.append({'seed':seed,'threshold':.5,'auprc_method':'average_precision',
                      'pooled':{k:v+seed*.001 for k,v in metric.items()},
                      'per_protein':{f'b{i}':{'MCC':.6,'AUPRC':.7} for i in range(25)}})
            u.append({'seed':seed,'threshold':.5,'auprc_method':'average_precision','pooled':metric,
                      'per_protein':{f'u{i}':{'MCC':.5,'AUPRC':.6} for i in range(25)}})
        r=paired_summary(b,u,pairs,bootstrap_seed=1,bootstrap_samples=100)
        self.assertAlmostEqual(r['paired_statistics']['MCC']['mean_delta'],-.1)
        self.assertAlmostEqual(r['pooled_run_summary']['bound']['MCC']['sample_sd'],np.std(np.arange(10)*.001,ddof=1))
        self.assertEqual(r['paired_statistics']['MCC']['lower_unbound'],25)

class EndToEndTests(unittest.TestCase):
    def test_training_cli_uses_smpi_defaults(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        import main_train
        dataset=SimpleNamespace(records=[{'id':'synthetic_only'}])
        argv=['main_train.py','--config',str(ROOT/'config/smpi.yaml'),'--output','unused_synthetic_run']
        with patch('sys.argv',argv), patch.object(main_train,'ProteinDataset',return_value=dataset), \
                patch.object(main_train,'train') as call:
            main_train.main()
        options=call.call_args.kwargs
        self.assertEqual(options['epochs'],20)
        self.assertEqual(options['seed'],2025)
        self.assertEqual(options['class_weights'],'balanced')
        self.assertEqual(options['auprc_method'],'average_precision')
        self.assertTrue(options['fit_threshold'])
        self.assertIsNone(options['threshold'])

    def test_ppi_refit_cli_imports_and_checks_cv_selection(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        import main_train
        dataset=SimpleNamespace(records=[{'id':'synthetic_only'}])
        c=load_config(ROOT/'config/ppi.yaml')
        selection={'seed':2025,'auprc_method':'average_precision',
                   'training_protein_ids':['synthetic_only'],'selected_candidate':0,
                   'candidates':[{'model_config':ModelConfig(**c['model']).to_dict(),
                                  'oof_MCC_threshold':.375,'suggested_refit_epochs':7}]}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'selection.json'
            path.write_text(json.dumps(selection),encoding='utf-8')
            argv=['main_train.py','--config',str(ROOT/'config/ppi.yaml'),'--output',str(Path(tmp)/'run'),
                  '--refit','--cv-selection',str(path)]
            with patch('sys.argv',argv), patch.object(main_train,'ProteinDataset',return_value=dataset), \
                    patch.object(main_train,'train') as call:
                main_train.main()
            options=call.call_args.kwargs
            self.assertEqual(options['epochs'],7)
            self.assertEqual(options['threshold'],.375)
            self.assertEqual(options['threshold_provenance'],'training_OOF_MCC')
            self.assertFalse(options['fit_threshold'])
            self.assertIsNone(call.call_args.args[2])
            with patch('sys.argv',argv+['--seed','2026']), \
                    patch.object(main_train,'ProteinDataset',return_value=dataset), \
                    patch.object(main_train,'train') as rejected:
                with self.assertRaisesRegex(ValueError,'seeds differ'):
                    main_train.main()
                rejected.assert_not_called()

    def test_validation_early_stopping_and_threshold(self):
        from unittest.mock import patch
        import trifusesite.engine as engine
        with tempfile.TemporaryDirectory() as tmp:
            train_row=dict(synthetic_inputs(), id='train_synthetic', sequence='ACDEFGHIK', labels=torch.arange(9)%2)
            valid_row=dict(synthetic_inputs(), id='valid_synthetic', sequence='ACDEFGHIK', labels=torch.arange(9)%2)
            rows=[{'id':'valid_synthetic','labels':valid_row['labels'].tolist(),
                   'probabilities':[.05,.8,.2,.7,.1,.9,.4,.6,.3]}]
            c=load_config(ROOT/'config/smpi.yaml')
            with patch.object(engine,'predict',return_value=rows):
                _,meta=engine.train(c,[train_row],[valid_row],Path(tmp)/'run',epochs=20,seed=2025,
                                    class_weights='balanced',auprc_method='average_precision',fit_threshold=True)
            self.assertEqual(meta['selected_epoch'],1)
            self.assertEqual(meta['epochs_completed'],6)
            self.assertEqual(meta['threshold_source'],'validation_MCC')
            self.assertEqual(meta['threshold'],validation_mcc_threshold(rows))
            self.assertEqual(meta['precision_used'],'float32')

    def test_structure_chain_mapping_and_atom_masks(self):
        from trifusesite.data.structures import map_structure
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'synthetic.pdb'
            lines, serial = ['MODEL        1'], 1
            for chain, names in [('A', ['ALA','CYS']), ('B', ['ALA','CYS','ASP','GLU'])]:
                for index, name in enumerate(names, 1):
                    lines.append(f'ATOM  {serial:5d}  CA  {name:3s} {chain}{index:4d}    '
                                 f'{float(index):8.3f}{0.:8.3f}{0.:8.3f}{1.:6.2f}{80.:6.2f}           C')
                    serial += 1
                lines.append('TER')
            lines += ['ENDMDL','END']
            path.write_text('\n'.join(lines)+'\n',encoding='ascii')
            xyz, mask, confidence, chain=map_structure('ACDE',path)
            self.assertEqual(chain,'B')
            self.assertEqual(int(mask[:,1].sum()),4)
            self.assertEqual(int(mask[:,0].sum()),0)
            np.testing.assert_array_equal(confidence,np.ones(4))
            _,mask,confidence,chain=map_structure('ACDE',path,'A','plddt')
            self.assertEqual(chain,'A')
            self.assertEqual(int(mask[:,1].sum()),2)
            self.assertAlmostEqual(float(confidence[0]),.8,places=6)
            with self.assertRaises(ValueError):
                map_structure('ACDE',path,'Z')

    def test_esm_window_average_and_cache_precision(self):
        from main_extract_esm import encode
        from types import SimpleNamespace
        class Batch(dict):
            def to(self, device):
                return self
        class Tokenizer:
            def __call__(self, sequence, **kwargs):
                return Batch(input_ids=torch.zeros(1,len(sequence)+2,dtype=torch.long))
        class Model:
            def __init__(self):
                self.calls=0
            def __call__(self, input_ids):
                self.calls+=1
                return SimpleNamespace(last_hidden_state=torch.full((1,input_ids.shape[1],1280),float(self.calls)))
        model=Model()
        result=encode('A'*1200,Tokenizer(),model,'cpu')
        self.assertEqual(model.calls,2)
        self.assertEqual(result.shape,(1200,1280))
        self.assertEqual(result.dtype,np.float16)
        self.assertTrue((result[:894]==1).all())
        self.assertTrue((result[894:1022]==1.5).all())
        self.assertTrue((result[1022:]==2).all())

    def test_ppi_feature_import_without_backbone(self):
        from main_prepare import prepare
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            fields={}
            for name,dim in [('pssm',20),('hmm',20),('dssp',14),('resAF',7)]:
                np.save(root/(name+'.npy'),np.ones((3,dim),np.float32))
                fields[name]=name+'.npy'
            np.save(root/'pseudo.npy',np.array([[0,0,0],[3,0,0],[6,0,0]],np.float32))
            np.save(root/'dist.npy',np.array([[0,3,6],[3,0,3],[6,3,0]],np.float32))
            fields.update(pseudopositions='pseudo.npy',distance_map='dist.npy')
            save_jsonl(root/'raw.jsonl',[{'id':'ppi_synthetic','sequence':'ACD','labels':[0,1,0],
                                        'structure_path':None,'ppi_features':fields}])
            prepare(root/'raw.jsonl',root/'prepared','ppi')
            with np.load(root/'prepared/ppi_synthetic.npz') as z:
                self.assertEqual(z['features'].shape,(3,62))
                np.testing.assert_allclose(z['features'][:,-1],[0,.2,.4])
                self.assertEqual(int(z['structure_mask'].sum()),0)
                self.assertEqual(z['global_edge_index'].shape,(2,9))

    def test_prepare_train_predict_checkpoint(self):
        from main_prepare import prepare
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root/'raw.jsonl'
            save_jsonl(raw,[{'id':'synthetic_only','sequence':'ACDEFGHIK',
                            'labels':[0,1,0,0,1,0,0,1,0], 'structure_path':None}])
            manifest=prepare(raw,root/'prepared','smpi')
            np.save(root/'prepared/synthetic_only.esm.npy',np.zeros((9,1280),np.float16))
            dataset=ProteinDataset(manifest,'smpi')
            c=load_config(ROOT/'config/smpi.yaml')
            model,meta=train(c,dataset,None,root/'run',epochs=1,seed=17,
                             class_weights=[1.,1.],auprc_method='average_precision',threshold=.5)
            self.assertEqual(meta['training_config']['optimizer'],'AdamW')
            loaded,_=load_checkpoint(root/'run/model.pt',expected_task='smpi')
            a,b=predict(model,dataset,'cpu'),predict(loaded,dataset,'cpu')
            self.assertEqual(a,b)
            self.assertEqual(len(a[0]['probabilities']),9)
            report=evaluate_rows(a,.5,'average_precision')
            self.assertEqual(report['pooled']['residues'],9)

if __name__ == '__main__':
    unittest.main()
