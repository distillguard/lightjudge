import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import data_gate_experiment as experiment
import report_data_gates as report

class ExternalGateTests(unittest.TestCase):
    def test_holdout_has_disjoint_questions_and_scenes(self):
        rows=[{'source_id':str(i),'question_sha256':str(i//2),'scene_id':str(i//4)} for i in range(40)]
        dev,test=experiment.pick_heldout(rows,3)
        self.assertEqual(len(dev),3); self.assertEqual(len(test),3)
        self.assertFalse({r['scene_id'] for r in dev}&{r['scene_id'] for r in test})
        self.assertFalse({r['question_sha256'] for r in dev}&{r['question_sha256'] for r in test})
        self.assertEqual((dev,test),experiment.pick_heldout(rows,3))

    def test_train_excludes_same_question_and_conflicting_labels(self):
        rows=[{'source_id':str(i),'question_sha256':str(i//2)} for i in range(8)]
        kept=experiment.clean_train(rows,[{'question_sha256':'0'}],{'2','3'})
        self.assertEqual([r['source_id'] for r in kept],['4','6'])

    def test_final_evaluation_without_manifest_is_blocked_before_model_loading(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(experiment,'require',return_value={}),patch.object(experiment,'PUBLIC',Path(tmp)),patch.object(experiment,'load_model') as loader:
            with self.assertRaises(FileNotFoundError): experiment.score('hellaswag','student','holdout')
            loader.assert_not_called()

    def test_paired_difference_preserves_matching_and_identical_results(self):
        rows=[{'source_id':str(i),'question_sha256':str(i),'correct':i%2==0,
               'gold':'A','choice_log_probabilities':[-1.]*4} for i in range(8)]
        result=report.paired(rows,rows)
        self.assertEqual(result['accuracy_paired_ci95_pp'],[0.,0.])
        with self.assertRaises(AssertionError): report.paired(rows,list(reversed(rows)))

    def test_two_position_gradient_audit_does_not_treat_eos_as_answer_entropy(self):
        import contextlib
        from types import SimpleNamespace
        import torch
        class Dummy(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight=torch.nn.Parameter(torch.zeros(2,5))
            def forward(self, **kwargs):
                return SimpleNamespace(logits=self.weight[None])
        model=Dummy(); tensor,zeros=torch.tensor,torch.zeros
        def cpu_tensor(*args,**kwargs):
            kwargs.pop('device',None); return tensor(*args,**kwargs)
        def cpu_zeros(*args,**kwargs):
            kwargs.pop('device',None); return zeros(*args,**kwargs)
        example={'input_ids':[0,1],'prediction_positions':[0,1],'target_ids':[0,4]}
        items=[{'source_id':str(i)} for i in range(8)]
        values=torch.log_softmax(torch.tensor([2.,1.,0.,0.,0.]),-1).repeat(8,1)
        with patch.object(experiment,'encode',return_value=example),patch.object(torch.Tensor,'cuda',lambda self:self),patch.object(torch,'tensor',side_effect=cpu_tensor),patch.object(torch,'zeros',side_effect=cpu_zeros),patch.object(torch,'autocast',return_value=contextlib.nullcontext()):
            result=experiment.gradient_audit(model,[model.weight],items,values,None)
        self.assertTrue(result['fp32_logit_decomposition_verified'])
        self.assertEqual(result['n'],8)
        self.assertTrue(model.training)
        self.assertIsNone(model.weight.grad)

if __name__=='__main__': unittest.main()
