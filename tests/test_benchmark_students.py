"""Scoring correctness and GPU work blocked until the upstream experiment finishes."""
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import benchmark_students as b


class BenchmarkTests(unittest.TestCase):
    def test_metrics_use_semantic_gold_label_not_token_id(self):
        rows=[{'labels':['A','B','C'],'gold':'C','predicted':'C','choice_log_probabilities':[math.log(.1),math.log(.2),math.log(.7)]},
              {'labels':['A','B','C','D','E'],'gold':'B','predicted':'A','choice_log_probabilities':[math.log(.5),math.log(.1),math.log(.1),math.log(.1),math.log(.2)]}]
        m=b.metrics(rows)
        self.assertEqual(m['accuracy'],.5)
        self.assertAlmostEqual(m['choice_nll'],(-math.log(.7)-math.log(.1))/2)

    def test_pending_training_blocks_before_model_loading(self):
        with patch.object(b,'read',return_value={'stage':'training'}),patch.object(b,'load_model') as loader:
            with self.assertRaisesRegex(RuntimeError,'Finish all six'):
                b.score('smollm2')
            loader.assert_not_called()

    def test_nonfinite_nll_rejected(self):
        with self.assertRaises(AssertionError):
            b.metrics([{'labels':['A','B','C'],'gold':'A','predicted':'A','choice_log_probabilities':[float('-inf'),0.,0.]}])


if __name__=='__main__': unittest.main()
