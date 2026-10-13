"""Pairing, effect-unit, dependence, numerical prediction and real-record checks."""
from pathlib import Path
import copy
import importlib.util
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import bayesian_analysis as bayes


class BayesianAnalysisTests(unittest.TestCase):
    def test_identical_methods_have_zero_paired_differences(self):
        rng=np.random.default_rng(12)
        y=np.repeat(rng.integers(0,2,size=(20,3,1)),3,axis=2)
        for result in bayes.paired_counts(y).values():
            self.assertEqual(result["observed_difference_pp"],0)
            self.assertEqual(result["questions_equal_mean_difference"],20)

    def test_counts_keep_seed_and_question_pairing(self):
        y=np.zeros((10,3,3),dtype=int)
        y[0,:,1]=1;y[1,:,2]=1;y[2,0,1]=1
        result=bayes.paired_counts(y)["Hard_minus_Soft"]
        self.assertAlmostEqual(result["observed_difference_pp"],100/30)
        self.assertEqual(result["questions_positive_mean_difference"],2)
        self.assertEqual(result["questions_negative_mean_difference"],1)
        self.assertEqual(result["per_seed"][0]["left_only_correct"],2)

    def test_credible_intervals_have_correct_units(self):
        result=bayes.summary_draws(np.array([-.5,0,.5,1.]))
        self.assertEqual(result["mean"],.25)
        self.assertEqual(result["probability_positive"],.5)

    def test_nonfinite_draws_are_rejected(self):
        with self.assertRaises(ValueError):
            bayes.summary_draws(np.array([0,float("nan")]))

    def test_score_validator_rejects_permuted_ids_and_false_accuracy(self):
        data=[{"source_id":"q","gold":"A","choices":[{"label":c} for c in "ABC"]}]
        record={"source_id":"q","gold":"A","predicted":"A","correct":True,
                "invalid":False,"hit_token_limit":False,
                "choice_probabilities":dict.fromkeys("ABC",1/3),
                "choice_log_probabilities":dict.fromkeys("ABC",float(-np.log(3)))}
        bayes.validate_scores([record],data)
        bad=copy.deepcopy(record);bad["source_id"]="different"
        with self.assertRaises(ValueError):bayes.validate_scores([bad],data)
        bad=copy.deepcopy(record);bad["correct"]=False
        with self.assertRaises(ValueError):bayes.validate_scores([bad],data)

    def test_posterior_pattern_checks_detect_seed_disagreement(self):
        y=np.ones((10,3,3),dtype=int);y[0,0,1]=0
        result=bayes.pattern_metrics(y)
        self.assertAlmostEqual(result["all_nine_correct_fraction"],.9)
        self.assertAlmostEqual(result["Hard_seed_disagreement_fraction"],.1)

    @unittest.skipUnless(importlib.util.find_spec("arviz"),"isolated Bayesian runtime required")
    def test_metadata_repair_preserves_posterior_values_in_netcdf(self):
        import arviz as az
        import tempfile
        trace=az.from_dict(posterior={"effect":np.arange(40).reshape(4,10)})
        trace.attrs={"sample_stats":{"inference_library":"nutpie","settings":{"chains":4}}}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"trace.nc"
            bayes.serializable_trace(trace).to_netcdf(path)
            import h5netcdf
            with h5netcdf.File(path,"r") as saved:
                np.testing.assert_array_equal(saved.groups["posterior"].variables["effect"][:],
                                              trace.posterior["effect"])
                self.assertIn("nutpie",str(saved.attrs["sample_stats"]))

    @unittest.skipUnless(importlib.util.find_spec("arviz"),"isolated Bayesian runtime required")
    def test_diagnostics_preserve_all_method_coordinates(self):
        import arviz as az
        draws=np.random.default_rng(17).normal(size=(4,500,3))
        trace=az.from_dict(posterior={"cohort_accuracy":draws},
                           coords={"method":list(bayes.LABELS)},dims={"cohort_accuracy":["method"]})
        table=az.summary(trace.posterior.rename({"method":"student_method"}),kind="diagnostics",round_to="none")
        self.assertEqual(len(table),3)
        self.assertTrue(np.isfinite(table.to_numpy()).all())

    @unittest.skipUnless(importlib.util.find_spec("pymc"),"isolated Bayesian runtime required")
    def test_model_initial_log_probability_is_finite(self):
        y=np.ones((8,3,3),dtype=np.int8);y[0]=0;y[1,0,1]=0
        model=bayes.model(y,np.zeros(8,dtype=bool),np.ones(8,dtype=bool),1)
        self.assertTrue(np.isfinite(model.compile_logp()(model.initial_point())))


if __name__=="__main__":
    unittest.main()
