"""Post-hoc paired Bayesian analysis of completed Qwen runs; no LLM training or new predictions."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from time import perf_counter

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=,openmp=False")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/bayesian_v1"
PUBLIC = ROOT / "results/bayesian_v1"
PROTOCOL = ROOT / "results/protocols/bayesian_analysis_protocol.json"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".matplotlib"))

import numpy as np

METHODS = ("fixed_mix", "adaptive_mix", "adaptive_soft")
LABELS = ("Fixed", "Hard", "Soft")
SEEDS = (42, 7, 123)
PAIRS = ((1, 2), (1, 0), (2, 0))
PROFILES = {"base": 1., "narrow": .5, "wide": 2.}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def validate_scores(records, data):
    if len(records) != len(data):
        raise ValueError("Wrong score count")
    for result, item in zip(records, data):
        labels = [c["label"] for c in item["choices"]]
        if result["source_id"] != item["source_id"] or result["gold"] != item["gold"]:
            raise ValueError("Question pairing or gold mismatch")
        if result["correct"] != (result["predicted"] == item["gold"]):
            raise ValueError("Inconsistent correctness")
        if result["invalid"] or result["hit_token_limit"]:
            raise ValueError("Original completed evaluation must be valid")
        if list(result["choice_probabilities"]) != labels:
            raise ValueError("Choice semantic order mismatch")
        p = np.array([result["choice_probabilities"][c] for c in labels])
        logp = np.array([result["choice_log_probabilities"][c] for c in labels])
        if not np.isfinite(logp).all() or not np.isfinite(p).all() or (p < 0).any():
            raise ValueError("Invalid probabilities")
        if not np.allclose(np.exp(logp), p, atol=1e-6) or not np.isclose(p.sum(), 1, atol=1e-5):
            raise ValueError("Unnormalized choice probabilities")


def load_completed():
    comparison = read(ROOT / "results/gate_comparison_v4/test/comparison.json")
    manifest = read(ROOT / "results/protocols/gate_test_manifest.json")
    for filename, checksum in comparison["score_files_sha256"].items():
        assert sha(ROOT / filename) == checksum, filename
    for filename, checksum in manifest["code_sha256"].items():
        assert sha(ROOT / "src" / filename) == checksum, filename
    preserved = read(ROOT / "results/protocols/gate_training_protocol.json")
    for filename, checksum in preserved["protected_artifacts_sha256"].items():
        assert sha(ROOT / filename) == checksum, filename
    data_file = ROOT / "data/pilot/test.jsonl"
    assert sha(data_file) == manifest["test_data_sha256"]
    data = rows(data_file)
    assert len(data) == 1172
    scores = ROOT / "outputs/gate_comparison_v4/scores/test"
    all_records = {}
    for name in comparison["conditions"]:
        all_records[name] = [rows(scores / f"{name}_{seed}.jsonl") for seed in SEEDS]
        for records in all_records[name]:
            validate_scores(records, data)
    teacher = rows(scores / "teacher_baseline.jsonl")
    student = rows(scores / "student_baseline.jsonl")
    validate_scores(teacher, data)
    validate_scores(student, data)
    y = np.stack([np.stack([[r["correct"] for r in records] for records in all_records[m]], axis=1)
                  for m in METHODS], axis=2).astype(np.int8)
    assert y.shape == (1172, 3, 3)
    teacher_wrong = np.array([not r["correct"] for r in teacher])
    original_correct = np.array([r["correct"] for r in student])
    return y, teacher_wrong, original_correct, data, all_records, comparison


def paired_counts(y):
    result = {}
    for a, b in PAIRS:
        name = f"{LABELS[a]}_minus_{LABELS[b]}"
        by_seed = []
        for s, seed in enumerate(SEEDS):
            left, right = y[:, s, a].astype(bool), y[:, s, b].astype(bool)
            counts = {"seed": seed, "both_correct": int((left & right).sum()),
                      "left_only_correct": int((left & ~right).sum()),
                      "right_only_correct": int((~left & right).sum()),
                      "both_wrong": int((~left & ~right).sum())}
            assert sum(counts[k] for k in counts if k != "seed") == len(y)
            counts["difference_pp"] = 100 * float((left.astype(int)-right.astype(int)).mean())
            by_seed.append(counts)
        d = (y[:, :, a].astype(float)-y[:, :, b]).mean(axis=1)
        result[name] = {"per_seed": by_seed, "observed_difference_pp": 100*float(d.mean()),
            "questions_positive_mean_difference": int((d > 0).sum()),
            "questions_negative_mean_difference": int((d < 0).sum()),
            "questions_equal_mean_difference": int((d == 0).sum())}
    return result


def nll_audit(all_records):
    result = {}
    for name, runs in all_records.items():
        values = []
        for seed, records in zip(SEEDS, runs):
            nll = np.array([-r["choice_log_probabilities"][r["gold"]] for r in records])
            correct = np.array([r["choice_predicted"] == r["gold"] for r in records])
            confidence = np.array([max(r["choice_probabilities"].values()) for r in records])
            baseline = np.array([np.log(len(r["choice_probabilities"])) for r in records])
            top = np.sort(nll)[-10:]
            values.append({"seed": seed, "mean_choice_nll": float(nll.mean()),
                "uniform_choice_reference_nll": float(baseline.mean()),
                "mean_nll_correct": float(nll[correct].mean()),
                "mean_nll_wrong": float(nll[~correct].mean()),
                "wrong_answers": int((~correct).sum()),
                "wrong_fraction_of_total_nll": float(nll[~correct].sum()/nll.sum()),
                "ten_largest_fraction_of_total_nll": float(top.sum()/nll.sum()),
                "wrong_with_confidence_at_least_90pct": int((~correct & (confidence >= .9)).sum()),
                "wrong_with_confidence_at_least_99pct": int((~correct & (confidence >= .99)).sum()),
                "max_choice_nll": float(nll.max())})
        result[name] = values
    return result


def summary_draws(values):
    values = np.asarray(values)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Finite scalar posterior draws required")
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "credible_interval95": np.quantile(values, [.025, .975]).tolist(),
            "probability_positive": float((values > 0).mean())}


def model(y, teacher_wrong, original_correct, scale):
    import pymc as pm
    import pytensor.tensor as pt
    n, s, m = y.shape
    assert (s, m) == (3, 3)
    coords = {"question": np.arange(n), "seed": list(SEEDS), "method": list(LABELS),
              "contrast": ["Hard_vs_Fixed", "Soft_vs_Fixed"]}
    with pm.Model(coords=coords) as graph:
        location = pm.Normal("location", 0, 2)
        sq = pm.HalfStudentT("sigma_question", nu=3, sigma=5*scale)
        qz = pm.Normal("question_z", 0, 1, dims="question")
        question = sq * (qz - qz.mean())
        beta = pm.Normal("method_effect", 0, scale, dims="contrast")
        si = pm.HalfNormal("sigma_item_method", 2*scale, dims="contrast")
        iz = pm.Normal("item_method_z", 0, 1, dims=("question", "contrast"))
        item_method = si * (iz - iz.mean(axis=0))
        ss = pm.HalfNormal("sigma_seed", .5*scale)
        sz = pm.Normal("seed_z", 0, 1, dims="seed")
        seed_effect = ss * (sz - sz.mean())
        sx = pm.HalfNormal("sigma_seed_method", .5*scale)
        xz = pm.Normal("seed_method_z", 0, 1, dims=("seed", "contrast"))
        methods = pt.concatenate([pt.zeros((1,)), beta])
        iq = pt.concatenate([pt.zeros((n, 1)), item_method], axis=1)
        xs = pt.concatenate([pt.zeros((s, 1)), sx*xz], axis=1)
        intercept = location * pt.sqrt(1+sq**2)
        eta = intercept + question[:, None, None] + methods[None, None, :] + iq[:, None, :] + seed_effect[None, :, None] + xs[None, :, :]
        probability = pm.math.sigmoid(eta)
        pm.Bernoulli("correct", logit_p=eta, observed=y, dims=("question", "seed", "method"))
        pm.Deterministic("cohort_accuracy", probability.mean(axis=(0, 1)), dims="method")
        if teacher_wrong.any():
            pm.Deterministic("teacher_wrong_accuracy", probability[teacher_wrong].mean(axis=(0, 1)), dims="method")
        selected = teacher_wrong & original_correct
        if selected.any():
            pm.Deterministic("initial_correct_teacher_wrong_accuracy", probability[selected].mean(axis=(0, 1)), dims="method")
    return graph


def pattern_metrics(values):
    values = np.asarray(values, dtype=bool)
    metrics = {"all_nine_correct_fraction": float(values.all(axis=(1, 2)).mean()),
               "all_nine_wrong_fraction": float((~values).all(axis=(1, 2)).mean())}
    for m, label in enumerate(LABELS):
        metrics[f"{label}_seed_disagreement_fraction"] = float((values[:, :, m].any(axis=1) != values[:, :, m].all(axis=1)).mean())
    for a, b in PAIRS:
        metrics[f"{LABELS[a]}_{LABELS[b]}disagreement_fraction"] = float((values[:, :, a] != values[:, :, b]).mean())
    return metrics


def posterior_probabilities(posterior, indices, new_seed=False, rng=None):
    def get(name):
        v = posterior[name]
        dims = [d for d in v.dims if d not in ("chain", "draw")]
        return np.asarray(v.stack(sample=("chain", "draw")).transpose("sample", *dims))[indices]
    sq = get("sigma_question")
    qz = get("question_z")
    q = sq[:, None] * (qz - qz.mean(axis=1, keepdims=True))
    location = get("location")
    beta = get("method_effect")
    iz = get("item_method_z")
    iq = get("sigma_item_method")[:, None, :] * (iz - iz.mean(axis=1, keepdims=True))
    n = q.shape[1]
    methods = np.column_stack((np.zeros(len(indices)), beta))
    iq = np.concatenate((np.zeros((len(indices), n, 1)), iq), axis=2)
    if new_seed:
        seed_effect = rng.normal(size=(len(indices), 1)) * get("sigma_seed")[:, None]
        xs = rng.normal(size=(len(indices), 1, 2)) * get("sigma_seed_method")[:, None, None]
    else:
        sz = get("seed_z")
        seed_effect = get("sigma_seed")[:, None] * (sz - sz.mean(axis=1, keepdims=True))
        xs = get("sigma_seed_method")[:, None, None] * get("seed_method_z")
    xs = np.concatenate((np.zeros((*xs.shape[:2], 1)), xs), axis=2)
    eta = (location * np.sqrt(1+sq**2))[:, None, None, None] + q[:, :, None, None] + methods[:, None, None, :] + iq[:, :, None, :] + seed_effect[:, None, :, None] + xs[:, None, :, :]
    return np.exp(-np.logaddexp(0, -eta))


def serializable_trace(trace):
    # nutpie 0.16 stores nested provenance attrs; NetCDF requires scalar/string attrs.
    for group in [trace] + [getattr(trace, name) for name in trace.groups()]:
        group.attrs = {key: json.dumps(value, default=str) if isinstance(value, (dict, bool)) else value
                       for key, value in group.attrs.items()}
    return trace


def fit(prior, attempt):
    import arviz as az
    import nutpie
    spec = read(PROTOCOL)
    assert spec["code_sha256"] == sha(Path(__file__))
    assert spec["test_already_seen"] and prior in PROFILES
    y, wrong, original, _, _, _ = load_completed()
    settings = spec["sampling_attempts"][attempt-1]
    folder = OUT / f"{prior}_attempt{attempt}"
    if folder.exists():
        raise FileExistsError("Previous fit preserved; use a separately recorded recovery attempt")
    folder.mkdir(parents=True)
    status = {"status": "running", "prior": prior, "attempt": attempt,
              "settings": settings, "protocol_sha256": sha(PROTOCOL)}
    write(folder/"status.json", status)
    started = perf_counter()
    try:
        graph = model(y, wrong, original, PROFILES[prior])
        compiled = nutpie.compile_pymc_model(graph)
        print(f"Compiled {prior}; sampling four chains on one CPU worker", flush=True)
        trace = nutpie.sample(compiled, draws=settings["draws"], tune=settings["tune"],
            chains=4, cores=1, seed=spec["sampling_seed"], target_accept=settings["target_accept"],
            progress_bar=False)
        serializable_trace(trace).to_netcdf(folder/"posterior.nc")
        table = az.summary(trace.posterior.rename({"method": "student_method"}), kind="diagnostics", round_to="none")
        table.to_csv(folder/"mcmc_diagnostics.csv")
        divergence = int(trace.sample_stats["diverging"].sum())
        stats = {"divergences": divergence, "max_rank_rhat": float(table["r_hat"].max()),
                 "min_bulk_ess": float(table["ess_bulk"].min()), "min_tail_ess": float(table["ess_tail"].min()),
                 "parameters_above_rhat_limit": int((table["r_hat"] > 1.01).sum()),
                 "parameters_below_ess_limit": int(((table["ess_bulk"] < 400)|(table["ess_tail"] < 400)).sum())}
        passed = divergence == 0 and stats["max_rank_rhat"] <= 1.01 and stats["min_bulk_ess"] >= 400 and stats["min_tail_ess"] >= 400
        status.update(status="passed" if passed else "diagnostics_failed", diagnostics=stats,
                      seconds=perf_counter()-started, posterior_sha256=sha(folder/"posterior.nc"))
        print(json.dumps(status, ensure_ascii=False), flush=True)
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}",
                      seconds=perf_counter()-started)
        raise
    finally:
        write(folder/"status.json", status)


def analyze(prior, attempt):
    import arviz as az
    folder = OUT / f"{prior}_attempt{attempt}"
    status = read(folder/"status.json")
    if status["status"] != "passed":
        raise ValueError("MCMC convergence checks have not passed")
    assert sha(folder/"posterior.nc") == status["posterior_sha256"]
    trace = az.from_netcdf(folder/"posterior.nc")
    y, wrong, original, _, records, comparison = load_completed()
    posterior = trace.posterior.load()
    values = np.asarray(posterior["cohort_accuracy"].stack(sample=("chain", "draw")).transpose("sample", "method"))
    result = {"status":"analyzed", "prior":prior, "attempt":attempt, "diagnostics":status["diagnostics"],
              "n_questions":len(y), "training_seeds":list(SEEDS), "student_methods":list(METHODS),
              "posthoc":True, "new_LLM_predictions":False, "LLM_retraining":False,
              "posterior_cohort_accuracy":{label:summary_draws(100*values[:,i]) for i,label in enumerate(LABELS)},
              "comparisons":{}, "paired_error_counts":paired_counts(y), "nll_audit":nll_audit(records)}
    rng = np.random.default_rng(20261014)
    new_seed = []
    for start in range(0,len(values),32):
        indices = np.arange(start,min(start+32,len(values)))
        new_seed.append(posterior_probabilities(posterior,indices,True,rng).mean(axis=(1,2)))
    new_seed = np.concatenate(new_seed)
    grid = np.linspace(0,2,81)
    for a,b in PAIRS:
        key = f"{LABELS[a]}_minus_{LABELS[b]}"
        d = 100*(values[:,a]-values[:,b])
        prediction = 100*(new_seed[:,a]-new_seed[:,b])
        result["comparisons"][key] = {"observed_cohort_expected_difference_pp":summary_draws(d),
             "one_hypothetical_new_seed_difference_pp":summary_draws(prediction),
             "effect_threshold_sensitivity":{"threshold_pp":grid.tolist(),
                 "probability_difference_exceeds_threshold":[float((d>delta).mean()) for delta in grid],
                 "probability_absolute_difference_below_threshold":[float((abs(d)<delta).mean()) for delta in grid]},
             "original_observed_difference_pp":result["paired_error_counts"][key]["observed_difference_pp"],
             "original_paired_ci95_pp":[100*x for x in comparison["comparisons"][f"{METHODS[a]}_minus_{METHODS[b]}"]["accuracy"]["ci95"]]}
    metrics = []
    indices = rng.choice(len(values),size=min(600,len(values)),replace=False)
    for start in range(0,len(indices),32):
        probabilities = posterior_probabilities(posterior,indices[start:start+32])
        simulated = rng.random(probabilities.shape) < probabilities
        metrics.extend(pattern_metrics(item) for item in simulated)
    observed = pattern_metrics(y)
    ppc = {}
    for key,value in observed.items():
        draws = np.array([v[key] for v in metrics])
        interval = np.quantile(draws,[.025,.975])
        ppc[key] = {"observed":value,"posterior_predictive_mean":float(draws.mean()),
                    "posterior_predictive_interval95":interval.tolist(),
                    "observed_inside_interval":bool(interval[0] <= value <= interval[1])}
    result["posterior_predictive_checks"] = ppc
    result["ppc_outside_interval"] = [key for key,v in ppc.items() if not v["observed_inside_interval"]]
    result["teacher_wrong_subset"] = {"n":int(wrong.sum()),"original_student_correct_n":int((wrong&original).sum()),
        "posterior_expected_accuracy":{label:summary_draws(100*np.asarray(posterior["teacher_wrong_accuracy"].sel({"method": label})).ravel()) for label in LABELS},
        "limits":"small, post-hoc subset; common posterior effects are not proof of teacher-error prevention"}
    result["limitations"] = ["Post-hoc model applied after original test results were seen; original primary CI remains unchanged.",
         "Credible intervals depend on logistic hierarchy, priors and observed question/seed support.",
         "Only three training seeds; hypothetical new-seed predictions are model-based, not new empirical results.",
         "No minimum practically meaningful effect was specified; threshold curves are illustrative, not an equivalence decision.",
         "Same Qwen pair and ARC; no evidence of cross-family or cross-dataset generalization.",
         "Shared GPU timings are not modeled as isolated causal speed effects."]
    write(folder/"analysis.json",result)
    destination = PUBLIC/f"{prior}_attempt{attempt}"
    destination.mkdir(parents=True,exist_ok=True)
    write(destination/"analysis.json",result)
    print(json.dumps({"prior":prior,"comparisons":result["comparisons"],"ppc_outside_interval":result["ppc_outside_interval"]},ensure_ascii=False),flush=True)


def prepare():
    if PROTOCOL.exists():
        raise FileExistsError("Existing Bayesian protocol preserved")
    y,wrong,original,_,records,comparison = load_completed()
    spec = {"status":"prepared_posthoc","test_already_seen":True,
        "model":"crossed hierarchical Bernoulli-logit / Rasch extension with question intercept, question-by-method effects, seed intercept and seed-by-method effects",
        "likelihood":"Y[q,s,m] ~ Bernoulli(sigmoid(location*sqrt(1+sigma_q^2)+question[q]+beta[m]+item_method[q,m]+seed[s]+seed_method[s,m]))",
        "methods":list(METHODS),"seeds":list(SEEDS),"n_questions":len(y),
        "priors":{"location":"Normal(0,2); standardized ability location",
          "method_effect":"Normal(0,1*scale) Hard/Soft relative to Fixed",
          "sigma_question":"HalfStudentT(df=3,scale=5*scale)",
          "sigma_item_method":"HalfNormal(2*scale), one per Hard/Soft contrast",
          "sigma_seed":"HalfNormal(.5*scale)",
          "sigma_seed_method":"HalfNormal(.5*scale)",
          "random_effects":"non-centered standard Normal; item effects and shared seed intercept centered for identification",
          "justification":"proper regularizing scale choices for this Rasch/logit extension, not literature defaults; zero-centered treatment effects; broad baseline prior and heavy-tailed item scale; sensitivity is mandatory"},
        "prior_profiles":PROFILES,"sampling_seed":20261013,
        "sampling_attempts":[{"draws":1000,"tune":1500,"target_accept":.95},
                             {"draws":2000,"tune":2500,"target_accept":.99}],
        "sampling":"four chains sequentially on one CPU worker; nutpie/PyMC; isolated runtime",
        "checks":{"max_rank_rhat":1.01,"min_bulk_ess":400,"min_tail_ess":400,"divergences":0,
                 "posterior_predictive":"all-agree fractions, within-method seed disagreements and matched-method disagreements"},
        "RQ":["posterior expected Accuracy differences and P(delta>0) for original three observed seeds",
              "same average Accuracy versus paired item-level gains/losses and heterogeneity",
              "conditional-choice NLL decomposition and high-confidence errors"],
        "primary_pair":"Hard minus Soft; Hard/Soft minus Fixed supporting and exploratory",
        "effect_threshold":"no user-defined ROPE; report curves over 0..2 percentage points without declaring practical equivalence",
        "scope":"post-hoc reanalysis of existing completed test; no new model predictions/checkpoint choices/retraining; later external generalization is separate",
        "source_comparison_sha256":sha(ROOT/"results/gate_comparison_v4/test/comparison.json"),
        "score_files_sha256":comparison["score_files_sha256"],"code_sha256":sha(Path(__file__)),
        "versions":{name:importlib.metadata.version(name) for name in ("pymc","arviz","nutpie","numpy","scipy","pytensor","numba")},
        "sources":[{"url":"https://mc-stan.org/docs/stan-users-guide/regression.html#item-response-theory-models","adopt":"Bernoulli-logit correctness, paired student-question hierarchy and identifiability"},
             {"url":"https://jmlr.org/papers/v18/16-305.html","adopt":"posterior effect probabilities and practical-equivalence thresholds; not copying multi-dataset CV likelihood"},
             {"url":"https://sites.stat.columbia.edu/gelman/research/published/taumain.pdf","adopt":"proper variance priors and sensitivity with few groups"},
             {"url":"https://arxiv.org/abs/1903.08008","adopt":"rank Rhat / bulk-tail ESS diagnostics"},
             {"url":"https://arxiv.org/abs/2011.01808","adopt":"prior/posterior predictive checks and model criticism"}],
        "thread_context_limit":"read_thread exposed user messages but assistant responses were content-reference placeholders; no claim to reproduce inaccessible detailed advice"}
    write(PROTOCOL,spec)
    write(PUBLIC/"descriptive_audit.json",{"paired_counts":paired_counts(y),"nll_audit":nll_audit(records),
         "teacher_wrong_n":int(wrong.sum()),"original_correct_on_teacher_wrong_n":int((wrong&original).sum()),
         "mean_generated_accuracy":{name:[float(np.mean([r["correct"] for r in run])) for run in runs] for name,runs in records.items()}})
    print("Post-hoc Bayesian protocol recorded; original source/results preserved",flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","fit","analyze"))
    parser.add_argument("--prior",choices=tuple(PROFILES),default="base")
    parser.add_argument("--attempt",type=int,choices=(1,2,3),default=1)
    args=parser.parse_args()
    if args.command=="prepare":prepare()
    elif args.command=="fit":fit(args.prior,args.attempt)
    else:analyze(args.prior,args.attempt)
