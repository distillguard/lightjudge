"""Figures and an auxiliary Bayesian bootstrap; completed Qwen results only."""
import argparse
from pathlib import Path
import json
import numpy as np
import bayesian_analysis as b

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

COLORS = ("#586b84", "#278473", "#bb6952")
NAMES = {"gold_sft":"Gold", "response_kd":"Response", "logit_kd":"Forward",
         "reverse_kl":"Reverse", "fixed_mix":"Fixed", "adaptive_mix":"Hard", "adaptive_soft":"Soft"}


def save(fig, name):
    b.PUBLIC.mkdir(parents=True, exist_ok=True)
    fig.savefig(b.PUBLIC/name, dpi=170, bbox_inches="tight")
    plt.close(fig)


def descriptive():
    audit = b.read(b.PUBLIC/"descriptive_audit.json")
    names = list(NAMES)
    mean = np.array([np.mean([r["mean_choice_nll"] for r in audit["nll_audit"][n]]) for n in names])
    wrong = np.array([np.mean([r["mean_choice_nll"]*r["wrong_fraction_of_total_nll"]
                               for r in audit["nll_audit"][n]]) for n in names])
    fig,ax = plt.subplots(figsize=(9,4.3))
    ax.bar(range(7),mean-wrong,label="Contribution from correct answers",color="#709ebc")
    ax.bar(range(7),wrong,bottom=mean-wrong,label="Contribution from wrong answers",color="#c47261")
    ax.set(xticks=range(7),xticklabels=[NAMES[n] for n in names],
           ylabel="Mean valid-choice NLL",title="Most NLL comes from incorrect predictions")
    ax.legend(frameon=False);ax.grid(axis="y",alpha=.2)
    save(fig,"nll_decomposition.png")
    pairs=list(audit["paired_counts"])
    gain=np.array([sum(v["left_only_correct"] for v in audit["paired_counts"][key]["per_seed"]) for key in pairs])
    loss=np.array([sum(v["right_only_correct"] for v in audit["paired_counts"][key]["per_seed"]) for key in pairs])
    fig,ax=plt.subplots(figsize=(8,4.2))
    x=np.arange(3)
    ax.bar(x-.18,gain,.36,color=COLORS[1],label="Left correct, right wrong")
    ax.bar(x+.18,loss,.36,color=COLORS[2],label="Left wrong, right correct")
    ax.set(xticks=x,xticklabels=[s.replace("_minus_"," vs ") for s in pairs],
           ylabel="Matched question-seed outcomes",title="Similar average accuracy does not imply identical errors")
    ax.legend(frameon=False);ax.grid(axis="y",alpha=.2)
    fig.text(.02,-.01,"Counts sum three matched training seeds; these are not independent question samples.",fontsize=9)
    save(fig,"paired_errors.png")


def bootstrap():
    target=b.PUBLIC/"nll_bayesian_bootstrap.json"
    if target.exists():
        return b.read(target)
    y,_,_,_,records,_=b.load_completed()
    names=list(records)
    matrix=np.column_stack([np.mean([[-r["choice_log_probabilities"][r["gold"]] for r in run]
                                     for run in records[name]],axis=0) for name in names])
    rng=np.random.default_rng(20261015)
    draws=[]
    for _ in range(40):
        weights=rng.exponential(size=(500,len(y)))
        weights/=weights.sum(axis=1,keepdims=True)
        draws.append(weights @ matrix)
    draws=np.concatenate(draws)
    assert draws.shape==(20000,len(names))
    result={"status":"computed","method":"Bayesian bootstrap: Dirichlet(1,...,1) paired question weights",
            "seed":20261015,"draws":20000,"conditions":names,
            "scope":"conditional on the three observed trained seeds; all methods share each question weight; does not estimate unseen training-seed variability",
            "source":"https://doi.org/10.1214/aos/1176345338","comparisons":{}}
    comparisons=[("adaptive_mix","adaptive_soft"),("adaptive_mix","fixed_mix"),
                 ("adaptive_soft","fixed_mix"),("logit_kd","gold_sft"),("logit_kd","response_kd")]
    for left,right in comparisons:
        d=draws[:,names.index(left)]-draws[:,names.index(right)]
        raw=matrix[:,names.index(left)]-matrix[:,names.index(right)]
        expected_variance=np.mean((raw-raw.mean())**2)/(len(raw)+1)
        assert abs(d.mean()-raw.mean()) < 6*np.sqrt(expected_variance/len(d)) + 1e-10
        assert abs(d.var()-expected_variance) < .08*expected_variance + 1e-12
        result["comparisons"][f"{left}_minus_{right}"]={**b.summary_draws(d),
            "probability_left_has_lower_nll":float((d<0).mean()),
            "observed_difference":float(raw.mean()),"analytic_posterior_variance":float(expected_variance)}
    b.write(target,result)
    return result


def accuracy_bootstrap():
    target=b.PUBLIC/"paired_accuracy_bayesian_bootstrap.json"
    if target.exists():return b.read(target)
    y,*_=b.load_completed()
    matrix=100*y.mean(axis=1)
    rng=np.random.default_rng(20261016)
    draws=[]
    for _ in range(40):
        weights=rng.exponential(size=(500,len(y)))
        weights/=weights.sum(axis=1,keepdims=True)
        draws.append(weights @ matrix)
    draws=np.concatenate(draws)
    result={"status":"computed","method":"paired question Bayesian bootstrap; Dirichlet(1,...,1)",
        "draws":len(draws),"seed":20261016,"training_seeds":list(b.SEEDS),
        "scope":"observed three-seed question averages; preserves every matched outcome; does not estimate unobserved seed variability",
        "reason":"auxiliary robustness check after hierarchical posterior predictive disagreement failed; not a new primary analysis",
        "source":"https://doi.org/10.1214/aos/1176345338","comparisons":{}}
    for a,c in b.PAIRS:
        delta=draws[:,a]-draws[:,c]
        raw=matrix[:,a]-matrix[:,c]
        variance=np.mean((raw-raw.mean())**2)/(len(raw)+1)
        assert abs(delta.mean()-raw.mean()) < 6*np.sqrt(variance/len(delta))+1e-10
        assert abs(delta.var()-variance) < .08*variance+1e-12
        result["comparisons"][f"{b.LABELS[a]}_minus_{b.LABELS[c]}"]={**b.summary_draws(delta),
            "observed_difference_pp":float(raw.mean()),"analytic_posterior_variance":float(variance)}
    b.write(target,result)
    return result


def prior_predictive():
    target=b.PUBLIC/"prior_predictive_reproducible.json"
    if target.exists():return b.read(target)
    import pymc as pm
    y,wrong,original,*_=b.load_completed()
    result={"status":"computed","draws":500,"sampling_seed":20261017,
        "code_sha256":b.sha(Path(__file__)),"analysis_model_sha256":b.sha(Path(b.__file__)),
        "scope":"prior plausibility only; same declared priors; no outcome-based prior selection",
        "original_auxiliary_snapshot_retained":"prior_predictive.json",
        "reason":"original auxiliary prior check did not record its random seed; add reproducible verification without replacing it",
        "prior_profiles":{}}
    for profile,scale in b.PROFILES.items():
        with b.model(y,wrong,original,scale):
            trace=pm.sample_prior_predictive(draws=500,random_seed=result["sampling_seed"],
                var_names=["cohort_accuracy","correct"])
        accuracy=np.asarray(trace.prior.cohort_accuracy).reshape(500,3)
        simulated=np.asarray(trace.prior_predictive.correct).reshape(500,*y.shape).mean(axis=(1,2))
        assert np.isfinite(accuracy).all() and np.isfinite(simulated).all()
        result["prior_profiles"][profile]={
            "expected_accuracy_percent":{label:b.summary_draws(100*accuracy[:,i]) for i,label in enumerate(b.LABELS)},
            "simulated_accuracy_percent":{label:b.summary_draws(100*simulated[:,i]) for i,label in enumerate(b.LABELS)},
            "Hard_minus_Soft_prior_pp":b.summary_draws(100*(accuracy[:,1]-accuracy[:,2]))}
    b.write(target,result)
    return result


def contrast_diagnostics(profile, attempt):
    import arviz as az
    import xarray as xr
    folder=b.OUT/f"{profile}_attempt{attempt}"
    trace=az.from_netcdf(folder/"posterior.nc")
    accuracy=trace.posterior.cohort_accuracy.load().values
    delta=np.stack([100*(accuracy[:,:,a]-accuracy[:,:,c]) for a,c in b.PAIRS],axis=2)
    dataset=xr.Dataset({"difference_pp":(("chain","draw","comparison"),delta)},
                       coords={"comparison":[f"{b.LABELS[a]}_minus_{b.LABELS[c]}" for a,c in b.PAIRS]})
    table=az.summary(dataset,kind="diagnostics",round_to="none")
    assert table.r_hat.max()<=1.01 and min(table.ess_bulk.min(),table.ess_tail.min())>=400
    result={"status":"passed","posterior_sha256":b.sha(folder/"posterior.nc"),
            "difference_diagnostics":table.to_dict(orient="index"),"probability_mcse":{}}
    for i,key in enumerate(dataset.comparison.values):
        indicator=(delta[:,:,i]>0).astype(float)
        ess=float(az.ess(indicator,method="mean"))
        probability=float(indicator.mean())
        result["probability_mcse"][str(key)]={"probability":probability,"indicator_ess":ess,
            "mcse":float(np.sqrt(probability*(1-probability)/ess))}
    b.write(b.PUBLIC/f"{profile}_attempt{attempt}/contrast_diagnostics.json",result)
    return result


def posterior_figures():
    fits={}
    for profile in b.PROFILES:
        candidates=[(attempt,b.PUBLIC/f"{profile}_attempt{attempt}/analysis.json") for attempt in (1,2,3)]
        available=[(a,p) for a,p in candidates if p.exists()]
        if not available:raise FileNotFoundError(f"Missing verified fit: {profile}")
        attempt,path=max(available)
        fits[profile]=b.read(path)
        fits[profile]["contrast_mcmc_diagnostics"]=contrast_diagnostics(profile,attempt)
        assert fits[profile]["diagnostics"]["max_rank_rhat"] <= 1.01
    fig,axes=plt.subplots(1,3,figsize=(12,4.2),sharey=True)
    keys=("Hard_minus_Soft","Hard_minus_Fixed","Soft_minus_Fixed")
    for ax,key in zip(axes,keys):
        for i,(profile,values) in enumerate(fits.items()):
            summary=values["comparisons"][key]["observed_cohort_expected_difference_pp"]
            low,high=summary["credible_interval95"]
            ax.errorbar(summary["median"],i,xerr=[[summary["median"]-low],[high-summary["median"]]],
                        fmt="o",capsize=4,color=COLORS[i])
            ax.text(high+.04,i,f"P(>0)={summary['probability_positive']:.2f}",va="center",fontsize=8)
        ax.margins(x=.45)
        ax.axvline(0,color="gray",ls="--",lw=1)
        ax.set(title=key.replace("_minus_"," - "),xlabel="Expected accuracy difference (pp)",
               yticks=range(3),yticklabels=list(fits))
        ax.grid(axis="x",alpha=.2)
    fig.suptitle("Posterior 95% credible intervals: observed Qwen cohort / prior sensitivity")
    fig.tight_layout();save(fig,"posterior_effects.png")
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax,key in zip(axes,keys):
        for i,(profile,values) in enumerate(fits.items()):
            sensitivity=values["comparisons"][key]["effect_threshold_sensitivity"]
            ax.plot(sensitivity["threshold_pp"],sensitivity["probability_difference_exceeds_threshold"],
                    label=profile,color=COLORS[i])
        ax.set(title=key.replace("_minus_"," - "),xlabel="Hypothetical minimum improvement (pp)",
               ylim=(0,1));ax.grid(alpha=.2)
    axes[0].set_ylabel("Posterior P(difference > threshold)");axes[-1].legend(frameon=False)
    fig.suptitle("Threshold sensitivity; no user-defined practical-equivalence decision")
    fig.tight_layout();save(fig,"effect_thresholds.png")
    base=fits["base"]
    ppc=base["posterior_predictive_checks"]
    fig,ax=plt.subplots(figsize=(10,5))
    labels=list(ppc)
    for i,key in enumerate(labels):
        v=ppc[key];low,high=v["posterior_predictive_interval95"]
        ax.plot([low,high],[i,i],color="#5b7d9d",lw=4)
        ax.scatter(v["observed"],i,color="#c36d4c",zorder=3)
    ax.set(yticks=range(len(labels)),yticklabels=[k.replace("_fraction","").replace("disagreement"," disagreement").replace("_"," ") for k in labels],
           xlabel="Fraction",title="Posterior predictive checks: intervals and observed outcomes")
    ax.set_xscale("log")
    ax.set_xticks([.02,.04,.06,.1,.2,.4,.8],labels=["2%","4%","6%","10%","20%","40%","80%"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Outcome fraction (log scale)")
    ax.plot([],[],color="#5b7d9d",lw=4,label="95% posterior predictive interval")
    ax.scatter([],[],color="#c36d4c",label="Observed")
    ax.legend(frameon=False,loc="upper right")
    ax.grid(axis="x",alpha=.2);fig.tight_layout();save(fig,"posterior_predictive.png")
    summary={"status":"reported","fits":fits,"posterior_predictive_warnings":{p:v["ppc_outside_interval"] for p,v in fits.items()},
             "paired_accuracy_bootstrap":accuracy_bootstrap(),
             "interpretation":"post-hoc Bayesian analysis; original primary CI and LLM models unchanged",
             "external_generalization":"deferred until this analysis is assessed"}
    b.write(b.PUBLIC/"summary.json",summary)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("descriptive","posterior"))
    args=parser.parse_args()
    descriptive()
    bootstrap()
    accuracy_bootstrap()
    if args.command=="posterior":
        prior_predictive()
        posterior_figures()
