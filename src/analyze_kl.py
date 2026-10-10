"""Paired question-level analysis and scientific plots for the fixed KL experiment."""
import json
import math
import os
from pathlib import Path
import numpy as np
import pilot
import kl_experiment as experiment
import monitor_training as diagnostic
OUT = experiment.OUT
os.environ.setdefault("MPLCONFIGDIR",str(OUT/".matplotlib"))
NAMES = {"gold_sft":"Gold SFT","response_kd":"Response KD","logit_kd":"Forward KL",
         "reverse_kl":"Reverse KL","fixed_mix":"Fixed 50:50","adaptive_mix":"Adaptive hard gate"}


def vectors(records):
    correct = np.array([r["correct"] for r in records],dtype=float)
    choice_correct = np.array([r["choice_predicted"] == r["gold"] for r in records],dtype=float)
    confidence = np.array([r["choice_confidence"] for r in records])
    brier = np.array([sum((p-(label == r["gold"]))**2 for label,p in r["choice_probabilities"].items()) for r in records])
    nll = np.array([-r["choice_log_probabilities"][r["gold"]] if "choice_log_probabilities" in r else
                    -math.log(r["choice_probabilities"][r["gold"]]) for r in records])
    assert np.isfinite(nll).all() and np.isfinite(brier).all()
    bins = np.minimum((confidence*10).astype(int),9)
    errors = (choice_correct-confidence)[:,None]*(bins[:,None] == np.arange(10)[None,:])
    return {"accuracy":correct,"brier":brier,"choice_nll":nll,"ece_errors":errors}


def aggregate(arrays,indices,weights):
    result = {}
    samples = {}
    for metric in ("accuracy","brier","choice_nll"):
        matrix = np.stack([a[metric] for a in arrays])
        per_seed = matrix.mean(1)
        sampled = matrix.mean(0)[indices].mean(1)
        result[metric] = {"mean":float(per_seed.mean()),"seed_sample_sd":float(per_seed.std(ddof=1)),
                         "per_seed":per_seed.tolist(),"ci95":np.quantile(sampled,[.025,.975]).tolist()}
        samples[metric] = sampled
    per_seed = np.array([np.abs(a["ece_errors"].mean(0)).sum() for a in arrays])
    sampled = np.mean([np.abs(weights @ a["ece_errors"]).sum(1) for a in arrays],axis=0)
    result["ece"] = {"mean":float(per_seed.mean()),"seed_sample_sd":float(per_seed.std(ddof=1)),
                    "per_seed":per_seed.tolist(),"ci95":np.quantile(sampled,[.025,.975]).tolist()}
    samples["ece"] = sampled
    return result,samples



def signal_summary(signals):
    assert signals and all(np.isfinite(list(r[k] for k in ("forward_kl","reverse_kl","teacher_confidence","student_confidence","adaptive_gate","forward_weight","loss"))).all() for r in signals)
    gates = [r["adaptive_gate"] for r in signals]
    assert set(gates) <= {0.0,1.0}
    n = len(signals)
    forward = sum(gates)
    return {"n_example_forwards":n,"entropy_gate_forward_n":int(forward),
            "entropy_gate_reverse_n":n-int(forward),"entropy_gate_forward_fraction":forward/n,
            "mean_effective_forward_weight":float(np.mean([r["forward_weight"] for r in signals])),
            "mean_teacher_confidence_T2_full_vocab":float(np.mean([r["teacher_confidence"] for r in signals])),
            "mean_student_confidence_T2_full_vocab":float(np.mean([r["student_confidence"] for r in signals]))}


def self_check():
    record = {"gold":"A","choice_predicted":"A","correct":True,"choice_confidence":.75,
              "choice_probabilities":{"A":.75,"B":.25},"choice_log_probabilities":{"A":math.log(.75),"B":math.log(.25)}}
    records=[record,{**record,"correct":False,"choice_predicted":"B","choice_confidence":.6,
                    "choice_probabilities":{"A":.4,"B":.6},
                    "choice_log_probabilities":{"A":math.log(.4),"B":math.log(.6)}}]
    a=vectors(records)
    indices=np.array([[0,0],[0,1],[1,1]])
    weights=np.array([[1.,0.],[.5,.5],[0.,1.]])
    result,sampled=aggregate([a,a,a],indices,weights)
    for i in range(3):
        assert abs(sampled["ece"][i]-diagnostic.calibration([records[j] for j in indices[i]])["ece"]) < 1e-9
    assert np.all((sampled["accuracy"]-sampled["accuracy"]) == 0)
    assert result["accuracy"]["seed_sample_sd"] == 0
    signals=[dict(forward_kl=.1,reverse_kl=.2,teacher_confidence=.5,student_confidence=.4,
                  adaptive_gate=1.,forward_weight=1.,loss=.2),
             dict(forward_kl=.2,reverse_kl=.1,teacher_confidence=.4,student_confidence=.5,
                  adaptive_gate=0.,forward_weight=0.,loss=.1)]
    gates=signal_summary(signals)
    assert gates["entropy_gate_forward_n"] == gates["entropy_gate_reverse_n"] == 1
    assert gates["entropy_gate_forward_fraction"] == gates["mean_effective_forward_weight"] == .5
    from tempfile import TemporaryDirectory
    from unittest.mock import patch
    with TemporaryDirectory() as directory:
        folder=Path(directory)
        (folder/"docs").mkdir()
        template="<!-- main-status:start -->pending<!-- main-status:end -->\n<!-- kl-results-live:start -->pending<!-- kl-results-live:end -->"
        for name in ("RESEARCH_PLAN.md","RESEARCH_REPORT.md","README.md"):
            (folder/(name if name == "README.md" else "docs/"+name)).write_text(template,encoding="utf-8")
        comparison={"accuracy":{"mean_difference":0.,"ci95":[-.1,.1]},
                    "choice_nll":{"mean_difference":.02,"ci95":[-.01,.04]}}
        fake={"scope":"full","split":"validation","n":299,"conditions":experiment.CONDITIONS,
              "aggregate":{c:{"accuracy":{"mean":.5,"seed_sample_sd":.01},"choice_nll":{"mean":1.}} for c in experiment.CONDITIONS},
              "primary_comparison":"adaptive_mix_minus_fixed_mix","comparisons":{"adaptive_mix_minus_fixed_mix":comparison}}
        with patch.object(pilot,"ROOT",folder):
            update_docs(folder,fake,True)
            assert (folder/"docs/RESEARCH_REPORT.md").read_text(encoding="utf-8") == template
            update_docs(folder,fake,False)
            text=(folder/"docs/RESEARCH_REPORT.md").read_text(encoding="utf-8")
            assert "선택지 NLL" in text and "각 64개" in text and "Brier" not in text
            fake["split"]="test";fake["n"]=1172
            update_docs(folder,fake,False)
        for name in ("RESEARCH_PLAN.md","RESEARCH_REPORT.md","README.md"):
            text=(folder/(name if name == "README.md" else "docs/"+name)).read_text(encoding="utf-8")
            assert "18회 학습·validation·동결된 test 평가 완료" in text and "파일럿 추가" not in text


def analyze(scope,split="validation",baseline_only=False):
    self_check()
    if split == "test":
        experiment.check_test(scope,"student",None,42)
    conditions = experiment.CONDITIONS[:3] if baseline_only else experiment.CONDITIONS
    source_rows,data_path = experiment.rows_for(scope,split)
    ids=[r["source_id"] for r in source_rows]
    def read(role=None,condition=None,seed=42):
        tag=f"{scope}_{condition}_{seed}" if condition else f"{role}_baseline"
        folder=OUT/"scores"/split
        summary_path=folder/f"{tag}_summary.json"
        assert experiment.ready(summary_path)
        summary=json.loads(summary_path.read_text(encoding="utf-8"))
        path=folder/f"{tag}.jsonl"
        assert diagnostic.sha(path) == summary["output_sha256"]
        records=pilot.read_rows(path)
        assert [r["source_id"] for r in records] == ids
        assert [r["gold"] for r in records] == [r["gold"] for r in source_rows]
        return records,summary
    teacher,teacher_summary=read(role="teacher")
    original,original_summary=read(role="student")
    rng=np.random.default_rng(42)
    n=len(ids)
    indices=rng.integers(0,n,size=(5000,n))
    weights=np.array([np.bincount(row,minlength=n) for row in indices],dtype=float)/n
    records_by_condition={}
    summaries_by_condition={}
    aggregate_by_condition={}
    bootstrap_by_condition={}
    wrong=[i for i,r in enumerate(teacher) if not r["correct"]]
    initially_correct=[i for i in wrong if original[i]["correct"]]
    error_patterns={}
    training_signals={}
    for condition in conditions:
        records_by_condition[condition]=[]
        summaries_by_condition[condition]=[]
        patterns=[]
        for seed in experiment.SEEDS:
            records,summary=read(condition=condition,seed=seed)
            records_by_condition[condition].append(records)
            summaries_by_condition[condition].append(summary)
            groups={}
            for threshold in (.90,.95):
                selected=[i for i in wrong if teacher[i]["choice_confidence"] >= threshold]
                groups[str(threshold)]={"n":len(selected),"student_correct":sum(records[i]["correct"] for i in selected),
                    "same_teacher_wrong":sum(records[i]["predicted"] == teacher[i]["predicted"] for i in selected)}
            patterns.append({"seed":seed,"teacher_wrong_n":len(wrong),
                "student_correct_on_teacher_wrong":sum(records[i]["correct"] for i in wrong),
                "same_teacher_wrong":sum(records[i]["predicted"] == teacher[i]["predicted"] for i in wrong),
                "initial_student_correct_n":len(initially_correct),
                "initial_student_correct_lost":sum(not records[i]["correct"] for i in initially_correct),
                "high_confidence_wrong":groups})
        error_patterns[condition]=patterns
        training_signals[condition]=[]
        for seed in experiment.SEEDS:
            signal_path=experiment.run_path(scope,condition,seed)/"signal_log.jsonl"
            if signal_path.exists():
                signals=pilot.read_rows(signal_path)
                assert len(signals) == (256 if scope == "pilot" else 1118)*3
                training_signals[condition].append({"seed":seed,**signal_summary(signals),
                    "epochs":[{"epoch":epoch,**signal_summary([r for r in signals if r["epoch"] == epoch])} for epoch in (1,2,3)]})
        aggregate_by_condition[condition],bootstrap_by_condition[condition]=aggregate(
            [vectors(r) for r in records_by_condition[condition]],indices,weights)
    comparisons={}
    pairs=[("adaptive_mix","fixed_mix"),("reverse_kl","logit_kd"),("adaptive_mix","logit_kd"),
           ("adaptive_mix","reverse_kl"),("logit_kd","response_kd"),("logit_kd","gold_sft")]
    for a,b in pairs:
        if a not in conditions or b not in conditions:continue
        comparisons[f"{a}_minus_{b}"]={}
        for metric in ("accuracy","brier","choice_nll","ece"):
            samples=bootstrap_by_condition[a][metric]-bootstrap_by_condition[b][metric]
            comparisons[f"{a}_minus_{b}"][metric]={
                "mean_difference":aggregate_by_condition[a][metric]["mean"]-aggregate_by_condition[b][metric]["mean"],
                "ci95":np.quantile(samples,[.025,.975]).tolist(),
                "per_seed_difference":(np.array(aggregate_by_condition[a][metric]["per_seed"])-
                                       np.array(aggregate_by_condition[b][metric]["per_seed"])).tolist()}
    name=f"{scope}_{split}"+("_baseline" if baseline_only else "")
    folder=OUT/"analysis"/name
    folder.mkdir(parents=True,exist_ok=True)
    summary={"status":"passed","stage":"existing baseline confidence" if baseline_only else "six-condition comparison",
        "scope":scope,"split":split,"n":n,"train_n":256 if scope == "pilot" else 1118,"seeds":experiment.SEEDS,
        "conditions":conditions,"aggregate":aggregate_by_condition,"comparisons":comparisons,
        "primary_comparison":"adaptive_mix_minus_fixed_mix" if not baseline_only else "logit_kd_minus_response_kd",
        "bootstrap_replicates":5000,"bootstrap_seed":42,
        "ci_scope":"paired questions conditional on the three observed training seeds; not independent 3*n observations",
        "secondary_comparisons":"exploratory unadjusted intervals; no joint significance or equivalence claim",
        "teacher_wrong_n":len(wrong),"initial_student_correct_on_teacher_wrong_n":len(initially_correct),
        "error_patterns":error_patterns,"error_pattern_scope":"descriptive held-out subsets; no causal attribution",
        "training_signals":training_signals,"signal_scope":"training forwards incl. dropout/repeated epochs; entropy gate counterfactual except adaptive_mix",
        "baselines":{"teacher":teacher_summary["metrics"],"student":original_summary["metrics"]},
        "source_sha256":diagnostic.sha(__file__),"test_evaluation":split == "test",
        "epoch_diagnostics":"seed 42 fixed 64 train/64 validation subsets; final evaluations use all held-out questions"}
    pilot.write_json(folder/"comparison.json",summary)
    plots(folder,summary,records_by_condition,summaries_by_condition)
    update_docs(folder,summary,baseline_only)
    print(json.dumps({"analysis":name,"metrics":aggregate_by_condition,"primary":comparisons.get(summary["primary_comparison"])},ensure_ascii=False),flush=True)


def plots(folder,summary,records_by_condition,scoring_summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conditions=summary["conditions"]
    labels=[NAMES[c] for c in conditions]
    positions=np.arange(len(conditions))
    def save(fig,name):
        fig.tight_layout()
        fig.savefig(folder/name,dpi=180,bbox_inches="tight")
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for metric,ax,scale in (("accuracy",axes[0],100),("choice_nll",axes[1],1)):
        values=np.array([summary["aggregate"][c][metric]["mean"] for c in conditions])*scale
        ci=np.array([summary["aggregate"][c][metric]["ci95"] for c in conditions])*scale
        ax.bar(positions,values,color="#5b87b2")
        ax.errorbar(positions,values,yerr=np.stack([values-ci[:,0],ci[:,1]-values]),fmt="none",color="black",capsize=3)
        for i,c in enumerate(conditions):
            ax.scatter(np.full(3,i),np.array(summary["aggregate"][c][metric]["per_seed"])*scale,color="#c05a30",s=18,zorder=3)
        ax.set_xticks(positions,labels,rotation=25,ha="right")
        ax.set_title("Generation accuracy (%)" if metric == "accuracy" else "Choice NLL (lower is better)")
        ax.grid(axis="y",alpha=.2)
    fig.suptitle(f"ARC {summary['scope']} / {summary['split']} (n={summary['n']}); conditional 95% paired-question CI")
    save(fig,"performance.png")
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    any_curves=False
    for condition in conditions:
        run=experiment.run_path(summary["scope"],condition,42)
        if summary["scope"] == "pilot" and condition in experiment.CONDITIONS[:3]:
            run=pilot.ROOT/"outputs/diagnostics/runs"/condition
        points=[]
        for epoch in range(4):
            path=run/f"epoch_{epoch}.json"
            if path.exists():points.append(json.loads(path.read_text(encoding="utf-8")))
        if not points:continue
        any_curves=True
        for ax,split,key in ((axes[0,0],"train","gold_ce"),(axes[0,1],"validation","gold_ce"),
                              (axes[1,0],"train","accuracy"),(axes[1,1],"validation","accuracy")):
            ax.plot([p["epoch"] for p in points],[p["metrics"][split][key] for p in points],marker="o",label=NAMES[condition])
            count = points[0]["metrics"][split]["n"]
            ax.set_title(f"Seed 42 {split} (n={count}): common gold CE" if key == "gold_ce" else f"Seed 42 {split} (n={count}): generation accuracy")
            ax.set_xticks(range(4));ax.set_xlabel("Epoch");ax.grid(alpha=.2)
    if any_curves:
        axes[0,0].legend(fontsize=8)
        save(fig,"learning_curves.png")
    else:plt.close(fig)
    fig,axes=plt.subplots(math.ceil(len(conditions)/3),3,figsize=(12,3.5*math.ceil(len(conditions)/3)))
    for ax,condition in zip(axes.flat,conditions):
        for seed in experiment.SEEDS:
            logs=pilot.read_rows(experiment.run_path(summary["scope"],condition,seed)/"train_log.jsonl")
            values=np.array([r["mean_loss"] for r in logs])
            width=min(8,len(values))
            ax.plot([r["optimizer_step"] for r in logs][width-1:],np.convolve(values,np.ones(width)/width,mode="valid"),label=f"seed {seed}")
        ax.set_title(NAMES[condition]);ax.set_xlabel("Optimizer step");ax.set_ylabel("Own training objective");ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(conditions):]:ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Actual objective, 8-step moving mean; panel scales/loss definitions differ")
    save(fig,"objective_curves.png")
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    gate_conditions=[];gate_weights=[];clips=[]
    for condition in conditions:
        clip=[]
        signals=[]
        for seed in experiment.SEEDS:
            run=experiment.run_path(summary["scope"],condition,seed)
            if (run/"train_log.jsonl").exists():
                logs=pilot.read_rows(run/"train_log.jsonl")
                clip.append(np.mean([r["grad_norm"] > 1 for r in logs]))
            if (run/"signal_log.jsonl").exists():
                signals+=pilot.read_rows(run/"signal_log.jsonl")
        clips.append(float(np.mean(clip)) if clip else np.nan)
        if signals:
            gate_conditions.append(NAMES[condition])
            gate_weights.append(float(np.mean([r["forward_weight"] for r in signals])))
    axes[0].bar(gate_conditions,gate_weights,color="#639b80")
    axes[0].set_ylim(0,1);axes[0].set_title("Mean effective Forward weight (new signal logs)")
    axes[0].tick_params(axis="x",rotation=25)
    axes[1].bar(labels,clips,color="#ba8955")
    axes[1].set_ylim(0,1);axes[1].set_title("Fraction of optimizer steps clipped")
    axes[1].tick_params(axis="x",rotation=25)
    save(fig,"gate_and_clipping.png")
    fig,axes=plt.subplots(math.ceil(len(conditions)/3),3,figsize=(12,3.5*math.ceil(len(conditions)/3)))
    for ax,condition in zip(axes.flat,conditions):
        ax.plot([0,1],[0,1],color="gray",linestyle="--")
        for seed,records in zip(experiment.SEEDS,records_by_condition[condition]):
            bins=diagnostic.calibration(records)["bins"]
            ax.plot([b["confidence"] for b in bins],[b["accuracy"] for b in bins],marker=".",label=f"seed {seed}")
        ax.set_title(NAMES[condition]);ax.set(xlim=(0,1),ylim=(0,1),xlabel="Choice confidence",ylabel="Choice accuracy")
    for ax in list(axes.flat)[len(conditions):]:ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Reliability: T=1, valid choices, fixed 10 bins; seeds shown separately")
    save(fig,"calibration.png")
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    recovered=[np.mean([p["student_correct_on_teacher_wrong"] for p in summary["error_patterns"][c]]) for c in conditions]
    lost=[np.mean([p["initial_student_correct_lost"] for p in summary["error_patterns"][c]]) for c in conditions]
    axes[0].bar(labels,recovered,color="#649580")
    axes[0].set_title(f"Correct answers on Teacher-wrong questions (n={summary['teacher_wrong_n']})")
    axes[1].bar(labels,lost,color="#bf7a65")
    axes[1].set_title(f"Initial Student correct answers lost (n={summary['initial_student_correct_on_teacher_wrong_n']})")
    for ax in axes:ax.tick_params(axis="x",rotation=25);ax.set_ylabel("Mean count over 3 seeds")
    save(fig,"teacher_errors.png")
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    train_times=[]
    latencies=[]
    for c in conditions:
        totals=[]
        for seed in experiment.SEEDS:
            training=json.loads((experiment.run_path(summary["scope"],c,seed)/"summary.json").read_text(encoding="utf-8"))
            totals.append(training["training_seconds"]/60)
        train_times.append(float(np.mean(totals)))
        latencies.append(float(np.mean([s["metrics"]["inference"]["latency_mean_seconds"] for s in scoring_summaries[c]])))
    axes[0].bar(labels,train_times,color="#8097af");axes[0].set_title("Training minutes (monitor time excluded)")
    axes[1].bar(labels,latencies,color="#9c8cb3");axes[1].set_title("Warm generation latency, batch 1, scores enabled (s)")
    for ax in axes:ax.tick_params(axis="x",rotation=25)
    save(fig,"costs.png")
    links="".join(f'<figure><figcaption>{p.stem}</figcaption><a href="{p.name}"><img src="{p.name}" style="max-width:100%"></a></figure>' for p in folder.glob("*.png"))
    (folder/"index.html").write_text(f'<!doctype html><meta charset="utf-8"><title>KL experiment</title><main style="max-width:1200px;margin:auto;font-family:sans-serif"><h1>{summary["scope"]} / {summary["split"]}</h1><p>Three training seeds; paired-question confidence intervals. Error subsets are descriptive.</p>{links}</main>',encoding="utf-8")


def update_docs(folder,summary,baseline_only):
    if baseline_only or summary["scope"] != "full":
        return
    assert tuple(summary["conditions"]) == experiment.CONDITIONS
    split=summary["split"]
    start="<!-- kl-results-live:start -->";end="<!-- kl-results-live:end -->"
    block=[start,f"### 본실험 결과: {split}","",
        f"학습 train 1,118개 · 평가 {summary['n']}개 전체 · seed 42·7·123.",
        "","| 조건 | Accuracy 평균±seed SD | 선택지 NLL |","|---|---:|---:|"]
    for condition in summary["conditions"]:
        m=summary["aggregate"][condition]
        block.append(f"| {NAMES[condition]} | {m['accuracy']['mean']*100:.2f}±{m['accuracy']['seed_sample_sd']*100:.2f}% | {m['choice_nll']['mean']:.4f} |")
    primary=summary["comparisons"][summary["primary_comparison"]]
    a=primary["accuracy"];low,high=a["ci95"]
    interpretation="우위를 확인하지 못했다." if low <= 0 <= high else ("양의 차이를 관측했다." if low > 0 else "음의 차이를 관측했다.")
    nll=primary["choice_nll"];nlo,nhi=nll["ci95"]
    block += ["",f"적응형−고정 혼합 Accuracy: **{a['mean_difference']*100:+.2f}%p**, paired 95% CI **{low*100:+.2f}~{high*100:+.2f}%p**. {interpretation}",
        f"선택지 NLL 차이: **{nll['mean_difference']:+.4f}**, 95% CI **{nlo:+.4f}~{nhi:+.4f}** (탐색적 부지표, 낮을수록 좋음).",
        "구간은 관측한 3개 seed에 조건부인 문항 bootstrap이다. 동등성·다른 모델/데이터의 일반적 우위를 주장하지 않는다.",
        "",f"[전체 수치·seed·CI·gate·오류 분석]({(folder/'comparison.json').as_posix()}) · [그래프·보조 자료]({(folder/'index.html').as_posix()})",
        "",f"![Accuracy와 NLL]({(folder/'performance.png').as_posix()})",
        "", "학습곡선은 seed 42의 고정된 train·validation 각 64개 진단 subset이며 checkpoint 선택에 사용하지 않았다.",
        f"[학습곡선]({(folder/'learning_curves.png').as_posix()})",end,""]
    p=pilot.ROOT/"docs/RESEARCH_REPORT.md";text=p.read_text(encoding="utf-8")
    a=text.index(start);b=text.index(end,a)+len(end)
    p.write_text(text[:a]+"\n".join(block)+text[b:],encoding="utf-8")
    state="전체 train 18회 학습·validation·동결된 test 평가 완료" if split == "test" else "전체 train 18회 학습·validation 완료; 동결 후 test 평가 진행"
    status_start="<!-- main-status:start -->";status_end="<!-- main-status:end -->"
    note=f"{status_start}\n**진행 상태:** {state}. [실측 결과]({(folder/'comparison.json').as_posix()})\n{status_end}"
    for name in ("RESEARCH_PLAN.md","RESEARCH_REPORT.md","README.md"):
        p=pilot.ROOT/(name if name == "README.md" else "docs/"+name);text=p.read_text(encoding="utf-8")
        a=text.index(status_start);b=text.index(status_end,a)+len(status_end)
        p.write_text(text[:a]+note+text[b:],encoding="utf-8")
