
"""Compare the fixed Gold SFT/Response KD pilots on the same validation examples."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pilot
from train_student import TRAINING, CONFIG, CONFIG_HASH

def analyze():
    training = {name: json.loads((TRAINING/name/"summary.json").read_text(encoding="utf-8"))
                for name in ("gold_sft","response_kd")}
    gold,kd = training.values()
    assert gold["status"] == kd["status"] == "passed"
    for field in ("config_sha256","initial_adapter_sha256","train_sha256","teacher_responses_sha256",
                  "epoch_order_sha256","trainable_parameters","optimizer_steps_completed",
                  "processed_input_tokens","supervised_target_tokens","train_rows","script_sha256"):
        assert gold[field] == kd[field], f"Conditions differ: {field}"
    assert gold["config_sha256"] == CONFIG_HASH and gold["optimizer_steps_completed"] == 96
    names = ["teacher","raw_student","gold_sft","response_kd"]
    stems = ["teacher_baseline","student_baseline","gold_sft_validation","response_kd_validation"]
    records = []
    for name,stem in zip(names,stems):
        summary=json.loads((pilot.OUTPUT/f"{stem}_summary.json").read_text(encoding="utf-8"))
        assert summary["status"]=="passed" and summary["split"]=="validation"
        if name in training:
            assert summary["training_config_sha256"]==CONFIG_HASH
        records.append(pilot.read_rows(pilot.OUTPUT/f"{stem}.jsonl"))
    validation=pilot.read_rows(pilot.DATA/"validation.jsonl")
    expected=[r["source_id"] for r in validation]
    for cohort in records:
        assert [r["source_id"] for r in cohort]==expected
        for source,row in zip(validation,cohort):
            assert row["gold"]==source["gold"] and row["correct"]==(row["predicted"]==source["gold"])
    values=np.array([[r["correct"] for r in cohort] for cohort in records],dtype=float)
    n=values.shape[1]
    assert n==299
    indices=np.random.default_rng(pilot.SEED).integers(0,n,size=(5000,n))
    bootstrap=values[:,indices].mean(axis=2)
    ci=np.percentile(bootstrap,[2.5,97.5],axis=1).T
    means=values.mean(axis=1)
    metrics={name:{"correct":int(values[i].sum()),"n":n,"accuracy":float(means[i]),
                   "ci95":ci[i].tolist(),"invalid":sum(r["invalid"] for r in records[i]),
                   "hit_token_limit":sum(r["hit_token_limit"] for r in records[i])}
             for i,name in enumerate(names)}
    differences={}
    for label,left,right in (("gold_minus_raw",2,1),("kd_minus_raw",3,1),("kd_minus_gold",3,2)):
        differences[label]={"difference":float(means[left]-means[right]),
            "ci95":np.percentile(bootstrap[left]-bootstrap[right],[2.5,97.5]).tolist(),
            "left_correct_right_wrong":int(((values[left]==1)&(values[right]==0)).sum()),
            "right_correct_left_wrong":int(((values[right]==1)&(values[left]==0)).sum())}
    result={"split":"validation","seed":pilot.SEED,"n":n,"bootstrap_replicates":5000,
        "ci_scope":"paired example bootstrap; one training seed, no training-seed CI",
        "metrics":metrics,"differences":differences,"matched_training_conditions":True,
        "config_sha256":CONFIG_HASH,"initial_adapter_sha256":gold["initial_adapter_sha256"],
        "student_training":True,"test_evaluation":False,
        "training_costs":{name:{k:record[k] for k in ("training_seconds","peak_allocated_bytes",
                          "adapter_bytes","epoch_mean_loss")} for name,record in training.items()}}
    pilot.write_json(TRAINING/"pilot_comparison.json",result)
    disagreements=[{"source_id":source["source_id"],"question":source["question"],
        "choices":source["choices"],"gold":source["gold"],"gold_sft":g["predicted"],
        "response_kd":k["predicted"],"gold_sft_correct":g["correct"],"response_kd_correct":k["correct"]}
        for source,g,k in zip(validation,records[2],records[3]) if g["predicted"]!=k["predicted"]]
    pilot.write_rows(TRAINING/"gold_vs_kd_disagreements.jsonl",disagreements)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    ax=axes[0]
    xlabels=["9B NF4\nTeacher","2B BF16\nRaw","2B LoRA\nGold SFT","2B LoRA\nResponse KD"]
    error=np.stack([means-ci[:,0],ci[:,1]-means])*100
    ax.bar(xlabels,means*100,color=["#8b9299","#b8bfc5","#33846c","#325d88"],yerr=error,capsize=4)
    ax.set(ylim=(0,100),ylabel="Accuracy (%)",title=f"ARC-Challenge validation (n={n})")
    for i,m in enumerate(means):
        ax.text(i,min(ci[i,1]*100+1.2,99),f"{m*100:.2f}%",ha="center",fontsize=9)
    for name,color in (("gold_sft","#33846c"),("response_kd","#325d88")):
        logs=pilot.read_rows(TRAINING/name/"train_log.jsonl")
        assert len(logs)==96
        axes[1].plot([r["optimizer_step"] for r in logs],[r["mean_loss"] for r in logs],
                     label=name,color=color,linewidth=1.2)
    axes[1].set(xlabel="Optimizer step",ylabel="Target token cross-entropy",
                title="Training loss (different targets)")
    axes[1].legend()
    fig.text(0.5,0.01,"95% paired example-bootstrap CI; seed 42; 256 train examples; final test not evaluated",
             ha="center",fontsize=8)
    fig.tight_layout(rect=(0,0.04,1,1))
    fig.savefig(TRAINING/"pilot_comparison.png",dpi=180)
    plt.close(fig)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

def paired_seed_statistics(values, names=("gold_sft","response_kd"), difference_name="kd_minus_gold"):
    assert values.ndim == 3 and values.shape[1] == 2
    per_seed = values.mean(axis=2)
    averaged = values.mean(axis=0)
    n = values.shape[2]
    indices = np.random.default_rng(pilot.SEED).integers(0, n, size=(5000, n))
    sampled = averaged[:, indices].mean(axis=2)
    gaps = per_seed[:, 1] - per_seed[:, 0]
    return per_seed, {
        names[0]: {"mean_accuracy": float(per_seed[:, 0].mean()),
                     "seed_sample_sd": float(per_seed[:, 0].std(ddof=1)),
                     "conditional_example_ci95": np.percentile(sampled[0], [2.5, 97.5]).tolist()},
        names[1]: {"mean_accuracy": float(per_seed[:, 1].mean()),
                        "seed_sample_sd": float(per_seed[:, 1].std(ddof=1)),
                        "conditional_example_ci95": np.percentile(sampled[1], [2.5, 97.5]).tolist()},
        difference_name: {"mean_difference": float(gaps.mean()),
                          "seed_sample_sd": float(gaps.std(ddof=1)),
                          "conditional_example_ci95": np.percentile(sampled[1]-sampled[0], [2.5, 97.5]).tolist()}
    }


def analyze_repeated():
    experiment_path = TRAINING / "repeated" / "experiment.json"
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    seeds = experiment["training_seeds"]
    assert seeds == [42, 7, 123] and experiment["data_sampling_seed"] == pilot.SEED
    for name, digest in experiment["protected_files_sha256"].items():
        assert hashlib.sha256((pilot.ROOT/name).read_bytes()).hexdigest() == digest, name
    validation = pilot.read_rows(pilot.DATA/"validation.jsonl")
    expected = [r["source_id"] for r in validation]
    assert len(expected) == 299
    baseline_config = {k: v for k, v in CONFIG.items() if k != "seed"}
    cohorts, seed_results, initial_hashes, eval_protocol = [], [], [], None
    matched = ("config_sha256", "initial_adapter_sha256", "train_sha256", "teacher_responses_sha256",
               "epoch_order_sha256", "trainable_parameters", "optimizer_steps_completed",
               "processed_input_tokens", "supervised_target_tokens", "train_rows", "sample_ids", "script_sha256")
    for seed in seeds:
        summaries, logs, predictions, eval_summaries = [], [], [], []
        for condition in ("gold_sft", "response_kd"):
            run = pilot.training_run_path(condition, seed)
            summary = json.loads((run/"summary.json").read_text(encoding="utf-8"))
            assert summary["status"] == "passed" and summary["config"]["seed"] == seed
            assert summary["config_sha256"] == hashlib.sha256(json.dumps(summary["config"],sort_keys=True).encode()).hexdigest()
            assert {k:v for k,v in summary["config"].items() if k != "seed"} == baseline_config
            assert summary["optimizer_steps_completed"] == 96 and summary["train_rows"] == 256
            assert summary["base_checkpoint_start"] and summary["previous_adapter"] is None
            assert not summary["validation_used_for_training"] and not summary["test_evaluation"]
            original = json.loads((pilot.OUTPUT/"teacher_responses_summary.json").read_text(encoding="utf-8"))
            assert summary["train_sha256"] == original["data_sha256"]
            assert summary["teacher_responses_sha256"] == original["output_sha256"]
            for name,digest in summary["adapter_files_sha256"].items():
                assert hashlib.sha256((run/"adapter"/name).read_bytes()).hexdigest() == digest
            summaries.append(summary)
            log = pilot.read_rows(run/"train_log.jsonl")
            assert len(log) == 96 and [r["optimizer_step"] for r in log] == list(range(1,97))
            from collections import Counter
            assert Counter(i for row in log for i in row["sample_ids"]) == Counter({i:3 for i in summary["sample_ids"]})
            logs.append(log)
            stem = pilot.validation_stem(condition, seed)
            evaluation = json.loads((pilot.OUTPUT/f"{stem}_summary.json").read_text(encoding="utf-8"))
            assert evaluation["status"] == "passed" and evaluation["completed"] == 299
            assert evaluation["split"] == "validation" and not evaluation["test_evaluation"]
            assert evaluation.get("training_seed", 42) == seed
            assert evaluation["training_config_sha256"] == summary["config_sha256"]
            assert evaluation["training_summary_sha256"] == hashlib.sha256((run/"summary.json").read_bytes()).hexdigest()
            protocol = {k:evaluation[k] for k in ("model_id","revision","data_sha256","thinking","do_sample",
                        "max_new_tokens","max_input_tokens","answer_rule","prompt_template","quantization")}
            if eval_protocol is None:
                eval_protocol = protocol
            assert protocol == eval_protocol and evaluation["sample_ids"] == expected
            rows = pilot.read_rows(pilot.OUTPUT/f"{stem}.jsonl")
            assert [r["source_id"] for r in rows] == expected
            for source,row in zip(validation, rows):
                assert row["gold"] == source["gold"]
                assert row["predicted"] == pilot.parse_answer(row["response"], source)
                assert row["correct"] == (row["predicted"] == source["gold"])
                assert row["invalid"] == (row["predicted"] is None)
            assert sum(r["correct"] for r in rows) == evaluation["correct"]
            predictions.append(rows)
            eval_summaries.append(evaluation)
        gold,kd = summaries
        for field in matched:
            assert gold[field] == kd[field], (seed,field)
        for a,b in zip(*logs):
            assert all(a[key] == b[key] for key in ("epoch","optimizer_step","learning_rate","sample_ids"))
        initial_hashes.append(gold["initial_adapter_sha256"])
        cohorts.append([[r["correct"] for r in rows] for rows in predictions])
        metrics = {condition: {"correct": e["correct"], "accuracy": e["accuracy"],
                    "invalid": e["invalid"], "hit_token_limit": e["hit_token_limit"],
                    "training_seconds": s["training_seconds"], "peak_allocated_bytes": s["peak_allocated_bytes"],
                    "adapter_bytes": s["adapter_bytes"], "inference_seconds": e["inference_seconds"]}
                   for condition,s,e in zip(("gold_sft","response_kd"),summaries,eval_summaries)}
        seed_results.append({"seed":seed,"metrics":metrics,
            "kd_minus_gold":metrics["response_kd"]["accuracy"]-metrics["gold_sft"]["accuracy"],
            "initial_adapter_sha256":gold["initial_adapter_sha256"]})
    assert len(set(initial_hashes)) == 3
    values = np.array(cohorts,dtype=float)
    per_seed, aggregate = paired_seed_statistics(values)
    result = {"status":"passed","split":"validation","n":299,"training_seeds":seeds,
              "data_sampling_seed":42,"fixed_train_rows":256,"per_seed":seed_results,"aggregate":aggregate,
              "bootstrap_replicates":5000,"bootstrap_seed":42,"ci_scope":experiment["ci_scope"],
              "seed_sd_definition":"sample SD (ddof=1) of 3 training-seed accuracies/differences",
              "matched_training_within_seed":True,"same_protocol_across_seeds":True,
              "original_artifacts_preserved":True,"test_evaluation":False,
              "experiment_sha256":hashlib.sha256(experiment_path.read_bytes()).hexdigest(),
              "analysis_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output = TRAINING/"repeated"/"comparison.json"
    pilot.write_json(output,result)
    fig,axes = plt.subplots(1,2,figsize=(10.4,4.7))
    for i,(seed,color) in enumerate(zip(seeds,("#33846c","#325d88","#ad7035"))):
        axes[0].plot([0,1],per_seed[i]*100,"o-",label=f"seed {seed}",color=color,alpha=.85)
    baseline=json.loads((TRAINING/"pilot_comparison.json").read_text(encoding="utf-8"))["metrics"]
    axes[0].axhline(baseline["raw_student"]["accuracy"]*100,color="#999999",linestyle="--",label="Raw Student")
    axes[0].axhline(baseline["teacher"]["accuracy"]*100,color="#aaaaaa",linestyle=":",label="Teacher")
    axes[0].set(xticks=[0,1],xticklabels=["Gold SFT","Response KD"],ylabel="Validation accuracy (%)",
                ylim=(72,96),xlim=(-.3,1.3),title="Same protocol; different training seeds")
    axes[0].legend(fontsize=8,loc="upper right")
    for x,name in enumerate(("gold_sft","response_kd")):
        axes[0].text(x,73.2,f"Mean {aggregate[name]['mean_accuracy']*100:.2f}%\nSD {aggregate[name]['seed_sample_sd']*100:.2f} pp",ha="center",fontsize=9)
    gaps=(per_seed[:,1]-per_seed[:,0])*100
    axes[1].scatter([0,1,2],gaps,color=["#33846c","#325d88","#ad7035"],s=45)
    mean=aggregate["kd_minus_gold"]["mean_difference"]*100
    lo,hi=np.array(aggregate["kd_minus_gold"]["conditional_example_ci95"])*100
    axes[1].errorbar([3],[mean],yerr=[[mean-lo],[hi-mean]],fmt="o",color="#333333",capsize=5)
    for x,gap in enumerate(gaps):
        axes[1].annotate(f"{gap:+.2f}",(x,gap),xytext=(0,10),textcoords="offset points",ha="center",fontsize=9)
    axes[1].axhline(0,color="#999999",linewidth=1)
    axes[1].set(xticks=[0,1,2,3],xticklabels=[str(s) for s in seeds]+["Mean + CI"],
                ylabel="KD minus Gold (percentage points)",title="Paired accuracy differences",xlim=(-.4,3.5))
    axes[1].text(.5,.04,f"Mean {mean:+.2f} pp; seed SD {aggregate['kd_minus_gold']['seed_sample_sd']*100:.2f} pp",
                 transform=axes[1].transAxes,ha="center",fontsize=9)
    fig.text(.5,.01,"95% CI: paired example bootstrap, conditional on these 3 seeds; fixed 256 train / 299 validation; test not evaluated",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.035,1,1))
    fig.savefig(TRAINING/"repeated"/"comparison.png",dpi=180)
    plt.close(fig)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)



def analyze_logit():
    import logit_kd
    from train_student import run_config
    folder=logit_kd.OUTPUT
    protocol=json.loads((folder/"protocol.json").read_text(encoding="utf-8"))
    previous=json.loads((folder/"previous_artifacts.json").read_text(encoding="utf-8"))
    for name,digest in previous["sha256"].items():
        assert hashlib.sha256((pilot.ROOT/name).read_bytes()).hexdigest()==digest,name
    seeds=protocol["training_seeds"]
    assert seeds==[42,7,123]
    validation=pilot.read_rows(pilot.DATA/"validation.jsonl")
    expected=[r["source_id"] for r in validation]
    assert len(expected)==299
    all_records,per_seed=[],[]
    condition_names=("gold_sft","response_kd","logit_kd")
    eval_protocol=None
    fixed_fields=("initial_adapter_sha256","epoch_order_sha256","train_sha256","teacher_responses_sha256",
                  "trainable_parameters","optimizer_steps_completed","processed_input_tokens",
                  "supervised_target_tokens","train_rows","sample_ids","versions")
    target_summary=json.loads((folder/"soft_targets_summary.json").read_text(encoding="utf-8"))
    assert target_summary["status"]=="passed"
    for seed in seeds:
        trained,records,metrics,logs={},{},{},{}
        for condition in condition_names:
            run=pilot.training_run_path(condition,seed)
            train=json.loads((run/"summary.json").read_text(encoding="utf-8"))
            assert train["status"]=="passed" and train["config"]["seed"]==seed
            config,config_hash=run_config(condition,seed)
            assert train["config"]==config and train["config_sha256"]==config_hash
            assert train["base_checkpoint_start"] and train["previous_adapter"] is None
            assert train["optimizer_steps_completed"]==96 and not train["validation_used_for_training"]
            assert not train["test_evaluation"]
            trained[condition]=train
            logs[condition]=pilot.read_rows(run/"train_log.jsonl")
            assert len(logs[condition])==96
            for name,digest in train["adapter_files_sha256"].items():
                assert hashlib.sha256((run/"adapter"/name).read_bytes()).hexdigest()==digest
            stem=pilot.validation_stem(condition,seed)
            evaluation=json.loads((pilot.OUTPUT/f"{stem}_summary.json").read_text(encoding="utf-8"))
            assert evaluation["status"]=="passed" and evaluation["split"]=="validation"
            assert evaluation["completed"]==299 and not evaluation["test_evaluation"]
            assert evaluation["training_config_sha256"]==config_hash
            assert evaluation["training_summary_sha256"]==hashlib.sha256((run/"summary.json").read_bytes()).hexdigest()
            rule={k:evaluation[k] for k in ("data_sha256","model_id","revision","thinking","do_sample",
                    "max_new_tokens","max_input_tokens","answer_rule","prompt_template","quantization")}
            if eval_protocol is None:
                eval_protocol=rule
            assert rule==eval_protocol and evaluation["sample_ids"]==expected
            cohort=pilot.read_rows(pilot.OUTPUT/f"{stem}.jsonl")
            assert [r["source_id"] for r in cohort]==expected
            for source,row in zip(validation,cohort):
                assert row["gold"]==source["gold"]
                assert row["predicted"]==pilot.parse_answer(row["response"],source)
                assert row["correct"]==(row["predicted"]==source["gold"])
            assert sum(r["correct"] for r in cohort)==evaluation["correct"]
            assert evaluation["accuracy"]==evaluation["correct"]/299
            records[condition]=cohort
            metrics[condition]={"correct":evaluation["correct"],"accuracy":evaluation["accuracy"],
                "invalid":evaluation["invalid"],"hit_token_limit":evaluation["hit_token_limit"],
                "training_seconds":train["training_seconds"],"inference_seconds":evaluation["inference_seconds"],
                "peak_allocated_bytes":train["peak_allocated_bytes"],"adapter_bytes":train["adapter_bytes"]}
        response=trained["response_kd"]
        for condition in ("gold_sft","logit_kd"):
            for key in fixed_fields:
                assert trained[condition][key]==response[key],(seed,condition,key)
            for a,b in zip(logs[condition],logs["response_kd"]):
                assert all(a[k]==b[k] for k in ("sample_ids","learning_rate","epoch","optimizer_step"))
        new=trained["logit_kd"]
        assert new["soft_targets_sha256"]==target_summary["output_sha256"]
        assert new["distillation_protocol_sha256"]==hashlib.sha256((folder/"protocol.json").read_bytes()).hexdigest()
        assert new["distillation_script_sha256"]==hashlib.sha256((folder/"distillation_source.py").read_bytes()).hexdigest()
        all_records.append([[r["correct"] for r in records[name]] for name in condition_names])
        per_seed.append({"seed":seed,"metrics":metrics})
    values=np.array(all_records,dtype=float)
    _,primary=paired_seed_statistics(values[:,[1,2]],("response_kd","logit_kd"),"logit_minus_response")
    _,secondary=paired_seed_statistics(values[:,[0,2]],("gold_sft","logit_kd"),"logit_minus_gold")
    aggregate={**primary,"gold_sft":secondary["gold_sft"],"logit_minus_gold":secondary["logit_minus_gold"]}
    result={"status":"passed","split":"validation","n":299,"training_seeds":seeds,"fixed_train_rows":256,
        "per_seed":per_seed,"aggregate":aggregate,"primary_comparison":"logit_minus_response",
        "secondary_comparison":"logit_minus_gold","bootstrap_replicates":5000,"bootstrap_seed":42,
        "ci_scope":"Paired example bootstrap of mean correctness over these 3 seeds; conditional on observed seeds, not training-seed CI",
        "same_initial_state_order_budget_within_seed":True,"original_artifacts_preserved":True,"test_evaluation":False,
        "protocol_sha256":hashlib.sha256((folder/"protocol.json").read_bytes()).hexdigest(),
        "analysis_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "soft_targets":{"bytes":target_summary["output_bytes"],"seconds":target_summary["extraction_seconds"],
                        "sha256":target_summary["output_sha256"]}}
    pilot.write_json(folder/"comparison.json",result)
    accuracies=values.mean(axis=2)*100
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    colors=("#33846c","#325d88","#ad7035")
    for i,(seed,color) in enumerate(zip(seeds,colors)):
        axes[0].plot([0,1,2],accuracies[i],"o-",label=f"seed {seed}",color=color,alpha=.85)
    baseline=json.loads((TRAINING/"pilot_comparison.json").read_text(encoding="utf-8"))["metrics"]
    axes[0].axhline(baseline["raw_student"]["accuracy"]*100,color="#999999",linestyle="--",label="Raw Student")
    axes[0].set(xticks=[0,1,2],xticklabels=["Gold SFT","Response KD","Logit KD"],
        ylabel="Validation accuracy (%)",ylim=(70,95),xlim=(-.3,2.3),title="Same inputs, Student, and training budget")
    axes[0].legend(fontsize=8,loc="upper right")
    for x,name in enumerate(condition_names):
        axes[0].text(x,71.0,f"{aggregate[name]['mean_accuracy']*100:.2f}%\nSD {aggregate[name]['seed_sample_sd']*100:.2f} pp",ha="center",fontsize=9)
    gaps=accuracies[:,2]-accuracies[:,1]
    for i,color in enumerate(colors):
        axes[1].scatter(i,gaps[i],color=color)
        axes[1].annotate(f"{gaps[i]:+.2f}",(i,gaps[i]),xytext=(0,10),textcoords="offset points",ha="center",fontsize=9)
    gap=aggregate["logit_minus_response"]
    mean=gap["mean_difference"]*100
    lo,hi=np.array(gap["conditional_example_ci95"])*100
    axes[1].errorbar([3],[mean],yerr=[[mean-lo],[hi-mean]],fmt="o",color="#333333",capsize=5)
    axes[1].axhline(0,color="#999999",linewidth=1)
    axes[1].set(xticks=[0,1,2,3],xticklabels=[str(s) for s in seeds]+["Mean + CI"],
        ylabel="Logit minus Response (percentage points)",title="Primary paired comparison",xlim=(-.4,3.5))
    axes[1].text(.5,.04,f"Mean {mean:+.2f} pp; seed SD {gap['seed_sample_sd']*100:.2f} pp",
        transform=axes[1].transAxes,ha="center",fontsize=9)
    fig.text(.5,.01,"T=2; full-vocabulary answer KL + shared EOS CE; 95% CI conditional on 3 seeds; test not evaluated",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.035,1,1))
    fig.savefig(folder/"comparison.png",dpi=180)
    plt.close(fig)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def plot_training_curves():
    folder=pilot.ROOT/"outputs/logit_kd"
    seeds=[42,7,123]
    conditions=("gold_sft","response_kd","logit_kd")
    titles=("Gold SFT: answer + EOS CE","Response KD: answer + EOS CE","Logit KD: scaled answer KL + EOS CE")
    fig,axes=plt.subplots(1,3,figsize=(13,4.5))
    plotted={}
    for condition,title,ax in zip(conditions,titles,axes):
        plotted[condition]={}
        for seed,color in zip(seeds,("#33846c","#325d88","#ad7035")):
            run=pilot.training_run_path(condition,seed)
            path=run/"train_log.jsonl"
            if not path.exists():
                continue
            lines=path.read_text(encoding="utf-8").splitlines(keepends=True)
            logs=[json.loads(line) for line in lines if line.endswith("\n") and line.strip()]
            if not logs:
                continue
            x=np.array([row["optimizer_step"] for row in logs])
            y=np.array([row["mean_loss"] for row in logs])
            assert np.isfinite(y).all() and np.all(np.diff(x)==1)
            summary=json.loads((run/"summary.json").read_text(encoding="utf-8"))
            state="complete" if summary["status"]=="passed" else "running"
            plotted[condition][str(seed)]={"steps":len(logs),"status":state}
            ax.plot(x,y,color=color,linewidth=.7,alpha=.25)
            if len(y)>=8:
                smooth=np.convolve(y,np.ones(8)/8,mode="valid")
                ax.plot(x[7:],smooth,color=color,linewidth=1.8,label=f"seed {seed} ({state})")
            else:
                ax.plot(x,y,color=color,linewidth=1.8,label=f"seed {seed} ({state})")
        for boundary in (32,64):
            ax.axvline(boundary,color="#aaaaaa",linestyle=":",linewidth=.8)
        ax.set(xlim=(1,96),ylim=(0,None),xlabel="Optimizer step",ylabel="Training objective loss",title=title)
        ax.legend(fontsize=8)
        ax.grid(axis="y",alpha=.15)
    fig.text(.5,.01,"Thin: actual batch loss; thick: 8-step moving mean; dotted: epoch boundaries; different objectives; no per-epoch validation",
        ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(folder/"training_curves.png",dpi=180)
    plt.close(fig)
    pilot.write_json(folder/"training_curves_summary.json",{
        "source":"actual train_log.jsonl; no interpolated missing steps","moving_mean_steps":8,
        "per_condition":plotted,"per_epoch_validation":False,"test_evaluation":False})
    print(json.dumps(plotted),flush=True)

def self_check():
    values=np.array([[[1,0],[0,1]],[[0,1],[1,0]],[[1,1],[1,1]]],dtype=float)
    per_seed,metrics=paired_seed_statistics(values)
    assert per_seed.shape == (3,2)
    assert metrics["gold_sft"]["mean_accuracy"] == 2/3
    assert metrics["kd_minus_gold"]["mean_difference"] == 0
    assert metrics["kd_minus_gold"]["seed_sample_sd"] == 0
    assert metrics["kd_minus_gold"]["conditional_example_ci95"] == [0.,0.]
    _,renamed=paired_seed_statistics(values,("response_kd","logit_kd"),"logit_minus_response")
    assert renamed["logit_minus_response"] == metrics["kd_minus_gold"]
    print("SELF-CHECK PASS: average across seeds; paired example resampling; identical differences have zero CI")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeated",action="store_true",help="Compare the fixed seeds 42,7,123")
    parser.add_argument("--logit",action="store_true",help="Compare Gold, Response, Logit at the same seeds")
    parser.add_argument("--curves",action="store_true",help="Plot actual loss logs, including unfinished runs")
    parser.add_argument("--self-check",action="store_true")
    args=parser.parse_args()
    if args.self_check:
        self_check()
    elif args.curves:
        plot_training_curves()
    elif args.logit:
        analyze_logit()
    elif args.repeated:
        analyze_repeated()
    else:
        analyze()

