"""Build research diagnostic figures and a local gallery from recorded measurements."""
import argparse
import json
from pathlib import Path
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pilot
from monitor_training import OUTPUT, CONDITIONS, calibration, sha

LABELS = {"teacher":"Teacher 9B NF4","raw_student":"Raw Student 2B",
          "gold_sft":"Gold SFT","response_kd":"Response KD","logit_kd":"Logit KD"}
COLORS = ("#64748b","#94a3b8","#33846c","#325d88","#ad7035")
LETTERS = list("ABCDE")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def confusion(gold, predictions):
    assert len(gold) == len(predictions)
    matrix = np.zeros((5, 6), dtype=float)
    for target, answer in zip(gold, predictions):
        assert target in LETTERS and (answer in LETTERS or answer is None)
        matrix[LETTERS.index(target), LETTERS.index(answer) if answer is not None else 5] += 1
    assert matrix.sum() == len(gold)
    return matrix


def save(fig, name, caption):
    fig.text(.5,.013,caption,ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.065,1,.98))
    fig.savefig(OUTPUT/name,dpi=170,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def collect():
    validation = pilot.read_rows(pilot.DATA/"validation.jsonl")
    ids = [r["source_id"] for r in validation]
    assert len(ids) == len(set(ids)) == 299
    cohorts = {}
    for name, stem in (("teacher","teacher_baseline"),("raw_student","student_baseline")):
        cohorts[name] = [pilot.read_rows(pilot.OUTPUT/f"{stem}.jsonl")]
    for condition in CONDITIONS:
        cohorts[condition] = [pilot.read_rows(pilot.OUTPUT/f"{pilot.validation_stem(condition,seed)}.jsonl")
                              for seed in (42,7,123)]
    for groups in cohorts.values():
        for group in groups:
            assert [r["source_id"] for r in group] == ids
            for source,row in zip(validation,group):
                assert row["gold"] == source["gold"]
                assert row["predicted"] == pilot.parse_answer(row["response"],source)
                assert row["correct"] == (row["predicted"] == source["gold"])
    diagnostic = {}
    for condition in CONDITIONS:
        points = []
        for epoch in range(4):
            path = OUTPUT/"runs"/condition/f"epoch_{epoch}.json"
            if not path.exists():
                continue
            try:
                point = read(path)
            except json.JSONDecodeError:
                continue  # ponytail: a running writer may be incomplete; rebuild when its epoch finishes.
            assert point["status"] == "passed" and point["epoch"] == epoch
            assert point["optimizer_step"] == 32*epoch and not point["test_evaluation"]
            for split, expected in (("train",256),("validation",299)):
                rows = pilot.read_rows(path.with_name(f"epoch_{epoch}_{split}.jsonl"))
                metrics = point["metrics"][split]
                assert len(rows) == metrics["n"] == expected
                source_rows=pilot.read_rows(pilot.DATA/f"{split}.jsonl")
                assert [r["source_id"] for r in rows]==[r["source_id"] for r in source_rows]
                for source,row in zip(source_rows,rows):
                    assert row["gold"]==source["gold"]
                    assert row["predicted"]==pilot.parse_answer(row["response"],source)
                    assert row["correct"]==(row["predicted"]==source["gold"])
                    assert list(row["choice_probabilities"])==[c["label"] for c in source["choices"]]
                assert np.isclose(np.mean([r["gold_ce"] for r in rows]),metrics["gold_ce"])
                assert np.isclose(np.mean([r["correct"] for r in rows]),metrics["accuracy"])
                assert calibration(rows) == metrics["calibration"]
            points.append(point)
        diagnostic[condition] = points
    return validation,cohorts,diagnostic


def accuracy_figure(cohorts):
    baseline = read(pilot.OUTPUT/"baseline_comparison.json")
    result = read(pilot.ROOT/"outputs/logit_kd/comparison.json")
    repeated = read(pilot.ROOT/"outputs/training/repeated/comparison.json")
    assert result["status"] == repeated["status"] == "passed"
    names = list(LABELS)
    means = np.array([np.mean([np.mean([r["correct"] for r in rows]) for rows in cohorts[name]]) for name in names])
    ci = [baseline[f"{name}_accuracy_ci95"] for name in ("teacher","student")]
    ci += [result["aggregate"][name]["conditional_example_ci95"] for name in CONDITIONS]
    ci = np.array(ci)
    for index,name in enumerate(CONDITIONS,2):
        assert np.isclose(means[index],result["aggregate"][name]["mean_accuracy"])
    fig,axes = plt.subplots(1,2,figsize=(12,4.6))
    axes[0].bar(range(5),means*100,color=COLORS,yerr=(np.stack((means-ci[:,0],ci[:,1]-means))*100),capsize=4)
    axes[0].set(xticks=range(5),xticklabels=["Teacher\n9B NF4","Raw\n2B","Gold SFT","Response KD","Logit KD"],
                ylim=(0,103),ylabel="Validation accuracy (%)",title="Final accuracy with 95% example CI")
    for index,value in enumerate(means):
        axes[0].text(index,ci[index,1]*100+1.5,f"{value*100:.2f}%",ha="center",fontsize=9)
    differences = [repeated["aggregate"]["kd_minus_gold"],result["aggregate"]["logit_minus_response"],
                   result["aggregate"]["logit_minus_gold"]]
    for index,row in enumerate(differences):
        mean = row["mean_difference"]*100
        lo,hi = np.array(row["conditional_example_ci95"])*100
        axes[1].errorbar(mean,index,xerr=[[mean-lo],[hi-mean]],fmt="o",color=COLORS[index+2],capsize=5)
        axes[1].annotate(f"{mean:+.2f} pp",(mean,index),xytext=(0,12),textcoords="offset points",ha="center",fontsize=9)
    axes[1].axvline(0,color="#999999",linewidth=1)
    axes[1].set(yticks=range(3),yticklabels=["Response - Gold","Logit - Response\n(primary)","Logit - Gold"],
                xlabel="Accuracy difference (percentage points)",ylim=(-.6,2.6),title="Paired differences: zero is included")
    axes[1].invert_yaxis()
    save(fig,"accuracy.png","Validation n=299. Trained methods: mean over seeds 42/7/123; CI conditional on these seeds. Teacher/raw: one run. Test not evaluated.")


def optimization_figure():
    fig,axes = plt.subplots(2,3,figsize=(13,6.2))
    for index,condition in enumerate(CONDITIONS):
        all_norms=[]
        for seed,color in zip((42,7,123),("#33846c","#325d88","#ad7035")):
            rows = pilot.read_rows(pilot.training_run_path(condition,seed)/"train_log.jsonl")
            assert len(rows)==96
            steps = [r["optimizer_step"] for r in rows]
            all_norms.extend(r["grad_norm"] for r in rows)
            axes[0,index].plot(steps,[r["learning_rate"] for r in rows],color=color,label=f"seed {seed}",alpha=.75)
            axes[1,index].plot(steps,[r["grad_norm"] for r in rows],color=color,alpha=.65,linewidth=1)
        axes[0,index].set(title=LABELS[condition],ylabel="Learning rate",xlabel="Optimizer step")
        axes[0,index].ticklabel_format(axis="y",style="sci",scilimits=(0,0))
        axes[0,index].legend(fontsize=8)
        fraction=np.mean(np.array(all_norms)>1)
        axes[1,index].set_title(f"Clipping applied: {fraction*100:.1f}% of 288 steps",fontsize=10)
        axes[1,index].set_yscale("symlog",linthresh=1)
        axes[1,index].axhline(1,color="#777777",linestyle="--",label="Clip threshold 1.0")
        axes[1,index].set(xlabel="Optimizer step",ylabel="Gradient norm before clipping",ylim=(0,None))
        for axis in axes[:,index]:
            for boundary in (32,64):axis.axvline(boundary,color="#aaaaaa",linestyle=":",linewidth=.8)
            axis.grid(axis="y",alpha=.15)
    save(fig,"optimization.png","Actual optimizer logs, 96 steps per run. Gradient norms use a symlog axis and are recorded before clipping; the dashed line is the cap applied to updates.")


def confusion_figure(validation,cohorts):
    fig,axes = plt.subplots(2,3,figsize=(13,8))
    support = np.array([sum(r["gold"]==label for r in validation) for label in LETTERS])
    for axis,(name,groups) in zip(axes.flat,LABELS.items()):
        rows = cohorts[name]
        matrices = [confusion([r["gold"] for r in group],[r["predicted"] for r in group]) for group in rows]
        counts = np.mean(matrices,axis=0)
        assert np.allclose(counts.sum(axis=1),support)
        percentages = counts/support[:,None]*100
        im = axis.imshow(percentages,cmap="Blues",vmin=0,vmax=100,aspect="auto")
        for i in range(5):
            for j in range(6):
                axis.text(j,i,f"{percentages[i,j]:.0f}%",ha="center",va="center",fontsize=8,
                          color="white" if percentages[i,j]>55 else "#334155")
        axis.set(xticks=range(6),xticklabels=LETTERS+["Invalid"],yticks=range(5),
                 yticklabels=[f"{label} ({n})" for label,n in zip(LETTERS,support)],
                 xlabel="Predicted answer position",ylabel="Gold position (support)",title=LABELS[name])
    legend = axes.flat[5]
    legend.axis("off")
    legend.text(.04,.75,"A-E are answer positions,\nnot semantic classes.\n\nE has only one validation example.\nTrained rows average 3 seeds.\nRows total 100% before rounding.",transform=legend.transAxes,fontsize=11,va="top")
    cax = legend.inset_axes([.04,.19,.8,.055])
    fig.colorbar(im,cax=cax,orientation="horizontal",label="Row percentage")
    save(fig,"confusion.png","Exploratory answer-position confusion, strict generated-response parsing. Support remains 299 examples; seed repeats do not add independent examples.")


def agreement_figure(cohorts):
    teacher = np.array([r["correct"] for r in cohorts["teacher"][0]],dtype=bool)
    fig,axes = plt.subplots(1,3,figsize=(12,4.1))
    for axis,condition in zip(axes,CONDITIONS):
        counts = np.zeros((2,2))
        for rows in cohorts[condition]:
            student = np.array([r["correct"] for r in rows],dtype=bool)
            for i in range(2):
                for j in range(2):counts[i,j] += ((teacher==i)&(student==j)).sum()/3
        assert np.isclose(counts.sum(),299)
        axis.imshow(counts,cmap="Blues",vmin=0,vmax=299)
        for i in range(2):
            for j in range(2):
                axis.text(j,i,f"{counts[i,j]:.1f}",ha="center",va="center",fontsize=14,
                          color="white" if counts[i,j]>150 else "#334155")
        axis.set(xticks=[0,1],xticklabels=["Student wrong","Student correct"],
                 yticks=[0,1],yticklabels=["Teacher wrong","Teacher correct"],title=LABELS[condition])
    save(fig,"agreement.png","Average validation counts over 3 student seeds. Upper-right: student solves teacher errors. Lower-left: remaining teacher/student gap.")


def learning_figure(diagnostic):
    fig,axes = plt.subplots(2,3,figsize=(13,6.4))
    loss_max=max((p["metrics"][split]["gold_ce"] for points in diagnostic.values() for p in points for split in ("train","validation")),default=1)*1.12
    for index,condition in enumerate(CONDITIONS):
        points = diagnostic[condition]
        if not points:
            for axis in axes[:,index]:axis.text(.5,.5,"Waiting for measured epochs",ha="center",transform=axis.transAxes)
        for split,color in (("train","#2563eb"),("validation","#ea580c")):
            epochs = [p["epoch"] for p in points]
            axes[0,index].plot(epochs,[p["metrics"][split]["gold_ce"] for p in points],"o-",color=color,label=split)
            axes[1,index].plot(epochs,[p["metrics"][split]["accuracy"]*100 for p in points],"o-",color=color,label=split)
        axes[0,index].set(title=LABELS[condition],ylabel="Common gold answer + EOS CE",ylim=(0,loss_max))
        axes[1,index].set(ylabel="Generated-response accuracy (%)",ylim=(0,103))
        if points:
            value=points[-1]["metrics"]["validation"]["accuracy"]*100
            axes[1,index].annotate(f"{value:.2f}%",(points[-1]["epoch"],value),xytext=(-2,-9),textcoords="offset points",ha="right",va="top",fontsize=9,color="#ea580c")
        for axis in axes[:,index]:
            axis.set(xlabel="Epoch (0 = before training)",xticks=[0,1,2,3],xlim=(-.1,3.1))
            axis.grid(axis="y",alpha=.15)
            axis.legend(fontsize=8)
    save(fig,"learning_curves.png","Diagnostic replay seed 42: real epoch-end measurements. Both splits use the same gold CE in eval mode and the same strict generation rule.\nValidation is measured without updates or checkpoint selection; this common CE is separate from each KD training objective.")


def calibration_figure(diagnostic):
    fig,axes = plt.subplots(2,3,figsize=(13,6.6))
    for index,condition in enumerate(CONDITIONS):
        if not diagnostic[condition]:
            for axis in axes[:,index]:axis.text(.5,.5,"Waiting for measured scores",ha="center",transform=axis.transAxes)
        else:
            point = diagnostic[condition][-1]
            epoch = point["epoch"]
            metrics = point["metrics"]["validation"]["calibration"]
            rows = pilot.read_rows(OUTPUT/"runs"/condition/f"epoch_{epoch}_validation.jsonl")
            buckets = metrics["bins"]
            axes[0,index].plot([r["confidence"] for r in buckets],[r["accuracy"] for r in buckets],"o-",color=COLORS[index+2])
            for bucket in buckets:
                axes[0,index].annotate(f"n={bucket['n']}",(bucket["confidence"],bucket["accuracy"]),
                    xytext=(3,5),textcoords="offset points",fontsize=7)
            axes[0,index].set(title=f"{LABELS[condition]}: epoch {epoch}\nECE {metrics['ece']:.3f}; Brier {metrics['multiclass_brier']:.3f}")
            for correct,color,label in ((True,"#33846c","correct choice"),(False,"#ad7035","wrong choice")):
                values = [r["choice_confidence"] for r in rows if (r["choice_predicted"]==r["gold"])==correct]
                axes[1,index].hist(values,bins=np.linspace(0,1,11),color=color,alpha=.6,label=label)
        axes[0,index].plot([0,1],[0,1],"--",color="#999999")
        axes[0,index].set(xlim=(0,1),ylim=(0,1.07),xlabel="Mean valid-choice confidence",ylabel="Empirical choice accuracy")
        axes[1,index].set(xlim=(0,1),ylim=(0,299),xlabel="Valid-choice confidence",ylabel="Validation examples")
        if diagnostic[condition]:axes[1,index].legend(fontsize=8)
    save(fig,"calibration.png","Seed 42 validation, n=299. T=1 first-step generation scores normalized over valid choices; calibration of choice-token argmax, not the full response.\n10 fixed confidence bins; empty bins omitted. Brier = sum of squared errors over valid choices, then mean over examples.")


def cost_figure():
    result = read(pilot.ROOT/"outputs/logit_kd/comparison.json")
    fig,axes = plt.subplots(1,3,figsize=(12,4.1))
    for axis,key,factor,title,ylabel in (
        (axes[0],"training_seconds",1,"Student training time","Seconds"),
        (axes[1],"peak_allocated_bytes",2**30,"Peak allocated VRAM","GiB"),
        (axes[2],"adapter_bytes",2**20,"Saved LoRA adapter","MiB")):
        values = np.array([[row["metrics"][name][key]/factor for row in result["per_seed"]] for name in CONDITIONS])
        means = values.mean(axis=1)
        deviations = values.std(axis=1,ddof=1)
        axis.bar(range(3),means,color=COLORS[2:],yerr=values.std(axis=1,ddof=1),capsize=4)
        axis.set(xticks=range(3),xticklabels=["Gold","Response","Logit"],title=title,ylabel=ylabel,ylim=(0,max(means)*1.3))
        for i,value in enumerate(means):axis.text(i,value+deviations[i]+max(means)*.03,f"{value:.2f}" if key!="training_seconds" else f"{value:.1f}",ha="center")
    save(fig,"costs.png","Original 3-seed runs, Student training only. Teacher hard-response generation: 161.8s; soft-target extraction: 37.7s; loading extra.\nDifferent execution times/GPU state and unmerged adapters: these measurements do not establish a speed advantage.")


def gallery(diagnostic):
    observed = {name:[p["epoch"] for p in points] for name,points in diagnostic.items()}
    complete = all(epochs==[0,1,2,3] for epochs in observed.values())
    title = "학습 진단 완료" if complete else "epoch별 학습 진단 측정 중"
    cards = [
        ("learning","learning_curves.png","Train·validation 학습곡선","같은 실제 정답 기준 CE와 실제 생성·파싱 기반 정답률을 함께 봅니다. seed 42 진단 재실행이며 epoch 0은 학습 전입니다. Teacher 응답을 전부 맞추면 정답 기준 train 정확도는 238/256=92.97%가 됩니다. Teacher 오답 18개도 학습 target으로 유지했기 때문입니다."),
        ("learning","training_objective.png","실제 학습 objective loss","기존 9개 실행의 batch loss와 이동평균입니다. CE와 KL+CE는 정의가 달라 방법 간 곡선 높이를 직접 비교하지 않습니다."),
        ("learning","optimization.png","학습률·gradient norm","warmup·학습률 감소와 업데이트 전 gradient norm을 확인합니다. 점선 1.0은 실제 적용한 clipping 상한입니다."),
        ("evaluation","accuracy.png","정확도·성능 차이·95% 신뢰구간","validation 299개에서 비교합니다. 학습한 세 방법은 3개 seed 평균이며 CI는 관측 seed에 조건부인 문항 bootstrap입니다."),
        ("evaluation","confusion.png","선택지 위치 혼동행렬","행 기준 백분율입니다. A–E는 문항마다 의미가 다른 답변 위치이며, E의 실제 정답 문항은 단 1개입니다."),
        ("evaluation","agreement.png","Teacher·Student 정오 패턴","Teacher 오답을 Student가 맞춘 경우와 Teacher가 맞췄지만 Student는 틀린 경우를 확인합니다. 3개 seed의 평균 문항 수입니다."),
        ("confidence","calibration.png","신뢰도·정답률·confidence 분포","유효 선택지로 정규화한 첫 생성 step score의 신뢰도를 평가합니다. 전체 생성 응답의 확률이 아닙니다. ECE와 Brier는 낮을수록 좋습니다."),
        ("confidence","costs.png","시간·VRAM·adapter 용량","기존 3개 seed의 Student 학습 비용입니다. Teacher 신호 생성 비용은 별도로 표기하며 실행 시간으로 속도 우위를 단정하지 않습니다.")
    ]
    sections = []
    for group in ("learning","evaluation","confidence"):
        content = "".join(f'<article><h2>{heading}</h2><p>{description}</p><a href="{file}" target="_blank" aria-label="{heading} 확대"><img src="{file}" alt="{heading}" loading="lazy"></a></article>'
                          for category,file,heading,description in cards if category==group)
        sections.append(f'<section data-group="{group}" {"hidden" if group!="learning" else ""}>{content}</section>')
    status = " · ".join(f"{LABELS[name]}: epoch "+(", ".join(map(str,epochs)) if epochs else "대기") for name,epochs in observed.items())
    html = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LightJudge 학습 진단</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#17212f;font:16px/1.65 system-ui,'Malgun Gothic',sans-serif}main{max-width:1450px;margin:auto;padding:28px 24px 70px}h1{margin:0;font-size:29px}header p{max-width:1040px;color:#4c596b;margin:8px 0}small{color:#64748b}.metrics{display:flex;gap:12px;margin:20px 0;flex-wrap:wrap}.metric{flex:1;min-width:180px;background:white;border:1px solid #dbe2ea;border-radius:12px;padding:16px}.metric strong{display:block;font-size:27px}.status{background:#e8eef5;border-radius:9px;padding:12px 16px;margin:16px 0}.tabs{display:flex;gap:8px;margin:20px 0;flex-wrap:wrap}button{font:inherit;cursor:pointer;padding:9px 18px;border:1px solid #b9c5d3;border-radius:8px;background:white;color:#24344b}button[aria-pressed=true]{color:white;background:#244665;border-color:#244665}button:focus-visible,a:focus-visible{outline:3px solid #f59e0b;outline-offset:3px}article{margin:20px 0;background:white;border:1px solid #dbe2ea;border-radius:12px;padding:20px}h2{font-size:21px;margin:0}article p{color:#4c596b;margin:7px 0 12px}img{display:block;width:100%;height:auto;border-radius:6px}a{color:#245a85}footer{margin-top:28px;color:#64748b;font-size:14px}[hidden]{display:none!important}@media(max-width:650px){main{padding:18px 10px}article{padding:12px}h1{font-size:24px}}</style></head><body><main><header><small>LightJudge / ARC-Challenge / Qwen3.5-9B NF4 → Qwen3.5-2B BF16 + LoRA</small><h1>학습과 평가를 함께 보는 진단 그래프</h1><p>학습 loss, 검증 성능, 일반화 간격, 출력 오류, 신뢰도와 자원 비용을 확인합니다. 그래프를 클릭하면 원본 PNG를 확대할 수 있습니다.</p></header>
<div class="metrics"><div class="metric">Gold SFT · 3 seed 평균<strong>82.05%</strong><small>seed SD 1.65%p</small></div><div class="metric">Response KD · 3 seed 평균<strong>83.61%</strong><small>seed SD 0.33%p</small></div><div class="metric">Logit KD · 3 seed 평균<strong>84.28%</strong><small>seed SD 0.58%p</small></div></div>
<div class="status"><strong>__TITLE__</strong><br>__STATUS__<br><small>원본 비교: train 256 / validation 299 · epoch 진단: 동일 seed 42 재실행 · test 미평가</small></div>
<nav class="tabs" aria-label="그래프 범주"><button data-target="learning" aria-pressed="true">학습 진단</button><button data-target="evaluation" aria-pressed="false">평가 결과</button><button data-target="confidence" aria-pressed="false">신뢰도·비용</button><button onclick="location.reload()">새로고침</button></nav>
__SECTIONS__
<footer>최종 checkpoint는 고정된 3 epoch입니다. validation은 측정에만 사용하며 업데이트·early stopping·checkpoint 선택에 사용하지 않습니다. 갱신: __TIME__<br><a href="summary.json">측정 범위·원본 기록</a> · <a href="protocol.json">진단 재실행 프로토콜</a></footer></main>
<script>document.querySelectorAll('button[data-target]').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('button[data-target]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));document.querySelectorAll('section[data-group]').forEach(s=>s.hidden=s.dataset.group!==button.dataset.target);location.hash=button.dataset.target}));const active=document.querySelector('button[data-target="'+location.hash.slice(1)+'"]');if(active)active.click();</script></body></html>"""
    primary=read(pilot.ROOT/"outputs/logit_kd/comparison.json")["aggregate"]
    metric_cards="".join(f'<div class="metric">{LABELS[name]} · 3 seed 평균<strong>{primary[name]["mean_accuracy"]*100:.2f}%</strong><small>seed SD {primary[name]["seed_sample_sd"]*100:.2f}%p</small></div>' for name in CONDITIONS)
    html=html.replace('<div class="metrics"><div class="metric">Gold SFT · 3 seed 평균<strong>82.05%</strong><small>seed SD 1.65%p</small></div><div class="metric">Response KD · 3 seed 평균<strong>83.61%</strong><small>seed SD 0.33%p</small></div><div class="metric">Logit KD · 3 seed 평균<strong>84.28%</strong><small>seed SD 0.58%p</small></div></div>',f'<div class="metrics">{metric_cards}</div>')
    html = html.replace("__TITLE__",title).replace("__STATUS__",status).replace("__SECTIONS__","".join(sections))
    html = html.replace("__TIME__",datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M KST"))
    (OUTPUT/"index.html").write_text(html,encoding="utf-8")
    return observed,complete


def build():
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    validation,cohorts,diagnostic = collect()
    accuracy_figure(cohorts);optimization_figure();confusion_figure(validation,cohorts)
    agreement_figure(cohorts);learning_figure(diagnostic);calibration_figure(diagnostic);cost_figure()
    shutil.copyfile(pilot.ROOT/"outputs/logit_kd/training_curves.png",OUTPUT/"training_objective.png")
    observed,complete = gallery(diagnostic)
    result = {"status":"complete" if complete else "recording", "n":299,
        "original_training_seeds":[42,7,123],"diagnostic_training_seed":42,
        "diagnostic_epochs":observed,"test_evaluation":False,
        "epoch_metrics":{name:[{"epoch":p["epoch"],"train":p["metrics"]["train"],"validation":p["metrics"]["validation"]} for p in points] for name,points in diagnostic.items()},
        "generation_accuracy_rule":"same greedy max8 strict parser as original evaluation",
        "loss_scope":"common gold answer+EOS CE in eval mode, separate from training objective",
        "calibration_scope":"valid-choice normalized first-step generation scores; choice-token argmax correctness",
        "figures":["learning_curves.png","training_objective.png","optimization.png","accuracy.png",
                   "confusion.png","agreement.png","calibration.png","costs.png"],
        "plot_source_sha256":sha(__file__),"protocol_sha256":sha(OUTPUT/"protocol.json")}
    pilot.write_json(OUTPUT/"summary.json",result)
    shutil.copyfile(__file__,OUTPUT/"plot_source.py")
    print(json.dumps({key:result[key] for key in ("status","diagnostic_epochs","figures")},ensure_ascii=False),flush=True)


def self_check():
    matrix=confusion(["A","B","E"],["A",None,"D"])
    assert matrix.shape==(5,6) and matrix.sum()==3
    assert matrix[0,0]==matrix[1,5]==matrix[4,3]==1
    try:confusion(["A"],["Z"])
    except AssertionError:pass
    else:raise AssertionError("Invalid prediction accepted")
    print("SELF-CHECK PASS: confusion totals, answer positions, invalid outputs and label validation")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check",action="store_true")
    if parser.parse_args().self_check:self_check()
    else:build()
