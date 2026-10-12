"""Frozen seven-condition statistics, observed curves and the three project documents."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import gate_experiment as gate
import gate_evaluation as evaluation
import kl_experiment as legacy
import pilot
from teacher_quantization import ROOT

os.environ.setdefault("MPLCONFIGDIR", str(gate.OUT / ".matplotlib"))
from analyze_kl import aggregate, vectors, NAMES

LABELS = {**NAMES, "adaptive_soft": "Adaptive soft gate"}


def paired_statistics(records):
    n = len(records[gate.CONDITIONS[0]][0])
    assert all(len(seeds) == 3 and all(len(rows) == n for rows in seeds) for seeds in records.values())
    rng = np.random.default_rng(evaluation.SPEC["bootstrap_seed"])
    indices = rng.integers(0, n, size=(evaluation.SPEC["bootstrap_replicates"], n))
    weights = np.array([np.bincount(row, minlength=n) for row in indices], dtype=float) / n
    summaries, draws = {}, {}
    for condition, seeds in records.items():
        summaries[condition], draws[condition] = aggregate([vectors(rows) for rows in seeds], indices, weights)
    comparisons = {}
    for a, b in evaluation.SPEC["pairs"]:
        comparisons[f"{a}_minus_{b}"] = {}
        for metric in ("accuracy", "choice_nll"):
            comparisons[f"{a}_minus_{b}"][metric] = {
                "mean_difference": summaries[a][metric]["mean"] - summaries[b][metric]["mean"],
                "ci95": np.quantile(draws[a][metric] - draws[b][metric], [.025, .975]).tolist(),
                "per_seed_difference": (np.array(summaries[a][metric]["per_seed"]) - np.array(summaries[b][metric]["per_seed"])).tolist()}
    return summaries, comparisons


def describe_signals(records, condition):
    assert records and all(np.isfinite([r[k] for k in ("forward_kl", "reverse_kl",
        "teacher_confidence", "student_confidence", "forward_weight", "loss")]).all() for r in records)
    gap = np.array([r["teacher_confidence"] - r["student_confidence"] for r in records])
    values = np.array([r["forward_weight"] for r in records])
    if condition == "fixed_mix":
        expected = np.full(len(records), .5)
    elif condition == "adaptive_mix":
        expected = (gap >= 0).astype(float)
    else:
        assert condition == "adaptive_soft"
        expected = 1 / (1 + np.exp(-(gap - .05)))
    np.testing.assert_allclose(values, expected, rtol=1e-6, atol=1e-6)
    return {"n_training_forwards": len(records), "forward_weight_mean": float(values.mean()),
        "forward_weight_sd": float(values.std(ddof=1)),
        "forward_weight_quantiles_0_25_50_75_100": np.quantile(values, [0, .25, .5, .75, 1]).tolist(),
        "confidence_gap_mean": float(gap.mean()), "counterfactual_hard_forward_fraction": float((gap >= 0).mean()),
        "scope": "T=2 full-vocabulary entropy during training including dropout; not answer correctness confidence"}


def analyze(split):
    assert split in ("validation", "test")
    manifest = evaluation.require_manifest()
    protocol = gate.require_protocol()
    rows, _ = legacy.rows_for("full", split)
    sources, records, baselines = {}, {}, {}
    for role in ("teacher", "student"):
        folder = gate.OUT if split == "test" else legacy.OUT
        path = folder / "scores" / split / f"{role}_baseline_summary.json"
        baseline_records, summary = evaluation.records_checked(path, rows, split)
        if split == "test":
            assert summary["test_manifest_sha256"] == gate.sha(gate.OUT / "test_manifest.json")
        baselines[role] = {"metrics": summary["metrics"], "records": baseline_records}
        sources.update({str(p.relative_to(ROOT)): gate.sha(p) for p in
            (path, path.with_name(path.name.replace("_summary.json", ".jsonl")))})
    for condition in gate.CONDITIONS:
        records[condition] = []
        for seed in legacy.SEEDS:
            path = evaluation.validation_path(protocol, condition, seed) if split == "validation" else (
                gate.OUT / "scores/test" / f"{condition}_{seed}_summary.json")
            values, summary = evaluation.records_checked(path, rows, split)
            if split == "test":
                assert summary["test_manifest_sha256"] == gate.sha(gate.OUT / "test_manifest.json")
            records[condition].append(values)
            sources.update({str(p.relative_to(ROOT)): gate.sha(p) for p in
                (path, path.with_name(path.name.replace("_summary.json", ".jsonl")))})
    folder = gate.OUT / "analysis" / split
    if folder.exists():
        saved = gate.read(folder / "comparison.json")
        assert saved["status"] == "passed" and saved["score_files_sha256"] == sources
        assert saved["test_manifest_sha256"] == gate.sha(gate.OUT / "test_manifest.json")
        gate.checked_files(saved["analysis_files_sha256"])
        destination = ROOT / "results/gate_comparison_v4" / split
        if not destination.exists():
            shutil.copytree(folder, destination)
        assert all(gate.sha(destination / p.name) == gate.sha(p) for p in folder.iterdir() if p.is_file())
        return saved
    summary, comparisons = paired_statistics(records)
    timing, curves, signals = {}, {}, {}
    for condition in ("fixed_mix", "adaptive_mix", "adaptive_soft"):
        timing[condition], curves[condition], signals[condition] = [], [], []
        for seed in legacy.SEEDS:
            run = gate.run_path(condition, seed)
            training = gate.read(run / "summary.json")
            timing[condition].append({"seed": seed, **{k: training[k] for k in (
                "training_seconds", "setup_seconds", "epoch_zero_monitor_seconds", "epoch_monitoring_seconds",
                "total_job_seconds", "peak_allocated_bytes", "peak_reserved_bytes", "measurement_only")}})
            points = [gate.read(run / f"epoch_{epoch}.json") for epoch in range(4)]
            assert [p["training_seconds"] for p in points] == sorted(p["training_seconds"] for p in points)
            curves[condition].append({"seed": seed, "observed_points": [{"epoch": p["epoch"],
                "optimizer_step": p["optimizer_step"], "training_seconds": p["training_seconds"],
                "accuracy": p["metrics"]["accuracy"], "gold_ce": p["metrics"]["gold_ce"]} for p in points]})
            values = pilot.read_rows(run / "signal_log.jsonl")
            signals[condition].append({"seed": seed, **describe_signals(values, condition),
                "epochs": [{"epoch": epoch, **describe_signals([r for r in values if r["epoch"] == epoch], condition)} for epoch in (1, 2, 3)]})
    wrong = [i for i, r in enumerate(baselines["teacher"]["records"]) if not r["correct"]]
    initially_correct = [i for i in wrong if baselines["student"]["records"][i]["correct"]]
    errors = {c: [{"seed": seed, "teacher_wrong_n": len(wrong),
        "student_correct_on_teacher_wrong": sum(records[c][s][i]["correct"] for i in wrong),
        "same_teacher_wrong": sum(records[c][s][i]["predicted"] == baselines["teacher"]["records"][i]["predicted"] for i in wrong),
        "initial_student_correct_n": len(initially_correct),
        "initial_student_correct_lost": sum(not records[c][s][i]["correct"] for i in initially_correct)}
        for s, seed in enumerate(legacy.SEEDS)] for c in gate.CONDITIONS}
    teacher_preparation = gate.read(legacy.OUT / "targets/full/summary.json")
    result = {"status": "building", "split": split, "n": len(rows), "conditions": gate.CONDITIONS,
        "seeds": legacy.SEEDS, "aggregate": summary, "comparisons": comparisons,
        "analysis_spec": manifest["analysis_spec"], "test_manifest_sha256": gate.sha(gate.OUT / "test_manifest.json"),
        "score_files_sha256": sources, "timing": timing, "observed_validation_curves": curves,
        "training_signals": signals, "teacher_error_patterns": errors,
        "baselines": {role: item["metrics"] for role, item in baselines.items()},
        "fixed42_replay": {"used_for_test": False, "used_for_timing": True,
            "tensor_sha256_matches_original": gate.read(gate.run_path("fixed_mix", 42) / "summary.json")["replay_adapter_tensor_sha256_matches_original"],
            "final_validation_prediction_mismatches": gate.read(gate.OUT / "scores/validation/fixed_mix_42_summary.json")["curve_prediction_mismatches"]},
        "teacher_cache_preparation": {"shared_full_train_extraction_seconds": teacher_preparation["extraction_seconds"],
            "hard_and_soft_costs_separately_measured": False,
            "scope": "legacy full-cache extraction includes response generation and probabilities; cannot split retrospectively"},
        "limits": ["single ARC/Qwen setting; public checkpoint ARC exposure unverified",
            "CI conditional on three observed seeds; small teacher-error subsets descriptive",
            "reference-kernel timing, including logging/final save, excluding loading/evaluation",
            "no arbitrary target or time-accuracy composite ranking; no causal smoothing-only claim"]}
    folder.mkdir(parents=True)
    plots(folder, result)
    result["analysis_files_sha256"] = {str(p.relative_to(ROOT)): gate.sha(p) for p in folder.iterdir() if p.is_file()}
    result["status"] = "passed"
    pilot.write_json(folder / "comparison.json", result)
    destination = ROOT / "results/gate_comparison_v4" / split
    assert not destination.exists(), "Published results preserved; inspect before replacement"
    shutil.copytree(folder, destination)
    return result


def plots(folder, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(folder / name, dpi=180, bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, condition in enumerate(gate.CONDITIONS):
        score = result["aggregate"][condition]["accuracy"]
        ax.bar(i, score["mean"] * 100, color="#6288ad")
        ax.scatter([i] * 3, np.array(score["per_seed"]) * 100, color="#b74936", s=22, zorder=3)
    ax.set_xticks(range(7), [LABELS[c] for c in gate.CONDITIONS], rotation=25, ha="right")
    for role, style in (("teacher", "--"), ("student", ":")):
        ax.axhline(result["baselines"][role]["accuracy"] * 100, linestyle=style, label=role)
    ax.set_ylabel("Generation accuracy (%)")
    ax.set_title(f"ARC-Challenge {result['split']}, n={result['n']}; dots: three training seeds")
    ax.legend(); ax.grid(axis="y", alpha=.2)
    save(fig, "accuracy.png")
    fig, ax = plt.subplots(figsize=(8, 3))
    names = []
    for i, (a, b) in enumerate(evaluation.SPEC["pairs"]):
        item = result["comparisons"][f"{a}_minus_{b}"]["accuracy"]
        low, high = np.array(item["ci95"]) * 100
        ax.plot([low, high], [i, i], color="#285a7c")
        ax.scatter(item["mean_difference"] * 100, i, color="#285a7c")
        names.append(LABELS[a] + " minus " + LABELS[b])
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_yticks(range(3), names); ax.set_xlabel("Accuracy difference (%p), paired 95% CI")
    ax.set_title("Hard-Soft primary; other comparisons exploratory\nConditional on the observed training seeds")
    save(fig, "paired_ci.png")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    colors = {"fixed_mix": "#487da5", "adaptive_mix": "#b45c2f", "adaptive_soft": "#3f885b"}
    for condition, color in colors.items():
        for i, item in enumerate(result["observed_validation_curves"][condition]):
            points = item["observed_points"]
            times = [p["training_seconds"] / 60 for p in points]
            label = LABELS[condition] if i == 0 else None
            axes[0, 0].plot(times, [100 * p["accuracy"] for p in points], "o-", color=color, alpha=.65, label=label)
            axes[0, 1].plot(times, [p["gold_ce"] for p in points], "o-", color=color, alpha=.65)
            log = pilot.read_rows(gate.run_path(condition, item["seed"]) / "train_log.jsonl")
            axes[1, 0].plot([r["optimizer_step"] for r in log], [r["mean_loss"] for r in log], color=color, alpha=.4)
            signals = result["training_signals"][condition][i]["epochs"]
            axes[1, 1].plot([p["epoch"] for p in signals], [p["forward_weight_mean"] for p in signals], "o-", color=color, alpha=.65)
    axes[0, 0].set_ylabel("Validation generation accuracy (%)"); axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_ylabel("Common validation Gold CE")
    for ax in axes[0]: ax.set_xlabel("Training phase time (min)")
    axes[1, 0].set_xlabel("Optimizer step"); axes[1, 0].set_ylabel("Batch training objective")
    axes[1, 0].set_title("Distinct losses; magnitudes not comparable", fontsize=10)
    axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("Mean forward KL weight during training")
    for ax in axes.flat: ax.grid(alpha=.2)
    fig.suptitle("Matched nine-run cohort; validation n=299 at epochs 0/1/2/3\nObserved points; epoch evaluation time excluded")
    save(fig, "learning_curves.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    for condition, color in colors.items():
        times = np.array([r["training_seconds"] for r in result["timing"][condition]]) / 60
        scores = np.array(result["aggregate"][condition]["accuracy"]["per_seed"]) * 100
        ax.scatter(times, scores, color=color, label=LABELS[condition])
        for x, y, seed in zip(times, scores, legacy.SEEDS): ax.annotate(str(seed), (x, y), fontsize=8)
    ax.set_xlabel("Training phase time (min), matched cohort"); ax.set_ylabel("Final generation accuracy (%)")
    ax.set_title(f"{result['split']}: observed time-accuracy tradeoff\nFixed42 accuracy uses the original final checkpoint")
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    save(fig, "time_accuracy.png")


def publish():
    result = gate.read(gate.OUT / "analysis/test/comparison.json")
    assert result["status"] == "passed" and result["n"] == 1172
    manifest = evaluation.require_manifest()
    gate.checked_files(manifest["public_checkpoint_files_sha256"])
    table = "| 조건 | Test Accuracy 평균±SD | seed 42 / 7 / 123 | 선택지 NLL |\n|---|---|---|---|\n"
    for condition in gate.CONDITIONS:
        item = result["aggregate"][condition]
        a = item["accuracy"]
        table += f"| {LABELS[condition]} | {a['mean']*100:.2f}±{a['seed_sample_sd']*100:.2f}% | " + " / ".join(f"{x*100:.2f}%" for x in a["per_seed"]) + f" | {item['choice_nll']['mean']:.4f} |\n"
    comparisons = "| 비교 | Accuracy 차이 | paired 95% CI |\n|---|---|---|\n"
    for a, b in evaluation.SPEC["pairs"]:
        item = result["comparisons"][f"{a}_minus_{b}"]["accuracy"]
        comparisons += f"| {LABELS[a]} − {LABELS[b]} | {item['mean_difference']*100:+.2f}%p | [{item['ci95'][0]*100:+.2f}, {item['ci95'][1]*100:+.2f}]%p |\n"
    time_table = "| 측정 조건 | 학습 시간 평균±SD | seed 42 / 7 / 123 |\n|---|---|---|\n"
    for c in ("fixed_mix", "adaptive_mix", "adaptive_soft"):
        values = np.array([r["training_seconds"] for r in result["timing"][c]]) / 60
        time_table += f"| {LABELS[c]} | {values.mean():.2f}±{values.std(ddof=1):.2f}분 | " + " / ".join(f"{x:.2f}" for x in values) + "분 |\n"
    status = "**v4.3 완료:** 본실험 21개 학습·validation과 Teacher·원본 Student·21개 adapter의 test 1,172개 평가 완료. Fixed seed 42 시간 재현 1회는 별도이며 원본 최종 checkpoint를 유지했다. Test 이후 재학습·checkpoint 변경은 수행하지 않았다."
    for name in ("README.md", "docs/RESEARCH_PLAN.md", "docs/RESEARCH_REPORT.md"):
        path = ROOT / name
        raw = path.read_bytes(); text = raw.decode("utf-8").replace("\r\n", "\n")
        text, count = re.subn(r"<!-- main-status:start -->.*?<!-- main-status:end -->", "<!-- main-status:start -->\n" + status + "\n<!-- main-status:end -->", text, flags=re.S)
        assert count == 1
        if name == "README.md":
            text = text.replace("문서 버전 **v4.3** · 새 프로토콜 동결·학습 pipeline 재개", "문서 버전 **v4.3** · 학습·동결 test·결과 검증 완료")
        if name == "docs/RESEARCH_REPORT.md":
            marker = "<!-- gate-final-results:start -->"
            primary = result["comparisons"][evaluation.SPEC["primary"]]["accuracy"]
            conclusion = "주 비교의 CI가 0을 포함해 Hard와 Soft의 Accuracy 우위를 확인하지 못했다. 동등성을 입증한 것은 아니다." if primary["ci95"][0] <= 0 <= primary["ci95"][1] else (
                "주 비교의 조건부 CI에서 Hard의 Accuracy가 높았다." if primary["mean_difference"] > 0 else "주 비교의 조건부 CI에서 Soft의 Accuracy가 높았다.")
            section = f"""{marker}
## v4.3 동결 test 최종 결과

Teacher {result['baselines']['teacher']['accuracy']*100:.2f}%, 원본 Student {result['baselines']['student']['accuracy']*100:.2f}%. 아래 세 seed의 SD와 paired 문항 CI는 서로 다른 불확실성이다.

{table}
{comparisons}
{conclusion} Hard−Fixed·Soft−Fixed는 사전 지정 보조 비교이며 미조정 탐색적 CI다. 모든 CI는 관측한 세 학습 seed에 조건부이며 seed 모집단까지 포괄하지 않는다.

{time_table}
시간은 모델/Teacher/cache 로딩·epoch 진단·최종 평가를 제외하고 학습 loop·로그·최종 저장을 포함한다. 기존 E1~E4 시간은 다른 날짜의 기술적 참조이며 위 동일 측정 cohort와 직접 순위를 합치지 않는다. Fixed seed 42 시간은 재현 run, 최종 성능은 기존 adapter에서 측정했다. 재현 tensor hash 일치: {result['fixed42_replay']['tensor_sha256_matches_original']}; final validation 답변 불일치: {result['fixed42_replay']['final_validation_prediction_mismatches']}개.

![Test Accuracy와 seed](../results/gate_comparison_v4/test/accuracy.png)
![Accuracy 차이와 paired CI](../results/gate_comparison_v4/test/paired_ci.png)
![전체 validation 학습곡선](../results/gate_comparison_v4/test/learning_curves.png)
![정확도와 학습 시간](../results/gate_comparison_v4/test/time_accuracy.png)

전체 Teacher 준비 작업의 기존 시간은 hard 응답 생성과 soft 확률 추출을 함께 측정했으므로 소급 분리하지 않았다. 추론 지연은 고정 warmup 후 문항별 생성 측정이며 reference kernel 환경에 한정된다. 선택지 NLL·ECE·Brier·gate 분포·Teacher 오류 분모·자원 상세값은 [전체 결과](../results/gate_comparison_v4/test/comparison.json)에 보존한다. 공개 checkpoint의 ARC 사전 노출과 원문 실행 config의 scaling은 여전히 미검증이다.
<!-- gate-final-results:end -->

"""
            if marker in text:
                text = re.sub(r"<!-- gate-final-results:start -->.*?<!-- gate-final-results:end -->\n*", section, text, flags=re.S)
            else:
                position = text.index("## 2026-10-12 계획 개정")
                text = text[:position] + section + text[position:]
            text = text.replace("E7 본학습·test 결과는 아직 없다.", "E7 본학습·test 결과는 위 최종 결과 절에 보고했다.")
        path.write_bytes(text.replace("\n", "\r\n").encode("utf-8") if b"\r\n" in raw else text.encode("utf-8"))
    pilot.write_json(ROOT / "results/protocols/gate_run_status.json", {
        "status": "passed", "main_training_complete": 21, "final_validation_complete": 21,
        "timing_replay_complete": 1, "test_models_complete": 23, "test_n": 1172,
        "no_post_test_selection_or_retraining": True,
        "test_manifest_sha256": gate.sha(gate.OUT / "test_manifest.json")})
    revision_path = ROOT / "results/protocols/research_revision_v4.json"
    raw = revision_path.read_bytes(); revision = json.loads(raw)
    revision["execution_status"] = "complete_training_validation_sealed_test_and_analysis"
    revision["verification"].update(v4_pipeline_running=False, test_still_unevaluated=False,
        full_new_training_completed=8, total_final_test_models_scored=23)
    content = json.dumps(revision, ensure_ascii=False, indent=2) + "\n"
    revision_path.write_bytes(content.replace("\n", "\r\n").encode("utf-8") if b"\r\n" in raw else content.encode("utf-8"))
    for name in ("README.md", "docs/RESEARCH_PLAN.md", "docs/RESEARCH_REPORT.md"):
        path = ROOT / name
        for _, target in re.findall(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            target = target.split("#")[0]
            if target and "://" not in target:
                assert (path.parent / target).exists(), (name, target)
    assert subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() == "main"
    subprocess.run(["git", "-c", "core.whitespace=blank-at-eol,space-before-tab,cr-at-eol,-blank-at-eof", "diff", "--check"], cwd=ROOT, check=True)
    subprocess.run(["git", "add", "--", "README.md", "docs/RESEARCH_PLAN.md", "docs/RESEARCH_REPORT.md",
        "results/gate_comparison_v4", "results/protocols/gate_run_status.json",
        "results/protocols/gate_test_manifest.json", "results/protocols/research_revision_v4.json"], cwd=ROOT, check=True)
    changed = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT)
    assert changed.returncode in (0, 1)
    if changed.returncode == 1:
        subprocess.run(["git", "commit", "-m", "Report sealed seven-condition gate comparison and matched timing"], cwd=ROOT, check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)
