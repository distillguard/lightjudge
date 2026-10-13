# LightJudge

Qwen3.5-9B Teacher와 Qwen3.5-2B Student에서 **Hard의 이진 선택·Soft의 연속 가중·Fixed 50:50**을 비교하는 지식 증류 연구다. 정답률과 학습 시간을 함께 평가한다. 갱신: 2026-10-14.

<!-- main-status:start -->
**v4.3 완료:** 본실험 21개 학습·validation과 Teacher·원본 Student·21개 adapter의 test 1,172개 평가 완료. Fixed seed 42 시간 재현 1회는 별도이며 원본 최종 checkpoint를 유지했다. Test 이후 재학습·checkpoint 변경은 수행하지 않았다.
<!-- main-status:end -->

## 현재 상태

| 단계 | 상태 | 확인한 범위 |
|---|---|---|
| ARC 본실험 | 완료 | 7조건×3 seeds, 별도 시간 재현 1회, 23개 모델의 test 평가 |
| 베이지안 후속 분석 | 완료 | 3가지 prior·paired Bayesian bootstrap; 모형 적합성 한계 있음 |
| 새 데이터 | 준비·초기 진단 완료 | OpenBookQA·HellaSwag 확보, 각 train 128문항에서 gate 진단 |
| 새 데이터 증류 학습 | 상태·결과 기록 참조 | 같은 Qwen에서 Fixed·Hard·Soft 6회; 아래 실행 기록 참조 |
| 다른 Student 계열 | 입력·손실·GPU 검사 완료 | 원본 성능 벤치마크·독립 본학습 미수행; SmolLM2 주 비교, Granite 보조 후보 |

ARC test는 Hard **85.38%**, Soft **85.04%**다. 차이 +0.34%p의 paired 95% CI는 [−0.23,+0.91]%p로, 우위와 동등성을 확정하지 못했다. 학습된 Soft의 평균 가중치 **0.4866**은 50:50에 가깝지만 문항별 변화와 selection gradient가 있어 Fixed와 동일하다고 단정하지 않는다.

확정 순서는 **새 데이터 6회 학습·최종 평가 → 원본 Student 3개 벤치마크 → 다른 계열 선택지 KD 비교**다. [벤치마크 사전 명세](results/model_benchmark_v1/protocol.json). 다른 계열 Student 비교도 합의된 범위다. 같은 문항·학습 횟수를 맞추는 준비안이며, 실제 입력 토큰 수까지 동일하지 않다(Qwen 37.9만 / SmolLM2 44.7만 / Granite 57.1만). Qwen도 공통 선택지 KD로 다시 학습해 비교한다. [파라미터·토큰 예산 실측](results/checks/student_comparison_budget.json).

## 문서·결과

- [연구계획서](docs/RESEARCH_PLAN.md): 연구 질문, 7조건 수식, 통제·평가 설계, 다음 단계.
- [연구보고서](docs/RESEARCH_REPORT.md): 실제 결과표, 그래프, Bayesian 해석, 새 데이터 진단.
- [ARC 최종 결과](results/gate_comparison_v4/test/comparison.json) · [Bayesian 결과](results/bayesian_v1/summary.json) · [새 데이터·gate 진단](results/gate_continuity/train_screening.json).

## 디렉토리

```text
src/          모델·데이터·학습·평가·분석 코드
docs/         연구계획서·연구보고서
results/      공유용 결과·그림·동결 프로토콜·검증 기록
requirements/ GPU 학습 환경과 별도 Bayesian 환경의 의존성
tests/        수식·데이터·학습·보존 검사
```

`data/`, `outputs/`, `.cache/`, `.venv/`는 로컬 실행용이다. 원본 데이터·모델/adapter 가중치·전체 확률 cache·실시간 로그는 Git에서 제외한다. `results/`는 기록 시점의 결과다.

## 현재 진단 재현

기존 로컬 cache·고정 입력이 있는 환경에서 실행한다. `screen`은 완료 추론을 검증·재사용하며, `prepare`는 완료 준비를 덮어쓰지 않는다.

```powershell
.\.venv\Scripts\python.exe -B -X utf8 src/gate_continuity.py screen
.\.venv\Scripts\python.exe -B -X utf8 src/report_gate_continuity.py
```

Bayesian 분석은 별도 `.cache/bayes-venv`와 [requirements/bayesian.txt](requirements/bayesian.txt)를 사용한다. 완료 posterior의 그림은 `src/report_bayesian.py posterior`로 재생성한다. 기존 ARC 학습·평가 경로는 [학습 프로토콜](results/protocols/gate_training_protocol.json)과 [test manifest](results/protocols/gate_test_manifest.json)에 보존한다. GPU 작업은 순차 실행하며 완료 checkpoint를 변경하지 않는다.

**현재 Teacher:** `Qwen/Qwen3.5-9B`, NF4 4-bit double quantization·BF16 compute, 가중치 동결. **현재 Student:** `Qwen/Qwen3.5-2B`, BF16 원본 동결·FP32 LoRA(rank 8, alpha 16) 학습. Fixed·Hard·Soft는 같은 원본 Student에서 각각 독립 시작한다. SmolLM2·Granite 본학습은 이후 단계다.

<!-- data-gates:start -->
## 현실적인 후속 범위 동결: 2026-10-14

**진행 점검:** OpenBookQA Fixed·Hard·Soft의 3회 학습과 adapter 재로딩을 검증했다. 개발용 Accuracy는 **82.20% / 82.80% / 82.20%**, NLL은 **0.5212 / 0.5265 / 0.5274**다. 동일 초기 adapter·학습 순서·각 420 steps와 3,354회 train 관측을 확인했다. Soft 최종 gradient 검사도 통과했다. HellaSwag 세 조건은 진행 중이며 최종 holdout 평가는 아직 시작하지 않았다.

[OpenBookQA 완료 검증](results/data_gates_v1/openbookqa_training_check.json)

**2개 데이터×Fixed·Hard·Soft×seed 42=6회 탐색 학습**으로 정했다. 각 train은 질문 단위 중복·validation 겹침·라벨 충돌을 제외한 1,118개, 3 epochs·420 steps이며 개발용 평가 500개를 사용한다. 기존 T=2·margin=0.05·BF16 LoRA·전체 어휘 KD·공통 EOS를 유지한다. Teacher·원본 Student의 개발용 기준선과 train 8문항의 gradient 크기를 먼저 측정한다.

HellaSwag는 공개 validation에서 질문·출처가 겹치지 않는 개발용 500개/최종 평가용 500개를 사전 분리했다. OpenBookQA는 공식 validation과 test를 쓰며, test는 6개 최종 adapter 동결 뒤 입력 겹침 제외 규칙을 적용하고 평가한다. 모든 평가는 T=1 선택지 문자 argmax/NLL로 통일하며 공식 completion 점수와 구분한다. 최종 평가 뒤 재학습·조건 변경은 하지 않는다.

원본 130개 산출물·12개 소스·21개 adapter는 보존한다. 다른 Student 9회와 추가 상수/detach 대조는 후속 단계로 둔다. 1 seed는 예산에 따른 탐색이며 안정적인 일반화를 확정하지 않는다. [실행·분할·코드 동결](results/data_gates_v1/protocol.json) · [현재 상태](results/data_gates_v1/status.json)
<!-- data-gates:end -->
