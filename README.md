# LightJudge

문서 버전 **v4.3** · 새 프로토콜 동결·학습 pipeline 재개

공개 post-trained Qwen3.5-9B NF4 Teacher와 Qwen3.5-2B BF16 LoRA Student에서 **confidence 기반 Hard·Soft Gate의 정확도와 학습 시간**을 비교한다. Fixed 50:50은 핵심 대조군이다.

<!-- main-status:start -->
**현재 v4.3 상태:** 순차 pipeline 실행 중. 본실험 학습·final validation **14/21회 완료**, 별도 Fixed seed 42 시간 재현 1회 완료. Hard seed 42 최종 validation은 87.29%이며 Soft seed 42를 학습 중이다. Test는 미평가다. [진행 snapshot](results/protocols/gate_run_status.json)
<!-- main-status:end -->

- [연구계획서](docs/RESEARCH_PLAN.md): 고정한 질문·조건·평가·종료 기준
- [연구보고서](docs/RESEARCH_REPORT.md): 핵심 결과·그래프·근거
- [중단 상태 기록](results/protocols/status.json)

| 항목 | 범위 |
|---|---|
| 데이터 | ARC-Challenge train 1,118 / validation 299 / test 1,172 |
| 개정 본실험 계획 | Gold·Response·Forward·Reverse·Fixed·Hard·Soft × 3 seeds = 21조건 |
| 시간 측정 계획 | E5~E7 전체 validation 정확도–시간 곡선, 기존 Fixed seed 42 측정용 재현 1회 별도 |
| 공통 학습 | 원본 Student부터 3 epochs·420 steps, LoRA rank 8, T=2 |
| 주 결과 | Hard−Soft Accuracy 차이·paired CI, 개별 seed·평균±SD, 학습 시간·정확도–시간 곡선 |
| 해석 분석 | 실제 gate 분기·Teacher 오류 패턴·학습곡선 |
| 보조 지표 | 선택지 NLL·Brier·ECE·추론 지연·VRAM은 부록 |
| 개정 최종 test 계획 | 동결 후 Teacher·원본 Student·21개 adapter = 23개 모델 |

**7조건 학습·validation 실행기를 별도로 구현하고 동결했다.** 기존 7개 소스·원본 프로토콜·130개 보호 산출물은 유지한다. 재사용 가능한 13회 학습과 12회 검증은 hash·설정·초기화·문항 순서로 확인했다. [v4.3 동결 기록](results/protocols/gate_training_protocol.json)

완료된 256개 파일럿은 탐색 기록으로 보존한다. 남은 파일럿 반복·파일럿 test·다른 모델/데이터 확장은 제외했다. 기존 epoch 진단은 seed 42의 train/validation 각 64개이며, 개정안의 E5~E7 정확도–시간 곡선은 세 seed 모두 validation 299개 전체에서 관측할 계획이다. **CI는 별도 성능 지표가 아니라 차이의 불확실성**이며, seed SD와 구분한다. 선택지 NLL은 확률 품질을 보는 보조 지표로 유지한다. [개정 평가·시간 정의](docs/RESEARCH_PLAN.md#evaluation) · [재정리한 중간 결과·그래프](results/main/index.html)

**근거 정책:** 원문 식·공식 자료·기존 통제값·실측 검증·미확인을 구분한다. T=2·학습률·rank·epoch·seed를 논문의 표준값으로 표기하지 않는다. 근거가 없던 80% 목표는 철회했으며 기존 결과는 그대로다. [항목별 근거 점검](results/protocols/evidence_audit.json)

수식·기존 누적 갱신과의 일치·평가 RNG 보존·덮어쓰기 방지 등 **22개 검사**와 새 실행 경로의 격리 GPU 검사가 통과했다. GPU 검사 모델은 저장·재사용하지 않았다. [GPU 검증 결과](results/checks/soft_gpu_probe.json)

## 디렉토리 구성

```text
lightjudge/
├── README.md
├── src/                 # 실행·모델·데이터·학습·평가·분석 코드
├── docs/                # 연구계획서·연구보고서
├── requirements/        # base.txt·gpu.txt
├── results/
│   ├── main/            # 전체 train 본실험의 중간 결과·그래프
│   ├── pilot/           # 256개 파일럿 결과·그래프
│   ├── checks/          # 수식·학습·보존 검증 결과
│   └── protocols/       # 고정 설정·데이터 기록·중단 상태
└── tests/               # 경로 변경과 동결 검증의 회귀 검사
```

로컬의 `data/`·`outputs/`·`.cache/`·`.venv/`·`_archive/`는 실행용이며 Git에서 제외한다. `results/`는 검토·공유용 snapshot이고 실행 중인 로그가 아니다. 모델·adapter 가중치와 전체 확률분포 cache는 포함하지 않는다.

| 코드 | 역할 |
|---|---|
| `src/kl_experiment.py` | 보존한 기존 6조건 실행·공통 평가 helper |
| `src/gate_experiment.py` | v4.3 동결·Fixed/Hard/Soft 학습·전체 validation·시간 계측 |
| `src/gate_evaluation.py`, `src/gate_reporting.py` | 학습 완료 후 23개 모델 test 동결·평가·paired 통계·그래프·문서 갱신 |
| `src/pilot.py` | 데이터 준비·프롬프트·기준 모델 평가 |
| `src/train_student.py` | Student LoRA 학습·CE |
| `src/logit_kd.py` | 기존 확률분포 cache·4가지 KL 손실·Hard gate |
| `src/soft_gate.py` | 원문 Soft gate·margin 0.05·selection gradient |
| `src/probe_soft_gate.py` | 본학습과 분리한 실제 GPU 검사 |
| `src/monitor_training.py` | epoch 진단·생성 평가·신뢰도 지표 |
| `src/teacher_quantization.py` | Teacher NF4 로딩·모델 설정 검사 |
| `src/analyze_kl.py`, `src/analyze_training.py`, `src/plot_diagnostics.py` | 통계·학습곡선·그래프 |

## 학습 방식: 7조건

| 방식 | 핵심 알고리즘 |
|---|---|
| Gold SFT | 실제 정답 + EOS에 CE 적용; Teacher 없는 기준선 |
| Response KD | Teacher 답변 + EOS에 CE 적용; hard label 증류 |
| Forward KL | Teacher → Student KL; Teacher 확률로 불일치 가중 |
| Reverse KL | Student → Teacher KL; Student 확률로 불일치 가중 |
| 고정 혼합 | Forward·Reverse 손실을 항상 50:50 평균 |
| Adaptive Hard (기존) | Teacher confidence ≥ Student면 Forward, 아니면 Reverse |
| Adaptive Soft (원문 수식 구현) | sigmoid(confidence 차이 − 0.05); Soft gate gradient 유지 |

기존 네 KL 조건과 신규 Soft 계획은 T=2·전체 어휘 분포·공통 EOS CE를 사용한다. 모든 조건에서 Teacher는 동결하고 Student LoRA만 학습한다. [손실함수·예시·gate·논문 연결의 자세한 설명](docs/RESEARCH_PLAN.md#methods)

## v4.3 실행 안내

기존 cache·데이터·완료 adapter가 있는 로컬 환경용이다. 실행기는 별도 `outputs/gate_comparison_v4/`에 저장하며, 미완료 run은 보존하고 자동 덮어쓰지 않는다. **GPU 학습이 이미 실행 중이면 같은 명령을 다시 실행하지 않는다.** Git의 `results/`는 기록 시점의 snapshot이다.

```powershell
# 구현·기존 결과 보존 검사
.\.venv\Scripts\python.exe -B -X utf8 -m unittest discover -s tests
# 기존 동결이 있으면 동일 소스·환경·재사용 hash 검증만 수행
.\.venv\Scripts\python.exe -B -X utf8 src/gate_experiment.py freeze
# Fixed·Hard·Soft 9회 순차 실행: 신규 본실험 8회 + 시간 재현 1회
.\.venv\Scripts\python.exe -B -X utf8 src/gate_experiment.py pipeline
```

seed 42는 Fixed→Hard→Soft, seed 7은 Hard→Soft→Fixed, seed 123은 Soft→Fixed→Hard 순서다. 모든 측정 run은 epoch 0·1·2·3에서 validation 299개를 평가하고 평가 시간을 학습 시간에서 제외한다. Fixed seed 42 재현은 기존 최종 checkpoint를 대체하지 않는다. 각 최종 adapter를 다시 로딩해 final validation을 검증한다.

학습 실행기는 **학습·validation 단계에서 종료**한다. 연결된 후속 실행기는 이 완료를 기다린 뒤 모든 산출물 검증 → 별도 23개 모델 test 동결·평가 → 통계·그래프·문서 갱신을 진행한다. 학습 코드 22개·후속 코드 11개, 총 33개 검사를 통과했다. Test를 보고 재학습하거나 checkpoint를 바꾸지 않는다. 원본 6조건 pipeline은 v4 실행에 사용하지 않는다. [동결 프로토콜](results/protocols/gate_training_protocol.json) · [새 학습 경로 GPU 검사](results/checks/trainer_gpu_check.json)

후속 실행기는 현재 대기 중이다. 같은 명령을 추가 실행하지 않는다. 신규 환경에서 후속 단계만 연결할 때는 다음 순서로 사용한다.

```powershell
.\.venv\Scripts\python.exe -B -X utf8 src/gate_evaluation.py prepare
.\.venv\Scripts\python.exe -B -X utf8 src/gate_evaluation.py pipeline
```

후속 실행은 예약 작업 없이 실행 프로세스 안에서 대기하며, 학습 실패·소스 변경·미완료 산출물이 있으면 test를 시작하지 않는다. [사전 평가 동결](results/protocols/gate_evaluation_protocol.json) · [학습 중간 곡선](results/gate_comparison_v4/progress/fixed_seed42_progress.png)

[파일럿 탐색 결과](results/pilot/index.html) · [범위 변경·원본 보존](results/protocols/scope_change.json)
