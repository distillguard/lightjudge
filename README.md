# LightJudge

문서 버전 **v3.2** · 전체 train 본실험에 집중

Qwen3.5-9B NF4 Teacher와 Qwen3.5-2B BF16 LoRA Student에서 **confidence 기반 KL 혼합이 고정 혼합보다 나은지** 비교한다.

<!-- main-status:start -->
**진행 상태:** 사용자 요청으로 실험 중단. 전체 train 학습 13/18회 완료, validation 검증·집계 12/18회 완료. 고정 혼합 seed 42는 학습 완료·validation 미완료이며 최종 test는 미평가다. [중단 상태 기록](results/protocols/status.json)
<!-- main-status:end -->

- [연구계획서](docs/RESEARCH_PLAN.md): 고정한 질문·조건·평가·종료 기준
- [연구보고서](docs/RESEARCH_REPORT.md): 핵심 결과·그래프·근거
- [중단 상태 기록](results/protocols/status.json)

| 항목 | 범위 |
|---|---|
| 데이터 | ARC-Challenge train 1,118 / validation 299 / test 1,172 |
| 본실험 | Gold·Response·Forward·Reverse·고정 혼합·적응형 혼합 × 3 seeds = 18회 |
| 공통 학습 | 원본 Student부터 3 epochs·420 steps, LoRA rank 8, T=2 |
| 주 결과 | Accuracy·개별 seed·평균±SD, 적응형−고정 혼합의 Accuracy 차이·paired 95% CI |
| 해석 분석 | 실제 gate 분기·Teacher 오류 패턴·학습곡선 |
| 보조 지표 | 선택지 NLL; Brier·ECE·비용은 부록 |
| 최종 test | 동결 후 Teacher·원본 Student·본실험 18개 모델 전체 |

완료된 256개 파일럿은 탐색 기록으로 보존한다. 남은 파일럿 반복·파일럿 test·다른 모델/데이터 확장은 제외했다. Epoch별 곡선 진단은 고정 train/validation 각 64개로 줄였고, 학습·최종 성능 평가는 각 전체 분할을 사용한다. **CI는 별도 성능 지표가 아니라 차이의 불확실성**이며, seed SD와 구분한다. 선택지 NLL은 확률 품질을 보는 보조 지표로 유지한다. [평가 정의·판단 근거](docs/RESEARCH_PLAN.md#평가와-보고) · [재정리한 중간 결과·그래프](results/main/index.html)

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
| `src/kl_experiment.py` | 6조건 pipeline·GPU 사전검사·동결·평가 |
| `src/pilot.py` | 데이터 준비·프롬프트·기준 모델 평가 |
| `src/train_student.py` | Student LoRA 학습·CE |
| `src/logit_kd.py` | 확률분포 cache·4가지 KL 손실·gate |
| `src/monitor_training.py` | epoch 진단·생성 평가·신뢰도 지표 |
| `src/teacher_quantization.py` | Teacher NF4 로딩·모델 설정 검사 |
| `src/analyze_kl.py`, `src/analyze_training.py`, `src/plot_diagnostics.py` | 통계·학습곡선·그래프 |

## 6가지 방식의 차이

| 방식 | 핵심 알고리즘 |
|---|---|
| Gold SFT | 실제 정답 + EOS에 CE 적용; Teacher 없는 기준선 |
| Response KD | Teacher 답변 + EOS에 CE 적용; hard label 증류 |
| Forward KL | Teacher → Student KL; Teacher 확률로 불일치 가중 |
| Reverse KL | Student → Teacher KL; Student 확률로 불일치 가중 |
| 고정 혼합 | Forward·Reverse 손실을 항상 50:50 평균 |
| 적응형 혼합 | Teacher confidence ≥ Student면 Forward, 아니면 Reverse |

네 KL 조건은 T=2·전체 어휘 분포·공통 EOS CE를 사용한다. 모든 조건에서 Teacher는 동결하고 Student LoRA만 학습한다. [손실함수·예시·gate·논문 연결의 자세한 설명](docs/RESEARCH_PLAN.md#methods)

## 실행

아래 명령은 모델 cache·데이터·학습 준비 산출물을 갖춘 기존 로컬 환경용이다. 저장소의 중간 결과 snapshot만으로 전체 pipeline을 바로 재개할 수는 없다. 현재 실험은 사용자 요청으로 중단돼 있으며 문서·보고 정리 중 추가 학습·test 평가는 수행하지 않았다.

```powershell
# 의존성 목록
.\.venv\Scripts\python.exe -m pip install -r requirements/gpu.txt

# 재개할 때 사용하는 진입점
.\.venv\Scripts\python.exe -B -X utf8 src/kl_experiment.py self-check
.\.venv\Scripts\python.exe -B -X utf8 src/kl_experiment.py pipeline

# CPU 경로·동결 검증 회귀 검사
.\.venv\Scripts\python.exe -B -m unittest discover -s tests
```

기존 프로토콜의 원본 hash를 덮어쓰지 않았다. 정확히 기록된 디렉토리 변경만 호환 처리하고, 그 밖의 소스 변경은 기존 동결 실험의 재개를 차단한다. [경로 변경 기록](results/protocols/layout_migration.json)

Pipeline은 준비·검사 → 전체 train 18회 → validation → test 동결 → 20개 모델 test → 핵심 보고 순서다. 완료한 산출물은 검사 후 재사용하고 미완료 결과를 자동 덮어쓰지 않는다. 최종 test 결과로 조건·checkpoint를 다시 선택하거나 재학습하지 않는다.

[파일럿 탐색 결과](results/pilot/index.html) · [범위 변경·원본 보존](results/protocols/scope_change.json)
