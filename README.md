# LightJudge

Qwen3.5-9B Teacher와 Qwen3.5-2B Student에서 **Hard의 이진 선택·Soft의 연속 가중·Fixed 50:50**을 비교하는 지식 증류 연구다. 정답률과 학습 시간을 함께 평가한다. 갱신: 2026-10-13.

<!-- main-status:start -->
**v4.3 완료:** 본실험 21개 학습·validation과 Teacher·원본 Student·21개 adapter의 test 1,172개 평가 완료. Fixed seed 42 시간 재현 1회는 별도이며 원본 최종 checkpoint를 유지했다. Test 이후 재학습·checkpoint 변경은 수행하지 않았다.
<!-- main-status:end -->

## 현재 상태

| 단계 | 상태 | 확인한 범위 |
|---|---|---|
| ARC 본실험 | 완료 | 7조건×3 seeds, 별도 시간 재현 1회, 23개 모델의 test 평가 |
| 베이지안 후속 분석 | 완료 | 3가지 prior·paired Bayesian bootstrap; 모형 적합성 한계 있음 |
| 새 데이터 | 준비·초기 진단 완료 | OpenBookQA·HellaSwag 확보, 각 train 128문항에서 gate 진단 |
| 새 데이터 증류 학습 | 미실행 | 같은 Qwen에서 Fixed·Hard·Soft의 6회 탐색 학습 계획 |
| 다른 Student 계열 | 준비·GPU 검사 완료 | SmolLM2·Granite 본학습은 새 데이터 비교 뒤에 진행 |

ARC test는 Hard **85.38%**, Soft **85.04%**다. 차이 +0.34%p의 paired 95% CI는 [−0.23,+0.91]%p로, 우위와 동등성을 확정하지 못했다. 학습된 Soft의 평균 가중치 **0.4866**은 50:50에 가깝지만 문항별 변화와 selection gradient가 있어 Fixed와 동일하다고 단정하지 않는다.

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
