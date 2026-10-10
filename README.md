# LightJudge

문서 버전 **v3.0** · 전체 train 본실험에 집중

Qwen3.5-9B NF4 Teacher와 Qwen3.5-2B BF16 LoRA Student에서 **confidence 기반 KL 혼합이 고정 혼합보다 나은지** 비교한다.

<!-- main-status:start -->
**진행 상태:** 사용자 요청으로 실험 중단. 전체 train 학습 13/18회 완료, validation 검증·집계 12/18회 완료. 고정 혼합 seed 42는 학습 완료·validation 미완료이며 최종 test는 미평가다. [중단 상태 기록](outputs/kl_extension/status.json)
<!-- main-status:end -->

- [연구계획서](RESEARCH_PLAN.md): 고정한 질문·조건·평가·종료 기준
- [연구보고서](RESEARCH_REPORT.md): 핵심 결과·그래프·근거
- [중단 상태 기록](outputs/kl_extension/status.json)

| 항목 | 범위 |
|---|---|
| 데이터 | ARC-Challenge train 1,118 / validation 299 / test 1,172 |
| 본실험 | Gold·Response·Forward·Reverse·고정 혼합·적응형 혼합 × 3 seeds = 18회 |
| 공통 학습 | 원본 Student부터 3 epochs·420 steps, LoRA rank 8, T=2 |
| 핵심 결과 | Accuracy·선택지 NLL·paired 95% CI |
| 최종 test | 동결 후 Teacher·원본 Student·본실험 18개 모델 전체 |

완료된 256개 파일럿은 탐색 기록으로 보존한다. 남은 파일럿 반복·파일럿 test·다른 모델/데이터 확장은 제외했다. Epoch별 곡선 진단은 고정 train/validation 각 64개로 줄였고, 학습·최종 성능 평가는 각 전체 분할을 사용한다. 보조 지표와 비용은 JSON·그래프 갤러리에 둔다.

## 저장소에 포함한 내용

실험 코드·의존성 목록·연구 문서 3개와 보고서에서 참조하는 작은 결과 JSON·그래프를 포함한다. 데이터 원문·모델 및 adapter 가중치·확률분포 cache·가상환경·과거 archive는 로컬에 보존하고 Git에서 제외한다. `outputs/`는 기본 제외하며 현재 문서의 참조 결과만 명시적으로 커밋했다.

## 실행

아래 명령은 모델 cache·데이터·학습 준비 산출물을 갖춘 기존 로컬 환경용이다. 저장소의 중간 결과 snapshot만으로 전체 pipeline을 바로 재개할 수는 없다.

```powershell
.\.venv\Scripts\python.exe -B -X utf8 kl_experiment.py self-check
.\.venv\Scripts\python.exe -B -X utf8 kl_experiment.py pipeline
```

Pipeline은 준비·검사 → 전체 train 18회 → validation → test 동결 → 20개 모델 test → 핵심 보고 순서다. 완료한 산출물은 검사 후 재사용하고 미완료 결과를 자동 덮어쓰지 않는다. 최종 test 결과로 조건·checkpoint를 다시 선택하거나 재학습하지 않는다.

[파일럿 탐색 결과](outputs/kl_extension/analysis/reverse_validation_interim/index.html) · [범위 변경·원본 보존](outputs/kl_extension/scope_change.json)
