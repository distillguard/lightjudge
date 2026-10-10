# 연구계획서 v3.0

<!-- main-status:start -->
**진행 상태:** 사용자 요청으로 실험 중단. 전체 train 학습 13/18회 완료, validation 검증·집계 12/18회 완료. 고정 혼합 seed 42는 학습 완료·validation 미완료이며 최종 test는 미평가다. [중단 상태 기록](outputs/kl_extension/status.json)
<!-- main-status:end -->

## 연구 질문

**Teacher·Student의 confidence를 이용한 KL 혼합이 고정 50:50 혼합보다 증류 성능을 개선하는가?**

주 비교는 적응형−고정 혼합의 생성 Accuracy다. 선택지 NLL을 핵심 부지표로 보고, Forward·Reverse와 Gold·Response는 해석을 위한 대조군으로 둔다. 개선을 가정하지 않고 부정적인 결과도 포함한다.

## 모델과 데이터

| 항목 | 고정 설정 |
|---|---|
| Teacher | Qwen/Qwen3.5-9B, NF4 double quantization, BF16 compute, 동결 |
| Student | Qwen/Qwen3.5-2B, BF16 원본 동결 + FP32 LoRA |
| 데이터 | allenai/ai2_arc, ARC-Challenge |
| 분할 | 정리된 train 1,118개 / validation 299개 / test 1,172개 |
| 반복 | 학습 seed 42·7·123, 각 조건을 원본 Student부터 독립 학습 |

Teacher revision은 `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, Student는 `15852e8c16360a2fea060d615a32b45270f8a8fc`, 데이터는 `210d026faf9955653af8916fad021475a3f00453`으로 고정한다. 공식 train 1,119개에서 입력 중복 1개를 제거했다. 선택지 3·4·5개를 유지하고 train/validation/test의 원본 ID·전체 입력 hash 교집합은 0이다. [분할·입력 hash](outputs/pilot/data_manifest.json)

## 본실험: 6조건 × 3 seeds = 18회

| 조건 | 지도 신호 |
|---|---|
| Gold SFT | 데이터셋 정답 letter + EOS |
| Response KD | Teacher의 원래 답변 letter + EOS |
| Forward KL | D_F = KL(P_teacher ∥ Q_student) |
| Reverse KL | D_R = KL(Q_student ∥ P_teacher) |
| 고정 혼합 | D = 0.5 D_F + 0.5 D_R |
| 적응형 혼합 | D = g D_F + (1−g) D_R |

네 KL 조건은 첫 답변 위치에서 **0.5 × (T² D + 공통 EOS CE)**를 사용한다. Teacher 분포는 T=2·전체 어휘 248,320개의 FP32 log probability로 오프라인 저장한다. Gold 정답 letter를 KL 손실에 넣지 않고 Teacher 오답도 필터링·대체하지 않는다.

Gate는 **c(X)=1−H(X)/ln(248,320), g=1[c(P)≥c(Q)]**다. 동률이면 Forward, gate gradient는 차단한다. Teacher는 eval mode이며 Student confidence는 LoRA dropout을 포함한 학습 forward에서 계산한다. 이는 [CaRE-KD 식 (1)–(3)](https://arxiv.org/html/2609.36734v1)의 hard-gate 손실을 이 과제에 적용하는 비교이고, 신규 손실 발명이나 Revival/BALD를 포함한 전체 프레임워크 재현을 주장하지 않는다.

## 공통 학습 조건

- 3 epochs, micro batch 1, gradient accumulation 8, **각 실행 420 optimizer steps**.
- LoRA rank 8·alpha 16·dropout 0.05·all-linear, AdamW lr 2e−4·weight decay 0.
- Linear scheduler·warmup 10 steps·gradient norm 상한 1, 최대 입력 1,024 tokens.
- 마지막 누적 그룹의 6개 문항도 버리지 않고 실제 그룹 크기로 정규화한다.
- 같은 seed 안에서 초기 adapter·문항 순서·입력·학습량을 맞추고 **고정 epoch 3**만 최종 checkpoint로 사용한다.

## 평가와 보고

| 구분 | 산출물 |
|---|---|
| 본문 | 6조건의 Accuracy·선택지 NLL, 3-seed 평균±sample SD, paired 차이의 95% CI |
| 학습곡선 | 실제 optimizer loss. seed 42에서 epoch 0·1·2·3의 고정 train/validation 각 64개 진단 |
| 보조 자료 | Brier·10-bin ECE·gate 분기·Teacher 오류·gradient clipping·시간·VRAM·모델 크기 |

64개 진단 문항은 분할별 source_id의 SHA-256 순서로 고정하며, 학습에 쓰는 train 1,118개와 별도다. 진단 평가의 RNG·모드·가중치를 보존하고 checkpoint 선택에 쓰지 않는다. **학습 후 성능 평가는 validation 299개와 test 1,172개 전체**에서 수행한다.

평가 confidence는 T=1 첫 생성 step을 유효 선택지로 정규화한다. 생성은 thinking=False·greedy·최대 새 token 8·batch 1·use_cache=False로 고정한다. 같은 문항을 모든 조건·seed에서 함께 5,000회 bootstrap(seed 42)한다. CI는 관측한 세 학습 seed에 조건부이며 평가 문항을 3배로 계산하지 않는다. NLL 등 부 분석은 미조정 탐색적 구간으로 표시하고, CI가 0을 포함하면 우위 미확인으로 기록한다. 동등성을 주장하지 않는다.

조건·checkpoint·지표·코드·입력·분석을 동결한 뒤 **Teacher·원본 Student + 본실험 18개 모델**의 test를 평가한다. 파일럿 모델은 최종 test 대상에서 제외한다. test 결과로 재학습·조건 선택·epoch 선택을 하지 않는다.

## 범위와 종료 기준

완료된 256개 파일럿은 탐색 기록으로 보존하고 남은 반복은 수행하지 않는다. 전체 train의 18회 학습·validation, 동결 후 test, 핵심 표·그래프·한계 보고가 끝나면 본실험을 종료한다. Temperature 변경·Reasoning KD·다른 모델/데이터는 이번 범위에서 제외한다.

결론은 이 Qwen 조합·영어 ARC·첫 답변 위치 증류·고정 학습 예산에 한정한다. 공개 checkpoint의 ARC 사전 노출은 확인하지 못했다. Confidence는 정답 판별기가 아니며 두 KL 모두 Teacher 분포를 목표로 한다. 적응형이 50:50보다 좋아도 모든 고정 혼합보다 낫거나 gate 자체가 원인이라고 주장하지 않는다.

[연구보고서](RESEARCH_REPORT.md) · [범위 변경·원본 보존 기록](outputs/kl_extension/scope_change.json)
