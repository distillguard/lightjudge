# 연구계획서 v3.1

<!-- main-status:start -->
**진행 상태:** 사용자 요청으로 실험 중단. 전체 train 학습 13/18회 완료, validation 검증·집계 12/18회 완료. 고정 혼합 seed 42는 학습 완료·validation 미완료이며 최종 test는 미평가다. [중단 상태 기록](../results/protocols/status.json)
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

Teacher revision은 `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, Student는 `15852e8c16360a2fea060d615a32b45270f8a8fc`, 데이터는 `210d026faf9955653af8916fad021475a3f00453`으로 고정한다. 공식 train 1,119개에서 입력 중복 1개를 제거했다. 선택지 3·4·5개를 유지하고 train/validation/test의 원본 ID·전체 입력 hash 교집합은 0이다. [분할·입력 hash](../results/protocols/data_manifest.json)

<a id="methods"></a>

## 6가지 학습 방식과 손실함수

6조건 × 학습 seed 42·7·123 = 18회다. Gold는 정답 라벨을 이용한 SFT 기준선, Response는 Teacher의 답변을 이용한 증류, 나머지 네 조건은 Teacher의 확률분포를 이용한 Logit KD다. 모두 같은 원본 Student의 LoRA를 학습하며 차이는 지도 신호와 손실함수다.

| 방식 | 학습에 쓰는 정보 | 알고리즘과 핵심 차이 |
|---|---|---|
| Gold SFT | 데이터셋의 정답 letter | 정답 token과 EOS에 Cross-Entropy(CE)를 적용한다. Teacher를 쓰지 않는 지도학습 기준선이다. |
| Response KD | Teacher가 생성한 답변 letter | Teacher 답변을 hard label로 삼아 CE를 적용한다. 정답 여부와 무관하게 같은 Teacher 답변을 모방하며 다른 답의 확률은 전달하지 않는다. |
| Forward KL | Teacher의 전체 어휘 분포 P | KL(P ∥ Q)를 줄인다. Teacher가 확률을 준 후보들을 Student도 표현하도록 유도한다. |
| Reverse KL | Teacher P와 현재 Student Q | KL(Q ∥ P)를 줄인다. Student가 높은 확률을 주지만 Teacher는 낮은 확률을 주는 후보의 불일치가 크게 반영된다. |
| 고정 혼합 | 두 방향의 KL | 매 문항마다 0.5 × Forward + 0.5 × Reverse를 사용한다. 가중치는 학습 내내 50:50이다. |
| 적응형 혼합 | 두 방향의 KL과 상대 confidence | Teacher confidence가 Student 이상이면 Forward, 낮으면 Reverse를 선택하는 hard gate다. 문항·학습 시점에 따라 방향이 바뀐다. |

### Gold와 Response: 정답 한 개를 가르치는 방식

입력을 x, 실제 정답을 y, Teacher가 생성한 답변을 a라고 하자. CE는 T=1의 Student 확률을 사용하며 답변 letter와 문장 종료 token(EOS)의 손실을 평균한다.

$$
L_{Gold}=\tfrac12\{ -\log Q(y\mid x)-\log Q(EOS\mid x,y)\}
$$

$$
L_{Response}=\tfrac12\{ -\log Q(a\mid x)-\log Q(EOS\mid x,a)\}
$$

예를 들어 실제 정답이 C인데 Teacher가 B라고 답하면 Gold는 C, Response는 B를 학습한다. 현재 Response는 설명이나 추론 과정을 생성하지 않는 **답변 letter 증류**다. Teacher 오답 52/1,118개도 제거하거나 Gold로 교체하지 않았다.

### Forward와 Reverse: 확률분포를 가르치는 방식

첫 답변 위치의 Teacher·Student logits를 T=2로 나누고 softmax한 분포를 P·Q로 둔다. 합은 A/B/C/D만이 아니라 **전체 어휘 V=248,320개**에 걸쳐 계산한다.

$$
D_F=KL(P\parallel Q)=\sum_v P(v)\log\frac{P(v)}{Q(v)},\qquad
D_R=KL(Q\parallel P)=\sum_v Q(v)\log\frac{Q(v)}{P(v)}
$$

Forward는 Teacher 확률 P로, Reverse는 Student 확률 Q로 불일치를 가중한다. KL은 방향을 바꾸면 값과 gradient가 달라진다. **두 방향 모두 Teacher는 동결하고 Student만 학습한다.** Reverse라는 이름이 Teacher·Student의 역할 교환을 뜻하지 않는다. Forward의 분포 포괄 경향과 Reverse의 낮은 Teacher 확률 영역 억제 경향은 [CaRE-KD §2.1](https://arxiv.org/html/2609.36734v1#S2.SS1)와 [MiniLLM](https://arxiv.org/abs/2306.08543)의 설명과 연결된다. 이것만으로 ARC 성능 우위가 보장되지는 않는다.

우리 네 KL 조건의 전체 목적함수는 다음과 같다.

$$
L_{KL}=\tfrac12\{T^2D+CE_{EOS}\}
$$

첫 답변에는 Teacher의 soft distribution을 쓰고 Gold 정답이나 hard-answer CE를 추가하지 않는다. EOS는 Teacher 답변 letter를 입력한 다음 위치에서 공통 hard CE로 학습한다. Teacher 분포는 오프라인 cache로 고정하고 gradient를 계산하지 않는다. 이 연구는 고정 문항의 첫 답변 위치를 증류하므로 MiniLLM의 on-policy 생성·최적화 전체를 재현하는 실험은 아니다.

### 고정 혼합과 적응형 혼합: KL 방향을 정하는 방식

$$
D_{Fixed}=0.5D_F+0.5D_R,\qquad D_{Adaptive}=gD_F+(1-g)D_R
$$

고정 혼합은 두 손실을 매번 평균한다. Teacher·Student 답변을 평균하거나 두 확률분포를 먼저 합치는 알고리즘은 아니다.

적응형 gate는 T=2 전체 어휘 분포의 정규화 entropy로 계산한다.

$$
c(X)=1-\frac{H(X)}{\ln V},\qquad g=\mathbf{1}[c(P)\ge c(Q)]
$$

Teacher가 상대적으로 확신하면 g=1로 Forward, Student가 더 확신하면 g=0으로 Reverse다. 동률은 Forward이며 gate의 gradient는 차단한다. Student confidence는 LoRA dropout을 포함한 같은 학습 forward에서 계산하므로 학습 중 바뀔 수 있다. 이 confidence는 분포의 뾰족함이며 정답일 확률을 보증하지 않는다. 평가에서 사용하는 **T=1 선택지 confidence**와도 구분한다.

현재 적응형은 [CaRE-KD 식 (1)–(3)](https://arxiv.org/html/2609.36734v1#S2.SS1)의 hard-gate 손실을 적용한다. 원 논문의 soft gate·Revival·BALD 기반 데이터 거절은 포함하지 않으며, Teacher가 덜 확신해도 학습 문항을 버리지는 않는다. 신규 손실 발명이나 전체 프레임워크 재현을 주장하지 않는다.

### 이 비교로 확인하는 것

Gold−Response는 실제 정답과 Teacher hard label의 차이, Response−Forward는 hard label과 soft distribution의 차이, Forward−Reverse는 KL 방향의 차이를 본다. **적응형−고정 혼합**이 주 비교이며 confidence에 따라 방향을 바꾸는 것이 고정 50:50보다 도움이 되는지 검증한다. 그 외 비교는 탐색적이며, 적응형이 모든 고정 가중치보다 우수하거나 gate만이 개선 원인이라고 주장하지 않는다.

실제 구현: [답변·EOS CE와 LoRA 학습](../src/train_student.py), [KL·entropy gate](../src/logit_kd.py), [6조건 실행·동결·평가](../src/kl_experiment.py).

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

[연구보고서](RESEARCH_REPORT.md) · [범위 변경·원본 보존 기록](../results/protocols/scope_change.json)
