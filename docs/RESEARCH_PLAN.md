# 연구계획서 v4.3

## 1. 연구 과제와 개정 목적

**대규모 언어모델 지식 증류에서 신뢰도 기반 Hard·Soft Gate의 성능과 시간 효율성 비교**

Performance and Time-Efficiency Comparison of Confidence-Based Hard and Soft Gating in Large Language Model Knowledge Distillation

작성일: 2026-10-12. 사용자 제공 「LightJudge 지식 증류 성능 및 시간 효율성 연구보고서」(11쪽)의 개정안을 바탕으로 작성했다. 기존 6조건의 중간 validation을 확인한 뒤 연구 질문을 개정한 **실행 전 설계안**으로 작성했다. 이후 진행 상태는 상단 snapshot과 연구보고서에 반영한다.

<!-- main-status:start -->
**현재 v4.3 상태:** 순차 pipeline 실행 중. 본실험 학습·final validation **18/21회 완료**, 별도 Fixed seed 42 시간 재현 1회 완료. Seed 42·7의 Fixed·Hard·Soft 검증을 마쳤고 Soft seed 123을 학습 중이다. Test는 미평가다. [진행 snapshot](../results/protocols/gate_run_status.json)
<!-- main-status:end -->

v3.2의 주 비교는 Hard−Fixed였다. v4.0은 **Hard−Soft를 주 비교**로 삼고 Hard−Fixed·Soft−Fixed를 핵심 대조로 둔다. Accuracy와 학습 시간을 함께 보고하되 임의의 종합 점수로 합산하지 않는다. v4.3 학습 프로토콜을 별도로 동결했으며 기존 소스·프로토콜은 유지한다. 학습·validation 완료 후 별도 test 동결 단계로 진행한다. [개정·보존 기록](../results/protocols/research_revision_v4.json)

## 2. 연구 배경·목적·질문

Forward KL과 Reverse KL은 각각 Teacher와 Student 확률로 분포 차이를 가중한다. 상대 confidence에 따라 방향을 이진 선택하는 Hard Gate와 연속 가중하는 Soft Gate를 동일 학생 모델·데이터·학습 예산에서 비교해 **정답률과 시간 비용의 차이**를 확인한다. 어느 방식이 우수할지는 사전에 단정하지 않는다.

| 구분 | 질문과 비교 | 판단 자료 |
|---|---|---|
| RQ1 주 연구 | E6 Hard와 E7 Soft는 정확도·학습 시간에서 어떤 차이를 보이는가? | 최종 test Hard−Soft Accuracy 차이·paired 95% CI, seed별 학습 시간·정확도–시간 곡선 |
| RQ2 핵심 대조 | 두 적응형 방식은 E5 Fixed 50:50보다 유리한가? | Hard−Fixed·Soft−Fixed Accuracy 차이·탐색적 CI, 동일 측정 범위의 시간 차이 |
| RQ3 보조 비교 | E1~E4 기준선 대비 성능·학습 비용은 어떠한가? | Accuracy·학습 시간과 보조 NLL·VRAM |

동일 Qwen3.5 계열의 공개 post-trained 9B Teacher와 2B Student를 사용하고, 조건 간 모델 구성을 고정한다. 원본 Student는 사전학습만 된 Base 모델이라는 뜻이 아니라 ARC 추가 학습 전 공개 checkpoint다. Teacher는 NF4, Student는 BF16+LoRA로 실행한다. 기존 validation의 Teacher 92.64%와 원본 Student 76.25%는 선정 후 확인한 참조 결과이며 선정 당시의 사전 근거로 소급하지 않는다.

연구 기여는 이 설정에서의 **출력 기반 KD와 gate 변형의 실증 비교**다. 신규 KD 알고리즘 발명·CaRE-KD 전체 재현·다른 모델 계열의 일반적 우위를 주장하지 않는다.

## 근거 대조와 설정 선택 원칙

수식·하이퍼파라미터·지표·결론을 **원문 근거 / 공식 자료 / 기존 통제 설정 / 실측 검증 / 미확인**으로 구분한다. 논문이나 모델 카드가 방법을 설명한다고 해서 우리의 숫자까지 정당화하는 인용을 붙이지 않는다. 새 수치는 근거·선정 절차를 먼저 기록하며, 근거가 없으면 표준값으로 생성하지 않는다. 모델 카드의 최신 설명보다 실제 사용한 revision의 config·코드를 우선 대조한다. [항목별 점검·출처·조치](../results/protocols/evidence_audit.json)

| 항목 | 확인한 근거와 이번 연구의 적용 범위 |
|---|---|
| Hard·Soft gate / m=0.05 | CaRE-KD 원본 PDF 식 (3)·식 (7)·§4.5(Fig.3). Gate 적용 비교이며 전체 프레임워크 재현은 아님 |
| 모델·어휘 크기·비전 분리 | 공식 고정 revision의 config가 로컬 cache와 byte 단위로 일치. 공개 checkpoint는 post-trained 모델이며 원본 Student는 ARC 추가 LoRA 전을 뜻함 |
| ARC 분할 | 공식 고정 dataset card: 1,119/299/1,172. Train 입력 중복 제거는 로컬 manifest에 근거해 1,118개로 정리 |
| LoRA·NF4 | 방법은 원문·공식 구현 자료로 확인. r=8·alpha=16·dropout=0.05와 Teacher-only NF4는 기존 실험 설정 |
| T=2·lr=2e−4·3 epochs·batch·warmup·seed | 기존 동결 프로토콜과 완료 로그의 통제값. 논문 기본값·튜닝 최적값으로 주장하지 않으며 새 조건만 바꾸지 않음 |
| 첫 위치·공통 EOS·손실 0.5 가중치 | ARC letter 출력에 맞춘 기존 과제 설계. CaRE-KD나 MiniLLM 전체 sequence 목적함수와 구분 |
| CI·반복 | NLP 비교·반복 변동 문헌이 방법 근거. 3 seeds·5,000 bootstrap은 고정 설계값이고 이 숫자의 보편적 표준성을 주장하지 않음 |
| 시간 | PyTorch의 warmup·GPU 동기화 원칙과 실측 구간 정의. 옛 시간 로그는 코드·환경·범위가 확인될 때만 비교 |
| 목표 80% | 인용 문헌·ARC 공식 기준에서 확인하지 못해 철회. 정당화된 별도 목표가 생기기 전 Time-to-Accuracy를 주 지표로 쓰지 않음 |

완료된 13회 결과의 재사용을 위해 기존 통제값을 유지한다. CaRE-KD의 lr=5e−5·LoRA r=16·과제별 epoch와 다르다는 점을 공개하며, 원문 숫자로 바꾸려면 모든 비교 조건을 새로 학습하는 별도 프로토콜이 필요하다. 3 seeds는 제한된 예산의 반복 수이며 학습 불확실성 전체를 충분히 추정한다는 보장은 없다.

## 3. 모델·데이터·공통 학습 조건

| 항목 | 유지할 설정 |
|---|---|
| Teacher | Qwen/Qwen3.5-9B, 4-bit NF4 double quantization, BF16 compute, 전체 동결 |
| Student | Qwen/Qwen3.5-2B, BF16 원본 동결 + FP32 LoRA 8,409,600개 학습 |
| 데이터 | allenai/ai2_arc, ARC-Challenge 영어 객관식 |
| 분할 | 정리된 train 1,118 / validation 299 / test 1,172 |
| 반복 | seed 42·7·123, 같은 seed의 초기 adapter·문항 순서 통제 |
| 학습 예산 | 원본 Student부터 독립 시작, 3 epochs·420 optimizer steps |
| Batch·입력 | micro batch 1, gradient accumulation 8, 최대 1,024 tokens |
| Optimizer | AdamW, lr 2e−4, weight decay 0, β=(0.9, 0.999), ε=1e−8 |
| Scheduler·안정화 | warmup 10 + linear decay, gradient max norm 1.0, gradient checkpointing(use_reentrant=False) |
| LoRA | rank 8, alpha 16, dropout 0.05, all-linear, bias=none |
| 최종 checkpoint | epoch 3 고정; validation 최고점으로 선택하지 않음 |
| Teacher soft target | 첫 답변 위치, T=2, 전체 어휘 V=248,320, 기존 FP32 log-probability cache 재사용 |

Teacher revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, Student `15852e8c16360a2fea060d615a32b45270f8a8fc`, 데이터 `210d026faf9955653af8916fad021475a3f00453`을 유지한다. 공식 train 1,119개 중 중복 입력 1개를 제거했으며 train/validation/test의 원본 ID·전체 입력 hash 교집합은 0이다. 선택지 3·4·5개를 유지한다. [데이터 기록](../results/protocols/data_manifest.json)

Teacher hard 응답의 train 오답 52/1,118개도 제거하거나 Gold로 바꾸지 않는다. 마지막 gradient accumulation 그룹 6개도 실제 그룹 크기로 정규화한다. Student는 4-bit 양자화를 적용하지 않으므로 Student 학습을 QLoRA로 표기하지 않는다.

<a id="methods"></a>

## 4. E1~E7 실험 조건과 손실함수

**7조건 × 3 seeds = 21개 본실험 학습 조건**이다. E1~E7은 순차 학습 단계가 아니며 모두 동일 원본 Student에서 독립 시작한다.

| 조건 | 방식 | 지도 신호·알고리즘 | 연구상 역할 |
|---|---|---|---|
| E1 | Gold SFT | 실제 정답 letter + EOS에 CE | Teacher 없는 비KD 기준선 |
| E2 | Response KD | Teacher가 생성한 letter + EOS에 CE | Hard-label KD 기준선 |
| E3 | Forward KL | Teacher 전체 어휘 P를 기준으로 KL(P∥Q) | 단독 KL 기준선 |
| E4 | Reverse KL | Student Q를 기준으로 KL(Q∥P) | 단독 KL 기준선 |
| E5 | Fixed 50:50 | Forward·Reverse 손실을 고정 가중치로 혼합 | 고정 대조군 |
| E6 | Adaptive Hard | 상대 confidence로 Forward 또는 Reverse를 선택 | 주 비교군 A |
| E7 | Adaptive Soft | 상대 confidence 차이로 연속 혼합 | 주 비교군 B, 신규 구현 대상 |

Gold와 Response는 각각 실제 정답 y, Teacher 답변 a를 사용해 letter·EOS의 CE를 평균한다. 현재 Response는 추론 과정이나 자유형 설명의 증류가 아니다.

$$
L_{Gold}=\tfrac12\{-\log Q(y\mid x)-\log Q(EOS\mid x,y)\},\qquad
L_{Response}=\tfrac12\{-\log Q(a\mid x)-\log Q(EOS\mid x,a)\}
$$

첫 답변 위치의 Teacher·Student logits를 T=2로 나누어 얻은 전체 어휘 분포를 P·Q로 둔다.

$$
D_F=\sum_vP(v)\log\frac{P(v)}{Q(v)},\quad
D_R=\sum_vQ(v)\log\frac{Q(v)}{P(v)},\quad
D(g)=gD_F+(1-g)D_R
$$

$$
L_{KL}=\tfrac12\{T^2D(g)+CE_{EOS}\},\qquad
c(X)=1-\frac{H(X)}{\ln V}
$$

| 조건 | Forward 계수 g |
|---|---|
| E3 Forward | 1 |
| E4 Reverse | 0 |
| E5 Fixed | 0.5 |
| E6 Hard | 1[c(P) ≥ c(Q)] (동률은 Forward) |
| E7 Soft | σ(c(P)−c(Q)−m), m=0.05, Student confidence를 통한 gradient 유지 |

다섯 KL 조건 모두 첫 답변에 Gold CE나 Teacher hard-letter CE를 추가하지 않는다. 공통 EOS CE는 Teacher 답변 letter 다음 위치에서 계산한다. Teacher는 동결하며 Student Q에는 KL gradient가 흐르고, Hard gate는 detach하고 **Soft gate의 Student confidence gradient는 유지**한다. Student confidence는 같은 학습 forward의 LoRA dropout을 포함한다. Confidence는 분포 집중도이며 정답일 확률이 아니다.

### E7: 원문 설정을 우선 적용

사용자 지적에 따라 v4.0의 `median(|Δc|)/ln(3)` 온도 선정안과 detached Soft 제안을 철회했다. 이 규칙을 이용한 calibration·학습은 수행하지 않았다. PDF의 τ_g 표기는 원문 수식과 다르므로 원문을 기준으로 정정한다.

[CaRE-KD §2.1 식 (3)](https://arxiv.org/html/2609.36734v1#S2.SS1)은 **g=σ(c(P)−c(Q)−m)**이며 별도 gate 온도로 나누지 않는다. 온도 표기로 다시 쓴다면 단위 스케일 τ_g=1과 같다는 것은 대수적 해석이며, 논문이 τ_g=1이라는 하이퍼파라미터를 따로 보고했다는 뜻은 아니다. 논문 §4.5는 margin ablation에서 **m=0.05**를 가장 좋은 값으로 보고하므로 첫 비교에 이 값을 그대로 고정한다. 이를 모든 실험의 공식 default나 ARC 최적값이라고 주장하지 않는다.

[§3.1 식 (7)](https://arxiv.org/html/2609.36734v1#S3.SS1)의 Soft gradient에는 `(D_F−D_R)∇g` selection 항이 있으므로 **Soft는 detach하지 않는다**. Teacher만 동결하고 Student Q·confidence에 gradient가 흐르게 한다. Hard는 이진 비교와 기존 gradient 차단을 유지한다. 따라서 두 변형의 차이에는 가중치 연속성뿐 아니라 margin·selection gradient가 포함되며 smoothing 하나의 인과 효과라고 해석하지 않는다.

| 설정 | 채택 값·근거 |
|---|---|
| Gate 식 | 원문 sigmoid(confidence gap − margin) |
| 별도 gate 온도 | 사용하지 않음 |
| Soft margin | m=0.05, 원문 §4.5에서 보고한 최상 ablation 값 |
| Soft gate gradient | 유지, 원문 §3.1의 selection 항 포함 |
| KD Temperature | 기존 T=2 유지; gate 온도나 Revival의 sampling temperature와 구분 |
| 추가 탐색 | 첫 실험에서는 없음. 별도 튜닝이 필요하면 후보·예산·선택 지표를 먼저 정의하고 validation에서 선택; test는 사용하지 않음 |

논문은 supplementary에 코드를 제공한다고 적었지만 이번 확인에서 공식 실행 코드의 저장소·config를 검증하지 못했다. 현재 근거는 공개 원문의 식 (3)·식 (7)·§4.5이며 코드 복제를 주장하지 않는다. `src/soft_gate.py`에 이 수식과 기존 공통 EOS 목적함수를 구현하고 selection gradient를 검사한다.

**원문 명세의 미확인점:** 원본 PDF p.3의 Figure 2는 식 (3)을 문자 그대로 계산한 unit sigmoid보다 가파르게 그려져 있다. Confidence 차이 −0.4~0.4에서 식의 가중치는 약 0.389~0.587인데 개념도는 더 넓은 전환을 보인다. 그림의 표현을 실행 설정으로 간주하지 않으며 공식 코드의 scaling을 확인하기 전에는 **공개 식 (3)의 적용 비교**라고만 명명한다. 임의의 gate 온도를 역추정하지 않는다. [대조 기록](../results/protocols/evidence_audit.json)

동률에서는 Hard=1이지만 Soft=σ(−0.05)≈0.4875다. Confidence가 [0,1]이면 Soft 가중치는 대략 0.259~0.721 범위여서 Hard처럼 0/1로 바뀌지 않는다. 실제 평균·SD·분위수·confidence gap을 기록하며, 50:50과 비슷해도 원문 설정을 성능 결과에 맞춰 재조정하지 않는다.

이번 비교는 원문 gate를 **ARC·T=2·첫 답변 위치·공통 EOS CE**에 적용하는 연구다. Revival·BALD·on-policy 생성·원문 전체 benchmark를 재현하지 않는다. 실제 수식 구현: [Soft loss](../src/soft_gate.py), [gradient·EOS 검사](../tests/test_soft_gate.py), [격리 GPU 검사](../src/probe_soft_gate.py).

Hard가 한 방향을 선택한다는 것은 손실 가중치 선택이다. 기존 구현은 두 KL을 계산하고 혼합하므로 Hard가 반드시 연산량 절반이거나 Soft보다 빠르다고 가정하지 않는다. E5~E7의 KL 계산·로그 범위를 동일하게 유지한다.

## 5. 성능·시간 효율성 평가 계획

<a id="evaluation"></a>

### 성능과 통계적 비교

| 우선순위 | 지표·분석 | 보고 원칙 |
|---|---|---|
| 주 성능 | Greedy 생성 Accuracy | 개별 seed, 3-seed 평균±sample SD(ddof=1) |
| 주 비교 | 최종 test Hard−Soft Accuracy 차이(%p)·paired 95% CI | RQ1의 단일 주 성능 비교 |
| 핵심 대조 | Hard−Fixed·Soft−Fixed 차이·95% CI | 사전 지정 보조 비교, 미조정 탐색적 구간으로 표시 |
| 주 효율성 | 학생 학습 단계 시간·정확도–시간 관측 곡선 | 정확도와 별도로 seed별·평균±SD, 고정 예산의 절충관계 |
| 설명 분석 | 실제 gate 가중치·Teacher 오류·학습곡선 | 분모와 측정 subset을 표시, 인과적 개선을 단정하지 않음 |
| 보조 품질·자원 | 선택지 NLL·ECE·Brier, 추론 지연·Peak VRAM | 부록; Accuracy 개선을 대신 입증하지 않음 |

생성은 thinking=False·greedy·max_new_tokens=8·batch 1·use_cache=False를 유지한다. 유효 letter 한 개만 엄격히 파싱하고 형식 오류는 오답, token 제한 도달은 별도 집계한다. 선택지 NLL은 **T=1 첫 생성 step의 유효 letter 확률**로 계산한다. 전체 어휘 KD 손실·문장 perplexity·calibration만의 지표가 아니다. ECE의 선택지 argmax 정오는 생성 답변 정오와 정의가 다를 수 있다.

Accuracy CI는 같은 문항을 모든 조건·seed에서 함께 5,000회 bootstrap(seed 42)하고, 문항별 관측 세 seed 결과를 평균한다. 관측한 세 seed에 조건부인 평가 문항 CI이며 seed 모집단까지 포괄하지 않는다. 299개 문항을 897개 독립 표본으로 계산하지 않는다. CI가 0을 포함하면 우위 미확인이고 동등성의 증거가 아니다. 여러 보조 비교에서 유리한 것만 골라 확증적 결론으로 쓰지 않는다.

### 시간 측정 범위와 공정성

학생 학습 단계 시간은 **첫 학습 loop 직전부터 마지막 adapter 저장·공통 종료 검사까지**의 wall-clock으로 정의해 기존 `training_seconds`와 맞춘다. 학습·optimizer 갱신·동일 로그 I/O·최종 저장을 포함하고, 모델/타깃 로딩·Teacher 캐시 생성·epoch 진단·별도 최종 평가는 제외한다. 모델 로딩 시간, 진단·평가 시간, 저장을 포함한 작업 전체 경과 시간을 별도 기록한다. 계측 구간 양끝에서 GPU를 동기화하고 시작·종료 시각을 남긴다. 기존 로그의 정확한 계측 범위와 실행 환경이 검증된 경우만 합친다.

Teacher hard 응답 생성과 soft target 추출 비용은 공통 준비 비용으로 각각 집계한다. E2는 hard 응답, E3~E7은 soft cache를 필요로 하므로 두 비용을 모든 KD에 일률 합산하지 않는다. 공유 cache 사용 시 한 번의 비용과 학생별 학습 시간을 분리하며, 조건별 독립 실행 비용을 추가 제시할 경우 필요한 준비 비용만 적용한다.

E5~E7은 같은 GPU·드라이버·라이브러리·정밀도·입력·평가 주기에서 측정한다. 새로운 측정 실행 순서는 seed 42: E5→E6→E7, seed 7: E6→E7→E5, seed 123: E7→E5→E6으로 순환해 순서 영향을 줄인다. 중복 GPU 작업 없이 부하·중단·실행 환경을 기록한다. 오래된 E1~E4 시간은 환경 확인 후 기술적 기준선으로 제시하고 작은 시간 차이를 gate의 구조적 개선으로 단정하지 않는다.

Peak allocated/reserved VRAM의 측정 구간을 함께 기록한다. 기존 peak가 진단 평가를 포함한 구간이면 순수 학습 peak로 바꿔 표기하지 않는다. 추론 지연은 동일 입력·warmup·batch·토큰 제한·동기화 조건에서 반복 측정해 평균·중앙값·p95를 보고하며 모델 로딩과 학습 비용은 분리한다. 모두 같은 2B Student 구조이므로 gate 자체가 배포 구조를 줄였다고 주장하지 않는다.

### 정확도–학습 시간 곡선: E5~E7

기존 80% 목표는 공식 ARC 기준이나 CaRE-KD의 설정에서 확인되지 않아 철회했다. [DAWNBench](https://arxiv.org/abs/1806.01427)는 정해진 품질 목표까지의 시간을 평가하지만 ARC의 80%를 정당화하는 자료는 아니다. 이번 본문은 목표를 새로 만들어 순위를 매기지 않고 **최종 Accuracy·총 학습 시간·관측된 학습 곡선**으로 비교한다. Time-to-Accuracy는 별도 목표·선정 근거·탐색 예산을 사전 명시할 수 있을 때만 후속 탐색 분석으로 추가한다.

- E5~E7의 세 seed 모두 epoch 0·1·2·3(step 0·140·280·420)에서 validation 299개 전체와 누적 학습 시간을 함께 기록한다. 4개 관측 시점은 기존 고정 epoch 설계에 맞춘 해상도이며 관측 사이 성능이나 최초 도달 시간을 추정하지 않는다.
- GPU 동기화 뒤 누적 학습 시간을 기록하고 진단·평가 비용은 별도 표시한다. RNG·훈련 모드·가중치를 보존하며 checkpoint 선택에 쓰지 않는다. 계측·평가 주기는 비교군에서 같게 유지한다. [PyTorch 2.11 벤치마크 원칙](https://docs.pytorch.org/docs/2.11/benchmark_utils.html)
- 기존 64개 진단 곡선은 그대로 보존하고 전체 validation 곡선과 섞지 않는다. E5 seed 42의 epoch 중간 checkpoint·전체 validation 곡선은 없으므로 소급 생성하지 않는다.

**E5 seed 42의 시간·곡선 측정용 재현 1회**는 기존 기록에 없는 관측을 동일 조건으로 수집하기 위한 계획이다. 원 논문이 요구한 실험 수라는 뜻은 아니다. 기존 성능 checkpoint를 대체하거나 test 모델을 추가하지 않으며, 재현성 불일치는 별도 기록한다. 새로운 E5~E7 측정끼리 시간 비교를 우선하고 기존 E1~E4 시간은 기술적 참조로 한정한다.

### 그래프와 결과 해석

본문에는 7조건 Accuracy·개별 seed, Hard−Soft 및 Fixed 대비 차이의 CI, E5~E7의 Accuracy–누적 학습 시간과 gate 가중치 변화 그림을 배치한다. 공통 Gold CE·Accuracy 곡선과 각 목적함수 loss는 구분하고 서로 다른 loss 크기로 우열을 판단하지 않는다. NLL·ECE·Brier·Teacher 오류·시간/VRAM 상세표는 부록에 둔다. A/B/C/D는 문항별 답 위치이므로 의미 클래스별 macro-F1이나 BLEU·ROUGE를 핵심 지표로 추가하지 않는다.

Accuracy 상승·시간 감소는 관측 설정에서 양쪽 개선, Accuracy 상승·시간 증가 또는 반대는 절충관계로 보고한다. Accuracy 차이 미확인은 동등성이 아니며, 시간은 세 측정 seed의 기술적 변동·차이를 함께 표시한다. 임의 가중치의 Accuracy–시간 종합 순위나 불안정한 '1%p당 비용'을 주 결과로 만들지 않는다.

## 6. 기존 결과 활용·추가 예산·실행 단계

| 항목 | 현재 기록과 새 계획 |
|---|---|
| 기존 6조건 목표 | 18회 중 학습 13회, final validation 검증 12회 완료 |
| 완료된 조건 | E1~E4 각 3 seeds; E5 seed 42는 학습만 완료 |
| 신규 본실험 | E7 3 seeds 추가, 전체 21조건. 호환성 확인 시 기존 13회 활용·남은 학습 8회·final validation 9회 |
| 시간 측정 재현 | E5 seed 42 1회 추가. 본실험 21조건과 분리해 **총 학습 실행 예산 22회(기존 포함)**로 표시 |
| 최종 test | Teacher·원본 Student + E1~E7의 21개 최종 adapter = 23개 모델 |

기존 13회는 **설정·초기화·순서·모델·데이터·adapter hash 호환성 검증 후 참조 경로로 재사용한다**. 원본 checkpoint·데이터·학습 설정·loss·초기화·문항 순서·평가 정의가 호환되는지 새 프로토콜에서 확인하고 참조 경로·hash로 재사용한다. 기존 프로토콜·소스·원본 103개와 Reverse 파일럿 27개 산출물은 유지하며, v4.0 출력은 별도 namespace에 저장한다. [기존 검증 결과](../results/checks/through_reverse_validation_verified.json)

1. **설계 동결 준비:** E7 원문 gate·m=0.05·Soft gradient 유지·시간 구간·관측 주기·비교 우선순위를 코드/설정에 명시한다. 기존 6조건 실행기는 보존하고 별도 v4 실행기를 사용한다.
2. **구현 검증:** E7 수식·gradient·g 범위, confidence 동률에서 Soft=σ(−0.05)/Hard=1, 식 (7)의 selection gradient와 공통 EOS·Student KL gradient 유지, finite loss·EOS·누적 batch·RNG 보존·GPU 메모리를 확인한다. 확인된 confidence 범위에 따라 Soft 가중치 범위도 검사한다.
3. **프로토콜 개정:** 소스·공통 설정·원문 margin·gradient 설정·타이머 정의·초기 adapter·데이터·target cache의 hash와 기존 실행 재사용 근거를 별도 고정한다. 기존 동결 hash를 덮어쓰지 않는다.
4. **학습·validation:** 남은 본실험 8회와 시간 재현 1회를 수행한다. 모든 조건의 epoch 3 final validation을 검증하고, E5~E7의 3-seed 시간·전체 validation 곡선·실제 gate 분포를 수집한다.
5. **test 동결·평가:** 21개 최종 adapter와 분석 코드를 동결한 뒤 23개 모델의 test 1,172개를 평가한다. 학습 곡선은 validation에서만 측정하며 test를 학습 도중 호출하지 않는다.
6. **보고 종료:** RQ1·RQ2·RQ3의 표·그림·불확실성·시간 포함 범위·한계를 작성한다. Test를 보고 gate 설정·조건·seed·checkpoint를 다시 선택하거나 재학습하지 않는다.

이미 확인한 중간 결과는 v3.2 기록으로 보존하며 E6·E7 결과를 추정해 채우지 않는다. [기존 연구보고서](RESEARCH_REPORT.md) · [기존 통계 원본](../results/main/comparison.json)

### v4.3 실행 동결 완료

`src/gate_experiment.py`는 기존 로더·encoder·Fixed/Hard 손실·평가 함수를 재사용하며 Soft만 검증된 원문 손실로 연결한다. 22개 검사와 실제 새 경로의 GPU 갱신 검사 후 소스 10개·runtime 버전·데이터/cache·재사용 13회/12회·보호 130개 산출물 hash를 동결했다. 기존 누적 학습과 새 loop의 parameter 갱신이 동일함을 마지막 부분 batch까지 검사했다. [동결 manifest](../results/protocols/gate_training_protocol.json) · [GPU 실측](../results/checks/trainer_gpu_check.json)

9개 시간 측정 run의 순서는 seed 42 Fixed→Hard→Soft, seed 7 Hard→Soft→Fixed, seed 123 Soft→Fixed→Hard로 고정한다. 하나의 GPU에서 순차 실행하며 CPU 코어 증가나 GPU 동시 학습으로 시간 비교 조건을 바꾸지 않는다. 새 실행기는 학습·validation 검증 완료에서 멈추고 별도 test 동결을 기다린다.

**후속 평가 준비:** 평가·분석 소스와 기존 5,000회 paired bootstrap(seed 42)·동일 generation·warmup 정의를 test 전에 사전 고정했다. 별도 test manifest는 학습 21회·final validation 21회·시간 측정 9회 검증 후에만 생성한다. Teacher·원본 Student·최종 adapter 21개를 평가하며 Fixed seed 42 재현은 제외한다. [평가 사전 동결](../results/protocols/gate_evaluation_protocol.json)

Teacher full-cache 준비의 기존 시간은 hard 응답과 soft 확률 추출의 합산 구간이다. 기존 기록에서 두 비용을 분리할 수 없어 공유 준비 비용으로 보고하고, 독립 조건별 hard/soft 준비 비용 비교는 수행하지 않는다.

## 7. 일정·논문 산출물

아래는 2026-10-12 기준 목표 일정이며 실행 완료를 뜻하지 않는다. 결과의 부호·유의성에 따라 제출 범위를 바꾸거나 미완료 결과를 완성된 것처럼 기록하지 않는다.

| 목표 기간 | 작업 |
|---|---|
| 10/12~10/15 | 계획 개정, E7·시간 계측 구현, 원문 gate·gradient 검증·새 프로토콜 동결 |
| 10/16~10/20 | 잔여 학습·시간 재현·전체 validation 검증 |
| 10/21~10/23 | 최종 adapter·분석 동결, 23개 모델 test·통계·그래프 |
| 10/24~10/25 | Accuracy·시간 중심 원고, 근거·양식·저자·결과 검토 |
| 10/26 | 완료 범위의 원고 제출 목표 |

한국디지털콘텐츠학회 공식 안내를 2026-10-12에 확인했다. 논문 접수는 **10/26 마감**, 대학생 원고는 **2단 1~2쪽**, 개최는 **11/19~20 수원컨벤션센터**다. 제출 시 최신 대학생 전용 양식과 접수 요건을 다시 확인한다. [학회 공식 안내](https://dcs.or.kr/homepage/custom/submit)

원고는 서론 → 방법(모델·7조건·gate) → 결과(Hard·Soft·Fixed의 Accuracy·CI·학습 시간/곡선) → 논의(가중치·확률 품질·절충) → 결론·참고문헌 순서로 구성한다. 상세 E1~E4 비교·로그·재현 기록은 저장소에 둔다. 제출·등록 자체는 이번 계획서 작성 범위에 포함하지 않는다.

## 8. 한계·향후 연구·종료 기준

- 영어 ARC·단일 Qwen 조합·첫 답변 위치·고정 예산에 한정된다. 공개 모델의 ARC 사전 노출은 확인하지 못했다.
- 시간은 공유 데스크톱에서 관측한 값이다. 2026-10-13 다른 앱의 GPU 엔진 활동을 한 번 확인했으며 이전 run 전체의 영향은 미측정이다. 알고리즘 자체의 속도 차이나 전용 GPU 성능으로 해석하지 않는다. [관측값](../results/checks/timing_environment_snapshot.json)
- 전체 어휘 entropy는 정답 confidence가 아니다. Soft와 Hard의 평균 Forward 기여율 차이가 성능을 설명할 수 있으며 가중치 일치 Fixed/Random 대조군 없이 gate 정보의 인과 효과를 단정하지 않는다.
- 원문 gate를 ARC에 적용한 비교다. 원문 실험 전체를 재현하지 않으며 공개 실행 코드 config는 미검증이다. 원문 PDF와 HTML을 대조했으며, 실제 GPU 검사 결과도 따로 기록한다.
- 3 seeds와 희소한 Teacher 오류 subset은 강한 일반화에 부족하다. 시간은 GPU 부하·I/O·캐시·실행 순서의 영향을 받으며 실측 환경을 함께 보고한다. 현재 reference kernel 환경이므로 최적 배포 성능으로 일반화하지 않는다.

평균 가중치 일치 Fixed/Random Gate, Feature/Relation KD, 다른 모델 계열·데이터·양자화·Reasoning KD는 **후속 연구**로 둔다. 이번에는 21개 성능 조건, 명시한 시간 측정, 동결 test, 핵심 표·그림·한계 보고를 완료하면 종료한다.

## 참고문헌·근거

- 사용자 제공 PDF: 「LightJudge 지식 증류 성능 및 시간 효율성 연구보고서」, 2026-10-12. 연구 개정·7조건·시간 효율성·제출 구성의 기준 자료.
- [CaRE-KD](https://arxiv.org/html/2609.36734v1): 상대 entropy confidence와 KL gate. 식 (3)·식 (7)·margin ablation을 근거로 설정을 고정.
- [MiniLLM](https://arxiv.org/abs/2306.08543): Reverse KL 기반 LLM 증류 관련 연구. On-policy 최적화 전체 재현은 아님.
- [Hinton 외(2015)](https://arxiv.org/abs/1503.02531): soft-target 지식 증류.
- [Reimers·Gurevych(2017)](https://aclanthology.org/D17-1035/) · [Dror 외(2018)](https://aclanthology.org/P18-1128/): 반복 성능 변동·NLP 통계 비교.
- [Guo 외(2017)](https://arxiv.org/abs/1706.04599): confidence calibration과 보조 품질 평가.

- [공식 9B revision](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a) · [공식 2B revision](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc) · [ARC dataset card](https://huggingface.co/datasets/allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453): 모델·분할 원본 대조.
- [LoRA](https://arxiv.org/abs/2106.09685) · [QLoRA/NF4](https://arxiv.org/abs/2305.14314) · [공식 bitsandbytes 설명](https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes): 방법 근거이며 우리 수치의 최적성 근거는 아님.
- [DAWNBench](https://arxiv.org/abs/1806.01427) · [PyTorch 2.11 Benchmark Utils](https://docs.pytorch.org/docs/2.11/benchmark_utils.html): 시간–품질 비교와 계측 원칙.
