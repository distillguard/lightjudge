# 연구계획서

**대규모 언어모델 지식 증류에서 신뢰도 기반 Hard·Soft Gate의 성능과 시간 효율성 비교**

v4.3 본실험의 동결 설계와 후속 탐색을 구분해 정리했다. 사용자 제공 「LightJudge 지식 증류 성능 및 시간 효율성 연구보고서」를 개정의 출발점으로 삼았다. 갱신: 2026-10-14.

<!-- main-status:start -->
**v4.3 완료:** 본실험 21개 학습·validation과 Teacher·원본 Student·21개 adapter의 test 1,172개 평가 완료. Fixed seed 42 시간 재현 1회는 별도이며 원본 최종 checkpoint를 유지했다. Test 이후 재학습·checkpoint 변경은 수행하지 않았다.
<!-- main-status:end -->

## 1. 연구 질문과 범위

Teacher·Student의 confidence 차이에 따라 Forward/Reverse KL을 **이진 선택하는 Hard**, **연속 혼합하는 Soft**, **고정 혼합하는 Fixed**를 동일 조건에서 비교한다.

| 구분 | 질문 | 판단 자료 |
|---|---|---|
| RQ1 주 비교 | Hard와 Soft의 정확도·학습 시간은 다른가? | Hard−Soft Accuracy·paired CI, seed별 시간·학습곡선 |
| RQ2 핵심 대조 | 두 gate는 Fixed 50:50보다 효과적인가? | Hard−Fixed·Soft−Fixed의 탐색적 차이·CI |
| RQ3 후속 탐색 | Soft가 50:50 부근에 집중될 때, 연속 가중이 얼마나 다르게 작동하는가? | 문항·epoch별 가중치/gradient, 새 데이터의 동일 모델 비교 |

현재 Hard−Soft 비교에는 연속성 외에 **margin과 selection gradient 차이**도 포함된다. 이 비교만으로 smoothing 하나의 인과 효과나 보편적인 gate 우위를 주장하지 않는다. 본실험 결과는 [연구보고서](RESEARCH_REPORT.md)에 둔다.

## 2. 모델·데이터·학습 통제

| 항목 | 완료된 v4.3의 고정 설정 |
|---|---|
| Teacher | Qwen3.5-9B, NF4 double quantization·BF16 compute, 전체 동결 |
| Student | Qwen3.5-2B, BF16 원본 동결·FP32 LoRA 8,409,600개 학습 |
| 데이터 | ARC-Challenge: train 1,118 / validation 299 / test 1,172 |
| 반복·초기화 | seed 42·7·123; 각 조건은 같은 원본 Student에서 독립 시작; 같은 seed의 초기 adapter·순서 통제 |
| 학습 예산 | 3 epochs·420 optimizer steps; 최종 epoch 3 고정 |
| Batch·입력 | micro batch 1·gradient accumulation 8·최대 1,024 tokens |
| Optimizer | AdamW, lr 2e−4·weight decay 0·betas=(0.9,0.999)·eps=1e−8 |
| Scheduler·안정화 | warmup 10·linear decay·max grad norm 1.0·gradient checkpointing |
| LoRA | rank 8·alpha 16·dropout 0.05·all-linear·bias=none |
| KL target | T=2·첫 답변 위치·전체 어휘 V=248,320·FP32 log-prob cache |

[Teacher 고정 revision](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a) · [Student 고정 revision](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc) · [ARC 고정 revision](https://huggingface.co/datasets/allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453).

공식 train 1,119개에서 동일 입력 1개를 제거했다. 분할 간 원본 ID·전체 입력 hash 교집합은 0이며 선택지 3~5개를 유지한다. Teacher의 train 오답 52개도 유지했다. Student는 4-bit가 아니므로 **BF16 LoRA 학습**으로 표기한다. 공개 checkpoint는 post-trained 모델이며 ‘원본 Student’는 ARC 추가 학습 전을 뜻한다. [데이터 기록](../results/protocols/data_manifest.json) · [학습·재사용·보존 동결](../results/protocols/gate_training_protocol.json)

<a id="methods"></a>

## 3. 7조건의 알고리즘

**7조건×3 seeds=21개 성능 조건**이다. Gold SFT 뒤 KD를 이어 하는 순차 단계가 아니라 모두 독립 학습이다.

| 조건 | 지도 신호와 손실 | 역할 |
|---|---|---|
| E1 Gold SFT | 실제 정답 letter+EOS의 CE 평균 | Teacher 없는 기준선 |
| E2 Response KD | Teacher 응답 letter+EOS의 CE 평균 | hard-label 증류 |
| E3 Forward KL | Teacher 확률로 가중한 KL(P∥Q), g=1 | 단독 Forward 기준선 |
| E4 Reverse KL | Student 확률로 가중한 KL(Q∥P), g=0 | 단독 Reverse 기준선 |
| E5 Fixed | 두 KL을 50:50 혼합, g=0.5 | 고정 대조군 |
| E6 Hard | c(P)≥c(Q)이면 g=1, 아니면 0 | 이진 선택; gate detach |
| E7 Soft | g=σ(c(P)−c(Q)−0.05) | 연속 혼합; gate gradient 유지 |

P·Q는 첫 답변 위치 logits를 T=2로 나눈 전체 어휘 분포다.

$$
D_F=\sum_vP(v)\log\frac{P(v)}{Q(v)},\quad
D_R=\sum_vQ(v)\log\frac{Q(v)}{P(v)},\quad
c(X)=1-\frac{H(X)}{\ln V}
$$

$$
L_{KL}=\tfrac12\{T^2[gD_F+(1-g)D_R]+CE_{EOS}\}
$$

다섯 KL 조건은 첫 답변에 별도 Gold/Teacher hard-letter CE를 붙이지 않는다. 공통 EOS CE는 Teacher 응답 letter 다음 위치에서 계산한다. Response KD는 추론 과정이나 자유형 설명을 증류한 실험이 아니다. Confidence는 **분포 집중도**이며 정답일 확률이 아니다.

Soft는 [CaRE-KD 식 (3)·(7)](https://arxiv.org/html/2609.36734v1)의 sigmoid와 selection 항 `(D_F−D_R)∇g`를 유지한다. 별도 gate 온도 τ_g는 사용하지 않는다. Margin 0.05는 원문의 ablation에서 보고한 값이고, T=2·lr·rank·epoch·seed는 기존 통제값으로서 ARC 최적값이나 논문 공통 표준값이 아니다.

Confidence 차이가 0이면 Hard=1, Soft≈0.4875다. Confidence 범위 [0,1]에서 Soft는 약 0.259~0.721이며 Hard처럼 0/1에 도달하지 않는다. `dg/dΔc=g(1−g)`이고, 평균이 0.5에 가까워도 가중치·gradient가 상수라고 단정하지 않는다. 구현은 [soft_gate.py](../src/soft_gate.py), 검사는 [test_soft_gate.py](../tests/test_soft_gate.py)에 둔다.

두 KL을 모두 계산하는 구현이므로 Hard의 연산량이 절반이라고 가정하지 않는다. 이 연구는 원문 gate의 과제별 적용이며 Revival·BALD·on-policy 생성·CaRE-KD 전체 실험의 재현은 아니다. 원문 Figure 2와 식의 scaling 차이는 공식 실행 config가 미확인인 한계로 남긴다. [근거 점검](../results/protocols/evidence_audit.json)

<a id="evaluation"></a>

## 4. 평가·통계·시간 정의

| 우선순위 | 지표 | 해석 |
|---|---|---|
| 주 성능 | Greedy 생성 Accuracy | 각 seed·3-seed 평균±sample SD |
| 주 비교 | Hard−Soft Accuracy 차이(%p)·paired 95% CI | 단일 주 비교; CI는 불확실성 구간 |
| 핵심 대조 | Hard−Fixed·Soft−Fixed 차이·CI | 미조정 탐색적 보조 비교 |
| 주 효율성 | 학습 시간·정확도–누적 시간 곡선 | 정확도와 별도 보고; 임의 종합 순위 없음 |
| 보조 해석 | gate·Teacher 오류·NLL·ECE·Brier·VRAM·추론 지연 | 주 비교의 우위를 대신 입증하지 않음 |

생성은 thinking=False·greedy·max_new_tokens=8·batch 1·use_cache=False를 유지한다. 유효 letter 한 개만 파싱하며 형식 오류는 오답이다. 선택지 NLL은 T=1의 첫 답변 위치에서 유효 문자에 재정규화한 확률로 계산한다. 전체 어휘 KD loss·문장 perplexity와 구분한다. A~E는 문항별 답 위치이므로 의미 클래스의 macro-F1·BLEU·ROUGE를 주 지표로 추가하지 않는다.

문항을 모든 방법·seed에서 함께 5,000회 bootstrap(seed 42)하고 문항별 세 seed 결과를 평균한다. CI는 **관측한 세 학습 seed에 조건부**이며 새 seed 변동을 포괄하지 않는다. CI가 0을 포함하면 우위 미확인이고 동등성 입증이 아니다. 서로 다른 목적함수의 loss 크기로 방법의 우열을 판단하지 않는다.

학습 시간은 첫 학습 loop 직전부터 최종 adapter 저장·공통 종료 검사까지의 wall-clock이다. 학습·optimizer·공통 로그·저장을 포함하고 모델/cache 로딩·Teacher target 준비·epoch 진단·별도 최종 평가는 제외한다. 계측 경계에서 GPU를 동기화한다. 기존 Teacher 준비 로그는 hard 응답 생성과 soft 확률 추출을 합산했으므로 소급 분리하지 않는다.

Fixed·Hard·Soft는 epoch 0·1·2·3의 validation 299개 전체와 누적 시간을 관측했다. 순서는 seed 42 Fixed→Hard→Soft, seed 7 Hard→Soft→Fixed, seed 123 Soft→Fixed→Hard다. Fixed seed 42의 시간 재현 1회는 성능 checkpoint를 대체하지 않았다. 공유 GPU·reference kernel 환경의 실측이며 알고리즘 자체의 속도 우위를 확정하지 않는다.

학습·validation 완료 뒤 21개 adapter와 분석을 동결하고 Teacher·원본 Student를 포함한 23개 모델의 test를 평가했다. Test 관측 후 재학습·checkpoint 변경·결과에 따른 설정 선택을 하지 않는다. [평가 사전 동결](../results/protocols/gate_evaluation_protocol.json) · [최종 test manifest](../results/protocols/gate_test_manifest.json)

<a id="data-followup"></a>
<a id="11-다른-데이터에서-이진-선택과-연속-가중의-비교-후속-탐색"></a>

## 5. 다음 단계: 같은 모델에서 다른 데이터 비교

현재 후속 질문은 **Hard의 이진 선택과 Soft의 연속 혼합이 Fixed와 얼마나 다르게 작동하는가**다. 데이터별 결과에 맞춰 sigmoid를 가파르게 만들거나 margin을 변경하지 않는다.

| 데이터 | 원본 train / validation | 정제 train 후보 | 선정 근거 |
|---|---|---|---|
| OpenBookQA main | 4,957 / 500 | 4,907 | 과학 지식·상식 4지선다로 ARC와 가까운 과제 |
| HellaSwag | 39,905 / 10,042 | 39,905 | 상황의 다음 전개 4지선다로 다른 영역 |

[OpenBookQA 논문](https://aclanthology.org/D18-1260/) · [HellaSwag 논문](https://aclanthology.org/P19-1472/). Train·validation만 고정 revision으로 확보했고, 원본을 유지하며 동일 입력 중복·validation 겹침·정답 선택지 충돌을 train 후보에서 제외했다. HellaSwag ID는 출처+번호를 사용한다. [데이터·EDA·hash](../results/gate_continuity/datasets.json)

**완료:** ARC와 새 두 데이터에서 미리 고른 train 128문항씩을 기존 Teacher·원본 Student로 진단했다. 공통 중립 문구·native template·T=2 전체 어휘 confidence를 사용했다. HellaSwag는 raw context/endings의 문자 선택형 변환이므로 공식 completion 점수가 아니다. 유효 선택지 entropy는 별도 측정 공간 진단이다. [선정·수식 동결](../results/gate_continuity/protocol.json)

**미실행:** 새 데이터 2개×Fixed·Hard·Soft×seed 42의 **6회 탐색 학습**. 먼저 전체 token 길이·공통 train 예산·validation·prompt·checkpoint 기준을 동결한다. 정제 후보 수를 자동으로 학습 예산으로 삼지 않는다. 두 데이터 결과를 모두 보고하며, 1 seed 결과로 안정적인 일반화를 확정하지 않는다. 새 test는 모든 선택을 고정한 뒤 접근한다.

Soft−Fixed에는 평균 가중치 이동·문항별 변화·selection gradient가 함께 포함된다. **연속성 자체의 인과 효과**를 주장하려면 train으로 정한 상수 가중치 대조와 Soft detach 대조가 추가로 필요하다. 현재 3조건의 질문은 실제 방법 차이이며 세 원인의 분리는 별도 확장이다.

<a id="bayesian-followup"></a>

## 6. 베이지안 분석의 역할

완료한 test 예측의 문항×seed×방법 Bernoulli-logit 계층모형, 3가지 prior 민감도와 paired Bayesian bootstrap을 보조 분석으로 수행했다. 기존 test 관측 이후의 탐색적 사후 분석이며 원래 주 비교·paired CI를 대체하지 않는다. 4개 MCMC chain과 3개 학습 seed는 다른 반복이다.

MCMC 수렴과 후방 예측 적합성을 구분한다. 현재 계층모형은 방법 간 정오 겹침을 충분히 재현하지 못하므로 탐색적 참고로 제한한다. 공통 Dirichlet 문항 가중치의 bootstrap은 관측한 세 seed에 조건부다. 이 모형을 다른 데이터에 그대로 확증 목적으로 확장하지 않는다. [분석 동결](../results/protocols/bayesian_analysis_protocol.json) · [결과·진단](../results/bayesian_v1/summary.json)

<a id="models-followup"></a>
<a id="9-다른-계열-확장-후보토크나이저-준비-2026-10-13"></a>

## 7. 이후 단계: 다른 Student 계열

| 준비한 Student | 실제 텍스트 가중치 | 모델 내 비교 |
|---|---|---|
| Qwen3.5-2B | 1.882B | Fixed·Hard·Soft |
| SmolLM2-1.7B-Instruct | 1.711B | Fixed·Hard·Soft |
| Granite-3.3-2B-Instruct | 2.534B | Fixed·Hard·Soft |

모델별 공식 tokenizer·native template를 유지하고 출력 의미만 A~E로 대응한다. 서로 다른 숫자 token ID를 같은 어휘라고 취급하지 않는다. 단일 답변 토큰·prefix 경계·입력 예산·실제 GPU 갱신 검사는 완료했다. 전체 후보·revision·LoRA 수·근거는 [준비 기록](../results/protocols/choice_kd_setup.json)에 둔다.

이 확장은 **유효 선택지에 재정규화한 별도 KD 손실**이며 기존 전체 어휘 KD와 구분한다. 이미 T=2인 Teacher cache에 온도를 다시 적용하지 않고 EOS CE를 붙이지 않는다. Qwen도 같은 새 손실로 다시 비교한다. [MCQA distillation](https://aclanthology.org/2025.findings-emnlp.452/)의 문자 logit 대응을 참고했으며, [DSKD](https://aclanthology.org/2024.emnlp-main.1010/)·[ULD](https://arxiv.org/html/2402.12030v3)의 어휘 정렬을 구현한 것은 아니다.

모델 3개×3조건×seed 42의 **9회 탐색 학습**은 새 데이터 비교 뒤로 보류한다. 초기 validation·독립 본학습·새 test는 미실행이다. 크기·template·LoRA 수가 달라 raw Accuracy 차이를 계열 자체의 인과 효과로 주장하지 않는다. [문헌 대조](../results/protocols/cross_family_evidence.json)

## 8. 근거·산출물·한계

수식·공식 모델/데이터 자료·기존 통제값·실측·미확인을 구분한다. 새 튜닝은 후보·예산·선택 지표를 먼저 정하고 validation에서 선택하며 test를 사용하지 않는다. 원본 130개 산출물(103+27)·동결 소스·완료 adapter를 보존하고 후속 실험은 별도 경로에 둔다.

현재 결론은 영어 ARC·단일 Qwen 조합·첫 답변 위치·3 seeds·공유 GPU에 한정된다. 공개 checkpoint의 사전 노출·의미상 데이터 중복·공식 CaRE 실행 config는 미검증이다. 최종 원고는 서론→방법→Accuracy·CI·시간/곡선→gate 해석·한계→결론으로 구성하며 상세 기록은 저장소에 둔다.

제출 목표는 **2026-10-26**이다. 2026-10-13 확인한 [학회 공식 안내](https://dcs.or.kr/homepage/custom/submit)는 대학생 원고 2단 1~2쪽, 행사 11/19~20을 명시한다. 제출·등록은 별도 작업이다.

- [CaRE-KD](https://arxiv.org/html/2609.36734v1): gate·margin·selection gradient의 근거.
- [Hinton 외](https://arxiv.org/abs/1503.02531) · [MiniLLM](https://arxiv.org/abs/2306.08543): 확률분포 증류·Reverse KL 관련 연구.
- [LoRA](https://arxiv.org/abs/2106.09685) · [QLoRA/NF4](https://arxiv.org/abs/2305.14314): 방법 근거이며 현재 수치의 최적성 근거는 아님.
- [Dror 외](https://aclanthology.org/P18-1128/) · [Reimers·Gurevych](https://aclanthology.org/D17-1035/): NLP 통계 비교·반복 변동.
- [Stan 계층 로짓](https://mc-stan.org/docs/stan-users-guide/regression.html#item-response-theory-models) · [Benavoli 외](https://jmlr.org/papers/v18/16-305.html) · [Rubin](https://doi.org/10.1214/aos/1176345338): Bayesian 구조·해석·bootstrap.
- [Guo 외](https://arxiv.org/abs/1706.04599) · [PyTorch 계측](https://docs.pytorch.org/docs/2.11/benchmark_utils.html): calibration·시간 계측 원칙.

<!-- data-gates:start -->
## 현실적인 후속 범위 동결: 2026-10-14

**2개 데이터×Fixed·Hard·Soft×seed 42=6회 탐색 학습**으로 정했다. 각 train은 질문 단위 중복·validation 겹침·라벨 충돌을 제외한 1,118개, 3 epochs·420 steps이며 개발용 평가 500개를 사용한다. 기존 T=2·margin=0.05·BF16 LoRA·전체 어휘 KD·공통 EOS를 유지한다. Teacher·원본 Student의 개발용 기준선과 train 8문항의 gradient 크기를 먼저 측정한다.

HellaSwag는 공개 validation에서 질문·출처가 겹치지 않는 개발용 500개/최종 평가용 500개를 사전 분리했다. OpenBookQA는 공식 validation과 test를 쓰며, test는 6개 최종 adapter 동결 뒤 입력 겹침 제외 규칙을 적용하고 평가한다. 모든 평가는 T=1 선택지 문자 argmax/NLL로 통일하며 공식 completion 점수와 구분한다. 최종 평가 뒤 재학습·조건 변경은 하지 않는다.

원본 130개 산출물·12개 소스·21개 adapter는 보존한다. 다른 Student 9회와 추가 상수/detach 대조는 후속 단계로 둔다. 1 seed는 예산에 따른 탐색이며 안정적인 일반화를 확정하지 않는다. [실행·분할·코드 동결](../results/data_gates_v1/protocol.json) · [현재 상태](../results/data_gates_v1/status.json)
<!-- data-gates:end -->
