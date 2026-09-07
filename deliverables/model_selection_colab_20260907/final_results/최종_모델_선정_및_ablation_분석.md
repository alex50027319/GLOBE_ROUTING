# SwitchGLOBE 최종 모델 선정 및 ablation 분석

## 1. 실험 완전성

- 실행 환경: Google Colab `NVIDIA A100-SXM4-40GB`, PyTorch `2.11.0+cu128`
- 학습 seed: 42, 77, 123, 314, 2718
- 평가 환경: 동일한 14개 시나리오
- 반복 수: 시나리오당 200 episodes
- 평가 정책: 10개
- 총 raw episode: `5 × 14 × 200 × 10 = 140,000`
- latency: CPU/A100, batch 1, warm-up 50회, 정책별·seed별 2,000회, randomized-block 순서와 장치 동기화 적용
- 추론 단위: 개별 episode가 아니라 **5개 training seed**
- 신뢰구간: seed-paired difference의 양측 95% Student-t CI
- raw key `(method, scenario, training_seed, evaluation_seed)`의 중복과 누락 없음

## 2. 비교한 ablation

### 최종 구조 ablation

1. Geo-Residual normal branch
2. Predictive Prior Only: residual을 0으로 둔 실제 predictive branch
3. Predictive Student (No Switch): predictive student 전체 출력
4. SwitchGLOBE Exact: normal·predictive 두 branch와 risk switch
5. FastSwitchGLOBE: 압축·근사 모델

### Policy-distillation 감사

6. Historical Untrained Local: negative control
7. Historical PPO-only Local: distillation 없음
8. Historical KD-only Local
9. Historical KD+PPO Local
10. Privileged Global Teacher: 배포 모델이 아닌 상한 참조

주의: PPO-only/KD-only/KD+PPO는 동일한 Phase-7 student 구조와 학습 세대를 비교하므로 distillation의 존재 여부를 평가하는 데 유효하다. 그러나 최종 Phase-11/12 구조를 KD 없이 새로 학습한 실험은 아니므로, 이를 최종 SwitchGLOBE에서 KD의 순수 인과효과라고 과장하면 안 된다.

## 3. Predictive-only 대 SwitchGLOBE Exact

| 지표 | Predictive-only | SwitchGLOBE | paired 효과와 95% CI | 해석 |
| --- | ---: | ---: | ---: | --- |
| Connected PDR | 0.911612 | 0.911612 | -0.000000 [-0.000902, 0.000902] | 전체 평균 동일 |
| Deadline delivery | 0.837571 | 0.837643 | -0.000071 [-0.001033, 0.000890] | 차이 불확실 |
| p95 성공 지연 | 6.0 steps | 6.0 steps | 0.000 [0.000, 0.000] | pooled p95에서 차이 없음 |
| Mean energy proxy | 1.775750 | 1.779117 | +0.003367 [-0.000326, 0.007060] | 양수는 predictive-only 우세, CI는 0 포함 |
| Energy/delivered | 2.078118 | 2.082755 | +0.004637 [-0.000574, 0.009849] | predictive-only가 평균상 약간 낮음 |

Predictive-only와 SwitchGLOBE는 전체 episode의 **95.3%**에서 완전한 action sequence와 node trajectory가 일치했다(95% CI 92.83–97.77%). 최초 분기 전 common-prefix 비율은 95.88%(95% CI 93.68–98.07%)였다. 즉 risk switch는 전체 routing의 약 4.7%에서만 실제 trajectory를 바꿨다.

## 4. 최악 시나리오와 risk switch의 역할

Predictive-only의 seed별 최악 connected-PDR 차이는 평균 -0.70%p(95% CI -1.04, -0.36%p)였다. 특히 `ood_fast_mobility`에서는 SwitchGLOBE가 다섯 seed에 걸쳐 평균 0.50%p 높은 PDR을 보였다. 반대로 `ood_link_loss_30`에서는 predictive-only가 평균 0.60%p 높았다.

따라서 risk switch는 평균 PDR을 높이는 일반적 구성요소라기보다, 일부 고속 이동 조건에서 작은 최악조건 손실을 완화하는 선택적 안전장치로 해석하는 편이 데이터에 부합한다. 현재 결과만으로 “dual branch가 모든 환경에서 우월하다”고 주장하면 reviewer의 지적을 받을 가능성이 높다.

## 5. CPU 및 A100 batch-1 latency

| 모델 | CPU p95, ms | A100 p95, ms | SwitchGLOBE 대비 CPU 감소 | SwitchGLOBE 대비 A100 감소 |
| --- | ---: | ---: | ---: | ---: |
| Predictive Prior Only | 0.5664 [0.5619, 0.5708] | 1.3900 [1.3824, 1.3977] | 75.9% | 77.0% |
| Predictive Student, no switch | 1.7483 [1.5938, 1.9029] | 2.7788 [2.7597, 2.7979] | 25.5% | 54.0% |
| SwitchGLOBE Exact | 2.3474 [2.3249, 2.3699] | 6.0462 [6.0108, 6.0816] | 기준 | 기준 |
| FastSwitchGLOBE | 0.9123 [0.9071, 0.9174] | 2.1101 [2.0951, 2.1252] | 61.1% | 65.1% |

Batch 1에서는 모든 주요 모델이 A100보다 단일-thread CPU에서 더 빨랐다. Predictive-only는 A100에서 CPU보다 약 2.45배, SwitchGLOBE는 약 2.58배 느렸다. 작은 신경망의 kernel-launch·host/device 동기화 비용이 산술 연산량보다 크기 때문이다. 따라서 실제 UAV의 단일 패킷 routing decision에는 CPU가 더 타당하고, GPU 결과는 대규모 batch 처리 가능성을 보여주는 보조 결과로 다루어야 한다.

## 6. Distillation 유무

KD-only는 PPO-only(no KD)에 비해 다음과 같은 direction-aligned paired 효과를 보였다.

- Connected PDR: +1.445%p, 95% CI -0.466–3.355%p
- Deadline delivery: +2.907%p, 95% CI +1.168–4.646%p
- p95 성공 지연 감소: 1.2 steps, 95% CI 0.645–1.755
- Mean energy proxy 감소: 0.4327, 95% CI 0.3828–0.4826
- Energy/delivered 감소: 0.7120, 95% CI 0.6421–0.7818

KD+PPO 역시 no-KD보다 deadline, delay 및 energy에서 명확히 우세했다. 반면 KD+PPO와 KD-only의 차이는 모든 핵심 지표에서 CI가 0을 포함했다. 이 결과는 이 역사적 동일구조 비교에서 성능 향상의 주된 원천이 PPO fine-tuning보다 distillation임을 시사한다.

## 7. FastSwitchGLOBE 판단

FastSwitchGLOBE는 SwitchGLOBE 대비 CPU p95 latency를 61.1% 낮췄지만, 평균 connected PDR 효과가 -1.91%p(95% CI -3.86, +0.04%p), deadline 효과가 -2.26%p(95% CI -4.62, +0.10%p)였다. 에너지 지표도 유의하게 악화됐다. seed별 최악 시나리오 PDR은 평균 -10.6%p였다. 따라서 현재 Fast checkpoint를 최종 모델로 채택하면 안 된다.

## 8. 최종 선정

사전 정의한 규칙은 다음과 같다.

> Predictive-only의 connected PDR 및 deadline-delivery paired 95% CI 하한이 각각 -0.5%p 이상이고 CPU p95 latency를 낮추면 predictive-only를 선택한다. 그렇지 않으면 SwitchGLOBE Exact를 유지한다.

전체 pooled 성능 기준으로 이 조건을 만족하므로 **최종 배포 모델은 Predictive Prior Only**로 선정된다. SwitchGLOBE Exact는 평균 효율 기준의 최종 모델이 아니라, 고속 이동 등 worst-case robustness가 최우선인 배포에서 선택할 수 있는 safety profile로 남긴다.

단, 논문의 핵심 novelty를 risk switch로 유지하려면 현재 데이터는 충분히 강하지 않다. 그 경우 다음 추가 실험이 필요하다.

1. 최종 Phase-11/12 구조를 동일한 curriculum과 compute budget으로 KD 없이 처음부터 다시 학습한다.
2. `ood_fast_mobility`의 속도·link-lifetime 구간을 세분화하고 사전에 임계값을 고정한 stress sweep를 수행한다.
3. 평균 PDR이 아니라 lower-tail PDR, CVaR, deadline miss tail을 risk switch의 1차 endpoint로 사전 등록한다.
4. switch가 실제로 trajectory를 바꾼 약 4.7% episode만 조건부 분석하여 위험 감소 메커니즘을 입증한다.
5. non-inferiority margin 0.5%p의 도메인 근거를 논문에 제시한다.

## 9. 산출물 해석 주의사항

- `energy proxy`는 실제 Joule 측정치가 아니라 simulator transmission proxy다.
- pooled p95 성공 지연은 모든 주요 모델에서 6 steps로 포화되어 구분력이 낮다. 시나리오별 delay와 tail CDF를 함께 제시해야 한다.
- global teacher는 full topology를 사용하므로 배포 가능한 baseline이 아니다.
- Phase-7 KD audit과 최종 구조 ablation은 서로 다른 질문에 답하며 한 표에서 동일한 causal comparison처럼 표현하면 안 된다.
- 95% CI는 다섯 training seed에 기반하므로 통계적 검정력을 과장하지 않는다.
