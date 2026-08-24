# Phase 12 실험 환경, Episode 구성, Method별 동작 원리 설명

작성일: 2026-07-28  
기준 실험: Phase 12 Risk-Switch Lite-GLOBE-P  
주요 코드 기준:

- `ResearchAIWorkspace/implementations/lite_globe/experiments/phase12_campaign.py`
- `ResearchAIWorkspace/implementations/lite_globe/scenarios/generalization_suite.py`
- `ResearchAIWorkspace/implementations/lite_globe/scenarios/structural_holes.py`
- `ResearchAIWorkspace/implementations/lite_globe/scenarios/predictive_traps.py`
- `ResearchAIWorkspace/implementations/lite_globe/env/fanet_env.py`
- `ResearchAIWorkspace/implementations/lite_globe/env/observation.py`
- `ResearchAIWorkspace/artifacts/lite_globe/phase12/tables/risk_switch_results.md`

## 1. Phase 12 실험의 큰 구조

Phase 12는 FANET 환경에서 하나의 패킷을 출발지 UAV에서 목적지 UAV까지 전달하는 라우팅 실험이다. 각 실험 단위는 다음과 같이 구성된다.

```text
Scenario 하나 선택
-> Method 하나 선택
-> Evaluation seed 하나 선택
-> Episode 하나 실행
-> 출발지에서 목적지까지 패킷 전달 성공/실패 기록
```

Phase 12 full run의 규모는 다음과 같다.

| 항목 | 값 |
| --- | ---: |
| Training seed 수 | 5개 |
| Training seeds | 42, 77, 123, 314, 2718 |
| Scenario 수 | 14개 |
| Method 수 | 6개 |
| Scenario별 evaluation episode 수 | 200개 |
| 총 episode row 수 | 84,000개 |

총 episode 수 계산은 다음과 같다.

```text
5 training seeds
* 14 scenarios
* 6 methods
* 200 episodes
= 84,000 episode rows
```

여기서 episode row는 "특정 training seed에서 학습된 정책 하나가, 특정 scenario에서, 특정 evaluation seed로 패킷 하나를 전달해본 결과"를 의미한다.

## 2. Episode 구성

### 2.1 Episode란 무엇인가

Episode는 하나의 패킷 배달 시도다. 쉽게 말하면 드론 네트워크 안에서 편지 하나를 목적지 드론까지 보내보는 한 판이다.

한 episode는 다음 순서로 진행된다.

```text
1. 환경 reset
2. UAV 위치, 속도, 링크, queue 상태 생성
3. 출발지 source와 목적지 destination 결정
4. source가 패킷을 들고 시작
5. 현재 패킷을 가진 UAV가 다음 hop을 선택
6. 선택한 next-hop으로 패킷 전달
7. 목적지에 도착하면 성공
8. loop, invalid action, TTL 초과, time limit 등은 실패
```

### 2.2 출발지와 목적지 설정

출발지와 목적지는 scenario의 `reset_options`에 따라 두 가지 방식으로 정해진다.

첫 번째는 고정 방식이다. 구조적 함정이나 predictive-break처럼 정확한 topology를 시험하는 scenario에서는 출발지와 목적지가 고정된다.

```text
structural_hole 계열: source = 0, destination = 5
predictive_break 계열: source = 0, destination = 8
```

두 번째는 랜덤 방식이다. 일반 OOD, mobility, sparse, node-count scenario에서는 `source`와 `destination`이 고정되어 있지 않다. 이 경우 episode reset 때 seed 기반으로 랜덤 선택된다.

```text
source: 0부터 num_nodes - 1 중 하나를 랜덤 선택
destination: source를 제외한 나머지 노드 중 하나를 랜덤 선택
```

단, `require_connected=True`가 있는 scenario에서는 시작 시점에 출발지와 목적지가 연결 가능한 pair가 되도록 후보를 샘플링한다. `unconditional_sparse`는 이 조건이 없기 때문에 애초에 연결 불가능한 episode도 포함될 수 있다.

### 2.3 Queue 설정

각 UAV의 queue는 episode reset 때 랜덤으로 생성된다.

```text
queue[node] ~ UniformInteger(0, max_queue_size)
```

Phase 12의 주요 scenario에서는 `max_queue_size = 8`이 많다. 따라서 각 UAV는 0부터 8 사이의 queue 값을 가진다.

Queue 여유도는 다음과 같이 계산된다.

```text
queue_headroom_j = 1 - queue_j / max_queue_size
```

예를 들어 `max_queue_size = 8`일 때:

| Queue 값 | Queue 여유도 | 의미 |
| ---: | ---: | --- |
| 0 | 1.0 | 매우 여유 있음 |
| 4 | 0.5 | 절반 정도 차 있음 |
| 8 | 0.0 | 꽉 차 있음 |

Queue 여유도는 후보 next-hop의 안정성 feature 중 하나로 들어간다. 즉 정책은 "목적지에 가까운 노드"뿐 아니라 "너무 붐비지 않는 노드"도 고려할 수 있다.

### 2.4 Link margin 설정

Link margin은 현재 두 UAV 사이의 통신 거리 여유도다.

```text
normalized_distance_ij = distance(i, j) / communication_radius
link_margin_ij = max(0, 1 - normalized_distance_ij)
```

의미는 다음과 같다.

| Link margin | 의미 |
| ---: | --- |
| 1에 가까움 | 두 UAV가 가까워서 링크가 안정적 |
| 0에 가까움 | 통신 반경 끝에 가까워서 링크가 아슬아슬함 |
| 0 | 통신 반경 한계 수준 |

예를 들어 통신 반경이 100m일 때:

```text
거리 20m  -> link margin = 0.8
거리 80m  -> link margin = 0.2
거리 100m -> link margin = 0.0
```

Episode 결과에는 실제로 선택되어 사용된 링크들 중 가장 낮은 margin이 `minimum_link_margin`으로 기록된다. 이것은 "이번 배달 경로에서 가장 위험했던 링크가 얼마나 아슬아슬했는가"를 나타낸다.

### 2.5 Link lifetime 설정

Link lifetime은 현재 상대 위치와 상대 속도가 유지된다고 가정했을 때, 두 UAV 사이의 링크가 앞으로 몇 step 동안 유지될지를 예측한 값이다.

계산 입력은 다음과 같다.

```text
relative_position = position_j - position_i
relative_velocity = velocity_j - velocity_i
communication_radius
time_step
horizon_steps = max_episode_steps
```

직관적으로는 다음 질문에 답한다.

```text
"현재 i와 j가 연결되어 있는데, 이대로 움직이면 몇 step 뒤에 통신 반경 밖으로 나갈까?"
```

이 값은 observation에서 `max_episode_steps`로 나누어 0부터 1 사이 값으로 정규화된다.

```text
normalized_lifetime_ij = link_lifetime_ij / max_episode_steps
```

값이 클수록 앞으로 오래 유지될 링크이고, 값이 작을수록 곧 끊길 위험이 큰 링크다.

### 2.6 Candidate risk feature 구성

Phase 12의 핵심 feature 중 하나는 `candidate_risk_features`다. 이는 현재 패킷을 가진 UAV가 선택할 수 있는 각 후보 next-hop에 대해 계산된다.

구성은 다음과 같다.

```text
candidate_risk_features[j] =
[
  link_margin_j,
  normalized_link_lifetime_j,
  queue_headroom_j,
  normalized_best_onward_lifetime_j
]
```

각 항목의 의미는 다음과 같다.

| Index | 이름 | 쉬운 의미 |
| ---: | --- | --- |
| 0 | Link margin | 이 next-hop까지의 링크가 거리상 얼마나 여유 있는가 |
| 1 | Link lifetime | 이 next-hop까지의 링크가 앞으로 얼마나 오래 유지될 것 같은가 |
| 2 | Queue headroom | 이 next-hop UAV의 queue가 얼마나 여유 있는가 |
| 3 | Best onward lifetime | 이 next-hop 이후로 이어지는 가장 안정적인 다음 링크가 얼마나 오래 유지될 것 같은가 |

이 feature의 핵심은 단순히 "목적지에 가까운가"만 보는 것이 아니라 다음 질문들을 함께 보게 하는 것이다.

```text
1. 지금 선택하려는 링크가 통신 반경 끝에 걸쳐 있지는 않은가?
2. 지금은 연결되어 있지만 곧 끊기지는 않는가?
3. 다음 UAV가 너무 혼잡하지는 않은가?
4. 다음 UAV에 도착한 뒤에도 계속 갈 길이 안정적인가?
```

## 3. Scenario 구성

Phase 12 평가 scenario는 총 14개다. Scenario는 "패킷 전달을 시험하는 환경 조건"이다. 각 scenario는 UAV 수, 이동 속도, 통신 반경, 링크 손실률, topology 고정 여부 등을 다르게 설정한다.

### 3.1 Scenario 전체 표

| Scenario | 쉬운 한글 이름 | UAV 수 | 출발지/목적지 | 핵심 환경 |
| --- | --- | ---: | --- | --- |
| `heldout_medium` | 기본 중간 난이도 테스트 | 8 | 랜덤 | 학습과 비슷하지만 평가용으로 따로 둔 일반 환경 |
| `ood_link_loss` | 링크가 자주 끊기는 환경 | 8 | 랜덤 | 링크 손실률 15% |
| `ood_fast_mobility` | 드론이 빠르게 움직이는 환경 | 8 | 랜덤 | 이동 속도 증가 |
| `ood_sparse` | 드론 사이가 듬성듬성한 환경 | 8 | 랜덤 | 통신 반경 감소, 이웃 수 감소 |
| `ood_nodes_10` | 드론 수 10개 확장 환경 | 10 | 랜덤 | 노드 수 증가 |
| `unconditional_sparse` | 연결 보장 없는 sparse 환경 | 8 | 랜덤 | 시작 연결성 보장 없음 |
| `structural_hole_45` | 구조적 막다른 길 45도 회전 | 8 | 0 -> 5 고정 | 가까워 보이는 길이 막다른 길 |
| `structural_hole_225_link_loss` | 구조적 막다른 길 + 링크 손실 | 8 | 0 -> 5 고정 | 225도 회전 + 링크 손실 10% |
| `predictive_break_45` | 곧 끊길 링크 예측 환경 | 9 | 0 -> 8 고정 | 현재 링크보다 미래 안정성이 중요 |
| `predictive_break_225_link_loss` | 곧 끊길 링크 + 링크 손실 | 9 | 0 -> 8 고정 | 225도 회전 + 링크 손실 10% |
| `ood_link_loss_30` | 심한 링크 손실 환경 | 8 | 랜덤 | 링크 손실률 30% |
| `ood_extreme_mobility` | 극단적으로 빠른 이동 환경 | 8 | 랜덤 | 이동 속도 매우 큼 |
| `ood_nodes_16` | 드론 수 16개 확장 환경 | 16 | 랜덤 | 큰 네트워크 확장성 |
| `ood_nodes_24` | 드론 수 24개 확장 환경 | 24 | 랜덤 | 가장 큰 네트워크 확장성 |

### 3.2 Scenario 그룹별 의미

#### 일반화 테스트

대상:

```text
heldout_medium
ood_link_loss
ood_fast_mobility
ood_sparse
unconditional_sparse
ood_link_loss_30
ood_extreme_mobility
```

이 그룹은 "평범한 FANET 환경이 조금씩 달라져도 정책이 잘 버티는가"를 본다.

예를 들어 `ood_fast_mobility`와 `ood_extreme_mobility`는 UAV들이 더 빠르게 움직인다. 이 경우 현재 이웃이 다음 step에도 이웃일 가능성이 낮아진다. 따라서 단순히 현재 거리만 보는 방식은 불리할 수 있다.

`ood_link_loss`와 `ood_link_loss_30`은 실제 무선 환경처럼 링크가 확률적으로 끊기는 상황을 만든다. `ood_link_loss_30`은 손실률이 30%라서 더 가혹하다.

`ood_sparse`는 통신 반경을 줄여 이웃이 적어지는 환경이다. 후보 next-hop 자체가 줄어들기 때문에 한 번 잘못 보내면 우회 경로가 사라질 수 있다.

`unconditional_sparse`는 시작 시점에 source와 destination이 반드시 연결되어 있어야 한다는 조건을 강제하지 않는다. 따라서 처음부터 배달이 불가능한 episode도 포함될 수 있고, deadline delivery가 낮게 나온다.

#### 노드 수 확장 테스트

대상:

```text
ood_nodes_10
ood_nodes_16
ood_nodes_24
```

이 그룹은 "UAV 수가 늘어나도 정책이 잘 작동하는가"를 본다. 기본 일반 환경은 8개 UAV를 중심으로 하지만, 확장성 테스트에서는 10개, 16개, 24개까지 늘린다.

중요한 점은 모델의 `max_nodes`가 32로 설정되어 있어 최대 32개 슬롯까지 처리 가능하다는 점이다. 실제 환경의 UAV 수는 `num_nodes`이고, 모델 입력/행동 공간의 최대 크기는 `max_nodes`다.

#### 구조적 막다른 길 테스트

대상:

```text
structural_hole_45
structural_hole_225_link_loss
```

이 그룹은 "목적지에 가까워 보이는 길이 사실은 막다른 길인 상황"이다.

GPSR처럼 목적지에 가장 가까운 이웃만 선택하는 방식은 다음과 같이 실패할 수 있다.

```text
S -> A1 -> A2

A2는 목적지에 가까워 보이지만, 그 다음으로 갈 수 있는 좋은 이웃이 없음
결과: routing loop 또는 drop
```

좋은 routing 방법은 처음에는 조금 돌아가는 것처럼 보여도 실제로 목적지까지 이어지는 길을 선택해야 한다.

```text
S -> B1 -> B2 -> B3 -> D
결과: 성공
```

`45`와 `225`는 topology 전체를 회전한 각도다. 학습 때 본 방향을 외운 것이 아니라 회전된 새 방향에서도 구조적 함정을 이해하는지 보기 위한 설정이다. `225_link_loss`는 여기에 링크 손실 10%까지 추가한 더 어려운 버전이다.

#### 미래 링크 단절 예측 테스트

대상:

```text
predictive_break_45
predictive_break_225_link_loss
```

이 그룹은 "지금은 연결되어 있지만 곧 끊길 링크"를 피할 수 있는지 본다.

FANET에서는 UAV가 움직이기 때문에 현재 연결된 링크도 몇 step 뒤에는 통신 반경 밖으로 나갈 수 있다.

```text
현재:    A ---- B   연결됨
잠시 후: A        B 연결 끊김
```

따라서 이 scenario에서는 단순히 현재 거리나 현재 연결 여부만 보면 부족하다. Link lifetime과 onward lifetime을 보고 "곧 끊길 링크인지"를 미리 판단해야 한다.

`predictive_break`에서는 위치뿐 아니라 velocity도 함께 회전된다. 그래야 topology의 모양만 회전한 것이 아니라, "곧 끊기는 링크"라는 동적 문제 구조가 유지된다.

## 4. Method 구성

Phase 12에서 직접 비교한 method는 6개다.

```text
1. GPSR
2. Predictive Geographic
3. Phase 8 Geo-Residual KD
4. Lite-GLOBE-P predictive-prior only
5. Lite-GLOBE-P no-switch
6. Risk-Switch Lite-GLOBE-P
```

아래 수식에서 현재 패킷을 가진 UAV를 `i`, 후보 next-hop을 `j`, 목적지를 `d`라고 둔다.

### 4.1 GPSR

GPSR은 가장 단순한 geographic routing이다. 현재 이웃 중 목적지에 가장 가까운 노드를 선택한다.

수식적으로는 다음과 같다.

```text
j* = argmin_j || position_j - position_d ||
```

또는 현재 위치에서 목적지까지의 거리 감소량을 최대화한다고 볼 수 있다.

```text
progress_j = ||position_i - position_d|| - ||position_j - position_d||
j* = argmax_j progress_j
```

강한 환경:

- 일반적으로 topology가 단순하고 목적지 방향으로 계속 가까워지는 경로가 있을 때
- 입력 정보량이 가장 작아야 할 때
- 링크 품질이나 queue 정보를 쓰지 않아도 되는 단순 환경

약한 환경:

- 구조적 막다른 길
- 곧 끊길 링크가 있는 predictive-break 환경
- 목적지에는 가까워지지만 이후 경로가 사라지는 환경

실험 결과:

- `structural_hole_45`에서 PDR 0.000
- `predictive_break_45`에서 PDR 0.000
- 전체 평균 PDR은 0.683으로 낮음

해석:

GPSR은 단순하고 비용이 낮지만, FANET의 어려운 부분인 "막다른 길"과 "곧 끊길 링크"를 처리하지 못한다.

### 4.2 Predictive Geographic

Predictive Geographic은 GPSR에 link quality와 forwardability를 더한 비학습 baseline이다.

점수 구조는 다음과 같다.

```text
score_j =
  w_g * geographic_progress_j
  + w_f * forwardability_j
  + w_r * risk_features_j

j* = argmax_j score_j
```

여기서 `risk_features_j`는 다음 네 가지다.

```text
[
  link margin,
  link lifetime,
  queue headroom,
  best onward lifetime
]
```

강한 환경:

- 곧 끊길 링크를 피해야 하는 predictive-break 환경
- 구조적 막다른 길처럼 onward 연결성을 봐야 하는 환경
- 학습 모델 없이도 deployable feature만으로 안정성을 반영해야 하는 경우

약한 환경:

- 일반 mobility 환경에서 risk feature 가중치가 과하게 작동할 수 있음
- 입력 정보량이 GPSR보다 큼
- 학습 기반 residual correction이 없어서 복잡한 일반화에서는 한계가 있음

실험 결과:

- `predictive_break_45`에서 PDR 1.000
- `predictive_break_225_link_loss`에서 PDR 0.512
- 전체 평균 PDR 0.892
- 전체 평균 input bytes 5,531으로 GPSR보다 큼

해석:

Predictive Geographic은 미래 링크 안정성을 잘 보는 장점이 있지만, 전체적으로는 Risk-Switch보다 입력 정보량이 크고 PDR이 조금 낮다.

### 4.3 Phase 8 Geo-Residual KD

Phase 8은 geographic progress를 기본 prior로 두고, 학습된 residual이 그 결정을 보정하는 방식이다. KD는 knowledge distillation을 의미한다.

간단히 표현하면 다음과 같다.

```text
score_j =
  alpha * geographic_progress_j
  + residual_theta(local_observation, j)
  + small_forwardability_bonus_j
```

강한 환경:

- 일반 OOD 환경
- node 수 확장 환경
- 구조적 막다른 길
- 빠르고 가벼운 local policy가 필요한 상황

약한 환경:

- 곧 끊길 링크를 예측해야 하는 predictive-break 환경
- link lifetime feature를 명시적으로 강하게 쓰지 않으면 현재 좋은 링크와 미래 안정 링크를 구분하기 어려움

실험 결과:

- 일반 낯선 조건 평균 PDR 0.948로 매우 강함
- node 수 확장 평균 PDR 0.967로 가장 강함
- `structural_hole_45`에서 PDR 1.000
- `predictive_break_45`에서 PDR 0.000
- `predictive_break_225_link_loss`에서 PDR 0.064

해석:

Phase 8은 일반 routing과 구조적 막다른 길에는 강하지만, 드론 이동으로 인해 곧 끊길 링크를 피하는 능력이 부족하다. Phase 12의 핵심 목표는 이 약점을 보완하는 것이다.

### 4.4 Lite-GLOBE-P predictive-prior only

이 방법은 Lite-GLOBE-P에서 학습 residual을 끄고, predictive prior만 사용하는 버전이다.

점수 구조는 다음과 같다.

```text
score_j =
  alpha * geographic_progress_j
  + beta * forwardability_j
  + gamma * predictive_features_j
  - lambda * gate_violation_j
```

여기서 gate violation은 link margin, link lifetime, onward lifetime이 기준보다 낮을 때 penalty를 주는 항이다.

```text
danger_j =
  [tau_m - margin_j]_+
  + [tau_l - lifetime_j]_+
  + [tau_o - onward_lifetime_j]_+
```

강한 환경:

- predictive-break
- link lifetime이 중요한 환경
- 학습 residual 없이도 규칙적으로 안정 경로를 고르고 싶은 경우

약한 환경:

- 일반 OOD에서 Phase 8만큼 깔끔하지 않을 수 있음
- predictive feature를 항상 쓰므로 입력 정보량이 큼

실험 결과:

- 전체 평균 PDR 0.905로 Risk-Switch와 같은 최고 수준
- `predictive_break_45`에서 PDR 1.000
- `predictive_break_225_link_loss`에서 PDR 0.512
- 전체 평균 input bytes 5,424로 Risk-Switch보다 큼

해석:

Predictive-prior only는 매우 강하지만 항상 predictive feature를 쓰기 때문에 비용이 크다. Risk-Switch는 이 장점을 필요할 때만 쓰는 방향이다.

### 4.5 Lite-GLOBE-P no-switch

Lite-GLOBE-P no-switch는 predictive prior와 학습 residual을 결합한 단일 정책이다. 즉 항상 같은 policy 안에서 geographic, predictive, residual 정보를 함께 사용한다.

점수 구조는 다음과 같이 볼 수 있다.

```text
score_j =
  geographic_prior_j
  + predictive_bonus_j
  - break_penalty_j
  + bounded_residual_theta(j)
```

강한 환경:

- predictive feature와 학습 residual을 함께 활용해야 하는 환경
- `predictive_break_45`에서 일부 성공

약한 환경:

- residual이 predictive warning을 덮어쓸 수 있음
- 위험할 때만 분기하는 구조가 아니어서 입력 비용이 큼
- seed별 편차가 커질 수 있음

실험 결과:

- 전체 평균 PDR 0.890
- `predictive_break_45`에서 PDR 0.800 ± 0.555로 seed 편차 큼
- 전체 평균 input bytes 5,775로 Phase 12 내부 method 중 높은 편

해석:

No-switch는 predictive 정보를 쓰지만, 항상 하나의 결합 정책으로 판단하기 때문에 일반 branch와 predictive branch의 장점을 분리하지 못한다. Risk-Switch는 이 문제를 해결하기 위해 hard switch를 사용한다.

### 4.6 Risk-Switch Lite-GLOBE-P

Risk-Switch Lite-GLOBE-P는 Phase 12의 주요 제안 기법이다. 핵심 아이디어는 다음과 같다.

```text
평소에는 Phase 8의 강한 일반 routing branch를 사용
Phase 8이 고른 next-hop이 위험해 보일 때만 predictive branch로 switch
```

Switch 여부는 Phase 8이 선택한 후보의 danger score와 predictive branch의 안전성 이득을 보고 결정한다.

Danger score는 다음과 같이 계산된다.

```text
danger_j =
  [tau_m - margin_j]_+
  + [tau_l - lifetime_j]_+
  + [tau_o - onward_lifetime_j]_+
```

여기서:

```text
tau_m: link margin gate
tau_l: link lifetime gate
tau_o: onward lifetime gate
```

Switch 조건은 쉽게 말하면 다음 중 하나다.

```text
1. Phase 8이 DROP을 선택한 경우
2. Phase 8이 선택한 후보의 danger가 threshold보다 큰 경우
3. Predictive branch가 Phase 8과 다른 후보를 골랐고, 그 후보가 더 안전한 경우
```

강한 환경:

- 전체 평균 reliability가 중요한 경우
- Phase 8이 강한 일반 환경과 predictive branch가 강한 link-break 환경을 동시에 다뤄야 하는 경우
- 입력 정보량을 predictive-only보다 줄이고 싶은 경우

약한 환경:

- Energy proxy는 DRAMA나 Phase 8보다 높을 수 있음
- `predictive_break_225_link_loss`에서는 Evo-QGeo보다 낮음
- 일반 node 확장 환경에서는 Phase 8이 약간 더 강함

실험 결과:

- 전체 평균 PDR 0.905로 최고 수준
- 전체 평균 deadline delivery 0.838로 최고
- Predictive Geographic보다 input bytes 12.8% 감소
- Evo-QGeo보다 input bytes 25.3% 감소
- DRAMA보다 input bytes 22.7% 감소
- Phase 8의 predictive-break 약점을 크게 보완

해석:

Risk-Switch는 "항상 무거운 predictive 판단을 하지 않고, 위험할 때만 predictive 판단을 켜는 구조"다. 이 때문에 전체 reliability와 입력 비용 사이의 균형이 가장 좋다.

## 5. Method별 전체 결과 해석

외부 baseline까지 포함한 Phase 12 full analysis의 전체 평균은 다음과 같다.

| Method | PDR | Deadline | Delay p95 | Energy | Input bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPSR | 0.683 | 0.637 | 4.163 | 1.166 | 2,284 |
| Predictive Geographic | 0.892 | 0.823 | 4.500 | 1.809 | 5,531 |
| Evo-QGeo | 0.887 | 0.815 | 4.557 | 1.812 | 6,452 |
| IQMR Q(lambda) | 0.516 | 0.385 | 6.289 | 1.946 | 7,953 |
| DRAMA | 0.891 | 0.822 | 4.504 | 1.724 | 6,240 |
| Phase 8 Geo-Residual KD | 0.803 | 0.740 | 4.110 | 1.424 | 3,956 |
| Lite-GLOBE-P predictive-prior only | 0.905 | 0.838 | 4.334 | 1.776 | 5,424 |
| Lite-GLOBE-P no-switch | 0.890 | 0.822 | 4.300 | 1.726 | 5,775 |
| Risk-Switch Lite-GLOBE-P | 0.905 | 0.838 | 4.264 | 1.779 | 4,821 |

이 표의 핵심 해석은 다음과 같다.

첫째, Risk-Switch Lite-GLOBE-P는 전체 평균 PDR과 deadline delivery에서 최고 수준이다.

둘째, Phase 8은 delay와 input bytes 측면에서는 가볍지만 predictive-break에서 거의 실패하므로 최종 방법으로는 약점이 분명하다.

셋째, Predictive Geographic과 Lite-GLOBE-P predictive-prior only는 predictive-break에 강하지만, 항상 predictive feature를 사용하기 때문에 입력 정보량이 크다.

넷째, Risk-Switch는 predictive 계열의 안정성을 얻으면서도 input bytes를 줄였다. 이것이 Phase 12의 가장 중요한 설계적 장점이다.

다섯째, energy proxy는 아직 약점이다. 안정적인 우회 경로를 선택하면 전달 성공률은 올라가지만 더 긴 거리의 transmission을 할 수 있기 때문이다.

## 6. Scenario별 패킷 전달 방식과 실패 상황

### 6.1 일반 환경: `heldout_medium`

이 환경은 Phase 12의 기본 시험지에 가깝다. UAV 8개, 보통 이동성, 링크 손실 없음, 출발지/목적지 랜덤이다.

패킷 전달 방식:

```text
source에서 시작
-> 현재 이웃 중 next-hop 선택
-> 목적지에 가까워지거나 안정적인 이웃으로 forwarding
-> destination에 도착하면 성공
```

결과:

- GPSR PDR 0.980
- Phase 8 PDR 0.984
- Risk-Switch PDR 0.985

해석:

대부분의 method가 잘한다. 이 환경에서는 구조적 함정이나 심한 링크 불안정이 없기 때문에 GPSR도 강하다. Risk-Switch는 불필요한 switch를 크게 늘리지 않고 기본 성능을 유지했다.

### 6.2 링크 손실 환경: `ood_link_loss`, `ood_link_loss_30`

이 환경은 실제 무선 FANET처럼 링크가 확률적으로 끊기는 조건이다.

패킷 전달에서 어려운 점:

```text
현재 거리상 연결되어 있어도 stochastic loss 때문에 링크가 사라질 수 있음
한 번 끊기면 invalid action 또는 우회 실패가 발생할 수 있음
```

결과:

- `ood_link_loss`: Risk-Switch PDR 0.911
- `ood_link_loss_30`: Risk-Switch PDR 0.778

해석:

손실률 15%에서는 모든 주요 방법의 PDR이 0.91 전후로 비슷하다. 손실률 30%에서는 전체적으로 성능이 떨어진다. 이 경우 link lifetime만으로는 stochastic loss를 완전히 설명할 수 없으므로, link-loss-aware danger score가 추가 개선점이 된다.

### 6.3 빠른 이동성 환경: `ood_fast_mobility`, `ood_extreme_mobility`

이 환경은 UAV들이 더 빠르게 움직인다.

패킷 전달에서 어려운 점:

```text
현재 이웃이 다음 step에도 이웃이라는 보장이 약해짐
목적지에 가까운 링크라도 곧 끊길 수 있음
```

결과:

- `ood_fast_mobility`: Risk-Switch PDR 0.966
- `ood_extreme_mobility`: Risk-Switch PDR 0.967

해석:

Risk-Switch는 빠른 이동성에서도 안정적이다. 다만 GPSR도 이 두 scenario에서는 높은 PDR을 보인다. 이는 해당 random mobility 환경에서는 구조적 함정보다 단순 경로가 자주 유효했기 때문으로 해석할 수 있다. 그러나 predictive-break처럼 의도적으로 곧 끊길 링크가 배치된 경우에는 GPSR이 크게 실패한다.

### 6.4 Sparse 환경: `ood_sparse`, `unconditional_sparse`

Sparse 환경은 통신 반경이 줄어 이웃 수가 적어진 조건이다.

패킷 전달에서 어려운 점:

```text
선택 가능한 next-hop이 적음
한 번 잘못 보내면 우회할 후보가 부족함
unconditional_sparse에서는 애초에 source-destination 연결이 없을 수도 있음
```

결과:

- `ood_sparse`: Risk-Switch PDR 0.980
- `unconditional_sparse`: Risk-Switch connected PDR 0.987, deadline delivery 0.380

해석:

`ood_sparse`는 시작 연결성을 보장하므로 대부분 method가 잘한다. `unconditional_sparse`는 연결 불가능한 episode까지 섞여 deadline delivery가 낮다. 이 scenario는 "정책이 못해서 실패한 경우"와 "환경적으로 처음부터 불가능한 경우"를 구분해서 설명해야 한다.

### 6.5 Node 수 확장 환경: `ood_nodes_10`, `ood_nodes_16`, `ood_nodes_24`

이 환경은 UAV 수가 늘어나는 확장성 테스트다.

패킷 전달에서 어려운 점:

```text
후보 next-hop 수 증가
경로가 길어질 수 있음
더 많은 노드 중 어떤 후보를 선택해야 하는지 판단 복잡도 증가
```

결과:

- `ood_nodes_10`: Risk-Switch PDR 0.980
- `ood_nodes_16`: Risk-Switch PDR 0.972
- `ood_nodes_24`: Risk-Switch PDR 0.938

해석:

노드 수가 증가해도 Risk-Switch는 높은 PDR을 유지한다. 다만 이 그룹에서는 Phase 8이 평균적으로 약간 더 강하다. 즉 Phase 8의 일반 routing branch가 node 확장에서는 매우 잘 동작하며, Risk-Switch는 predictive 안정성을 얻는 대신 일부 일반 확장성에서 아주 작은 손실이 있다.

### 6.6 구조적 막다른 길: `structural_hole_45`, `structural_hole_225_link_loss`

이 환경은 가까워 보이는 길이 막다른 길인 고정 topology다.

패킷 전달에서 어려운 점:

```text
목적지에 가장 가까운 이웃을 고르면 막다른 곳으로 감
처음에는 돌아가는 것처럼 보이는 detour를 선택해야 성공
```

실패 예시:

```text
GPSR:
현재 목적지에 가장 가까운 후보 선택
-> 막다른 노드 도착
-> 더 갈 수 있는 이웃 없음 또는 loop
-> 실패
```

성공 예시:

```text
Risk-Switch / Phase 8 / Predictive Geographic:
forwardability와 learned residual 또는 onward 안정성 고려
-> 막다른 길 대신 우회 경로 선택
-> destination 도착
```

결과:

- `structural_hole_45`: GPSR PDR 0.000, Risk-Switch PDR 1.000
- `structural_hole_225_link_loss`: GPSR PDR 0.079, Risk-Switch PDR 0.698

해석:

이 scenario는 단순 거리 기반 routing의 한계를 명확히 보여준다. Risk-Switch는 Phase 8이 이미 학습한 구조적 우회 능력을 유지한다.

### 6.7 미래 링크 단절: `predictive_break_45`, `predictive_break_225_link_loss`

이 환경은 현재는 연결되어 있지만 곧 끊길 링크가 있는 고정 topology다.

패킷 전달에서 어려운 점:

```text
현재 가장 좋아 보이는 링크가 몇 step 뒤 끊김
단순 거리나 현재 연결 여부만 보면 실패
link lifetime과 onward lifetime을 봐야 함
```

실패 예시:

```text
GPSR 또는 Phase 8:
현재 목적지에 가까운 후보 선택
-> 다음 step 이후 링크가 끊김
-> 더 갈 수 없음 또는 loop/drop
-> 실패
```

성공 예시:

```text
Risk-Switch:
Phase 8 후보의 link lifetime 또는 onward lifetime이 낮음을 감지
-> predictive branch로 switch
-> 조금 돌아가더라도 더 오래 유지되는 링크 선택
-> destination 도착
```

결과:

- `predictive_break_45`: Phase 8 PDR 0.000, Risk-Switch PDR 1.000
- `predictive_break_225_link_loss`: Phase 8 PDR 0.064, Risk-Switch PDR 0.512

해석:

이 scenario가 Phase 12의 핵심 성공 사례다. Risk-Switch는 Phase 8의 결정적 약점이었던 "곧 끊길 링크 문제"를 크게 보완했다. 다만 `predictive_break_225_link_loss`에서는 Evo-QGeo가 더 높은 PDR을 보였으므로, 링크 손실이 결합된 predictive-break는 향후 개선 대상이다.

## 7. Episode 실패 조건

Episode는 다음 경우 실패로 기록된다.

| 실패 조건 | 의미 |
| --- | --- |
| `invalid_action` | 선택한 next-hop이 현재 통신 가능한 이웃이 아님 |
| `routing_loop` | 이미 방문한 노드로 다시 가서 loop 발생 |
| `ttl_expired` | 패킷 TTL을 모두 사용 |
| `time_limit` | 최대 episode step을 초과 |
| `agent_drop` | 정책이 명시적으로 DROP action 선택 |

성공은 `packet.current == destination`이 되는 순간이다.

Deadline delivery는 단순 성공보다 더 엄격하다. 초기 shortest path 길이를 기준으로 deadline step을 계산하고, 그 안에 도착해야 deadline을 만족한다.

```text
deadline_steps = ceil(1.5 * initial_shortest_hops) + 1
```

따라서 PDR은 높지만 deadline delivery가 낮으면 "도착은 했지만 너무 늦게 도착한 episode가 많다"는 의미다.

## 8. 교수님/박사님 발표용 설명 흐름

### 8.1 첫 번째 슬라이드: 실험 질문

발표에서는 먼저 다음 질문을 던지는 것이 좋다.

```text
FANET 라우팅에서 단순히 목적지에 가까운 UAV를 고르면 충분한가?
```

답은 아니다. 이유는 두 가지다.

```text
1. 가까워 보이는 길이 막다른 길일 수 있다.
2. 지금 연결된 링크가 곧 끊길 수 있다.
```

Phase 12는 이 두 문제를 동시에 다루기 위한 실험이다.

### 8.2 두 번째 슬라이드: Episode 구조

한 episode는 패킷 하나를 보내는 한 번의 시도라고 설명한다.

```text
출발지 UAV -> next-hop 선택 반복 -> 목적지 UAV 도착
```

각 step에서 정책은 주변 이웃 후보를 보고 다음 UAV를 고른다. 이때 후보별로 거리, queue, link margin, link lifetime, onward lifetime 등이 관측된다.

### 8.3 세 번째 슬라이드: Scenario 구성

Scenario는 14개이고, 네 그룹으로 묶어 설명하면 쉽다.

```text
1. 일반 OOD 환경: 링크 손실, 빠른 이동, sparse 조건
2. 노드 수 확장: 10, 16, 24 UAV
3. 구조적 막다른 길: structural_hole
4. 미래 링크 단절: predictive_break
```

핵심은 단순 평균 환경만 본 것이 아니라, FANET 라우팅의 어려운 케이스를 의도적으로 넣었다는 점이다.

### 8.4 네 번째 슬라이드: Method 구성

Method는 다음 흐름으로 설명하면 논리적이다.

```text
GPSR:
  목적지에 가장 가까운 이웃 선택

Predictive Geographic:
  거리 + link quality + onward 가능성 고려

Phase 8 Geo-Residual KD:
  GPSR prior + 학습 residual로 일반 routing 강화

Lite-GLOBE-P predictive-prior only:
  predictive feature를 명시적으로 강하게 사용

Lite-GLOBE-P no-switch:
  predictive prior와 residual을 항상 결합

Risk-Switch Lite-GLOBE-P:
  평소에는 Phase 8, 위험할 때만 predictive branch 사용
```

이 흐름은 "단순 거리 기반 -> 안정성 feature 추가 -> 학습 기반 보정 -> 위험 시 branch 전환"으로 자연스럽게 이어진다.

### 8.5 다섯 번째 슬라이드: Risk-Switch 핵심 아이디어

Risk-Switch는 다음 한 문장으로 설명할 수 있다.

```text
일반 상황에서는 Phase 8의 강한 local routing을 유지하고,
선택된 링크가 위험할 때만 predictive link-stability 판단으로 전환한다.
```

수식은 너무 복잡하게 보여주기보다 danger score만 제시하는 것이 좋다.

```text
danger =
  부족한 link margin
  + 부족한 link lifetime
  + 부족한 onward lifetime
```

정확한 형태는 다음과 같다.

```text
danger_j =
  [tau_m - margin_j]_+
  + [tau_l - lifetime_j]_+
  + [tau_o - onward_lifetime_j]_+
```

### 8.6 여섯 번째 슬라이드: 핵심 결과

전체 결과는 다음 메시지로 정리한다.

```text
Risk-Switch Lite-GLOBE-P는 전체 평균 PDR과 deadline delivery에서 최고 수준이며,
Predictive Geographic, Evo-QGeo, DRAMA보다 적은 입력 정보량으로 유사하거나 더 높은 신뢰성을 달성했다.
```

숫자는 다음 세 개만 강조해도 충분하다.

```text
전체 평균 PDR: 0.905
전체 평균 deadline delivery: 0.838
전체 평균 input bytes: 4,821
```

비교 포인트:

```text
GPSR 대비 PDR +32.5%
Predictive Geographic 대비 input bytes -12.8%
Evo-QGeo 대비 input bytes -25.3%
DRAMA 대비 input bytes -22.7%
```

### 8.7 일곱 번째 슬라이드: 한계와 향후 개선

한계도 명확히 말하는 것이 좋다.

```text
1. Energy proxy는 DRAMA보다 높다.
2. Node 확장 일반 환경에서는 Phase 8이 약간 더 강하다.
3. predictive_break_225_link_loss에서는 Evo-QGeo가 더 높은 PDR을 보인다.
```

향후 개선 방향:

```text
1. link-loss-aware danger score 추가
2. energy-aware switch penalty 추가
3. top-k onward lifetime 기반 안정성 평가 강화
4. 위험할 때만 full risk feature를 계산해 input bytes 추가 절감
```

## 9. 최종 발표용 핵심 문장

Phase 12의 핵심은 다음 문장으로 정리할 수 있다.

```text
Phase 12 Risk-Switch Lite-GLOBE-P는 FANET 라우팅에서 일반적인 local geographic routing의 장점은 유지하면서,
구조적 막다른 길과 곧 끊길 링크 같은 위험 상황에서는 link margin, link lifetime, queue headroom, onward lifetime을 이용해
더 안정적인 next-hop으로 전환하는 방법이다.
```

더 짧게 말하면 다음과 같다.

```text
평소에는 가볍고 빠른 Phase 8 방식으로 보내고,
위험한 링크가 감지될 때만 predictive 안정성 판단으로 바꾸는 routing 방법이다.
```

결과적으로 Phase 12는 다음을 보여준다.

```text
1. 단순히 목적지에 가까운 노드를 고르는 GPSR은 FANET의 어려운 상황에서 실패한다.
2. Link margin과 link lifetime 같은 안정성 feature는 predictive-break 문제 해결에 중요하다.
3. Predictive feature를 항상 쓰면 강하지만 비용이 크다.
4. Risk-Switch는 위험할 때만 predictive branch를 사용해 reliability와 input cost의 균형을 맞춘다.
```

