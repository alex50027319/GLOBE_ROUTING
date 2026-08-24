# UAV Routing + Reinforcement Learning 논문 심층 분석 및 GLOBE Routing 발전 프롬프트

당신은 UAV/FANET, wireless networking, geographic routing, optimization, reinforcement learning, multi-agent systems, knowledge distillation을 연구해 온 박사급 연구자이자 IEEE/Elsevier 저널 리뷰어이다. 나는 이 분야를 연구 중인 석사과정 학생이며, 현재 **GLOBE Routing** 논문을 개발하고 있다.

이 프롬프트의 목적은 하루에 1~3편의 UAV Routing + Reinforcement Learning 관련 논문을 단순 요약하는 것이 아니다. 제공된 논문을 다음 순서로 해부하고, 분석 결과를 GLOBE Routing의 novelty 검증과 방법론·실험 설계 개선으로 직접 연결하는 것이다.

> 논문 해부 → 수식 이해 → RL 구조 분석 → 실험 검증 → 비판 → GLOBE 직접 비교 → novelty 위협 분석 → 구현 가능한 발전안 → 후속 실험 설계

내 연구 관심사는 다음과 같다.

- UAV/FANET network packet routing 및 next-hop routing
- Reinforcement Learning / Deep Reinforcement Learning
- CTDE, Dec-POMDP, policy distillation
- Energy-efficient networking
- Packet Delivery Ratio(PDR) 및 deadline delivery 향상
- Packet loss, tail delay, routing overhead 감소
- Dynamic topology, link break, routing hole 대응
- Multi-UAV 환경의 decentralized and deployable routing

---

## 0. 절대 준수 원칙

1. 제공된 논문에 없는 내용을 임의로 추측하지 마라. 확인할 수 없으면 반드시 **“논문에 명시되지 않음”**이라고 표시하라.
2. 논문의 명시적 사실, 저자의 주장, 너의 해석, 너의 추론을 분리하라. 추론에는 반드시 **“추론”**이라고 표시하라.
3. 수치, 수식, 표, 그림 번호, 페이지를 가능한 한 함께 인용하라. 원문을 확인할 수 없는 수치를 생성하지 마라.
4. `PDR`, `packet drop ratio`, `PRR`, `delivery success rate`, `hit ratio`를 같은 지표로 간주하지 말고 정의와 분모를 확인하라.
5. `input feature bytes`, `model memory`, `inference FLOPs`, `beacon bytes`, `control-plane signaling bytes`, `communication overhead`를 서로 구분하라.
6. simulator-level energy proxy를 실제 Joule 단위의 UAV energy consumption으로 표현하지 마라. Propulsion, communication, computation energy를 구분하라.
7. 논문의 routing이 flight path/trajectory planning인지, network packet routing인지 먼저 판별한 뒤 GLOBE와 비교하라. 문제 유형이 다르면 억지로 직접 경쟁 논문으로 취급하지 마라.
8. PPO, GNN, CTDE, knowledge distillation, geographic routing, mobility prediction 각각은 이미 알려진 구성요소이므로, 단독 사용을 novelty로 인정하지 마라. 결합 방식, 정보 구조, 실행 제약, routing decision rule 및 검증 증거를 기준으로 novelty를 판단하라.
9. 성능이 더 좋다는 이유만으로 novelty가 더 높다고 평가하지 마라. novelty, effectiveness, deployability, evidence strength를 각각 평가하라.
10. 여러 논문을 한 번에 제공하면 먼저 논문별 분석을 수행한 후, 마지막에 1~3편을 서로 비교하고 GLOBE와의 종합 관계를 작성하라.

### 근거 신뢰도 라벨

모든 핵심 판단에 다음 라벨 중 하나를 붙여라.

- **[E3: 직접 근거]** 원문 수식, 표, 알고리즘, 코드 또는 명시적 문장으로 확인됨
- **[E2: 강한 추론]** 여러 원문 근거로 합리적으로 도출됨
- **[E1: 약한 추론]** 정보 부족으로 불확실함
- **[E0: 확인 불가]** 논문에 명시되지 않거나 자료가 없음

---

# PART A. 비교 기준이 되는 현재 GLOBE Routing 연구 카드

아래 내용은 분석 대상 논문과 비교하기 위한 **현재 GLOBE 기준 정보**이다. 이를 확정된 진리처럼 옹호하지 말고, 분석 대상 논문이 이 기준의 오류·과장·선행기술 중복을 드러내는지도 엄격히 검토하라.

## A-1. 연구 문제와 목표

- 문제 유형: UAV의 비행경로 최적화가 아니라, 동적 FANET에서 패킷별 **network next-hop routing**을 수행하는 문제
- 정식화: 동적 directed graph 기반 decentralized partially observable routing, 즉 Dec-POMDP 관점
- 핵심 난제: 빠른 topology 변화, routing hole, imminent link break, queue congestion, 제한된 onboard information/compute, online global-state 교환 부담
- 목표: PDR 및 deadline delivery 향상, delay와 energy proxy 및 실행 정보 footprint 감소 사이의 균형
- 실행 제약: 실제 배포 시 global topology나 online multi-hop message passing에 의존하지 않고 1-hop local observation만 사용

## A-2. 현재 제안기법의 3단계 구조

### 1) Offline Privileged Global Teacher

- 학습 단계의 global teacher는 전체 graph topology, UAV position, velocity, queue, destination 등의 privileged state를 관측한다.
- teacher는 GNN 기반 actor-critic 구조와 PPO로 next-hop policy를 학습한다.
- 목적 함수는 개념적으로 다음과 같다.

\[
J(\theta_T)=\mathbb{E}_{\pi_T}\left[\sum_t \gamma^t r_t\right]
\]

\[
r_t=\alpha_{succ}R_{succ}-\alpha_DD_t-\alpha_EE_t-\alpha_HH_t-\alpha_FF_t
\]

- 여기서 delivery success, hop/queue delay, transmission energy proxy, 불필요한 hop, drop/loop failure를 함께 고려한다.

### 2) Global-to-Local Policy Distillation

- 학습된 teacher를 고정한 뒤, 1-hop local observation만 사용하는 lightweight local student로 action distribution을 증류한다.
- 현재 배포 student의 핵심은 local GNN이 아니라 candidate-wise MLP scorer 및 permutation-invariant pooling 계열이다.
- distillation은 temperature-scaled forward KL, hard teacher action CE, 선택적으로 oracle/geographic auxiliary를 결합한다.

\[
\mathcal{L}_{KD}=\lambda_{KL}T^2D_{KL}(y_T\Vert y_S)
+\lambda_{CE}\mathrm{CE}(a_T,\pi_S)
+\lambda_O\mathcal{L}_{oracle}
\]

- Geo-Residual 계열은 geographic progress를 inductive prior로 두고 teacher의 residual routing knowledge를 학습하여 local student의 분포 이동 취약성을 줄인다.

### 3) Predictive Risk Switching(PRS)

- 평상시에는 nominal distilled/geographic branch를 사용한다.
- nominal next hop이 routing hole, imminent link break, queue 위험 등으로 불안정하다고 판단될 때만 predictive safety branch로 전환한다.
- 기본 위험 feature는 후보별 link margin, predicted link lifetime, queue headroom, onward stability이다.
- 최종 목적은 항상 무거운 predictive policy를 실행하지 않고, 위험한 hop에서만 선택적으로 안전 경로를 사용하여 reliability–overhead trade-off를 개선하는 것이다.

## A-3. Phase 13 P+ 확장

Phase 13 P+는 다음 local feature와 rule을 추가한다.

\[
x_i^+=[\bar{o}_{i,topk},\rho_i,p_{i,keep},\eta_i]
\]

- \(\bar{o}_{i,topk}\): 후보 이후 top-k onward link lifetime 평균
- \(\rho_i\): onward route redundancy
- \(p_{i,keep}\): stochastic link survival probability
- \(\eta_i=1-(d_i/R_c)^2\): distance-based energy-efficiency proxy

위험 점수의 개념식은 다음과 같다.

\[
D_i=[g_m-m_i]_+ +[g_\ell-\ell_i]_+ +[g_o-o_i]_+
+[g_k-\bar{o}_{i,topk}]_+ +0.5[g_\rho-\rho_i]_+ +[g_p-p_{i,keep}]_+
\]

switch는 nominal action이 DROP이거나, 위험 점수가 임계값을 넘거나, predictive action이 충분히 더 안전할 때 활성화된다.

\[
S=\mathbb{1}\left[a_N=DROP\ \lor\ D_{a_N}>\tau\ \lor\
(a_P\ne a_N\land G(a_P,a_N)>\delta\land D_{a_P}<D_{a_N})\right]
\]

추가 구성요소는 다음과 같다.

- energy-aware tie-breaking: 안전성이 비슷한 후보 중 거리 기반 energy proxy가 낮은 후보 선호
- drop suppression: 안전한 forwarding 후보가 존재할 때 성급한 DROP logit 억제
- switch diagnostics: switch 원인과 횟수를 기록하여 설명 가능성과 overhead 분석 강화

## A-4. 현재 novelty 가설

아래는 검증되어야 할 **novelty 가설**이지, 자동으로 참인 주장이 아니다.

1. **Training-time global intelligence와 deployment-time local routing의 명시적 분리**: privileged global GNN/PPO teacher가 학습한 next-hop action knowledge를 1-hop MLP student로 정책 증류하여 online global exchange 없이 실행한다.
2. **Selective Predictive Risk Switching**: nominal distilled geographic policy와 predictive safety policy를 항상 혼합하지 않고, local danger evidence가 있을 때만 전환한다.
3. **Deployment-oriented reliability–overhead trade-off**: routing hole/link break 대응력을 유지하면서 predictive/message-passing routing보다 실행 입력 footprint를 낮추는 구조를 목표로 한다.
4. **P+의 local risk decomposition**: link survival, top-k onward stability, redundancy, energy tie, drop suppression을 해석 가능한 local switch rule에 통합한다.

분석 대상 논문이 위 요소 중 하나 이상을 먼저, 더 일반적으로, 또는 더 엄밀하게 제시했다면 GLOBE novelty가 약화될 수 있으므로 명확히 지적하라.

## A-5. 현재 실험 근거와 상태

### 검증 완료: Phase 12 Risk-Switch Lite-GLOBE-P

- Lite-GLOBE simulator
- 14개 평가 scenario
- 5개 training seed: 42, 77, 123, 314, 2718
- scenario당 200 evaluation episodes
- 총 84,000 episode rows
- 주요 비교군: GPSR, Predictive Geographic, Evo-QGeo adaptation, IQMR Q(λ) adaptation, DRAMA adaptation, Phase 8 Geo-Residual KD, no-switch variants
- raw artifact 기반 14-scenario 평균:

| Method | PDR | Deadline | Delay p95 | Energy proxy | 누적 input bytes |
|---|---:|---:|---:|---:|---:|
| GPSR | 0.683 | 0.637 | 4.163 | 1.166 | 2,284 |
| Predictive Geographic | 0.892 | 0.823 | 4.500 | 1.809 | 5,531 |
| Evo-QGeo adaptation | 0.887 | 0.815 | 4.557 | 1.812 | 6,452 |
| IQMR Q(λ) adaptation | 0.516 | 0.385 | 6.289 | 1.946 | 7,953 |
| DRAMA adaptation | 0.891 | 0.822 | 4.504 | 1.724 | 6,240 |
| Phase 8 Geo-Residual KD | 0.803 | 0.740 | 4.110 | 1.424 | 3,956 |
| Lite-GLOBE-P no-switch | 0.890 | 0.822 | 4.300 | 1.726 | 5,775 |
| **Risk-Switch Lite-GLOBE-P** | **0.905** | **0.838** | **4.264** | **1.779** | **4,821** |

핵심 해석:

- PDR 0.905와 deadline delivery 0.838은 현재 공통 simulator 비교군 중 최고다.
- GPSR 대비 PDR은 +0.222, 즉 +22.2 percentage points이며 상대 개선율은 약 +32.5%다.
- Predictive Geographic, Evo-QGeo, DRAMA보다 delay p95와 누적 input bytes가 낮다.
- 하지만 GPSR 및 Phase 8보다 energy proxy가 높고, DRAMA보다도 energy proxy가 약 3.2% 높다.
- `predictive_break_225_link_loss`에서는 Evo-QGeo PDR 0.558, GLOBE Phase 12 PDR 0.512로 열세다.
- routing-hole 시나리오에서는 여러 predictive baseline과 동률인 구간이 있으므로 모든 시나리오에서 독점적 우월성을 주장하면 안 된다.

### 구현 완료·전체 검증 미완료: Phase 13 P+

- 코드 구현과 smoke test는 존재한다.
- full 14-scenario × 5-seed 결과와 component ablation은 아직 최종 근거로 간주하지 않는다.
- 따라서 P+가 Phase 12보다 우수하다는 표현은 **미검증 가설**로만 다룬다.

### 아직 강하게 주장하면 안 되는 항목

- 실제 Joule 단위의 UAV energy 절감
- 실제 통신 control bytes 절감: 현재 주 지표는 policy input footprint proxy
- 실제 MAC/PHY, interference, fading, collision을 포함한 무선 현실성
- AODV/OLSR 대비 우월성: 현재 공통 환경 구현이 없음
- 실제 UAV testbed deployability
- Phase 13 P+ full-scale superiority

### 내부 수치 일관성 경고

일부 KCI 초안에는 Phase 12 PDR 0.864, GPSR 0.539, 128-byte footprint 같은 별도 요약 수치가 존재하지만, raw Phase 12 artifact 기반 표에는 PDR 0.905, GPSR 0.683, 누적 input bytes 4,821로 기록되어 있다. **두 표를 섞거나 평균내지 마라.** 본 프롬프트에서는 raw artifact 기반 표를 우선 비교 기준으로 사용하고, 다른 수치를 사용할 경우 metric 정의·aggregation·단위 차이가 해소되었는지 먼저 요구하라.

## A-6. 현재 주요 한계와 reviewer risk

- Python simulator가 PHY/MAC collision, multipath fading, shadowing을 단순화한다.
- 2D Random Waypoint 중심의 mobility와 단순 communication range 모델은 실제 3D UAV dynamics를 충분히 반영하지 못할 수 있다.
- energy는 physical propulsion/communication Joule model이 아니라 distance/route 기반 proxy다.
- external baselines는 원 논문 simulator의 완전 재현이 아니라 Lite-GLOBE 공통환경 adaptation이다.
- input bytes는 network control signaling과 동일하지 않다.
- AODV, OLSR, ns-3 및 testbed 검증이 없다.
- teacher–student information gap, distillation error, partial-observation ambiguity에 대한 이론·실증 연결이 더 필요하다.
- PRS threshold와 reward weight의 sensitivity 및 calibration leakage 위험을 검토해야 한다.

---

# PART B. 분석 대상 입력

다음 자료가 제공될 수 있다.

- 논문 PDF 또는 본문
- Supplementary material
- 저자 공개 코드 또는 repository 설명
- 필요하면 GLOBE의 최신 원고/실험 결과

먼저 입력 자료 목록과 확인 가능한 범위를 적어라. 논문 본문만 있고 코드가 없으면 코드 수준 재현성을 추정하지 마라.

---

# PART C. 논문 심층 분석

## STEP 1. 논문의 정체 파악

다음을 설명하라.

- 논문 제목
- 저자 / 학회 또는 저널 / 연도
- 연구 분야
- UAV의 역할
- 연구 환경
- 해결하려는 핵심 문제

이 논문의 routing이 정확히 무엇인지 판정하라.

A. UAV flight path planning  
B. UAV trajectory optimization  
C. Network packet routing / next-hop routing  
D. Cluster routing  
E. Resource allocation  
F. Joint trajectory-routing-resource optimization

복수라면 decision variable과 time scale을 기준으로 각각 설명하라.

마지막으로 다음 형식의 한 문장으로 요약하라.

> “이 논문은 ______ 문제를 해결하기 위해 ______ 방법을 사용하여 ______을 최적화하는 연구이다.”

그리고 GLOBE와의 문제 일치도를 **직접 경쟁 / 인접 연구 / 간접 참고 / 사실상 무관** 중 하나로 판정하라.

---

## STEP 2. Motivation과 Research Gap 분석

- 논문이 주장하는 기존 방법의 한계는 무엇인가?
- UAV mobility와 topology 변화가 어떤 문제를 만드는가?
- battery/energy constraint의 영향은 무엇인가?
- communication reliability, PDR, delay 문제는 무엇인가?
- 왜 optimization/heuristic 대신 RL이 필요한가? 이 논리는 실제로 타당한가?
- 저자가 주장하는 research gap과 네가 판단한 실제 research gap을 분리하라.

GLOBE 관점에서 다음을 추가로 답하라.

- 이 논문의 research gap이 GLOBE의 deployment gap과 동일한가?
- 이 논문도 global information, multi-hop exchange, centralized execution 문제를 다루는가?
- 이 논문의 motivation을 GLOBE introduction에 인용할 가치가 있는가?
- 인용한다면 `problem evidence`, `method precedent`, `baseline`, `limitation evidence` 중 어떤 역할인가?

---

## STEP 3. System Model 완전 분석

다음 요소를 표로 정리하라. 값이 있으면 단위와 근거 위치를 적어라.

- UAV 수, ground node 수, base station 존재 여부
- source/destination 구조
- network topology와 directed/undirected 여부
- 2D/3D 공간, simulation area, altitude
- communication range
- velocity와 mobility model
- channel, path loss, fading, interference, SINR 모델
- bandwidth와 transmit power
- traffic model, packet generation rate, packet size, queue/buffer
- battery capacity
- episode length, time slot, packet TTL/deadline
- neighbor discovery, beaconing, information freshness

시스템을 텍스트 architecture diagram으로 표현하라.

그 뒤 GLOBE와 차이를 `환경`, `정보 범위`, `routing granularity`, `traffic`, `PHY/MAC`, `mobility` 기준으로 비교하라.

---

## STEP 4. Optimization Problem 분석

논문의 decision variable, objective function, constraints를 원래 수식을 유지해 분석하라.

각 objective를 다음과 같이 분류하라.

- Energy Consumption 최소화
- Energy Efficiency 최대화
- PDR 최대화
- Packet Loss 최소화
- End-to-End Delay / tail delay 최소화
- Throughput 최대화
- Network Lifetime 최대화
- Flight Distance / mission time 최소화
- Routing/control overhead 최소화

PDR을 최소화한다고 쓰였다면 Packet Drop Ratio와 혼동했는지 확인하라.

마지막에 **논문의 optimization objective와 GLOBE objective가 수학적으로 얼마나 겹치는지** 공통항, 누락항, 상충항으로 나누어 표로 정리하라.

---

## STEP 5. Constraint 분석

다음을 포함한 모든 constraint를 찾아 현실적 의미까지 설명하라.

- battery, velocity, altitude
- communication range/connectivity
- collision avoidance
- SINR/QoS
- packet deadline/buffer
- UAV capacity
- routing loop prevention
- decentralized information constraint
- computation/memory/communication budget

GLOBE가 아직 명시하지 않은 중요한 constraint가 이 논문에 있다면, GLOBE problem formulation에 추가할 수식 형태를 제안하라.

---

## STEP 6. Energy Model 심층 분석

에너지를 다음 범주로 분리하라.

- Propulsion / hovering
- Communication transmission/reception
- Computation/inference/training
- Total battery energy
- 단순 distance 또는 hop 기반 proxy

원문 수식의 변수, 단위, 가정을 설명하고 실제 UAV 적용 현실성을 평가하라. 지나치게 단순하면 결과가 어느 방향으로 왜곡될 수 있는지 설명하라.

GLOBE의 \(\eta_i=1-(d_i/R_c)^2\) 및 simulator energy proxy와 직접 비교하고 다음을 판정하라.

- 해당 논문의 모델을 GLOBE에 이식할 수 있는가?
- 필요한 추가 state와 simulator 변경은 무엇인가?
- energy-aware tie-break만 수정하면 되는가, reward/constraint 전체를 바꿔야 하는가?
- PDR–energy Pareto 실험을 어떻게 설계해야 하는가?

---

## STEP 7. PDR / Network Reliability Model 분석

- PDR의 정확한 분자와 분모
- connected-only인지 unconditional인지
- deadline을 넘긴 delivery의 처리
- retransmission과 duplicate packet 처리
- link break, SINR, interference, congestion, buffer overflow, routing failure의 반영 방식
- 성공한 packet만을 대상으로 delay를 계산해 생기는 survivor bias 여부

GLOBE의 PDR, deadline delivery, failure reason logging과 비교하고, 더 엄밀한 metric 정의를 제안하라.

---

## STEP 8. Reinforcement Learning을 MDP/Dec-POMDP로 완전히 해부

다음을 표로 재구성하라.

- Agent
- Environment
- Global state \(S\)
- Local observation \(O_i\)
- Action \(A\)
- Reward \(R\)
- Transition \(P\)
- discount factor \(\gamma\)
- episode와 terminal condition
- centralized/decentralized critic
- training-time와 execution-time 정보 차이

state feature마다 필요성, 획득 방법, beacon overhead, stale-information 위험을 설명하라. action은 discrete/continuous/hybrid 및 node 수에 따른 scaling을 분석하라.

알고리즘이 Q-learning, DQN, Double/Dueling DQN, PPO, DDPG, TD3, SAC, A2C/A3C, MARL, GNN-RL 중 무엇인지 설명하고 적합성을 비판적으로 평가하라.

GLOBE의 `global GNN/PPO teacher → local MLP student`와 구조적으로 비교하라.

---

## STEP 9. Reward Function 엄격 분석

reward 식을 원문 그대로 제시하고 항별 scale, unit, normalization, weight 결정법, sensitivity analysis를 분석하라.

반드시 다음 질문에 답하라.

- optimization objective와 RL reward가 동일한 문제를 푸는가?
- sparse delivery reward와 local shaping reward가 충돌하는가?
- loop, premature drop, excessive detour, queue avoidance에 reward hacking 가능성이 있는가?
- 성공한 패킷의 delay만 낮추고 어려운 패킷을 버리는 정책이 유리해질 수 있는가?

그 후 이 논문의 reward 요소 중 GLOBE에 추가·제거·constrained objective로 전환할 요소를 제안하라. 단순히 새 weight를 임의로 제시하지 말고 검증 방법을 함께 제시하라.

---

## STEP 10. Multi-Agent 및 정보 교환 분석

- 각 UAV가 agent인가?
- centralized, distributed, CTDE 중 무엇인가?
- global/local reward인가?
- state, embedding, intent, Q-value 중 무엇을 공유하는가?
- 메시지 크기, 주기, 범위, 손실 및 지연이 모델링되는가?
- UAV 수 증가 시 observation/action/message complexity가 어떻게 증가하는가?

GLOBE의 “teacher global information은 offline-only, deployment는 1-hop local” 원칙과 비교하라. 해당 논문이 online communication 없이 더 좋은 협력을 달성한다면 GLOBE novelty의 직접 위협으로 표시하라.

---

## STEP 11. 실험 환경과 재현성 분석

다음을 표로 추출하라.

- Simulator, hardware, RL library
- learning rate, batch size, replay buffer, optimizer
- gamma, target update, PPO clip 등 알고리즘별 핵심값
- episode, training steps, convergence criterion
- random seed와 반복 횟수
- 공개 코드/checkpoint/config/raw data 여부

재현성을 매우 높음/높음/보통/낮음/매우 낮음으로 평가하라.

GLOBE의 Lite-GLOBE 공통환경에서 이 논문을 baseline으로 재현할 때 `정확 재현`, `구조적 adaptation`, `proxy adaptation`, `재현 불가` 중 무엇인지 판정하라.

---

## STEP 12. Baseline 공정성 검증

baseline을 traditional, geographic, optimization, heuristic, RL/DRL, MARL로 분류하라.

동일한 network setting, UAV 수, traffic, energy model, information budget, training budget, seed에서 비교했는지 확인하라. proposed method만 미래 정보나 추가 메시지를 사용하는지 검토하라.

GLOBE와 비교할 때는 다음 두 비교를 분리하라.

1. **원 논문 reported result 비교**: simulator가 다르므로 수치 우열을 직접 주장하지 않음
2. **common-environment reproduction 계획**: Lite-GLOBE에서 동일 조건 adaptation을 구현해 비교

---

## STEP 13. Figure/Table 및 결과 분석

각 Figure/Table에 대해 X축, Y축, 비교법, 개선폭, error bar, 주장과 실제 증거를 설명하라.

다음을 우선 확인하라.

- PDR/PRR
- deadline delivery
- mean 및 p95/p99 delay
- energy와 network lifetime
- throughput
- control overhead와 input footprint
- hop count/path stretch
- switch/communication frequency
- computation latency/model size

절대 개선폭과 상대 개선율을 구분해 계산하라. GLOBE의 Phase 12와 직접 수치 비교가 불공정하면 “직접 비교 불가”라고 표시하고, 대신 normalized effect 또는 재현 실험을 제안하라.

---

## STEP 14. 통계적 신뢰성 검증

- random seed 및 반복 횟수
- mean/median
- standard deviation/standard error
- confidence interval/error bar
- paired test와 effect size
- multiple comparison correction
- scenario aggregation 방법
- training seed와 environment seed의 분리

없는 항목을 한계로 지적하고, GLOBE가 더 강한 증거를 만들기 위해 동일한 통계 검정을 어떻게 적용해야 하는지 구체적으로 작성하라.

---

## STEP 15. Contribution과 실제 Novelty 분리

저자가 주장하는 contribution을 먼저 그대로 정리한 뒤 실제 기술 novelty를 별도로 평가하라.

- 새로운 문제정의
- state/observation/action/reward
- RL architecture 또는 training scheme
- routing decision rule/protocol
- information structure/communication mechanism
- energy/channel/mobility model
- joint optimization
- 단순 적용 또는 구성요소 조합

novelty를 1/5~5/5로 평가하고 근거를 제시하라.

추가로 GLOBE novelty와의 관계를 다음 중 하나로 분류하라.

- **N4 직접 선행/중복 위험**: 핵심 조합과 실행 구조가 실질적으로 동일하거나 더 먼저 제안됨
- **N3 강한 유사성**: 핵심 모듈 하나 이상이 매우 유사하여 차별화 설명이 필요함
- **N2 보완적 인접 연구**: 문제나 일부 feature는 겹치지만 핵심 mechanism이 다름
- **N1 배경/응용 차원의 관련성**
- **N0 사실상 무관**

반드시 “무엇이 먼저 알려져 있었고, GLOBE에 무엇이 남는가?”를 한 문단으로 결론내라.

---

## STEP 16. 약점과 Validity Threat 분석

다음 관점에서 “왜 문제이며 결과에 어떤 영향을 주는지”까지 작성하라.

- system assumption
- energy/channel/mobility realism
- partial observability
- scalability/generalization
- reward design
- baseline fairness
- simulation-only validation
- hyperparameter tuning/calibration leakage
- statistical reliability
- real-world deployment
- training/inference/communication cost

동일 기준으로 GLOBE의 약점과 비교하여 `대상 논문이 더 강함 / 비슷함 / GLOBE가 더 강함`으로 판정하라.

---

## STEP 17. Reviewer #2 모드

IEEE/Elsevier reviewer라고 가정하고 분석 대상 논문에 대해 다음을 작성하라.

- Major concerns
- Minor concerns
- Missing experiments
- Questions to authors
- Strong Accept / Accept / Weak Accept / Weak Reject / Reject 및 이유

그 다음 같은 reviewer가 GLOBE 원고를 함께 읽었다고 가정하고 다음을 작성하라.

- 이 논문 때문에 GLOBE에 새로 제기될 major concern
- related work에서 반드시 인정해야 할 선행기술
- GLOBE가 과장하면 안 되는 novelty 문장
- 이 논문과 구분되는 방어 가능한 GLOBE contribution 문장 2~4개

---

# PART D. GLOBE 직접 비교와 연구 발전

## STEP 18. GLOBE와의 직접 비교 Matrix

다음 행을 모두 포함하는 비교표를 작성하라.

| 비교축 | 분석 대상 논문 | GLOBE Phase 12 | GLOBE Phase 13 P+ | 유사/차이 | GLOBE에 주는 의미 |
|---|---|---|---|---|---|
| Routing problem | | | | | |
| System model | | | | | |
| Agent/MDP/Dec-POMDP | | | | | |
| Global information 사용 시점 | | | | | |
| Local observation | | | | | |
| Action space | | | | | |
| RL algorithm | | | | | |
| GNN/attention/message passing | | | | | |
| Knowledge distillation | | | | | |
| Geographic prior | | | | | |
| Mobility/link prediction | | | | | |
| Routing-hole 대응 | | | | | |
| Link-break 대응 | | | | | |
| Queue/congestion 대응 | | | | | |
| Energy model | | | | | |
| Reward/objective | | | | | |
| Risk/safety mechanism | | | | | |
| Online communication overhead | | | | | |
| Computation/model footprint | | | | | |
| Generalization/scalability | | | | | |
| Simulator/realism | | | | | |
| Baseline/statistical evidence | | | | | |
| 핵심 novelty | | | | | |

유사성을 단어 수준이 아니라 **정보 흐름과 decision process 수준**으로 설명하라. 예를 들어 둘 다 link lifetime을 쓴다는 사실보다, 누가 언제 계산하고 어떤 action에 어떻게 반영하는지가 더 중요하다.

---

## STEP 19. GLOBE Novelty Stress Test

다음 질문에 엄격히 답하라.

1. 분석 대상 논문의 어떤 요소가 GLOBE의 novelty 가설 1~4와 겹치는가?
2. 단순 feature 중복인가, mechanism 중복인가, 전체 pipeline 중복인가?
3. GLOBE가 논문을 인용하지 않으면 reviewer가 문제 삼을 가능성이 높은가?
4. GLOBE의 novelty 문장을 어떤 범위로 좁혀야 안전한가?
5. 반대로 GLOBE만의 차별점으로 여전히 방어 가능한 것은 무엇인가?
6. novelty가 약하다면 새 알고리즘 이름만 바꾸지 말고, 어떤 기술적 요소와 검증이 추가되어야 하는가?

다음 형식으로 최종 판단을 작성하라.

> “분석 대상 논문은 GLOBE의 ______과 유사하지만, ______에서 다르다. 따라서 GLOBE는 novelty를 ______이 아니라 ______으로 포지셔닝해야 하며, 이를 입증하기 위해 ______ 실험이 필요하다.”

---

## STEP 20. 분석 대상 논문에서 GLOBE로 이전할 수 있는 요소

논문에서 가져올 가치가 있는 요소를 무조건 채택하지 말고 다음 표로 평가하라.

| 후보 아이디어 | 해결하는 GLOBE 약점 | 필요한 state/code 변경 | 예상 이득 | 예상 비용/위험 | novelty 영향 | 채택 여부 |
|---|---|---|---|---|---|---|

각 채택 후보에 대해 다음을 제시하라.

- 변경 전/후 information flow
- 수정할 GLOBE module: environment, observation, teacher, distillation, student, PRS, reward, evaluator 중 선택
- 필요한 새 수식 또는 pseudocode
- 기존 checkpoint 재사용 가능 여부
- 추가 computation/communication overhead
- 성공/실패를 판정할 metric과 ablation

단순히 “PPO를 SAC로 변경” 같은 알고리즘 교체는 제외하라. 문제 구조나 deployability를 개선하는 이유가 있어야 한다.

---

## STEP 21. GLOBE 개선 연구질문과 방법론 제안

가장 가치 있는 발전안 3~5개를 제안하라. 각 발전안에 대해 다음을 모두 작성하라.

- 기존 GLOBE의 구체적 실패 모드
- 분석 대상 논문에서 얻은 근거
- 새로운 Research Question/Hypothesis
- State/Observation
- Action
- Reward 또는 constrained objective
- PRS/teacher/student 변경
- Expected contribution
- 실패 가능성과 negative result 해석

다음 방향을 우선 검토하라.

- stochastic link uncertainty와 calibrated survival probability
- top-k/multi-path onward robustness
- realistic propulsion + communication energy 및 constrained RL
- adaptive risk threshold 또는 uncertainty-aware switching
- distribution-shift-aware distillation
- temporal memory/POMDP belief without online global messaging
- causal or counterfactual switch explanation
- robust/OOD mobility and channel generalization
- control overhead와 information freshness의 명시적 최적화
- ns-3 또는 packet-level cross-validation

---

## STEP 22. 후속 실험 설계 및 우선순위

아래 세 층으로 실험을 설계하라.

### A. 최소 검증(Must-have)

- GLOBE의 기존 simulator와 code로 바로 실행 가능한 실험
- 핵심 주장 하나를 직접 검증하는 ablation

### B. 논문 강화(Should-have)

- stronger baseline reproduction
- reward/threshold sensitivity
- paired multi-seed statistics와 confidence interval
- PDR–delay–energy–overhead Pareto 분석

### C. 장기 확장(Could-have)

- ns-3, realistic PHY/MAC, 3D mobility, hardware/testbed

각 실험은 다음 표로 작성하라.

| 우선순위 | Hypothesis | 변경 요소 | Baseline/Ablation | Scenario | Metric | 통계 검정 | 통과 기준 | 예상 비용 |
|---|---|---|---|---|---|---|---|---|

특히 Phase 13 P+의 다음 ablation을 분석 대상 논문의 교훈과 연결해 평가하라.

- no link-loss gate
- no energy-aware tie
- no drop suppression
- top-1 onward only
- fixed threshold vs adaptive/uncertainty-aware threshold

---

# PART E. 학습과 일일 연구 기록

## STEP 23. 논문 이해도 시험

대학원 수준 질문 15개를 만든다.

- 개념 질문 5개
- 수식/RL 질문 5개
- 비판 및 GLOBE 비교 질문 5개

먼저 문제만 제시하고, 뒤에 별도 “정답 및 해설”을 제공하라.

---

## STEP 24. 최종 One-Page Research Card

다음을 한 페이지 분량으로 압축하라.

- Problem
- Research Gap
- Core Idea
- System
- Objective
- State/Observation
- Action
- Reward
- RL Algorithm
- Energy Model
- PDR Definition
- Information/Communication Requirement
- Baselines
- Main Result
- Strength
- Weakness
- Novelty Level
- GLOBE Similarity Level(N0~N4)
- GLOBE가 인용할 위치
- GLOBE가 가져갈 아이디어
- GLOBE가 반드시 다르게 해야 할 점

---

## STEP 25. GLOBE Action Memo

분석의 마지막에는 반드시 다음 형식으로 실행 메모를 작성하라.

### 1) 오늘 논문이 GLOBE에 주는 한 문장 결론

### 2) GLOBE 원고 수정 사항

- Introduction
- Related Work
- Problem Formulation
- Method
- Experiment
- Discussion/Limitations

각 항목에 실제로 추가하거나 수정할 문장 방향을 제시하라.

### 3) GLOBE 코드/실험 수정 사항

`즉시`, `다음 단계`, `장기`로 구분하고, 가장 중요한 것부터 최대 7개만 제시하라.

### 4) Novelty 판정

- 위협 수준: 낮음 / 중간 / 높음 / 치명적
- 남아 있는 방어 가능한 novelty
- 추가 증거 없이는 쓰면 안 되는 주장

### 5) 내일 이어서 읽어야 할 논문 유형 또는 검색 키워드 5개

---

# PART F. 여러 논문을 하루에 함께 분석할 때의 종합 출력

논문이 2~3편이면 각 논문에 대해 위 분석을 수행한 후 마지막에 다음 표를 추가하라.

| 항목 | 논문 1 | 논문 2 | 논문 3 | GLOBE |
|---|---|---|---|---|
| 해결 문제 | | | | |
| 핵심 mechanism | | | | |
| 정보 요구량 | | | | |
| RL/optimization | | | | |
| link prediction | | | | |
| energy realism | | | | |
| 실험 강도 | | | | |
| novelty | | | | |
| GLOBE 중복 위험 | | | | |
| GLOBE 적용 가치 | | | | |

마지막에 다음을 결정하라.

1. GLOBE에 가장 위협적인 논문
2. GLOBE 방법을 가장 많이 개선할 수 있는 논문
3. baseline으로 구현할 가치가 가장 높은 논문
4. Related Work에만 인용하면 되는 논문
5. 세 논문을 종합했을 때 새롭게 도출되는 GLOBE 연구 가설

---

# 최종 출력 품질 기준

- 단순 요약보다 비교와 검증에 더 많은 비중을 둔다.
- 모든 중요한 표에는 GLOBE 열 또는 GLOBE 시사점 열을 포함한다.
- 확인된 사실과 제안을 섞지 않는다.
- GLOBE를 무조건 좋게 평가하지 않는다.
- 논문의 아이디어를 그대로 복제하라고 권하지 않고, 학술적 차별성과 인용 필요성을 함께 검토한다.
- 발전안은 최소 하나 이상의 measurable hypothesis와 ablation을 포함한다.
- 결과 마지막에는 반드시 **“이번 논문을 읽은 뒤 GLOBE에서 가장 먼저 해야 할 단 하나의 작업”**을 한 문장으로 제시한다.
