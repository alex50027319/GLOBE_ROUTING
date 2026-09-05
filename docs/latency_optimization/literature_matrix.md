# UAV/FANET RL routing 문헌 행렬과 SwitchGLOBE latency 적용 분석

## 1. 조사 범위

- 기준 데이터베이스: [UAV/FANET RL 데일리 논문 분석](https://app.notion.com/p/0a291669e68449f6918ddca01538d040?v=65eabce9cdf745be98bf89e9c7ff1d7e)
- data source: `collection://82c87922-d396-48b4-b6b0-27d3f7217c90`
- 2026-09-05 현재 접근 가능 행: 45편
- 논문 ID: 1, 3–46. ID 2는 현재 data source 결과에 없음.
- metadata 상 45편 모두 `근거 범위=전문`이거나 전문 기반으로 작성된 행이다. 단 ID 45–46은 근거 범위 property가 비어 있어 정량 결과를 보수적으로 취급한다.

문헌의 latency 수치는 정의가 다르다. 아래 표와 상세 카드에서는 `per-packet inference`, `slot-level optimization`, `path update`, `network end-to-end delay`, `control overhead`를 구분한다. 서로 다른 정의를 직접 비교하지 않는다.

## 2. 45편 전수 census

| ID | 논문 | 연도 | 알고리즘 | Routing 의미 | SwitchGLOBE latency 관점 |
|---:|---|---:|---|---|---|
| 1 | [Joint UAV Flight and Opportunistic Routing](https://app.notion.com/3c4ad79990e681ceb0abd6243a8fbe61) | 2026 | PPO, MARL | packet/joint | CTDE 학습과 local observation 실행 분리를 참고. 직접 decision latency는 핵심 근거 아님 |
| 3 | [Predictive-Q Dual-Path Routing](https://app.notion.com/3c6ad79990e681e999fdc7b22c9cefab) | 2026 | Q-learning | packet | 경량 Q-table, primary/backup, per-packet timing을 제공하는 최우선 latency 참고 |
| 4 | [GA-HAPPO UAV Routing](https://app.notion.com/3c6ad79990e6815aba2dfae117f591f5) | 2025 | PPO, MARL | packet/resource/joint | GAT central critic은 training 전용으로 격리할 근거. 재현성 매우 낮음 |
| 5 | [Interference-Aware Multi-Flow GCN Actor-Critic](https://app.notion.com/3c6ad79990e68110a280e120d1855e3b) | 2026 | actor-critic | packet/joint | multi-flow global graph 비용이 있어 online local path와 분리 필요 |
| 6 | [AoI-Aware Sampling-Buffering-Routing](https://app.notion.com/3c6ad79990e681449c8fc41bbd0ba40b) | 2026 | PPO, MARL | packet/joint | joint action의 latency/quality trade-off, strong ablation 설계 참고 |
| 7 | [DP-MADDPG Dual-Path Redundant Routing](https://app.notion.com/3c6ad79990e681eeb328ec7c7e81f895) | 2026 | DDPG, MARL | packet | learned primary/backup과 passive switch. Fast+Top2와 직접 연결, 재현성 매우 낮음 |
| 8 | [Joint Trajectory, Frequency and Routing](https://app.notion.com/3c6ad79990e6816982a8f17bed2dd105) | 2024 | DDPG, MARL | flight/packet/resource | LSTM-attention 2-hop state는 강하지만 online 비용·상태 수집량 분리 필요 |
| 9 | [TRUAV](https://app.notion.com/3c6ad79990e681c480d4ffde3f4d508e) | 2026 | Q-learning, MARL | trajectory/joint | distributed state/reward 설계 참고; explicit next-hop latency 직접성 낮음 |
| 10 | [Multi-Scale Radio Map Guided Cooperation](https://app.notion.com/3c6ad79990e681f797a2f6d9f140359b) | 2026 | PPO, MARL | packet/resource/joint | radio-map 생성과 online routing 추론을 별도 비용으로 측정해야 함 |
| 11 | [RDQN-HERP Adaptive Intelligent Routing](https://app.notion.com/3c7ad79990e6811da0e3ca071599b67b) | 2026 | DQN, Double DQN | packet | 분산 recurrent inference의 직접 decision timing을 제공 |
| 12 | [Adaptive MPR Selection](https://app.notion.com/3c7ad79990e681d399f2cfe8f0239539) | 2025 | Q-learning, PPO | cluster/control | data-plane inference가 아니라 OLSR MPR 제어면 최적화 |
| 13 | [DRL-AdCAR](https://app.notion.com/3c7ad79990e681639626f9ec833e2b70) | 2025 | DDPG | packet | GRU-LSTM global topology 비용. coding gain과 decision latency 분리 필요 |
| 14 | [Intelligent Maintenance and Routing](https://app.notion.com/3c7ad79990e681728f38ec98a825cb0e) | 2026 | DQN | packet/cluster/joint | GCN+DQN과 backbone repair의 two-stage latency 측정 필요 |
| 15 | [Multipath Routing for Multi-Hop UAV Networks](https://app.notion.com/3c8ad79990e681878966d5289b7fd4ff) | 2026 | PPO, MARL | packet | Dirichlet continuous split은 single next-hop보다 action head가 복잡 |
| 16 | [Explainable MARL for Secure FANETs](https://app.notion.com/3c9ad79990e681a79a9ddaab0c00c6c4) | 2026 | DDPG, MARL | packet/cluster | trust/XAI 계산을 critical path 밖으로 이동할 필요 |
| 17 | [Traffic-Adaptive Per-Hop Multipath Routing](https://app.notion.com/3caad79990e68174ad2ed9abe277f6bd) | 2026 | PPO, MARL | packet | 공개 코드·50 seeds가 강점. 분할 action의 runtime benchmark 추가 가치 높음 |
| 18 | [RoutePPO eBPF/P4](https://app.notion.com/3cbad79990e68170b83fd5fae294030a) | 2026 | PPO | packet | inference와 forwarding-plane update를 분리하는 핵심 deployment 문헌 |
| 19 | [GAT Double-DQN Resilient Routing](https://app.notion.com/3cbad79990e681be8be2c9ef0433ccdd) | 2025 | Double DQN, MARL | packet | dynamic top-k/shared heads/GhostConv의 경량화 아이디어, 직접 latency 부재 |
| 20 | [Trusted Routing with Blockchain](https://app.notion.com/3cbad79990e681b58966e151171b538a) | 2025 | Double DQN, MARL | packet | trust 합의와 routing inference 시간을 분리해야 함 |
| 21 | [Large-Scale UAV Swarms with IEN](https://app.notion.com/3cbad79990e68175a20ae2049f061fa4) | 2026 | DDPG | packet | 200–1000 UAV scaling. global MPNN을 onboard critical path에 두기 어려움 |
| 22 | [KG-DDRL](https://app.notion.com/3cbad79990e681bea59ac607ef1e8627) | 2026 | DQN | joint | K-neighbor KAN-GNN과 action masking이 sparse local encoder 후보 |
| 23 | [HCPMR](https://app.notion.com/3cbad79990e681dda9dbc01402974c7e) | 2026 | PPO, MARL | packet/cluster | 계층 분리로 local online head를 작게 유지할 수 있으나 수치 불일치 주의 |
| 24 | [Distributed Routing Optimization with MARL](https://app.notion.com/3ccad79990e681dcb50fcfdc0a71e3d6) | 2024 | DDPG, MARL | packet | route mode/HELLO interval 제어로 compute보다 control overhead를 줄임 |
| 25 | [Priority-Aware Packet Routing](https://app.notion.com/3ccad79990e681f590f1ddfbe5374b55) | 2025 | DDPG, MARL | packet | packet lifetime/queue feature는 SwitchGLOBE gate calibration에 유용 |
| 26 | [RLFR/DRLFR Energy-Efficient Fast Routing](https://app.notion.com/3ccad79990e6811ba6d1cca21b43db45) | 2024 | Q-learning, Double DQN | packet/joint | local neighbor sharing, risk, power, Raspberry Pi testbed의 최강 배치 참고 |
| 27 | [UAV-Assisted Geo-Edge Routing](https://app.notion.com/3ccad79990e68120bd56c90474ba7d07) | 2026 | Double DQN | packet | local shared model, position prediction, routing-hole fallback 참고 |
| 28 | [Graph RL Routing and Power](https://app.notion.com/3ccad79990e681e980d3cf64f34084f4) | 2025 | DQN | packet/joint | local GAT/GRU; KG-DDRL 계보와 중복성 확인 필요 |
| 29 | [SDN-Blockchain Secure Routing](https://app.notion.com/3ccad79990e6813d81f4ebbd52c76473) | 2026 | PPO | packet | beam-search 후보 제한 후 PPO. pruning-first 설계 참고 |
| 30 | [Anti-Jamming FANET Q-Routing](https://app.notion.com/3cdad79990e681b79368f8d92b16db2e) | 2025 | Q-learning | packet/resource | OGM interval과 power 제어. signaling freshness–overhead trade-off |
| 31 | [MARL Clustering Routing](https://app.notion.com/3cdad79990e6815d8caaccbe4a6c2cb0) | 2025 | DQN/DDQN/MARL | cluster | analytic link lifetime과 energy model을 guard feature에 활용 가능 |
| 32 | [Improved Q-learning UAV Routing](https://app.notion.com/3cdad79990e681c7b8d5f2d8513ca54d) | 2026 | Q-learning | packet | QGeo의 경량 strong baseline; reward와 actual update cost 비교 필요 |
| 33 | [Q-Learning with Visual Information](https://app.notion.com/3cdad79990e681e7ac5cebf0248a2886) | 2025 | Q-learning | packet | visual obstacle risk hard-filter와 실제 routing bytes 증가를 함께 보고 |
| 34 | [SARRA](https://app.notion.com/3cdad79990e6818e8eb0e5a5331e9dad) | 2026 | Q-learning, DQN | packet/cluster | 복합 pipeline이라 attribution과 재현성이 약함 |
| 35 | [Topology-Aware Resilient Q-Routing](https://app.notion.com/3cdad79990e681f8af32f48c75d65788) | 2022 | Q-learning | packet | adaptive table 기반 strong classic baseline; sensing overhead 포함 필요 |
| 36 | [QLR-FANET](https://app.notion.com/3cead79990e68124b48cf4f3fb44ce0d) | 2025 | Q-learning | packet/resource | retry/PER feedback와 bitrate control, 경량 cross-layer 대안 |
| 37 | [Improved Q-Learning Multi-Hop Routing](https://app.notion.com/3d0ad79990e6818c96e3dff5999fb7ca) | 2025 | Q-learning | packet | Q(λ), adaptive HELLO, mode switching. Phase 12 baseline 공정성 확인 |
| 38 | [Dual-Layer DRL UAV-Satellite Routing](https://app.notion.com/3d0ad79990e681beaa7ee1a0621b4be6) | 2026 | PPO | packet/resource/joint | multipath+compute allocation은 action-space 비용이 큼 |
| 39 | [DFS-PPO Robust Routing/Scheduling](https://app.notion.com/3d0ad79990e681bf8c07cc73161ea8db) | 2026 | PPO | packet/resource/joint | DFS hard pruning/action shielding은 적용성 높으나 global online model은 무거움 |
| 40 | [Q-Learning and Fuzzy Logic Routing](https://app.notion.com/3d0ad79990e681c4b1d9ee65cd0952a4) | 2024 | Q-learning | packet | inference는 가볍지만 destination tables/HELLO가 O(N²)로 커질 수 있음 |
| 41 | [Hydro-AI MANET Routing](https://app.notion.com/3d0ad79990e681a4a069d5506d87222f) | 2025 | DQN | packet | predictive LSTM+DQN 아이디어는 관련되나 표/서술 충돌로 성능 근거 제외 |
| 42 | [Adversarial GCN-Transformer Routing](https://app.notion.com/3d1ad79990e681a48e69e1bf4ceb548f) | 2026 | PPO | packet/joint | robustness training 참고, global online 실행과 재현성 한계 |
| 43 | [Resilient Packet Routing with DRL](https://app.notion.com/3d1ad79990e681e48d38f566ff8b5e0b) | 2024 | Dueling DDQN, MARL | packet | local per-packet resilience baseline. recurrent/value ensemble 비용 점검 |
| 44 | [Congestion-Aware UAV Deployment/Q-Routing](https://app.notion.com/3d1ad79990e681ed9bd4d9c75f69cf73) | 2025 | Q-learning | flight/packet/joint | survivability 기반 proactive drop과 SwitchGLOBE drop suppression 비교 |
| 45 | [RL Joint Routing and TSN Slot Scheduling](https://app.notion.com/3d1ad79990e681319485fcb26c6fed19) | 2026 | PPO | packet/resource/joint | route와 slot offset의 계층적 분리 참고. 근거 범위 property 비어 있음 |
| 46 | [ML Congestion-Aware Adaptive Routing](https://app.notion.com/3d1ad79990e6817bab25ff34d292def0) | 2025 | supervised ML | packet | 경량 predictor+weighted cost 후보이나 시제/구현 불일치로 수치 신뢰 매우 낮음 |

## 3. latency 직접성이 높은 핵심 논문 카드

### 3.1 Predictive-Q dual-path routing — ID 3

출처: [공식 전문](https://www.techscience.com/cmc/v88n3/68155/html), [DOI](https://doi.org/10.32604/cmc.2026.084301)

핵심은 mobile-edge assisted swarm에서 interference/mobility feature로 Q 값을 갱신하고 primary와 backup path를 유지하는 것이다. 논문이 보고한 Q-table memory는 약 79.7 KB이며, per-packet decision은 제안법 1.00 ms, 비교법 1.62 ms 수준이다. 반면 slot-level computation은 약 120.3 ms로 다른 단위다.

SwitchGLOBE 적용:

- Fast+Top2처럼 한 inference에서 backup을 산출하는 구조는 적합하다.
- table lookup의 1 ms와 neural CUDA call의 2–6 ms를 같은 hardware 조건 없이 직접 비교할 수는 없다.
- dual-path는 장애 대응 지연을 줄이지만 policy approximation error를 해결하지 않는다.

### 3.2 RDQN-HERP — ID 11

출처: [Notion 전문 분석](https://app.notion.com/3c7ad79990e6811da0e3ca071599b67b), [DOI](https://doi.org/10.1109/TVT.2026.3668740)

50-node 조건에서 mean decision time 4.92 ms, maximum 6.13 ms가 보고되어 있다. 같은 문서의 AODV 값 0.0103은 측정 범위/단위가 크게 다를 가능성이 있으므로 neural policy와 직접 우열 비교에 사용하지 않는다.

SwitchGLOBE 적용:

- recurrent hidden state를 매 packet 재계산하지 않고 node-local state로 유지하는 방향은 검토 가치가 있다.
- 평균뿐 아니라 maximum/p95를 함께 보고해야 한다.
- recurrent model의 state freshness와 topology reset 규칙을 명시해야 한다.

### 3.3 RoutePPO eBPF/P4 — ID 18

출처: [연구 페이지](https://avesis.hacettepe.edu.tr/yayin/d19f4f82-b239-4558-8e4d-a0420543a5c0/routeppo-ebpf-based-proximal-policy-optimization-for-adaptive-routing-in-uav-swarm-networks), [DOI](https://doi.org/10.1109/SMARTNETS69662.2026.11604730)

eBPF/P4 path update는 1 ms 미만, simulation latency는 약 14 ms에서 9.6 ms로 개선됐다고 보고한다. 그러나 live 환경에서는 learned routing이 static routing보다 나빴던 결과가 있어 sim-to-deployment gap이 중요하다.

SwitchGLOBE 적용:

\[
T_{reaction}=T_{observe}+T_{infer}+T_{install}+T_{propagate}
\]

현재 실험은 주로 \(T_{infer}\)를 측정한다. 후속 실기기 검증은 routing table/eBPF map update인 \(T_{install}\)을 분리해야 한다.

### 3.4 GAT Double-DQN 경량화 — ID 19

출처: [Notion 전문 분석](https://app.notion.com/3cbad79990e681be8be2c9ef0433ccdd), [DOI](https://doi.org/10.1109/WCSP68525.2025.1010249)

dynamic top-k, shared heads, GhostConv를 사용해 graph policy를 줄이는 방향을 제시하지만 직접 decision latency를 보고하지 않는다.

SwitchGLOBE 적용:

- top-k는 post-hoc backup 선택보다 **encoder 이전의 candidate pruning**에 둘 때 계산량을 줄일 수 있다.
- learned top-k가 Exact action을 제거할 위험이 있으므로 conservative physical mask부터 시작한다.
- FLOPs/parameter 감소를 latency 감소로 대신 보고하면 안 된다.

### 3.5 KG-DDRL — ID 22

출처: [Notion 전문 분석](https://app.notion.com/3cbad79990e681bea59ac607ef1e8627), [DOI](https://doi.org/10.1109/CCDC69976.2026.11560534)

local K-neighbor KAN-GNN encoder의 복잡도는 개략적으로 \(O(n_tKd_v)\), dense attention은 \(O(n_t^2d_h)\)로 정리된다. action masking도 사용한다. 직접 latency 수치는 없어 complexity evidence와 runtime evidence를 구분해야 한다.

SwitchGLOBE 적용:

\[
N\times N\;graph\rightarrow N\times K\;local\ neighborhood
\]

이는 대규모 UAV에서 유력하지만 observation schema와 checkpoint를 바꾸므로 재학습이 필요하다.

### 3.6 RLFR/DRLFR — ID 26

출처: [저자 공개 전문](https://yongjinliu.github.io/files/2024_Reinforcement_Learning_based_Energy-Efficient_Fast_Routing_for_FANETs.pdf), [DOI](https://doi.org/10.1109/TCOMM.2024.3409561)

local neighbor information sharing, latency-risk 학습, forward/drop과 transmit-power 공동 결정을 결합한다. 60-UAV simulation과 5-UAV Raspberry Pi testbed를 제공한다. 현재 추출 근거에서 SwitchGLOBE와 같은 정의의 onboard decision p95 수치는 확인되지 않았다.

SwitchGLOBE 적용:

- global teacher는 학습에만 사용하고 execution feature는 local neighbor로 제한한다.
- 실기기에서는 compute latency, packet delivery, radio energy를 함께 측정한다.
- power action을 추가하면 action space가 커지므로 routing latency 연구와 cross-layer 연구를 분리한다.

### 3.7 DFS-PPO — ID 39

출처: [Notion 전문 분석](https://app.notion.com/3d0ad79990e681bf8c07cc73161ea8db), [DOI](https://doi.org/10.1109/JIOT.2026.3687225)

global GAT/Transformer와 DFS hard pruning/action shielding을 사용하며 i7 환경의 flow online inference 약 0.451 s가 보고된다. 이는 SwitchGLOBE의 per-decision millisecond benchmark와 대상 범위가 다르다.

SwitchGLOBE 적용:

- DFS/constraint pruning을 policy 앞에 두되 Exact action 보존을 먼저 검증한다.
- global planning은 느린 control loop, local forwarding은 빠른 data loop로 계층화한다.

### 3.8 Q-learning + fuzzy logic — ID 40

출처: [Notion 전문 분석](https://app.notion.com/3d0ad79990e681c4b1d9ee65cd0952a4), [DOI](https://doi.org/10.1109/WCSP62071.2024.10826772)

table/fuzzy inference 자체는 가볍지만 destination별 Q-table과 HELLO signaling은 구현에 따라 \(O(N^2)\) 상태·메시지 비용이 생길 수 있다.

SwitchGLOBE 적용:

- neural inference time만 비교하지 말고 memory per destination, HELLO bytes/s, update frequency를 함께 보고한다.
- SwitchGLOBE의 `policy_input_bytes`는 무선 overhead가 아니므로 별도 계측이 필요하다.

### 3.9 Lightweight congestion predictor — ID 46

출처: [Notion 전문 분석](https://app.notion.com/3d1ad79990e6817bab25ff34d292def0), [DOI](https://doi.org/10.1109/SISIMPACT67725.2025.11439231)

queue/link/mobility feature를 사용하는 감독학습 혼잡 예측기와 weighted next-hop cost를 결합한다. 1.82 ms라는 수치가 있으나 문서 내 결과 시제와 구현 설명이 충돌하고 재현성이 매우 낮아 benchmark target으로만 사용하고 사실 확정 근거로 사용하지 않는다.

SwitchGLOBE 적용:

- full policy를 압축하기보다 small disagreement/risk predictor만 학습해 Exact escalation gate로 쓰는 방향이 더 안전하다.
- predictor calibration은 accuracy가 아니라 false-safe rate와 worst-scenario coverage로 평가한다.

## 4. 문헌에서 도출한 후보 우선순위

| 우선순위 | 설계 | 문헌 근거 | 현재 상태 |
|---:|---|---|---|
| 1 | Fused single-pass Exact | 모든 neural routing의 공통 runtime 원칙 | 구현·채택 |
| 2 | CPU/device-aware dispatch | 작은 batch latency와 deployment literature | A100에서 검증·채택 |
| 3 | primary+backup one-pass | Predictive-Q, DP-MADDPG | Fast+Top2 구현, quality 보류 |
| 4 | conservative hard pruning | DFS-PPO, visual Q-routing, KG-DDRL | 미구현, 다음 우선순위 |
| 5 | confidence cascade | lightweight congestion predictor, hierarchical schemes | 미구현, calibration 필요 |
| 6 | sparse K-neighbor encoder | KG-DDRL, large-scale GNN routing | 중기 재학습 과제 |
| 7 | forwarding-plane integration | RoutePPO eBPF/P4 | 실기기 측정 과제 |
| 8 | recurrent state reuse | RDQN-HERP | state freshness 검증 후 후보 |

## 5. 문헌 해석의 한계

1. 서로 다른 논문은 decision, path computation, slot optimization, network delay를 같은 `latency` 용어로 부른다.
2. hardware, batch, warm-up, synchronization이 없는 수치는 SwitchGLOBE 결과와 직접 비교하지 않는다.
3. complexity 감소와 실제 runtime 감소는 다르다. 현재 Early Exit가 그 반례다.
4. 최신 conference/preprint 중 일부는 코드·checkpoint가 없고 재현성이 낮다.
5. database metadata의 관련성/참신성 점수는 screening 보조값이며 통계적 evidence가 아니다.
6. ID 45–46은 `근거 범위` property가 비어 있어 상세 결과를 보수적으로 사용했다.

## 6. 실험 설계에 반영한 사항

- primary/backup: Fast+Top2 stale-primary 합성 검증
- local execution: batch-1 CPU/CUDA 동시 측정
- hard guard: Early Exit margin sweep과 43,467-decision divergence audit
- deployment gap: A100뿐 아니라 host CPU도 같은 session에서 측정
- evidence discipline: p95, seed-paired CI, quality gate, figure별 source CSV 제공
- overhead taxonomy: model compute, policy input bytes, simulator energy proxy, wireless control overhead를 분리
