"""Convert reviewed wiki content to Korean while preserving original titles and links."""

from __future__ import annotations

import re
from pathlib import Path

from utils import ROOT, VAULT_DIR, read_markdown, yaml_document

PAPER_DIR = VAULT_DIR / "03_Papers" / "Paper_Cards"
SOURCE_DIR = VAULT_DIR / "01_Sources" / "Papers"

EVIDENCE_KO = {
    "Protocol definition and reported headline gains": "프로토콜 정의와 보고된 주요 성능 향상",
    "Two-hop routing-hole bypass and shaped reward": "2-hop routing-hole 우회와 설계된 보상",
    "10 repetitions and simulation/HIL settings": "10회 반복 및 simulation/HIL 설정",
    "PDR, energy, delay, ETX and HIL results": "PDR, 에너지, 지연, ETX 및 HIL 결과",
    "POSG, CTDE, objectives and contributions": "POSG, CTDE, 목적함수 및 기여",
    "Training algorithm and complexity": "학습 알고리즘과 복잡도",
    "Parameters, five seeds and baselines": "파라미터, 5개 seed 및 baseline",
    "Ablation, robustness and scalability": "Ablation, 강건성 및 확장성",
    "Task definition and contributions": "문제 정의와 기여",
    "REINFORCE and greedy rollout baseline": "REINFORCE와 greedy rollout baseline",
    "Settings and solver baselines": "설정과 solver baseline",
    "Small/large-instance results and limitations": "소규모·대규모 instance 결과와 한계",
    "POMDP, DE-MADDPG contributions and CTDE": "POMDP, DE-MADDPG 기여 및 CTDE",
    "Training settings and network architecture": "학습 설정과 network architecture",
    "Repeated tests and comparison with MADDPG": "반복 시험과 MADDPG 비교",
    "Single-connectivity limitation": "단일 연결성의 한계",
    "Emergent communication and topology-flexible routing": "Emergent communication과 topology-flexible routing",
    "Simulator, metrics and six baselines": "Simulator, 지표 및 6개 baseline",
    "Load, ablation and overhead results": "부하, ablation 및 overhead 결과",
    "Failure, node addition, ATT topology and limitations": "고장, 노드 추가, ATT topology 및 한계",
    "Protocol components and motivation": "프로토콜 구성요소와 동기",
    "Analytical overhead and delay model": "분석적 overhead·지연 모델",
    "NS-3 setup and baseline variants": "NS-3 설정과 baseline 변형",
    "Delay, PDR, overhead, stability and limitations": "지연, PDR, overhead, 안정성 및 한계",
    "IQMR motivation and contributions": "IQMR의 동기와 기여",
    "Algorithm and engineered decision variables": "알고리즘과 설계된 의사결정 변수",
    "MATLAB simulation parameters": "MATLAB simulation 파라미터",
    "Energy/throughput comparison and limitations": "에너지·throughput 비교와 한계",
    "Draft title, architecture and claimed contributions": "초안 제목, architecture 및 주장한 기여",
    "Latent L2 distillation and MLP actor": "Latent L2 distillation과 MLP actor",
    "Planned baselines, metrics and seeds": "계획된 baseline, 지표 및 seed",
    "No validated results and explicit limitations": "검증된 결과의 부재와 명시된 한계",
    "Dissertation identity and scope": "학위논문의 서지 식별 정보와 범위",
    "Three proposed routing/control systems": "제안된 3개 routing/control system",
    "Reported aggregate improvements": "보고된 종합 성능 향상",
    "Conclusions, limitations and future directions": "결론, 한계 및 향후 연구",
}

KO = {
    "Reinforcement-Learning-Based Geographic Routing Considering Future Evolution of Link States for UAV Networks": {
        "summary": "Evo-QGeo는 미래 링크 상태를 예측한 수작업 점수와 2-hop 라우팅 홀 우회 규칙을 Q-learning에 결합한 FANET 지리 라우팅 방식이다.",
        "problem": "고속 UAV 이동으로 링크가 급변하고 지역적 라우팅 홀이 발생하는 환경에서 신뢰성 있는 다음 홉을 선택하는 문제를 다룬다.",
        "method": "예측 거리, 수신 전력, 링크 지속시간을 융합한 링크 상태값으로 Q 값을 초기화·갱신한다. 비콘으로 얻은 2-hop 지역 토폴로지와 구역별 우회 규칙으로 라우팅 홀을 처리한다.",
        "training": "온라인 Q-learning과 epsilon 기반 행동 선택을 사용한다. 목적지 도달은 +10, 라우팅 홀은 -10, 그 외에는 추정 링크 상태를 보상으로 사용한다.",
        "execution": "전역 토폴로지 없이 로컬 및 2-hop 비콘 정보를 사용하지만, 링크 점수와 우회 구역 규칙이 행동을 직접 유도한다.",
        "experiment": "20~60대 UAV, 5~60 m/s, 250~450 m 통신 거리, 여러 배치 영역을 변화시켰다. RWP 이동성, log-normal/Rayleigh 손실, IEEE 802.11b를 사용하며 시나리오별 10회 반복과 실제 UAV 궤적 기반 HIL 검증을 수행했다.",
        "results": "GPSR 대비 PDR 3.15~44.61%, QGeo 대비 0.81~31.35% 향상을 보고했다. 에너지, 지연, end-to-end ETX도 두 baseline보다 낮았고 HIL 결과도 동일한 경향을 보였다.",
        "theory": "Dec-POMDP나 정책 성능 이론은 없으며 링크 예측과 라우팅 홀 규칙에 대한 설계적 정당화가 중심이다.",
        "assumptions": "GPS·속도 정보, 주기적 비콘, 2-hop 이웃 요약, RWP 이동성, 고정 무선 파라미터, 설계된 링크 상태 보상을 가정한다.",
        "strengths": "FANET 라우팅과 직접 관련되고 이동성·밀도·통신거리 sweep이 넓다. 10회 반복, HIL 궤적 검증, PDR·지연·에너지·ETX 보고가 장점이다.",
        "weaknesses": "baseline이 GPSR과 QGeo뿐이고, 미래 링크 융합과 2-hop 우회 규칙을 분리한 ablation이 없다. 신뢰구간도 제시하지 않는다.",
        "fragile": "성능은 정확하고 최신인 2-hop 비콘과 이동·링크 예측에 의존한다. HIL도 전체 무선 네트워크를 실제 비행으로 검증한 것은 아니다.",
        "connection": "GLOBE++의 핵심 heuristic-rich baseline이다. GLOBE++는 engineered score 없이 ego-graph 정책이 성능을 회복하는지, 그리고 오버헤드가 실제로 감소하는지를 입증해야 한다.",
        "supports": "RL 기반 FANET 지리 라우팅 관련연구, baseline 선정, 이동성 sweep, 라우팅 홀 평가 설계를 뒷받침한다.",
        "cannot": "정책 증류, Dec-POMDP 이론, CTDE, GNN 실행, 부분 관측 정보 격차는 뒷받침하지 못한다.",
    },
    "GLo-MAPPO: Multi-Agent Deep Reinforcement Learning for Energy-Efficient UAV-Assisted LoRa Networks": {
        "summary": "GLo-MAPPO는 다중 UAV LoRa 게이트웨이의 궤적과 무선 자원 최적화를 부분 관측 확률 게임으로 모델링하고 CTDE 기반 MAPPO로 해결한다.",
        "problem": "QoS, 추진 에너지, 충돌 회피, 임무 제약 아래 spreading factor, 송신 전력, UAV 궤적, 단말 연결을 함께 최적화한다.",
        "method": "분산 GRU actor는 로컬 관측을 사용하고 중앙 critic이 학습을 안정화한다. 단말 연결은 행동 공간을 줄이기 위해 MARL 밖의 gain 기반 알고리즘으로 분리한다.",
        "training": "MAPPO, GAE, PPO clipping을 사용해 200만 환경 step을 학습하고 5개 고정 seed와 hyperparameter 민감도 분석을 보고한다.",
        "execution": "각 UAV actor는 로컬 관측으로 실행하지만 단말 연결은 별도 gain 기반 규칙을 사용한다. CTDE이지만 정책 증류는 아니다.",
        "experiment": "2~5대 UAV, 1 km 정사각형 영역, 임무당 400 step, 5개 seed를 사용한다. MAA2C, COMA, VDN, QMIX, IPPO와 비교하고 자원별 ablation, noisy CSI, 100~400 단말 scalability를 시험한다.",
        "results": "시험한 모든 밀도에서 높은 누적 보상과 에너지 효율을 보고하며, 최선 baseline 대비 에너지 효율 향상은 단말 수에 따라 18.56~152.72%다.",
        "theory": "POSG 정식화와 계산 복잡도는 있으나 부분 관측 또는 정책 오차를 성능과 연결하는 bound는 없다.",
        "assumptions": "LoRa 채널 모델, 고정 고도, 이산화된 행동, MARL과 분리된 association, perfect/noisy CSI, simulation-only 평가를 가정한다.",
        "strengths": "부분 관측과 CTDE가 명시적이고 5개 seed, MARL baseline, ablation, robustness, 민감도, scalability를 함께 제시한다.",
        "weaknesses": "패킷 next-hop 라우팅이 아니라 게이트웨이 이동·자원 할당 문제다. association은 수작업 규칙이며 PDR·end-to-end routing overhead를 측정하지 않는다.",
        "fragile": "joint optimization 일부를 MARL 밖으로 분리했으며 결과는 LoRa와 추진 에너지 모델의 정확성에 민감하다.",
        "connection": "MAPPO/CTDE 설계, recurrent local policy, multi-seed 보고, noisy observation 시험에 유용하지만 직접적인 FANET 라우팅 baseline은 아니다.",
        "supports": "CTDE MAPPO 설계와 재현성 높은 실험 보고 형식을 뒷받침한다.",
        "cannot": "next-hop 라우팅, global-to-local 증류, routing overhead, heuristic-free forwarding은 뒷받침하지 못한다.",
    },
    "Deep Reinforcement Learning for Online Routing of Unmanned Aerial Vehicles with Wireless Power Transfer": {
        "summary": "배터리 소진과 무선 충전을 고려한 단일 UAV 순회 경로를 attention 기반 sequence model과 REINFORCE로 생성한다.",
        "problem": "단일 UAV가 작업 지점을 방문하고 필요할 때 충전 기지로 복귀하는 총 임무 시간을 최소화한다.",
        "method": "Transformer 계열 encoder-decoder가 경로를 순차 생성하며 invalid node는 mask한다. 가장 좋은 greedy rollout 정책을 baseline으로 REINFORCE를 학습한다.",
        "training": "100 epoch, epoch당 32만 synthetic instance, batch 256, Adam, 128차원 hidden layer를 사용한다. greedy, sampling, beam search를 비교한다.",
        "execution": "중앙집중식 combinatorial solver이며 FANET 패킷 라우팅이나 분산 MARL 실행이 아니다.",
        "experiment": "20~200 node의 무작위 문제에서 100개 test instance를 사용하고 모든 solver를 동일한 Python 환경과 장비에서 비교한다.",
        "results": "OR-Tools와 유사한 비용을 훨씬 짧은 시간에 산출한다. 100 node에서는 0.7%, 150 node에서는 1.1% 비용이 나쁘지만 각각 5.6배와 592배 빠르다.",
        "theory": "최적성 또는 일반화 보장은 없으며 경험적 결과만 제공한다.",
        "assumptions": "단일 UAV, 고정 속도, synthetic Euclidean instance, 알려진 방문 지점, 단일 충전 기지, 단순화된 WPT를 가정한다.",
        "strengths": "목적함수가 명확하고 강한 비학습 solver와 runtime-quality trade-off를 비교하며 큰 instance까지 확장한다.",
        "weaknesses": "여기서 routing은 차량 순회 경로이며 FANET 패킷 forwarding이 아니다. 100개 test instance뿐이고 신뢰구간이 없다.",
        "fragile": "arbitrary topology 일반화는 동일 synthetic generator 안에서만 검증되며 고정 속도와 단순 충전 모델을 사용한다.",
        "connection": "graph/attention 정책, action masking, learned heuristic의 확장성에는 참고할 수 있지만 직접 FANET routing 연구로 인용하면 안 된다.",
        "supports": "attention 기반 그래프 의사결정 구조와 runtime-quality 평가를 뒷받침한다.",
        "cannot": "MARL, CTDE, 패킷 지표, 부분 관측, 분산 FANET 실행은 뒷받침하지 못한다.",
    },
    "Building a Connected Communication Network for UAV Clusters Using DE-MADDPG": {
        "summary": "DE-MADDPG는 보상 분해, reward shaping, virtual leader-follower 좌표계를 사용해 단절된 UAV 군집의 연결성을 복구한다.",
        "problem": "고장이나 링크 단절 이후 이웃 운동 정보를 완전히 얻지 못하는 상황에서 단일 연결 네트워크를 재구축한다.",
        "method": "MADDPG를 global/local critic으로 분해하고 중간 보상을 shaping한다. virtual navigator로 이동 좌표계와 상태 공간을 안정화한다.",
        "training": "20,000 episode, replay buffer 200,000, batch 1024, actor/critic learning rate 0.001의 CTDE를 사용한다.",
        "execution": "각 UAV가 로컬 관측으로 이산화된 3차원 이동 행동을 출력한다. 패킷 next-hop이 아니라 topology control 문제다.",
        "experiment": "100 m 이동 큐브의 Python/PaddlePaddle simulation에서 100-episode test, 10회 training, UAV 수 변화로 MADDPG와 비교한다.",
        "results": "MADDPG보다 빠르게 학습하고 상태 공간 이탈을 줄이며 더 높은 연결 성공률을 보인다. UAV 수가 늘면 수렴이 어려워진다.",
        "theory": "POMDP는 제시하지만 수렴이나 부분 관측 성능 bound는 없다.",
        "assumptions": "동질 UAV, 단순 이산 방향, virtual navigator, 이상화된 통신 반경, 단일 연결 목표를 가정한다.",
        "strengths": "CTDE와 로컬 실행을 명확히 구분하고 반복 학습 결과와 hyperparameter, UAV 수 증가에 따른 난점을 제시한다.",
        "weaknesses": "baseline이 MADDPG 하나이며 패킷 지표, 에너지, 실제 무선 stack이 없다. 단일 연결은 외란에 취약하다.",
        "fragile": "reward shaping과 virtual coordinate가 해법을 상당 부분 내장할 수 있으며 제한된 공간의 연결 성공이 동적 라우팅으로 직접 이전되지 않는다.",
        "connection": "CTDE와 reward shaping 비판, topology control과 packet routing의 구분에 유용하다.",
        "supports": "CTDE 동기, local execution, partial observation, scalability 논의를 뒷받침한다.",
        "cannot": "PDR·delay, 정책 증류, graph routing, next-hop 선택은 뒷받침하지 못한다.",
    },
    "DRAMA: A Dynamic Packet Routing Algorithm using Multi-Agent Reinforcement Learning with Emergent Communication": {
        "summary": "DRAMA는 학습된 router 간 메시지와 topology-flexible Q-network로 부하, 링크·노드 장애, router 추가에 적응한다.",
        "problem": "고정 입력·출력 크기의 neural policy가 처리하기 어려운 혼잡과 동적 topology, 특히 새 router 추가 상황의 분산 패킷 라우팅을 다룬다.",
        "method": "각 router가 로컬 관측을 message로 encoding하고 attention으로 C-hop message를 집계해 이웃별 Q 값을 예측한다. TD loss에 weighted shortest-path 기반 auxiliary loss를 더한다.",
        "training": "replay, target network, emergent communication, auxiliary cost supervision을 사용하는 off-policy Q-learning 계열 MARL이다.",
        "execution": "분산 실행이지만 runtime에 학습된 메시지 교환이 필요하다. 메시지 round와 quantization이 overhead-performance trade-off를 만든다.",
        "experiment": "Poisson traffic과 고정 link bandwidth의 Python simulator를 사용한다. 기본 시험은 512 step 10회, 장애·확장 시험은 50회이며 synthetic topology와 실제 ATT topology를 평가한다.",
        "results": "가장 높은 toy load에서 delivery rate 100%, latency 18.15 ms를 보고한다. 재학습 없이 장애와 router 추가에 적응하며, quantized 10-step 메시지는 원래 통신량의 0.3%로 성능 저하가 작다.",
        "theory": "Dec-POMDP나 성능 bound는 없으며 topology-compatible network와 ATT 실험으로 scalability를 주장한다.",
        "assumptions": "단순 queue와 고정 link, 실행 중 message 교환, shortest-path auxiliary target, delivered packet만의 latency 계산을 가정한다.",
        "strengths": "baseline이 다양하고 동적 node/link, 실제 topology, message overhead ablation을 포함해 패킷 라우팅과 직접 관련된다.",
        "weaknesses": "UAV PHY/MAC simulation이 아니고 runtime communication이 strict local execution과 충돌한다. shortest-path supervision도 수작업 bias다.",
        "fragile": "메시지 동기화와 신뢰성 있는 이웃 통신에 의존하며 더 큰 topology scalability는 future work로 남는다.",
        "connection": "동적 topology와 graph-structured local routing의 핵심 비교 대상이다. GLOBE++는 DRAMA의 runtime 메시지와 자신의 ego-graph 획득 overhead를 명확히 비교해야 한다.",
        "supports": "동적 topology 시험, communication overhead, graph policy, routing baseline 설계를 뒷받침한다.",
        "cannot": "global teacher 증류, FANET 무선 성능, irreducible observation gap 이론은 뒷받침하지 못한다.",
    },
    "EDP Protocol: Advancing Mobility-Aware Drone Network Connectivity with Adaptive Routing": {
        "summary": "EDP는 mobility prediction, coverage-aware probabilistic flooding, link/load/lifetime 기반 route maintenance를 결합한 비학습 FANET protocol이다.",
        "problem": "희소하고 빠르게 이동하는 drone network에서 flooding overhead, route break, delay를 줄인다.",
        "method": "adaptive Kalman predictor로 이동을 예측하고 conical neighbor-coverage forwarding으로 RREQ flooding을 줄인다. 거리, traffic load, expected delay, route lifetime으로 경로를 선택·복구한다.",
        "training": "학습 정책은 없고 모든 파라미터와 수식이 분석적으로 설계된다.",
        "execution": "주기적 mobility/link 정보를 사용하는 분산 reactive routing이며 route discovery와 maintenance가 수작업으로 설계되어 있다.",
        "experiment": "NS-3, IEEE 802.11b, YansWifiPhy, Friis loss, custom 3D Gauss-Markov를 사용한다. 10~80대, 4~20 m/s, 10~50 KB/s를 평가하지만 seed 수는 명시하지 않는다.",
        "results": "AODV 대비 long-hop delay 17.6% 감소, 40대에서 PDR 15.8% 향상, 20 m/s에서 PDR 0.65 이상, 고속에서 overhead ratio 0.18 미만을 보고한다.",
        "theory": "단순 가정 아래 overhead와 M/G/1 delay를 유도하지만 RL policy 성능 이론은 아니다.",
        "assumptions": "충분히 정확한 이동 예측, multi-metric normalization, custom Gauss-Markov, 단순 path loss, 저자가 조정한 component baseline을 가정한다.",
        "strengths": "직접적인 FANET NS-3 연구이며 overhead와 안정성 지표, mechanism-aligned variant, 극단적 mobility 한계를 제시한다.",
        "weaknesses": "정식 출판 연도와 seed가 불명확하고 confidence interval이 없다. baseline variant는 독립 구현이 아니다.",
        "fragile": "극고속에서 route lifetime은 여전히 급감하며 비동기 대규모 mobility estimation 비용을 충분히 평가하지 않는다.",
        "connection": "중요한 비학습 baseline이자 mobility/overhead stress test 출처다. 강한 성능이 engineered prediction과 route score에서 나온다는 점도 보여준다.",
        "supports": "NS-3 설정, overhead·route stability 지표, predictive baseline을 뒷받침한다.",
        "cannot": "MARL, 정책 증류, GNN 실행, partial-observability theory는 뒷받침하지 못한다.",
    },
    "Improved Q-learning based Multi-hop Routing for UAV-Assisted Communication": {
        "summary": "IQMR는 에너지, ACK, coverage, collision 정보를 포함한 다목적 Q(lambda)-learning으로 UAV-to-TBS multi-hop forwarding을 수행한다.",
        "problem": "사전 UAV 경로 없이 intermittent connectivity, 에너지 고갈, 충돌 위험, network fragmentation 상황에서 surveillance data를 전달한다.",
        "method": "잔여 에너지, ACK 상태, coverage·collision probability, source-destination 정렬로 tabular Q 값을 갱신한다. UAV는 discovery, receive, transmit, charge mode를 전환한다.",
        "training": "온라인 epsilon-greedy Q(lambda)를 사용하고 learning/exploration rate를 분석하며 500 episode 이내 수렴을 보고한다.",
        "execution": "분산 neighbor-table routing이지만 objective, constraint, geometry tie-break가 강하게 수작업 설계되어 있다.",
        "experiment": "MATLAB에서 50대 UAV, 3D Gauss-Markov, Nakagami fading, 802.11n, 2 Mbps CBR를 사용하고 fragmentation과 rejoin을 시험한다.",
        "results": "QMR과 Q-FANET 대비 에너지 효율 32.27~36.35%, throughput 25.19~32.05% 향상을 보고한다.",
        "theory": "MARL·Dec-POMDP 분석이나 성능 보장은 없다.",
        "assumptions": "GPS, ACK, 에너지·충돌 추정, 단일 TBS, 50대 UAV, MATLAB network model, 수작업 objective priority를 가정한다.",
        "strengths": "직접적인 UAV multi-hop routing이고 에너지, fragmentation, rejoin, 충돌 제약을 다룬다.",
        "weaknesses": "baseline이 두 개뿐이고 seed와 confidence interval이 없다. PDR, delay, routing overhead가 빠져 있고 simulator fidelity가 불명확하다.",
        "fragile": "정규화된 reward weight와 여러 추정량에 민감하고 tabular Q는 확장성이 낮을 수 있다.",
        "connection": "GLOBE++의 직접 Q-learning baseline이다. engineered state·reward·constraint가 실제 이득 원인인지 ablation해야 한다.",
        "supports": "에너지·fragmentation scenario와 Q-learning baseline을 뒷받침한다.",
        "cannot": "CTDE, GNN, 정책 증류, 엄밀한 통계, partial-observability bound는 뒷받침하지 못한다.",
    },
    "GLOBE-Routing: Global-to-Local Knowledge-Distilled Graph-MAPPO for Decentralized Routing in UAV Swarm FANETs": {
        "summary": "현재 GLOBE-Routing 초안은 global graph context의 latent distillation과 2-hop forwardability를 제안하지만 검증된 실험 결과가 없다.",
        "problem": "학습 중 LEO/BS의 global visibility를 활용하면서 실행 중에는 online global command 없이 FANET next-hop을 선택하려 한다.",
        "method": "global GNN encoder의 latent z를 local history predictor가 L2 loss로 맞춘다. MLP actor는 local/2-hop feature와 예측 context를 사용하며 reward에 다수의 engineered routing term이 포함된다.",
        "training": "central critic의 MAPPO와 latent L2 distillation을 계획하지만 완료된 training campaign은 없다.",
        "execution": "local/2-hop observation, predicted global context, action mask, 2-hop forwardability를 사용한다. 제시된 actor는 local GNN이 아니라 MLP다.",
        "experiment": "Python training 후 NS-3/ns3-gym validation, 20~100 UAV, 5~10 seed, traditional/RL baseline, component ablation을 계획만 한다.",
        "results": "수치 결과는 없으며 validated simulation log에서 표와 그림을 만들어야 한다고 명시한다.",
        "theory": "복잡도 식은 있으나 Dec-POMDP, information-gap decomposition, KL-performance bound는 없다.",
        "assumptions": "training-time global observer, local/2-hop summary, local history로 global latent를 복원할 수 있다는 가정, 복잡한 shaped reward와 quality threshold mask를 사용한다.",
        "strengths": "training/execution을 구분하고 결과 조작 금지를 명시하며 폭넓은 baseline·metric 계획을 제시한다.",
        "weaknesses": "latent L2는 decision-aligned가 아니고 actor는 MLP다. 2-hop forwardability, rich reward, quality-threshold mask가 강한 heuristic이며 결과와 저자 정보가 없다.",
        "fragile": "동일 local history로 global context를 복원할 수 있는지가 핵심인데 irreducible information loss와 predictor error를 분리하지 않는다.",
        "connection": "GLOBE++의 직접 전신이다. policy KL, local ego-graph GNN, invalid-action-only mask, heuristic 제거, partial observability 정식화가 명확한 변경점이다.",
        "supports": "내부 설계 이력과 실험 계획에만 사용할 수 있고 외부 근거는 아니다.",
        "cannot": "경험적 우월성이나 peer-reviewed novelty를 뒷받침할 수 없다.",
    },
    "Routing Algorithms Based on Reinforcement Learning for Unmanned Aerial Vehicle Swarm Networks": {
        "summary": "이 박사학위논문은 JTCR, QRIFC, JTFR의 세 시스템으로 mobility·topology·resource control과 UAV packet routing을 결합한다.",
        "problem": "동적 링크, 지연, 간섭, 에너지 제약 아래 swarm mobility와 topology, 무선 자원, packet routing을 함께 관리한다.",
        "method": "JTCR은 virtual-force mobility, fuzzy clustering, Q-routing을 결합한다. QRIFC는 adaptive flocking과 2-hop Q-learning을, JTFR은 LSTM actor와 multi-head attention critic의 DMA-DDPG를 사용한다.",
        "training": "시스템별 online Q-learning 또는 DMA-DDPG를 사용하며 link duration, transmission success, SINR, queue delay, residual energy를 포함한 다목적 reward를 설계한다.",
        "execution": "분산·로컬 의사결정이 최대 2-hop neighbor 정보와 mobility/topology control coordination을 사용한다.",
        "experiment": "node 수와 velocity를 변화시키는 custom simulation에서 mission·communication metric을 측정하고 각 제안 시스템을 별도 baseline과 비교한다.",
        "results": "JTCR은 PDR 7~21%, delay 9~37%, energy 15~23% 개선을, QRIFC는 PDR 9~23%, delay 21~40% 개선을, JTFR은 PDR 15~32%, delay 30~60%, energy 20~46% 개선을 보고한다.",
        "theory": "최적화 식과 복잡도 논의는 있으나 Dec-POMDP information gap이나 policy distillation theory는 없다.",
        "assumptions": "swarm behavior mobility, 2-hop control exchange, predictive link duration, 다목적 reward weight, custom simulation을 가정한다.",
        "strengths": "FANET routing을 폭넓게 다루고 여러 알고리즘, coverage·connectivity·overhead·energy metric을 포함한다.",
        "weaknesses": "모듈이 많아 인과 기여를 분리하기 어렵고 수작업 mobility·reward가 많다. 전체 기여를 같은 modern MARL baseline으로 비교하지 않는다.",
        "fragile": "routing learner보다 topology control과 2-hop 정보가 성능을 만들 수 있으며 joint mobility는 baseline의 task distribution 자체를 바꾼다.",
        "connection": "GLOBE++의 주요 설계·baseline 출처다. 2-hop predictive feature와 다목적 reward의 효과를 인정하되 정보 비용과 clean ablation이 필요하다.",
        "supports": "FANET 설계 이슈, metric, Q-learning/MARL baseline, topology-control confound, 2-hop overhead 질문을 뒷받침한다.",
        "cannot": "global-to-local policy distillation, local GNN novelty, irreducible observation-gap theory는 뒷받침하지 못한다.",
    },
}

HEADINGS = {
    "## 1. One-line Summary": "## 1. 한 줄 요약",
    "## 2. Problem Setting": "## 2. 문제 설정",
    "## 3. Core Contribution": "## 3. 핵심 기여",
    "### Paper Claim": "### 논문 주장",
    "### Agent Assessment": "### Agent 평가",
    "## 4. Method": "## 4. 방법",
    "### 4.1 Model / Algorithm": "### 4.1 모델 / 알고리즘",
    "### 4.2 Training Setup": "### 4.2 학습 설정",
    "### 4.3 Execution Assumption": "### 4.3 실행 단계 가정",
    "## 5. Experimental Setup": "## 5. 실험 설정",
    "## 6. Main Results": "## 6. 주요 결과",
    "## 7. Theoretical Claims": "## 7. 이론적 주장",
    "## 8. Assumptions": "## 8. 가정",
    "## 9. Limitations": "## 9. 한계",
    "## 10. Reviewer-level Critique": "## 10. Reviewer 관점 비판",
    "### Strengths": "### 강점",
    "### Weaknesses": "### 약점",
    "### Hidden Fragile Assumptions": "### 숨겨진 취약 가정",
    "### Reproducibility Risk": "### 재현성 위험",
    "## 11. Connection to GLOBE++": "## 11. GLOBE++와의 연결",
    "## 12. What This Paper Can Support in My Manuscript": "## 12. 내 논문에서 뒷받침할 수 있는 내용",
    "## 13. What This Paper Cannot Support": "## 13. 뒷받침할 수 없는 내용",
    "## 14. Follow-up Questions": "## 14. 후속 질문",
    "## 15. Source Evidence": "## 15. 출처 근거",
    "### Source Files": "### 원본 파일",
}

GENERIC = {
    "# Research Wiki Index": "# 연구 Wiki 인덱스",
    "# Research Status": "# 연구 상태",
    "# Wiki Lint Report": "# Wiki 품질 검사 보고서",
    "# GLOBE++ Novelty Claims": "# GLOBE++ 신규성 주장",
    "# GLOBE++ Ablation Plan": "# GLOBE++ Ablation 계획",
    "# GLOBE++ Algorithm Design": "# GLOBE++ 알고리즘 설계",
    "# GLOBE++ Baseline Matrix": "# GLOBE++ Baseline 매트릭스",
    "# GLOBE++ Experiment Plan": "# GLOBE++ 실험 계획",
    "# GLOBE++ Problem Formulation": "# GLOBE++ 문제 정식화",
    "# GLOBE++ Reviewer Attack List": "# GLOBE++ Reviewer 공격 목록",
    "# GLOBE++ Theory Notes": "# GLOBE++ 이론 노트",
    "# GLOBE++ Threat Model": "# GLOBE++ 위협 모델",
    "# Literature Matrix": "# 문헌 매트릭스",
    "# Baseline Comparison": "# Baseline 비교",
    "# Distillation Comparison": "# 증류 방식 비교",
    "# Method Comparison Matrix": "# 방법 비교 매트릭스",
    "# Routing Protocol Comparison": "# 라우팅 프로토콜 비교",
    "# Experiment Registry": "# 실험 레지스트리",
    "# Experiment Metrics": "# 실험 지표",
    "# Result Interpretation": "# 결과 해석",
    "# Simulator Notes": "# Simulator 노트",
    "# Manuscript Discussion": "# 원고 Discussion",
    "# Manuscript Experiments": "# 원고 Experiments",
    "# Manuscript Introduction": "# 원고 Introduction",
    "# Manuscript Method": "# 원고 Method",
    "# Manuscript Outline": "# 원고 개요",
    "# Manuscript Problem Formulation": "# 원고 문제 정식화",
    "# Manuscript Related Work": "# 원고 Related Work",
    "# Manuscript Theory": "# 원고 Theory",
    "# Active Ideas": "# 진행 중인 아이디어",
    "# Idea Inbox": "# 아이디어 수집함",
    "# Rejected Ideas": "# 기각된 아이디어",
    "# Claim Template": "# 주장 템플릿",
    "# Comparison Template": "# 비교 템플릿",
    "# Concept Template": "# 개념 템플릿",
    "# Experiment Template": "# 실험 템플릿",
    "# Manuscript Section Template": "# 원고 섹션 템플릿",
    "# Paper Card Template": "# 논문 카드 템플릿",
    "## 1. Definition": "## 1. 정의",
    "## 2. Why It Matters": "## 2. 중요성",
    "## 3. Technical Details": "## 3. 기술적 세부사항",
    "## 4. Mathematical Formulation": "## 4. 수학적 정식화",
    "## 5. Relevance to FANET Routing": "## 5. FANET 라우팅과의 관련성",
    "## 6. Relevance to GLOBE++": "## 6. GLOBE++와의 관련성",
    "## 7. Common Misuse or Misinterpretation": "## 7. 흔한 오용 또는 오해",
    "## 8. Related Papers": "## 8. 관련 논문",
    "## 9. Open Questions": "## 9. 열린 질문",
    "## 10. Source Evidence": "## 10. 출처 근거",
    "TODO: add a source-backed definition.": "TODO: 출처가 뒷받침하는 정의를 추가한다.",
    "TODO: needs-verification.": "TODO: 검증이 필요하다.",
    "TODO in ": "TODO: 다음 문서에서 정리한다: ",
    "TODO: define": "TODO: 정의한다:",
    "TODO: add": "TODO: 추가한다:",
    "TODO: document": "TODO: 문서화한다:",
    "TODO: specify": "TODO: 명시한다:",
    "TODO: distinguish": "TODO: 구분한다:",
    "TODO: record": "TODO: 기록한다:",
    "TODO: verify": "TODO: 검증한다:",
    "TODO: assess": "TODO: 평가한다:",
    "TODO: connect": "TODO: 연결한다:",
    "TODO: map": "TODO: 대응시킨다:",
    "TODO: identify": "TODO: 식별한다:",
    "TODO: no result should be recorded without evidence.": "TODO: 근거 없이 결과를 기록하지 않는다.",
    "Source Evidence": "출처 근거",
    "Paper Claim": "논문 주장",
    "Agent Assessment": "Agent 평가",
    "Needs Verification": "검증 필요",
    "Research Dashboard": "연구 대시보드",
    "Active Research Question": "현재 연구 질문",
    "Current GLOBE++ Claims": "현재 GLOBE++ 주장",
    "Reading Queue": "읽기 대기열",
    "Recently Ingested Sources": "최근 처리한 원본",
    "Weak Claims Needing Evidence": "근거가 부족한 주장",
    "Experiments to Run": "수행할 실험",
    "Concepts Needing Cleanup": "정리가 필요한 개념",
    "Reviewer Risks": "Reviewer 위험",
    "No issues found.": "문제가 발견되지 않았다.",
    "## Current Phase": "## 현재 단계",
    "Wiki initialized; evidence collection and formal claim validation are pending.": "Wiki 초기화는 완료되었으며 근거 수집과 정식 주장 검증이 남아 있다.",
    "## Active Workstreams": "## 진행 중인 연구 흐름",
    "## Evidence Gaps": "## 근거 공백",
    "- No source-backed novelty comparison yet.": "- 출처 기반 novelty 비교가 아직 없다.",
    "- No multi-seed experimental results yet.": "- multi-seed 실험 결과가 아직 없다.",
    "- No verified KL-to-performance bound yet.": "- 검증된 KL-to-performance bound가 아직 없다.",
    "Can a decentralized UAV routing policy, observing only a local ego-graph, reliably approximate a global topology-aware teacher policy under dynamic FANET conditions, and can we quantify the remaining performance gap caused by partial observability?": "로컬 ego-graph만 관측하는 분산 UAV 라우팅 정책이 동적 FANET에서 전역 topology-aware teacher 정책을 신뢰성 있게 근사할 수 있는가? 또한 부분 관측 때문에 남는 성능 격차를 정량화할 수 있는가?",
    "See [[GLOBE++ Problem Formulation]] and [[GLOBE++ Theory Notes]].": "[[GLOBE++ Problem Formulation]]과 [[GLOBE++ Theory Notes]]를 참조한다.",
    "## FANET Routing as a Dec-POMDP": "## FANET 라우팅의 Dec-POMDP 정식화",
    "Model decentralized [[UAV Routing]] in a dynamic [[FANET]] as a [[Dec-POMDP]]. This is a research specification, not yet a source-backed theoretical claim.": "동적 [[FANET]]의 분산 [[UAV Routing]]을 [[Dec-POMDP]]로 모델링한다. 현재는 연구 명세이며 출처로 검증된 이론적 주장은 아니다.",
    "## Key Separation": "## 핵심 분리",
    "- Global teacher:": "- 전역 teacher:",
    "- Local student:": "- 로컬 student:",
    "- Optimization and model capacity induce a separate imitation error.": "- 최적화와 모델 용량은 별도의 모방 오차를 만든다.",
    "## Open Decisions": "## 결정이 필요한 항목",
    "- Packet-level versus periodic decision process.": "- 패킷 단위 의사결정과 주기적 의사결정 중 무엇을 사용할지 결정한다.",
    "- Shared versus agent-specific policy.": "- 공유 정책과 agent별 정책 중 무엇을 사용할지 결정한다.",
    "- How stale neighbor information enters the observation.": "- 오래된 이웃 정보가 관측에 어떻게 포함되는지 정의한다.",
    "- Whether reward is shared, local, or mixed.": "- 보상을 전역 공유, 로컬, 혼합 중 어떻게 구성할지 결정한다.",
    "- How invalid action masks preserve comparable teacher/student support.": "- invalid action mask 이후 teacher와 student의 action support를 일치시키는 방법을 정의한다.",
    "## Evidence Needed": "## 필요한 근거",
    "TODO: source citations for the Dec-POMDP formulation and simulator-specific state transitions.": "TODO: Dec-POMDP 정식화와 simulator별 상태 전이에 대한 출처를 추가한다.",
    "## Claim 1: Behavior-level Policy Distillation": "## 주장 1: 행동 수준 정책 증류",
    "Replace latent matching with action-distribution alignment using [[Policy Distillation]] and [[KL Divergence]].": "[[Latent Distillation]] 대신 [[Policy Distillation]]과 [[KL Divergence]]를 이용해 행동 분포를 정렬한다.",
    "## Claim 2: Local Ego-Graph GNN Execution": "## 주장 2: 로컬 Ego-Graph GNN 실행",
    "Use a lightweight [[Graph Neural Network]] over an [[Ego-Graph]] as the executable decentralized policy.": "[[Ego-Graph]] 위의 경량 [[Graph Neural Network]]를 실제 분산 실행 정책으로 사용한다.",
    "## Claim 3: Partial Observability-Aware Distillation Gap": "## 주장 3: 부분 관측을 고려한 증류 격차",
    "Separate the irreducible information gap caused by [[Partial Observability]] from reducible imitation error.": "[[Partial Observability]]에서 발생하는 비가역적 정보 격차와 줄일 수 있는 모방 오차를 분리한다.",
    "## Claim 4: Heuristic-Free Decentralized Routing": "## 주장 4: Heuristic-free 분산 라우팅",
    "Remove the 2-hop routing-score heuristic while retaining only invalid-action masking.": "2-hop routing score heuristic을 제거하고 invalid-action masking만 유지한다.",
    "## Reviewer Burden": "## Reviewer 관점의 입증 부담",
    "- Demonstrate that this is more than a combination of known techniques.": "- 알려진 기술을 단순 조합한 것 이상임을 입증한다.",
    "- Compare against [[Latent Distillation]], no-KD, MLP, GNN, and heuristic variants.": "- [[Latent Distillation]], no-KD, MLP, GNN, heuristic variant와 비교한다.",
    "- Specify exactly what information the global teacher receives.": "- global teacher가 받는 정보를 정확히 명시한다.",
    "- Avoid claiming an oracle upper bound without a defensible definition.": "- 방어 가능한 정의 없이 oracle upper bound라고 주장하지 않는다.",
    "## Required Evidence": "## 필요한 근거",
    "See [[GLOBE++ Baseline Matrix]], [[GLOBE++ Ablation Plan]], and [[GLOBE++ Reviewer Attack List]].": "[[GLOBE++ Baseline Matrix]], [[GLOBE++ Ablation Plan]], [[GLOBE++ Reviewer Attack List]]를 참조한다.",
    "## Initial Literature Audit": "## 초기 문헌 검토",
    "The first reviewed corpus is synthesized in [[Initial FANET Routing Literature Synthesis]].": "첫 번째 검토 문헌 묶음은 [[Initial FANET Routing Literature Synthesis]]에 종합했다.",
    "## Global GNN Teacher": "## 전역 GNN Teacher",
    "- Input: global dynamic graph and packet/routing context.": "- 입력: 전역 동적 그래프와 패킷·라우팅 context.",
    "- Output: masked next-hop distribution": "- 출력: mask가 적용된 next-hop 분포",
    "- Training: candidate [[MAPPO]]/[[CTDE]] setup; details TODO.": "- 학습: [[MAPPO]]/[[CTDE]] 후보 설정. 세부사항은 TODO.",
    "## Local GNN Student": "## 로컬 GNN Student",
    "- Input: bounded [[Ego-Graph]] and locally available packet context.": "- 입력: 제한된 [[Ego-Graph]]와 로컬에서 이용 가능한 패킷 context.",
    "- Deployment: decentralized, without global graph access.": "- 배포: 전역 그래프 접근 없이 분산 실행한다.",
    "## Policy Distillation": "## 정책 증류",
    "Candidate objective:": "후보 목적함수:",
    "TODO: 검증한다: KL direction, temperature, coefficient, and occupancy distribution.": "TODO: KL 방향, temperature, 계수, occupancy distribution을 검증한다.",
    "## On-policy KD": "## On-policy KD",
    "Collect student-generated states, query the teacher during training, and optimize on the resulting [[Student-induced Distribution]].": "student가 생성한 상태를 수집하고 학습 중 teacher에 질의하여 [[Student-induced Distribution]]에서 최적화한다.",
    "## Invalid Action Masking": "## Invalid Action Masking",
    "Use masks only for structurally invalid actions. Do not reintroduce a hand-crafted routing score through the mask.": "구조적으로 불가능한 행동에만 mask를 적용한다. mask를 통해 수작업 routing score를 다시 도입하지 않는다.",
    "## Training Objective": "## 학습 목적함수",
    "TODO: 명시한다: whether the student receives RL loss, KD loss, or both:": "TODO: student가 RL loss, KD loss 또는 둘 다 받는지 명시한다.",
    "## Execution Audit": "## 실행 단계 감사",
    "Every student feature must be locally measurable and its communication cost counted in [[Routing Overhead]].": "모든 student feature는 로컬에서 측정 가능해야 하며 통신 비용을 [[Routing Overhead]]에 포함해야 한다.",
    "## Observation Mismatch": "## 관측 불일치",
    "The teacher conditions on global graph $G$, while the student conditions on local observation $o$. TODO: formalize observation equivalence classes where multiple global states map to the same local observation.": "teacher는 전역 그래프 $G$에 조건화되고 student는 로컬 관측 $o$에 조건화된다. 여러 전역 상태가 동일 로컬 관측으로 매핑되는 관측 동치류를 정식화한다.",
    "## Irreducible Distillation Gap": "## 비가역적 증류 격차",
    "Candidate decomposition:": "후보 분해:",
    "This notation is provisional. TODO: 정의한다: the loss, conditioning distribution, and decomposition rigorously.": "이 표기는 잠정적이다. loss, 조건 분포, 분해를 엄밀하게 정의한다.",
    "## KL-Performance Connection": "## KL과 성능의 연결",
    "Candidate route:": "후보 유도 경로:",
    "## Student-induced Distribution": "## Student-induced Distribution",
    "Offline teacher trajectories may not cover states reached by the student. Consider on-policy queries under [[Student-induced Distribution]].": "offline teacher trajectory는 student가 도달하는 상태를 포함하지 못할 수 있다. [[Student-induced Distribution]]에서 on-policy teacher query를 고려한다.",
    "## Required Checks": "## 필수 검증 항목",
    "- Direction of KL.": "- KL 방향.",
    "- Action-mask support.": "- action mask 이후 support.",
    "- Joint versus per-agent policy.": "- joint policy와 agent별 policy의 차이.",
    "- Finite horizon versus discounted infinite horizon.": "- finite horizon과 discounted infinite horizon의 차이.",
    "- Teacher disagreement conditional on identical local observations.": "- 동일 로컬 관측에 조건화된 teacher disagreement.",
    "## Prohibited Shortcut": "## 금지할 단축 논리",
    "Do not state a performance guarantee until every assumption and source is recorded.": "모든 가정과 출처를 기록하기 전에는 성능 보장을 주장하지 않는다.",
    "## Baselines": "## Baseline",
    "## Metrics": "## 지표",
    "## Multi-seed Protocol": "## Multi-seed 절차",
    "- Separate training and evaluation seeds.": "- 학습 seed와 평가 seed를 분리한다.",
    "- Pre-register seed count and aggregation before results.": "- 결과 확인 전에 seed 수와 집계 방식을 사전 등록한다.",
    "- Report mean, dispersion, confidence intervals, and paired tests when justified.": "- 평균, 산포, 신뢰구간, 필요 시 paired test를 보고한다.",
    "- Keep environment budgets comparable across learned methods.": "- 학습 방법 간 환경 interaction budget을 동일하게 맞춘다.",
    "## Scalability": "## 확장성",
    "Test node counts outside the training range and report inference/communication cost.": "학습 범위 밖의 node 수를 시험하고 inference·communication cost를 보고한다.",
    "## OOD Tests": "## OOD 시험",
    "Vary mobility, density, traffic, radio/link conditions, and failure patterns. See [[OOD Generalization]].": "mobility, density, traffic, radio/link 조건, failure pattern을 변화시킨다. [[OOD Generalization]]을 참조한다.",
    "## Ablations": "## Ablation",
    "## Reproducibility": "## 재현성",
    "Record simulator version, scenario generation, hyperparameters, checkpoints, seeds, hardware, and evaluation scripts.": "simulator version, scenario 생성법, hyperparameter, checkpoint, seed, hardware, 평가 script를 기록한다.",
    "## Literature-derived Stress Tests": "## 문헌에서 도출한 Stress Test",
    "## Source-backed Candidates": "## 출처 기반 후보",
    "## Fairness Checks": "## 공정성 점검",
    "- Comparable training steps and tuning budget.": "- 학습 step과 tuning budget을 동일하게 맞춘다.",
    "- Same simulator scenarios and action validity rules.": "- 동일 simulator scenario와 action validity rule을 사용한다.",
    "- Report parameter count and information access.": "- parameter 수와 접근 정보를 보고한다.",
    "- Do not label global teacher an upper bound without proof.": "- 증명 없이 global teacher를 upper bound라고 부르지 않는다.",
    "## Manuscript Rule": "## 원고 작성 규칙",
    "Do not hide unresolved attacks. Route them to [[GLOBE++ Experiment Plan]], [[GLOBE++ Theory Notes]], or the discussion section.": "해결되지 않은 공격 지점을 숨기지 않는다. [[GLOBE++ Experiment Plan]], [[GLOBE++ Theory Notes]], discussion으로 연결한다.",
    "## Configuration TODO": "## 설정 TODO",
    "- Simulator name and version.": "- Simulator 이름과 version.",
    "- Mobility, propagation, MAC, queue, and traffic models.": "- Mobility, propagation, MAC, queue, traffic model.",
    "- Time step and policy decision frequency.": "- Time step과 policy decision 주기.",
    "- Failure and reconnection behavior.": "- Failure와 reconnection 동작.",
    "- Seed control and scenario serialization.": "- Seed 제어와 scenario serialization.",
    "## Reporting Rules": "## 결과 해석 규칙",
    "- Separate observed result from causal interpretation.": "- 관측 결과와 인과 해석을 분리한다.",
    "- Report variance and failed seeds.": "- 분산과 실패한 seed를 보고한다.",
    "- Discuss metric tradeoffs, not only wins.": "- 승리한 지표뿐 아니라 지표 간 trade-off를 논의한다.",
    "- Distinguish in-distribution and [[OOD Generalization]].": "- in-distribution과 [[OOD Generalization]]을 구분한다.",
    "- Connect each result to a claim and [[Ablation Study]].": "- 각 결과를 주장과 [[Ablation Study]]에 연결한다.",
    "- Mark unexpected results and alternative explanations.": "- 예상 밖 결과와 대안 설명을 표시한다.",
    "Use [[GLOBE++ Baseline Matrix]] as the active design table.": "[[GLOBE++ Baseline Matrix]]를 현재 설계 표로 사용한다.",
    "See [[GLOBE++ Baseline Matrix]].": "[[GLOBE++ Baseline Matrix]]를 참조한다.",
    "See [[GLOBE++ Ablation Plan]].": "[[GLOBE++ Ablation Plan]]을 참조한다.",
    "See [[GLOBE++ Experiment Plan]] and [[GLOBE++ Reviewer Attack List]].": "[[GLOBE++ Experiment Plan]]과 [[GLOBE++ Reviewer Attack List]]를 참조한다.",
    "See [[ns-3 Notes]] and [[GLOBE++ Threat Model]].": "[[ns-3 Notes]]와 [[GLOBE++ Threat Model]]을 참조한다.",
    "See [[GLOBE++ Baseline Matrix]].": "[[GLOBE++ Baseline Matrix]]를 참조한다.",
    "See [[Literature Matrix]] and [[Method Comparison Matrix]].": "[[Literature Matrix]]와 [[Method Comparison Matrix]]를 참조한다.",
    "No source-based claim should enter these pages without citation metadata.": "인용 메타데이터가 없는 출처 기반 주장을 원고에 넣지 않는다.",
    "All ideas require literature and experimental validation.": "모든 아이디어는 문헌과 실험 검증이 필요하다.",
    "Record rejected ideas with date, reason, evidence, and conditions for reconsideration.": "기각한 아이디어의 날짜, 이유, 근거, 재검토 조건을 기록한다.",
    "- Conditional teacher disagreement as an empirical proxy for irreducible local-information gap.": "- 조건부 teacher disagreement를 비가역적 로컬 정보 격차의 경험적 proxy로 사용한다.",
    "- On-policy KD under [[Student-induced Distribution]].": "- [[Student-induced Distribution]]에서 on-policy KD를 수행한다.",
    "- Cost-aware [[Ego-Graph]] radius selection.": "- 비용을 고려해 [[Ego-Graph]] 반경을 선택한다.",
    "- TODO: capture unreviewed research ideas here before promotion.": "- TODO: 검토되지 않은 연구 아이디어를 승격 전에 여기에 기록한다.",
    "> All claims are provisional until the literature matrix and experiments provide evidence.": "> 모든 주장은 문헌 매트릭스와 실험 근거가 확보될 때까지 잠정적이다.",
    "**Agent 평가:** These claims must remain provisional. Absence from this small corpus is not novelty proof.": "**Agent 평가:** 이 주장들은 잠정적으로 유지해야 한다. 소규모 문헌 묶음에서 발견되지 않았다는 사실은 신규성의 증명이 아니다.",
    "## In Scope": "## 범위에 포함",
    "## Needs Decision": "## 결정 필요",
    "## Assumption Audit": "## 가정 점검",
    "## Evaluation Links": "## 평가 연결",
    "- Mobility-driven link changes.": "- 이동성에 따른 링크 변화.",
    "- Packet loss, queue variation, and delayed local observations.": "- 패킷 손실, queue 변화, 지연된 로컬 관측.",
    "- Node-density and traffic shifts.": "- 노드 밀도와 트래픽 변화.",
    "- Local topology aliasing under [[Partial Observability]].": "- [[Partial Observability]]에서의 로컬 토폴로지 aliasing.",
    "- Benign versus adversarial UAV behavior.": "- 정상 UAV와 적대적 UAV 행동의 구분.",
    "- Position spoofing or sensor errors.": "- 위치 spoofing 또는 센서 오류.",
    "- Compromised nodes.": "- 침해된 노드.",
    "- Jamming and denial of service.": "- Jamming 및 서비스 거부 공격.",
    "- Byzantine routing announcements.": "- Byzantine 라우팅 공지.",
    "Use the same seed protocol and tuning budget for each [[Ablation Study]].": "모든 [[Ablation Study]]에 동일한 seed protocol과 tuning budget을 사용한다.",
    "See [[Initial FANET Routing Literature Synthesis]] for evidence quality and caveats.": "근거 품질과 주의사항은 [[Initial FANET Routing Literature Synthesis]]를 참조한다.",
    "Reviewer-facing details should feed [[GLOBE++ Reviewer Attack List]].": "Reviewer에게 제시할 세부사항은 [[GLOBE++ Reviewer Attack List]]에 반영한다.",
    "## Evidence TODO": "## 근거 TODO",
    "- Verify protocol configurations.": "- 프로토콜 설정을 검증한다.",
    "- Verify learned-baseline tuning budgets.": "- 학습 baseline의 tuning budget을 검증한다.",
    "- Record simulator compatibility and information access.": "- simulator 호환성과 정보 접근 범위를 기록한다.",
    "TODO: attach paper evidence before making comparative claims.": "TODO: 비교 주장을 하기 전에 논문 근거를 연결한다.",
    "TODO: limitations, observation gap, simulator validity, failure cases, deployment cost, and reviewer risks.": "TODO: 한계, 관측 격차, simulator 타당성, 실패 사례, 배포 비용, reviewer 위험을 정리한다.",
    "TODO: report registered settings and completed results from [[Experiment Registry]] and [[GLOBE++ Experiment Plan]].": "TODO: [[Experiment Registry]]와 [[GLOBE++ Experiment Plan]]에 등록된 설정과 완료된 결과를 보고한다.",
    "TODO: evidence-backed motivation, gap, research question, and contribution summary.": "TODO: 근거 기반 동기, 연구 공백, 연구 질문, 기여 요약을 작성한다.",
    "TODO: derive the final method from [[GLOBE++ Algorithm Design]], including masks, objectives, and execution information.": "TODO: mask, 목적함수, 실행 정보를 포함해 [[GLOBE++ Algorithm Design]]에서 최종 방법을 도출한다.",
    "TODO: convert [[GLOBE++ Problem Formulation]] into source-backed notation and assumptions.": "TODO: [[GLOBE++ Problem Formulation]]을 출처 기반 표기와 가정으로 변환한다.",
    "TODO: synthesize verified paper cards by routing, MARL, graph policy, and distillation themes.": "TODO: 검증된 논문 카드를 routing, MARL, graph policy, distillation 주제별로 종합한다.",
    "TODO: include only fully derived and source-backed results from [[GLOBE++ Theory Notes]].": "TODO: [[GLOBE++ Theory Notes]]에서 완전히 유도되고 출처가 확인된 결과만 포함한다.",
    "Use [[GLOBE++ Reviewer Attack List]] and [[GLOBE++ Threat Model]].": "[[GLOBE++ Reviewer Attack List]]와 [[GLOBE++ Threat Model]]을 사용한다.",
    "Use [[GLOBE++ Novelty Claims]] only after literature validation.": "문헌 검증 후에만 [[GLOBE++ Novelty Claims]]를 사용한다.",
    "## Scope": "## 범위",
    "## Comparison": "## 비교",
    "## Purpose": "## 목적",
    "## Evidence-backed Draft": "## 근거 기반 초안",
    "## Claims Used": "## 사용한 주장",
    "## Figures and Tables": "## 그림과 표",
    "TODO: every source-based claim requires source file, path, hash, and page or section.": "TODO: 출처 기반 주장마다 원본 파일, 경로, hash, 페이지 또는 섹션을 기록한다.",
    "TODO: 기록한다: version, modules, routing integration, wireless model, energy model, tracing, reproducibility, and known simulator limitations.": "TODO: version, module, routing integration, wireless model, energy model, tracing, 재현성, 알려진 simulator 한계를 기록한다.",
    "TODO: 검증한다: every protocol property from sources and implementation.": "TODO: 모든 protocol 속성을 출처와 구현에서 검증한다.",
    "TODO: 기록한다: which local features are trusted, synchronized, and available without hidden global communication.": "TODO: 어떤 local feature가 신뢰 가능하고 동기화되며 숨겨진 global communication 없이 이용 가능한지 기록한다.",
    "## 1. Claim Statement": "## 1. 주장 문장",
    "## 2. Why This Claim Matters": "## 2. 주장의 중요성",
    "## 3. Evidence Supporting This Claim": "## 3. 주장을 뒷받침하는 근거",
    "## 4. Evidence Against This Claim": "## 4. 주장에 반하는 근거",
    "## 5. Reviewer Attack": "## 5. Reviewer 공격",
    "## 6. Defense Strategy": "## 6. 방어 전략",
    "## 7. Required Experiment": "## 7. 필요한 실험",
    "## 8. Required Theory": "## 8. 필요한 이론",
    "## 9. Manuscript Location": "## 9. 원고 위치",
    "## 1. Purpose": "## 1. 목적",
    "## 2. Hypothesis": "## 2. 가설",
    "## 3. Claim Tested": "## 3. 검증할 주장",
    "## 4. Environment": "## 4. 환경",
    "## 5. Compared Methods": "## 5. 비교 방법",
    "## 6. Metrics": "## 6. 지표",
    "## 7. Expected Result": "## 7. 예상 결과",
    "## 8. Failure Case": "## 8. 실패 사례",
    "## 9. Result Summary": "## 9. 결과 요약",
    "## 10. Interpretation": "## 10. 해석",
    "## 11. Figure/Table Output": "## 11. 그림·표 출력",
    "## 12. Manuscript Usage": "## 12. 원고 활용",
    "## 15. Source Notes": "## 15. 출처 메모",
    "- Source file:": "- 원본 파일:",
    "- Source path:": "- 원본 경로:",
    "- Source hash:": "- 원본 해시:",
    "- Page/section references:": "- 페이지/섹션 참조:",
    "- Introduction:": "- Introduction:",
    "- Related Work:": "- Related Work:",
    "- Method:": "- Method:",
    "- Theory:": "- Theory:",
    "- Experiments:": "- Experiments:",
    "- Discussion:": "- Discussion:",
    "| Item | Description |": "| 항목 | 설명 |",
    "| Item | Setting |": "| 항목 | 설정 |",
    "| Aspect | Relevance |": "| 관점 | 관련성 |",
    "| Evidence | Source | Strength | Notes |": "| 근거 | 출처 | 강도 | 메모 |",
    "| Evidence | Source | Risk | Notes |": "| 근거 | 출처 | 위험 | 메모 |",
    "| Method | Assumptions | Strengths | Weaknesses | Evidence | GLOBE++ Relevance |": "| 방법 | 가정 | 강점 | 약점 | 근거 | GLOBE++ 관련성 |",
    "| Approach | Matched Object | Decision Alignment | Observation Mismatch Risk | GLOBE++ Role |": "| 접근법 | 정렬 대상 | 의사결정 정렬 | 관측 불일치 위험 | GLOBE++ 역할 |",
    "| Method | Training Information | Execution Information | Policy Model | Objective | Key Risk |": "| 방법 | 학습 정보 | 실행 정보 | 정책 모델 | 목적함수 | 핵심 위험 |",
    "| Protocol | Family | Required State | Adaptation | Metrics to Report |": "| 프로토콜 | 계열 | 필요한 상태 | 적응 방식 | 보고 지표 |",
    "| ID | Experiment | Claim | Status | Seeds | Artifact |": "| ID | 실험 | 주장 | 상태 | Seed | 산출물 |",
    "| Metric | Definition Status | Aggregation | Caveat |": "| 지표 | 정의 상태 | 집계 | 주의사항 |",
    "| Axis | Variants | Question |": "| 축 | 변형 | 질문 |",
    "| Method | Purpose | Information at Execution | Distillation | Heuristic | Verification |": "| 방법 | 목적 | 실행 시 정보 | 증류 | Heuristic | 검증 |",
    "| Candidate | Source | Why Include |": "| 후보 | 출처 | 포함 이유 |",
    "| Component | GLOBE++ Candidate Definition | Status |": "| 구성요소 | GLOBE++ 후보 정의 | 상태 |",
    "| Attack | Why It Is Plausible | Required Defense |": "| 공격 지점 | 가능한 이유 | 필요한 방어 |",
    "| Claim | Current Evidence | Status |": "| 주장 | 현재 근거 | 상태 |",
    "| Priority | Source | Reason | Status |": "| 우선순위 | 출처 | 이유 | 상태 |",
    "| Simulator | |": "| Simulator | |",
    "| Number of nodes | |": "| 노드 수 | |",
    "| Mobility model | |": "| 이동성 모델 | |",
    "| Traffic model | |": "| 트래픽 모델 | |",
    "| Baselines | |": "| Baseline | |",
    "| Metrics | |": "| 지표 | |",
    "| Seeds | |": "| Seed | |",
    "| Statistical reporting | |": "| 통계 보고 | |",
    "TODO: isolate one design choice while controlling training budget.": "TODO: 학습 예산을 통제하면서 하나의 설계 선택만 분리한다.",
    "Planned in [[GLOBE++ Ablation Plan]].": "[[GLOBE++ Ablation Plan]]에서 계획한다.",
    "TODO: under-tuned variants do not isolate causal contribution.": "TODO: tuning이 부족한 변형은 인과적 기여를 분리하지 못한다.",
    "- Which variants separate architecture, information, and objective effects?": "- 어떤 변형이 architecture, 정보, 목적함수 효과를 분리하는가?",
    "TODO: mean, median, tail percentiles, and confidence intervals.": "TODO: 평균, 중앙값, tail percentile, 신뢰구간을 정의한다.",
    "Measures timeliness under mobility and route breakage.": "이동성과 경로 단절 상황의 적시성을 측정한다.",
    "Required metric in [[GLOBE++ Experiment Plan]].": "[[GLOBE++ Experiment Plan]]의 필수 지표다.",
    "TODO: survivorship bias when only delivered packets are measured.": "TODO: 전달된 패킷만 측정할 때 발생하는 생존자 편향을 점검한다.",
    "- Which tail latency statistic is appropriate?": "- 어떤 tail latency 통계량이 적절한가?",
    "TODO: node count, mobility, traffic, radio, and topology shifts.": "TODO: 노드 수, 이동성, 트래픽, 무선 환경, 토폴로지 변화를 정의한다.",
    "Dynamic deployments can differ from training scenarios.": "동적 배포 환경은 학습 시나리오와 달라질 수 있다.",
    "Required evaluation axis in [[GLOBE++ Experiment Plan]].": "[[GLOBE++ Experiment Plan]]의 필수 평가 축이다.",
    "TODO: random seeds alone are not OOD evaluation.": "TODO: random seed 변화만으로는 OOD 평가가 되지 않는다.",
    "- Which shifts isolate graph-policy generalization?": "- 어떤 변화가 graph policy의 일반화 성능을 분리하는가?",
    "TODO: warm-up, duplicate packets, drops, and confidence intervals.": "TODO: warm-up, 중복 패킷, drop, 신뢰구간 처리 방식을 정의한다.",
    "Primary reliability metric.": "주요 신뢰성 지표다.",
    "- How should disconnected periods be counted?": "- 네트워크 단절 기간을 어떻게 집계해야 하는가?",
    "Captures protocol cost under dynamic topology.": "동적 토폴로지에서 프로토콜 비용을 측정한다.",
    "TODO: incomparable definitions across papers.": "TODO: 논문마다 다른 정의로 인해 직접 비교가 어려움을 명시한다.",
    "- Does ego-graph construction require control traffic?": "- ego-graph 구성에 제어 트래픽이 필요한가?",
    "This is the target network setting for [[UAV Routing]].": "이는 [[UAV Routing]]의 대상 네트워크 환경이다.",
    "Defines the operating environment in [[GLOBE++ Problem Formulation]].": "[[GLOBE++ Problem Formulation]]의 운용 환경을 정의한다.",
    "- Which mobility and radio models are representative?": "- 어떤 이동성·무선 모델이 대표성을 갖는가?",
    "TODO: construction cost, stale information, boundary effects.": "TODO: 구성 비용, 오래된 정보, 경계 효과를 정리한다.",
    "Candidate decentralized observation for [[UAV Routing]].": "[[UAV Routing]]을 위한 분산 관측 후보다.",
    "Student input in [[GLOBE++ Algorithm Design]].": "[[GLOBE++ Algorithm Design]]의 student 입력이다.",
    "- What hop radius is locally available without hidden overhead?": "- 숨겨진 overhead 없이 로컬에서 확보 가능한 hop 반경은 얼마인가?",
    "TODO: node, edge, graph features and permutation properties.": "TODO: node, edge, graph feature와 permutation 특성을 정리한다.",
    "TODO: topology representation and neighbor-level actions.": "TODO: 토폴로지 표현과 이웃 단위 행동을 정리한다.",
    "Used by global teacher and local [[Ego-Graph]] student in [[GLOBE++ Algorithm Design]].": "[[GLOBE++ Algorithm Design]]의 global teacher와 local [[Ego-Graph]] student가 사용한다.",
    "- What depth balances local receptive field and latency?": "- 어떤 깊이가 로컬 receptive field와 latency의 균형을 이루는가?",
    "TODO: objectives, temperature, online/offline data distributions.": "TODO: 목적함수, temperature, online/offline 데이터 분포를 정리한다.",
    "Parent concept for [[Policy Distillation]] in [[GLOBE++ Algorithm Design]].": "[[GLOBE++ Algorithm Design]]의 [[Policy Distillation]] 상위 개념이다.",
    "- Which information can be transferred under observation mismatch?": "- 관측 불일치 상황에서 어떤 정보를 전달할 수 있는가?",
    "Candidate baseline: $\\lVert z-\\hat z\\rVert_2^2$. Needs source and implementation verification.": "후보 baseline은 $\\lVert z-\\hat z\\rVert_2^2$이다. 출처와 구현 검증이 필요하다.",
    "Comparison target in [[Distillation Comparison]] and [[GLOBE++ Novelty Claims]].": "[[Distillation Comparison]]과 [[GLOBE++ Novelty Claims]]의 비교 대상이다.",
    "TODO: latent closeness does not automatically imply action agreement.": "TODO: latent representation의 근접성이 행동 일치를 자동으로 의미하지 않음을 명시한다.",
    "- Which representations are identifiable across teacher and student?": "- teacher와 student 사이에서 식별 가능한 representation은 무엇인가?",
    "TODO: compare decision alignment with [[Latent Distillation]].": "TODO: [[Latent Distillation]]과 의사결정 정렬 정도를 비교한다.",
    "TODO: direction of KL, action masks, temperature, and sampling distribution.": "TODO: KL 방향, action mask, temperature, sampling distribution을 정리한다.",
    "Central mechanism in [[GLOBE++ Novelty Claims]].": "[[GLOBE++ Novelty Claims]]의 핵심 메커니즘이다.",
    "- Forward or reverse [[KL Divergence]]?": "- Forward와 reverse [[KL Divergence]] 중 무엇을 사용할 것인가?",
    "TODO: action-space alignment, masking, and deployment assumptions.": "TODO: 행동 공간 정렬, masking, 배포 가정을 정리한다.",
    "Global-to-local architecture in [[GLOBE++ Algorithm Design]].": "[[GLOBE++ Algorithm Design]]의 global-to-local architecture다.",
    "TODO: teacher performance is not automatically a deployable upper bound.": "TODO: teacher 성능이 배포 가능한 upper bound를 자동으로 의미하지 않음을 명시한다.",
    "- How is teacher access restricted to training?": "- teacher 접근을 학습 단계로 어떻게 제한하는가?",
    "Training framework for [[GLOBE++ Algorithm Design]].": "[[GLOBE++ Algorithm Design]]의 학습 framework다.",
    "TODO: global information at execution would violate decentralized execution.": "TODO: 실행 단계의 global 정보 사용은 분산 실행 조건을 위반함을 명시한다.",
    "- Is the teacher part of CTDE or a separate privileged-information model?": "- teacher는 CTDE의 일부인가, 별도의 privileged-information model인가?",
    "TODO: actor sharing, critic inputs, clipping, advantage estimation.": "TODO: actor 공유, critic 입력, clipping, advantage estimation을 정리한다.",
    "Candidate optimization backbone in [[GLOBE++ Algorithm Design]].": "[[GLOBE++ Algorithm Design]]의 최적화 backbone 후보다.",
    "- Which distribution supplies the distillation samples?": "- 어떤 분포에서 distillation sample을 얻는가?",
    "TODO: forwarding action, candidate neighbors, loops, and link failures.": "TODO: forwarding 행동, 후보 이웃, loop, 링크 고장을 정리한다.",
    "See [[GLOBE++ Problem Formulation]].": "[[GLOBE++ Problem Formulation]]을 참조한다.",
    "Primary decision problem under [[FANET]] dynamics.": "[[FANET]] 동역학에서의 주요 의사결정 문제다.",
    "Target behavior of [[GLOBE++ Algorithm Design]].": "[[GLOBE++ Algorithm Design]]이 학습할 대상 행동이다.",
    "- Is the policy per-packet, per-flow, or periodic?": "- 정책은 packet별, flow별, 주기적 방식 중 무엇인가?",
    "Core framework for [[GLOBE++ Theory Notes]] and [[Partial Observability]].": "[[GLOBE++ Theory Notes]]와 [[Partial Observability]]의 핵심 framework다.",
    "- What information belongs in global state but not local observation?": "- global state에는 있지만 local observation에는 없는 정보는 무엇인가?",
    "TODO: forward/reverse direction, masks, zeros, and temperature.": "TODO: forward/reverse 방향, mask, 0 확률, temperature를 정리한다.",
    "Behavior-matching objective for [[Policy Distillation]].": "[[Policy Distillation]]의 행동 정렬 목적함수다.",
    "TODO: asymmetric directions have different behavior.": "TODO: 비대칭 KL 방향에 따라 동작이 달라짐을 명시한다.",
    "- How are invalid actions normalized before KL computation?": "- KL 계산 전에 invalid action을 어떻게 정규화하는가?",
    "TODO: observation aliasing and history dependence.": "TODO: observation aliasing과 history dependence를 정리한다.",
    "Local routers cannot directly observe all global topology dynamics.": "로컬 router는 전체 global topology dynamics를 직접 관측할 수 없다.",
    "Defines the irreducible gap studied by [[GLOBE++ Novelty Claims]].": "[[GLOBE++ Novelty Claims]]에서 다루는 제거 불가능한 격차를 정의한다.",
    "- What global distinctions are aliased by the same ego-graph?": "- 동일 ego-graph에 의해 구분되지 않는 global 상태 차이는 무엇인가?",
    "TODO: occupancy measure, advantage, horizon, and boundedness assumptions.": "TODO: occupancy measure, advantage, horizon, boundedness 가정을 정리한다.",
    "Candidate bridge from policy mismatch to return gap.": "policy mismatch와 return gap을 연결하는 후보 이론이다.",
    "TODO: do not claim a KL bound without deriving all inequalities and constants.": "TODO: 모든 부등식과 상수를 유도하기 전에는 KL bound를 주장하지 않는다.",
    "- Which occupancy distribution appears in the final bound?": "- 최종 bound에는 어떤 occupancy distribution이 사용되는가?",
    "TODO: distribution shift can separate offline imitation from deployed behavior.": "TODO: distribution shift가 offline imitation과 실제 배포 행동을 분리할 수 있음을 정리한다.",
    "Student forwarding decisions alter later topology encounters and packet paths.": "Student의 forwarding 결정은 이후 마주치는 토폴로지와 패킷 경로를 변화시킨다.",
    "Motivates on-policy distillation in [[GLOBE++ Algorithm Design]].": "[[GLOBE++ Algorithm Design]]에서 on-policy distillation이 필요한 이유다.",
    "- How should teacher labels be queried on student-generated trajectories?": "- student가 생성한 trajectory에서 teacher label을 어떻게 질의해야 하는가?",
    "| High | TODO | Evidence for policy-level distillation under observation mismatch | queued |": "| 높음 | TODO | 관측 불일치에서 policy-level distillation을 뒷받침하는 근거 | 대기 |",
    "| High | TODO | Dec-POMDP performance bounds under local observations | queued |": "| 높음 | TODO | 로컬 관측 조건의 Dec-POMDP 성능 bound | 대기 |",
    "| High | TODO | FANET routing baselines and reproducible simulator settings | queued |": "| 높음 | TODO | FANET routing baseline과 재현 가능한 simulator 설정 | 대기 |",
    "- Formalization:": "- 정식화:",
    "- Theory:": "- 이론:",
    "- Algorithm:": "- 알고리즘:",
    "- Evaluation:": "- 평가:",
    "- Manuscript:": "- 원고:",
    "| Distillation | none / [[Latent Distillation]] / [[Policy Distillation]] | Is behavior alignment responsible? |": "| 증류 | 없음 / [[Latent Distillation]] / [[Policy Distillation]] | 행동 정렬이 성능 원인인가? |",
    "| Data distribution | offline teacher / mixed / on-policy student | Does covariate shift matter? |": "| 데이터 분포 | offline teacher / 혼합 / on-policy student | covariate shift가 중요한가? |",
    "| Student | MLP / local GNN | Does graph structure improve execution? |": "| Student | MLP / local GNN | 그래프 구조가 실행 성능을 높이는가? |",
    "| Observation | 1-hop / 2-hop / bounded radius | How does information affect the gap? |": "| 관측 | 1-hop / 2-hop / 제한 반경 | 정보 범위가 격차에 어떤 영향을 주는가? |",
    "| Heuristic | original score / mask-only | Does heuristic removal preserve performance? |": "| Heuristic | 기존 score / mask-only | heuristic 제거 후에도 성능이 유지되는가? |",
    "| KD weight | scheduled values | Is performance robust to $\\lambda_{KD}$? |": "| KD 가중치 | 예정된 값 | 성능이 $\\lambda_{KD}$ 변화에 강건한가? |",
    "| Teacher input | full / restricted global context | Which privileged information matters? |": "| Teacher 입력 | 전체 / 제한된 global context | 어떤 privileged information이 중요한가? |",
    "| AODV | Reactive protocol reference | Local/protocol state | No | Protocol-defined | TODO |": "| AODV | reactive protocol 기준 | 로컬·protocol 상태 | 없음 | protocol 정의 | TODO |",
    "| OLSR | Proactive protocol reference | Distributed topology state | No | Protocol-defined | TODO |": "| OLSR | proactive protocol 기준 | 분산 topology 상태 | 없음 | protocol 정의 | TODO |",
    "| GPSR | Geographic routing reference | Position/neighbors | No | Geographic | TODO |": "| GPSR | geographic routing 기준 | 위치·이웃 | 없음 | geographic | TODO |",
    "| MAPPO-local | MARL control | Local vector | No | No | TODO |": "| MAPPO-local | MARL 대조군 | 로컬 vector | 없음 | 없음 | TODO |",
    "| MAPPO-GNN | Architecture control | Local graph | No | No | TODO |": "| MAPPO-GNN | architecture 대조군 | 로컬 graph | 없음 | 없음 | TODO |",
    "| GLOBE-original | Prior method | Local MLP plus designed components | Latent | 2-hop score | TODO |": "| GLOBE-original | 선행 방법 | local MLP와 설계 요소 | latent | 2-hop score | TODO |",
    "| GLOBE++ no-KD | Distillation control | [[Ego-Graph]] | No | Mask only | TODO |": "| GLOBE++ no-KD | 증류 대조군 | [[Ego-Graph]] | 없음 | mask-only | TODO |",
    "| GLOBE++ offline-KD | Data-distribution control | [[Ego-Graph]] | Offline policy KD | Mask only | TODO |": "| GLOBE++ offline-KD | 데이터 분포 대조군 | [[Ego-Graph]] | offline policy KD | mask-only | TODO |",
    "| GLOBE++ on-policy-KD | Proposed method | [[Ego-Graph]] | On-policy KD | Mask only | TODO |": "| GLOBE++ on-policy-KD | 제안 방법 | [[Ego-Graph]] | on-policy KD | mask-only | TODO |",
    "| Global teacher | Privileged reference | Global graph | N/A | Mask only | TODO |": "| Global teacher | privileged reference | global graph | 해당 없음 | mask-only | TODO |",
    "| Evo-QGeo | [[Reinforcement-Learning-Based Geographic Routing Considering Future Evolution of Link States for UAV Networks]] | Direct RL geographic FANET routing with future-link and two-hop heuristic |": "| Evo-QGeo | [[Reinforcement-Learning-Based Geographic Routing Considering Future Evolution of Link States for UAV Networks]] | 미래 링크와 2-hop heuristic을 사용하는 직접 RL geographic FANET routing |",
    "| EDP | [[EDP Protocol: Advancing Mobility-Aware Drone Network Connectivity with Adaptive Routing]] | Strong non-learning predictive NS-3 baseline with overhead/stability metrics |": "| EDP | [[EDP Protocol: Advancing Mobility-Aware Drone Network Connectivity with Adaptive Routing]] | overhead·안정성 지표를 포함한 강한 비학습 예측형 NS-3 baseline |",
    "| IQMR | [[Improved Q-learning based Multi-hop Routing for UAV-Assisted Communication]] | Energy/collision/fragmentation-aware Q-learning baseline |": "| IQMR | [[Improved Q-learning based Multi-hop Routing for UAV-Assisted Communication]] | 에너지·충돌·fragmentation을 고려한 Q-learning baseline |",
    "| DRAMA | [[DRAMA: A Dynamic Packet Routing Algorithm using Multi-Agent Reinforcement Learning with Emergent Communication]] | Dynamic topology, graph-compatible actions, runtime communication comparison |": "| DRAMA | [[DRAMA: A Dynamic Packet Routing Algorithm using Multi-Agent Reinforcement Learning with Emergent Communication]] | 동적 topology, graph-compatible 행동, runtime communication 비교 |",
    "| GLOBE-original | [[GLOBE-Routing: Global-to-Local Knowledge-Distilled Graph-MAPPO for Decentralized Routing in UAV Swarm FANETs]] | Direct predecessor with latent distillation and two-hop forwardability |": "| GLOBE-original | [[GLOBE-Routing: Global-to-Local Knowledge-Distilled Graph-MAPPO for Decentralized Routing in UAV Swarm FANETs]] | latent distillation과 2-hop forwardability를 사용하는 직접 선행 방법 |",
    "- Throughput": "- Throughput",
    "- Energy per delivered packet": "- 전달 패킷당 에너지",
    "- Link-break recovery time": "- 링크 단절 복구 시간",
    "- Teacher-student [[KL Divergence]]": "- Teacher-student [[KL Divergence]]",
    "- Return gap": "- Return gap",
    "- Runtime, parameter count, memory, and communication cost": "- Runtime, parameter 수, memory, 통신 비용",
    "- Evo-QGeo-aligned sweeps: 20-60 UAVs, 5-60 m/s, communication range, and deployment density.": "- Evo-QGeo 정렬 sweep: UAV 20~60대, 5~60 m/s, 통신 범위, 배치 밀도.",
    "- DRAMA-aligned topology changes: random link failure, node failure, and new-node addition without retraining.": "- DRAMA 정렬 topology 변화: 무작위 링크 고장, 노드 고장, 재학습 없는 신규 노드 추가.",
    "- EDP-aligned control analysis: route lifetime, discovery frequency, and routing-overhead ratio.": "- EDP 정렬 제어 분석: route lifetime, discovery frequency, routing-overhead ratio.",
    "- IQMR-aligned disruptions: energy depletion, fragmentation, and gradual versus simultaneous rejoin.": "- IQMR 정렬 장애: 에너지 고갈, fragmentation, 점진적·동시 rejoin.",
    "- GLo-MAPPO-aligned reporting: at least five fixed seeds, noisy observations, hyperparameter sensitivity, and scalability.": "- GLo-MAPPO 정렬 보고: 최소 5개 고정 seed, noisy observation, hyperparameter 민감도, scalability.",
    "| Behavior-level policy distillation | No directly matching external paper in this batch | needs-broader-search |": "| 행동 수준 policy distillation | 이번 문헌 묶음에는 직접 일치하는 외부 논문이 없음 | 추가 검색 필요 |",
    "| Local ego-graph GNN execution | [[DRAMA: A Dynamic Packet Routing Algorithm using Multi-Agent Reinforcement Learning with Emergent Communication]] supports graph-compatible dynamic routing, but uses runtime communication | partially-supported |": "| Local ego-graph GNN 실행 | [[DRAMA: A Dynamic Packet Routing Algorithm using Multi-Agent Reinforcement Learning with Emergent Communication]]은 graph-compatible 동적 routing을 뒷받침하지만 runtime communication을 사용함 | 부분 뒷받침 |",
    "| Partial-observability-aware gap | [[GLo-MAPPO: Multi-Agent Deep Reinforcement Learning for Energy-Efficient UAV-Assisted LoRa Networks]] formulates a POSG but does not decompose the information gap | unsupported |": "| 부분 관측을 고려한 격차 | [[GLo-MAPPO: Multi-Agent Deep Reinforcement Learning for Energy-Efficient UAV-Assisted LoRa Networks]]는 POSG를 정식화하지만 정보 격차를 분해하지 않음 | 미지원 |",
    "| Heuristic-free routing | Existing strong FANET methods use engineered prediction, scores, or two-hop logic | high empirical burden |": "| Heuristic-free routing | 기존의 강한 FANET 방법은 설계된 예측, score, 2-hop logic을 사용함 | 높은 실증 부담 |",
    "| Agents | UAV forwarding nodes | TODO |": "| Agent | UAV forwarding node | TODO |",
    "| Global state $s_t$ | Full topology, mobility, queues, links, destinations | TODO |": "| Global state $s_t$ | 전체 topology, 이동성, queue, 링크, 목적지 | TODO |",
    "| Local observation $o_t^i$ | Node features plus bounded [[Ego-Graph]] | TODO |": "| Local observation $o_t^i$ | node feature와 제한된 [[Ego-Graph]] | TODO |",
    "| Action $a_t^i$ | Next-hop choice under invalid-action mask | TODO |": "| Action $a_t^i$ | invalid-action mask 아래의 next-hop 선택 | TODO |",
    "| Transition $P$ | Mobility, channel, queue, and forwarding dynamics | TODO |": "| Transition $P$ | 이동성, channel, queue, forwarding dynamics | TODO |",
    "| Reward $r_t$ | Delivery, latency, loss, overhead, and energy tradeoff | TODO |": "| Reward $r_t$ | 전달, latency, 손실, overhead, 에너지 trade-off | TODO |",
    "| Observation map $O$ | Global state to local information | TODO |": "| Observation map $O$ | global state에서 local information으로의 mapping | TODO |",
    "- [[Partial Observability]] may induce an irreducible information gap.": "- [[Partial Observability]]는 제거 불가능한 정보 격차를 만들 수 있다.",
    "1. Measure teacher/student [[KL Divergence]] under a specified occupancy distribution.": "1. 명시된 occupancy distribution에서 teacher/student [[KL Divergence]]를 측정한다.",
    "2. Convert KL to total variation only with explicit assumptions and inequality.": "2. 명시적 가정과 부등식이 있을 때만 KL을 total variation으로 변환한다.",
    "3. Apply a source-backed [[Performance Difference Lemma]].": "3. 출처가 뒷받침하는 [[Performance Difference Lemma]]를 적용한다.",
    "4. Report horizon, reward bounds, and distribution-shift terms.": "4. horizon, reward bound, distribution-shift 항을 보고한다.",
    "Use [[Literature Matrix]] and [[Method Comparison Matrix]].": "[[Literature Matrix]]와 [[Method Comparison Matrix]]를 사용한다.",
    "- Mobility, propagation, MAC, queue, traffic model.": "- Mobility, propagation, MAC, queue, traffic model을 기록한다.",
    "| Novelty is only a combination | GNN, MAPPO, and KD are established components | Literature gap plus factorized ablations |": "| 신규성이 단순 조합에 불과함 | GNN, MAPPO, KD는 이미 확립된 구성요소임 | 문헌 공백과 요인별 ablation 제시 |",
    "| ns-3 validation is insufficient | Simplified simulation may not transfer | Scenario realism, ns-3 details, limitations |": "| ns-3 검증이 불충분함 | 단순화된 simulation은 실제 환경으로 이전되지 않을 수 있음 | scenario 현실성, ns-3 세부사항, 한계 제시 |",
    "| Teacher is an oracle | Global graph may be unrealistic or too strong | Training-only access and precise reference status |": "| Teacher가 oracle임 | global graph는 비현실적이거나 지나치게 강할 수 있음 | 학습 단계만의 접근과 정확한 reference 지위 명시 |",
    "| Heuristic removal lowers performance | Prior heuristic may encode valuable structure | Mask-only comparison across seeds |": "| Heuristic 제거가 성능을 낮춤 | 기존 heuristic이 유용한 구조를 내장할 수 있음 | 여러 seed에서 mask-only 비교 |",
    "| Local observation cannot match teacher | [[Partial Observability]] creates aliasing | Quantify irreducible disagreement and return gap |": "| Local observation이 teacher를 따라갈 수 없음 | [[Partial Observability]]가 aliasing을 만듦 | 제거 불가능한 disagreement와 return gap 정량화 |",
    "| Scalability is unproven | GNN cost and neighbor growth may dominate | Node-count OOD tests and runtime/overhead |": "| Scalability가 입증되지 않음 | GNN 비용과 이웃 증가가 지배적일 수 있음 | node 수 OOD 시험과 runtime/overhead 보고 |",
    "| KL does not imply routing gain | Objective may be behaviorally misaligned | Correlate KL, action agreement, and network metrics |": "| KL 감소가 routing 성능 향상을 의미하지 않음 | 목적함수가 실제 행동과 어긋날 수 있음 | KL, action agreement, network 지표의 상관 분석 |",
    "| Baselines are under-tuned | Learned methods are sensitive to budget | Equal tuning protocol and disclosed search |": "| Baseline tuning이 부족함 | 학습 방법은 budget에 민감함 | 동일 tuning protocol과 탐색 범위 공개 |",
    "| Multi-seed evidence is weak | High variance can reverse conclusions | Confidence intervals and paired scenario analysis |": "| Multi-seed 근거가 약함 | 높은 분산이 결론을 뒤집을 수 있음 | 신뢰구간과 paired scenario 분석 |",
    "| [[Latent Distillation]] | Internal representation | Indirect | High unless representations align | Baseline |": "| [[Latent Distillation]] | 내부 representation | 간접 | representation이 정렬되지 않으면 높음 | Baseline |",
    "| [[Policy Distillation]] | Action distribution | Direct | Irreducible gap remains | Proposed |": "| [[Policy Distillation]] | 행동 분포 | 직접 | 제거 불가능한 격차가 남음 | 제안 방식 |",
    "| No KD | Environment return only | Task-dependent | No teacher transfer | Control |": "| KD 없음 | 환경 return만 사용 | task 의존적 | teacher 지식 전달 없음 | 대조군 |",
    "| GLOBE-original | TODO | TODO | Local MLP | RL plus latent matching | Needs source verification |": "| GLOBE-original | TODO | TODO | Local MLP | RL과 latent matching | 출처 검증 필요 |",
    "| GLOBE++ | Global teacher plus local student | [[Ego-Graph]] | Local GNN | RL plus policy KL | Observation gap |": "| GLOBE++ | global teacher와 local student | [[Ego-Graph]] | Local GNN | RL과 policy KL | 관측 격차 |",
    "| AODV | Reactive | TODO | TODO | PDR, delay, overhead |": "| AODV | Reactive | TODO | TODO | PDR, 지연, overhead |",
    "| OLSR | Proactive | TODO | TODO | PDR, delay, overhead |": "| OLSR | Proactive | TODO | TODO | PDR, 지연, overhead |",
    "| GPSR | Geographic | TODO | TODO | PDR, delay, overhead |": "| GPSR | Geographic | TODO | TODO | PDR, 지연, overhead |",
    "| GLOBE++ | Learned local graph policy | [[Ego-Graph]] | Policy inference | Full metric suite |": "| GLOBE++ | 학습된 local graph policy | [[Ego-Graph]] | Policy inference | 전체 지표 묶음 |",
    "| EXP-001 | Baseline suite | Overall method | planned | TODO | [[GLOBE++ Experiment Plan]] |": "| EXP-001 | Baseline 전체 비교 | 전체 방법 | 계획 | TODO | [[GLOBE++ Experiment Plan]] |",
    "| EXP-002 | Distillation ablation | Claim 1 | planned | TODO | [[GLOBE++ Ablation Plan]] |": "| EXP-002 | Distillation ablation | 주장 1 | 계획 | TODO | [[GLOBE++ Ablation Plan]] |",
    "| EXP-003 | OOD scaling | Claims 2-3 | planned | TODO | [[OOD Generalization]] |": "| EXP-003 | OOD 확장성 | 주장 2~3 | 계획 | TODO | [[OOD Generalization]] |",
    "| [[Packet Delivery Ratio]] | TODO | Per scenario and seed | Denominator consistency |": "| [[Packet Delivery Ratio]] | TODO | scenario·seed별 | 분모 일관성 |",
    "| [[End-to-End Delay]] | TODO | Mean and tail | Delivered-packet bias |": "| [[End-to-End Delay]] | TODO | 평균과 tail | 전달 패킷 편향 |",
    "| Throughput | TODO | Time normalized | Offered-load dependence |": "| Throughput | TODO | 시간 정규화 | offered load 의존성 |",
    "| [[Routing Overhead]] | TODO | Packets and bytes | Include graph maintenance |": "| [[Routing Overhead]] | TODO | packet과 byte | graph 유지 비용 포함 |",
    "| Energy per delivered packet | TODO | Per scenario | Energy model validity |": "| 전달 패킷당 에너지 | TODO | scenario별 | 에너지 모델 타당성 |",
    "| Teacher-student [[KL Divergence]] | TODO | Occupancy weighted | Mask and KL direction |": "| Teacher-student [[KL Divergence]] | TODO | occupancy 가중 | mask와 KL 방향 |",
    "| Return gap | TODO | Paired teacher/student | Teacher is not automatically optimal |": "| Return gap | TODO | paired teacher/student | teacher가 자동으로 optimal인 것은 아님 |",
    "- Global teacher upper bound": "- Global teacher reference",
    "- Energy per Delivered Packet": "- 전달 패킷당 에너지",
    "- Link Break Recovery Time": "- 링크 단절 복구 시간",
    "- Teacher-Student KL": "- Teacher-Student KL",
    "- Return Gap": "- Return gap",
    "| UAV count | |": "| UAV 수 | |",
    "| Traffic pattern | |": "| 트래픽 패턴 | |",
    "| Link model | |": "| 링크 모델 | |",
    "| Training seeds | |": "| 학습 seed | |",
    "| Evaluation seeds | |": "| 평가 seed | |",
}


def extract_section(body: str, starts: tuple[str, ...], ends: tuple[str, ...]) -> str:
    start_positions = [(body.find(marker), marker) for marker in starts if marker in body]
    if not start_positions:
        return ""
    start_index, marker = min(start_positions, key=lambda item: item[0])
    content_start = start_index + len(marker)
    end_positions = [
        body.find(end_marker, content_start)
        for end_marker in ends
        if body.find(end_marker, content_start) >= 0
    ]
    content_end = min(end_positions) if end_positions else len(body)
    return body[content_start:content_end].strip()


def render_card(path: Path) -> None:
    meta, old_body = read_markdown(path)
    title = str(meta["title"])
    data = KO.get(title)
    if not data:
        return
    source_section = extract_section(
        old_body,
        ("### Source Files", "### 원본 파일"),
        (),
    )
    evidence_section = extract_section(
        old_body,
        ("## 15. Source Evidence", "## 15. 출처 근거"),
        ("### Source Files", "### 원본 파일"),
    )
    evidence_section = (
        evidence_section
        .replace("| Evidence | Page/Section | Evidence Type | Confidence |", "| 근거 | 페이지/섹션 | 근거 유형 | 신뢰도 |")
        .replace("direct-claim", "직접 주장")
        .replace("experimental-result", "실험 결과")
        .replace("theoretical-claim", "이론적 주장")
        .replace("limitation", "한계")
        .replace("assumption", "가정")
        .replace("Source file:", "원본 파일:")
        .replace("Source path:", "원본 경로:")
        .replace("Source hash:", "원본 해시:")
    )
    for english, korean in EVIDENCE_KO.items():
        evidence_section = evidence_section.replace(english, korean)
    source_section = (
        source_section
        .replace("Source file:", "원본 파일:")
        .replace("Source path:", "원본 경로:")
        .replace("Source hash:", "원본 해시:")
    )
    body = f"""# {title}

## 1. 한 줄 요약

{data["summary"]}

## 2. 문제 설정

{data["problem"]}

## 3. 핵심 기여

### 논문 주장

{data["method"]}

### Agent 평가

{data["connection"]}

## 4. 방법

### 4.1 모델 / 알고리즘

{data["method"]}

### 4.2 학습 설정

{data["training"]}

### 4.3 실행 단계 가정

{data["execution"]}

## 5. 실험 설정

{data["experiment"]}

| 항목 | 내용 |
| --- | --- |
| Baseline | {", ".join(str(x) for x in meta.get("baselines", []))} |
| 지표 | {", ".join(str(x) for x in meta.get("metrics", []))} |
| 재현성 | {meta.get("reproducibility", "")} |

## 6. 주요 결과

**논문 주장:** {data["results"]}

## 7. 이론적 주장

{data["theory"]}

## 8. 가정

{data["assumptions"]}

## 9. 한계

{data["weaknesses"]}

## 10. Reviewer 관점 비판

### 강점

{data["strengths"]}

### 약점

{data["weaknesses"]}

### 숨겨진 취약 가정

{data["fragile"]}

### 재현성 위험

**{meta.get("reproducibility", "")}**. 서지정보와 주장은 제공된 PDF 본문에서 확인했다. 불명확한 seed나 출판 정보는 검증 필요로 유지한다.

## 11. GLOBE++와의 연결

{data["connection"]}

관련 개념: {", ".join(f"[[{x}]]" for x in meta.get("related_concepts", []))}.

## 12. 내 논문에서 뒷받침할 수 있는 내용

{data["supports"]}

## 13. 뒷받침할 수 없는 내용

{data["cannot"]}

## 14. 후속 질문

- 분산 실행 시 실제로 사용할 수 있는 정보는 무엇인가?
- 동일한 학습 예산과 multi-seed 조건에서도 이득이 유지되는가?
- 성능 향상의 원인은 학습, privileged information, engineered heuristic 중 무엇인가?

## 15. 출처 근거

{evidence_section}

### 원본 파일

{source_section}
"""
    meta["language"] = "ko"
    path.write_text(yaml_document(meta, body.strip()), encoding="utf-8")


def render_source(path: Path) -> None:
    meta, _ = read_markdown(path)
    papers = meta.get("related_papers", [])
    if not papers or papers[0] not in KO:
        return
    title = papers[0]
    data = KO[title]
    body = f"""# {meta["title"]}

## 1. 출처 메타데이터

| 항목 | 값 |
| --- | --- |
| 실제 논문 제목 | [[{title}]] |
| 원본 파일 | `{meta["source_file"]}` |
| 원본 경로 | `{meta["source_path"]}` |
| 원본 해시 | `{meta["source_hash"]}` |
| 처리 날짜 | {meta["updated"]} |

## 2. 핵심 요약

{data["summary"]}

## 3. 핵심 개념

{", ".join(f"[[{x}]]" for x in meta.get("related_concepts", []))}

## 4. 중요 세부사항

- 방법: {data["method"]}
- 실험: {data["experiment"]}
- 해시가 다른 동일 본문 PDF도 source record는 각각 유지하고 canonical Paper Card에 연결한다.

## 5. 추출된 주장

- 주요 결과: {data["results"]}
- 이론 범위: {data["theory"]}
- 상세 페이지 근거는 [[{title}]]의 출처 근거 표를 참조한다.

## 6. 내 연구와의 관련성

### GLOBE++에 직접 유용

{data["connection"]}

### 간접적으로 유용

{data["supports"]}

### 관련성이 낮거나 뒷받침하지 못하는 범위

{data["cannot"]}

## 7. 생성 또는 갱신할 개념 페이지

{", ".join(f"[[{x}]]" for x in meta.get("related_concepts", []))}

## 8. 연결된 Paper Card

[[{title}]]

## 9. 기존 Wiki와의 긴장 또는 한계

{data["weaknesses"]}

## 10. 후속 질문

- 어떤 결과를 독립적으로 재현해야 하는가?
- 어떤 가정이 mask-only local ego-graph 실행과 충돌하는가?

## 11. 인용 메모

- 원본 파일: `{meta["source_file"]}`
- 원본 경로: `{meta["source_path"]}`
- 원본 해시: `{meta["source_hash"]}`
- 페이지별 evidence type은 연결된 Paper Card에 기록되어 있다.
"""
    meta["language"] = "ko"
    path.write_text(yaml_document(meta, body.strip()), encoding="utf-8")


def localize_generic(path: Path) -> None:
    meta, body = read_markdown(path)
    for old, new in {**HEADINGS, **GENERIC}.items():
        body = body.replace(old, new)
    body = (
        body
        .replace("[[연구 대시보드]]", "[[Research Dashboard]]")
        .replace("[[읽기 대기열]]", "[[Reading Queue]]")
    )
    if meta.get("title"):
        body = re.sub(r"^# .+$", f"# {meta['title']}", body, count=1, flags=re.MULTILINE)
    meta["language"] = "ko"
    path.write_text(yaml_document(meta, body.strip()), encoding="utf-8")


def main() -> int:
    for path in PAPER_DIR.glob("*.md"):
        render_card(path)
    for path in SOURCE_DIR.glob("*.md"):
        render_source(path)
    for path in VAULT_DIR.rglob("*.md"):
        if path.parent in {PAPER_DIR, SOURCE_DIR}:
            continue
        localize_generic(path)
    print("한국어 현지화 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
