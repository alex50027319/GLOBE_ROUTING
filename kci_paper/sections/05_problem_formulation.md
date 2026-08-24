본 장에서는 무인 비행체 애드혹 네트워크(FANET) 상에서의 분산 라우팅 문제를 부분관측 다중 에이전트 마르코프 의사결정 과정(Decentralized Partially Observable Markov Decision Process; Dec-POMDP)으로 엄밀하게 정식화한다.

\subsection{네트워크 모델 및 상태 정의}
3차원 가상 공중 환경 $M \times M \times M$ 내에 존재하는 모든 통신 노드들의 집합은 $E = \{e_s, e_1, e_2, \dots, e_N, e_d\}$로 정의된다. 여기서 $e_s$와 $e_d$는 각각 패킷 전송을 개시하는 소스 노드(Source Node)와 패킷의 종단 수신 노드인 목적지 노드(Destination Node)를 의미하는 고정된 지상 기반 시설이다. 공중에 배치되어 패킷 중계를 담당하는 $N$개의 무인 비행체(UAV)들은 $e_1, e_2, \dots, e_N$으로 지칭되며, 모든 노드는 무선 통신 반경 $\delta$를 구비하여 두 노드 $e_i$와 $e_j$의 유클리디안 거리가 $d(i,j) \le \delta$를 만족할 때에만 직접적인 1-hop 무선 링크를 형성할 수 있다.

특정 타임스텝 $t$에서의 전역 상태 $s_t \in \mathcal{S}$는 네트워크 내 존재하는 모든 에이전트의 3차원 위치 벡터와 속도 벡터, 그리고 각 노드가 내부 버퍼에 대기 중인 라우팅 패킷 큐 길이의 집합으로 묘사된다:
\[
s_t = [ \mathbf{x}_t, \mathbf{v}_t, \mathbf{q}_t ]^\top
\]
여기서 $\mathbf{x}_t = [x_s, x_1, \dots, x_N, x_d]^\top$는 모든 통신 노드들의 3차원 위치 공간 좌표이다.

\subsection{Dec-POMDP 요소의 정식화}
UAV들의 통신 및 연산 제약을 고려하여 라우팅 제어 문제를 다음과 같은 Dec-POMDP 튜플로 정의한다:
\[
\mathcal{M} = \langle \mathcal{I}, \mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{P}, \mathcal{R}, \gamma \rangle
\]

1) **에이전트 집합 ($\mathcal{I}$)**: 패킷 포워딩 및 경로 우회 결정을 내릴 수 있는 중계용 UAV 에이전트들의 인덱스 집합 $\mathcal{I} = \{1, 2, \dots, N\}$이다.

2) **행동 공간 ($\mathcal{A}$)**: 특정 무인기 $e_i$가 패킷을 소지하고 있을 때 내릴 수 있는 개별 행동 $a_{i,t} \in \mathcal{A}_i$는 1-hop 통신 반경 내에 존재하는 인접 이웃 노드 인덱스들 중 하나를 선택하여 패킷을 송신하거나, 혹은 더 이상 라우팅을 진행할 수 없다고 판단하여 패킷을 완전히 폐기하는 드롭($\dropact$) 행동의 이산 집합으로 규정된다:
\[
\mathcal{A}_i = \{ j \in E \mid d(i,j) \le \delta \} \cup \{ \dropact \}
\]

3) **관측 공간 ($\mathcal{O}$)**: 배포된 개별 에이전트 $e_i$는 전역 토폴로지 구조 $s_t$를 직접 관측할 수 없으며, 오직 자신의 1-hop 영역 내에 인접한 이웃 노드들의 지역적 특징(Local Ego-graph)만을 관측값 $o_{i,t} \in \mathcal{O}_i$로 획득한다:
\[
o_{i,t} = [ x_i, v_i, q_i, D_{i,t} ]^\top
\]
여기서 $D_{i,t} = \{ (d(i,k), v_k, q_k) \mid d(i,k) \le \delta \}$는 1-hop 인접 이웃 노드들의 상대 거리, 속도 및 큐 상태의 집합이다.

4) **전이 확률 ($\mathcal{P}$)**: 현재 전역 상태 $s_t$에서 에이전트의 공동 행동(Joint Action) $\mathbf{a}_t = [a_{1,t}, \dots, a_{N,t}]$가 수행되었을 때 다음 상태 $s_{t+1}$로 전이되는 무선 환경 상태 전이 확률 분포 $\mathcal{P}(s_{t+1} \mid s_t, \mathbf{a}_t)$이다.

5) **보상 구조 ($\mathcal{R}$)**: 에이전트의 완전 분산적인 협력 학습을 촉진하기 위해 개별 보상이 아닌 동일한 팀 보상 $r_t = \mathcal{R}(s_t, \mathbf{a}_t)$을 공유한다. 보상 함수는 패킷의 성공적 종단 전달, 홉별 누적 지연 시간 억제, 그리고 드롭에 따른 실패 비용을 차등 부과하는 세 가지 항의 합으로 조율된다:
\[
r_t = R_{\mathrm{delivery}} \cdot \mathbb{1}_{\mathrm{delivery}} - R_{\mathrm{delay}} - R_{\mathrm{failure}} \cdot \mathbb{1}_{\mathrm{failure}}
\]
- $\mathbb{1}_{\mathrm{delivery}}$: 목적지 노드 $e_d$에 패킷이 안전하게 도착했음을 나타내는 지시 함수.
- $\mathbb{1}_{\mathrm{failure}}$: 패킷이 드롭되거나 TTL(Time to Live) 만료 등으로 전송에 완전히 실패했음을 지시하는 함수.
- $R_{\mathrm{delay}}$: 패킷이 공중 노드 사이에서 머무르는 지연 시간당 부과되는 패널티 비용.

6) **목적 함수 ($J(\pi)$)**: 공동 정책 $\pi = [\pi_1, \dots, \pi_N]$ 하에서 각 에이전트가 오직 자신의 로컬 관측 $o_{i,t}$만을 활용하여 기대 누적 감쇄 보상을 극대화하는 최적의 분산 제어 정책을 도출하는 것을 목표로 한다:
\[
J(\pi) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]
\]
여기서 $\gamma \in [0,1)$는 미래 보상에 대한 시간 감쇄 인자이다.
