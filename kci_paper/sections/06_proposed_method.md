본 장에서는 제안하는 \method\ 라우팅 프레임워크의 상세 아키텍처와 핵심 제어 알고리즘을 설명한다. 제안 기법은 전역 정보를 다루는 오프라인 교사 모델, 지식을 이전받는 로컬 학생 모델, 그리고 온라인에서 실행 오버헤드를 극적으로 억제하는 예측적 위험 스위칭 게이트의 유기적 결합으로 이루어진다.

\subsection{전역 교사 정책 (Global Teacher Policy)}
오프라인 환경 학습 단계에서 구동되는 교사 정책 $\pi_T$는 FANET의 전체 연결 관계를 그래프 $G_t = (\mathcal{V}, \mathcal{E}_t)$로 입력받는 특권적인(Privileged) 구조를 지닌다. 교사 모델은 에지 조건부 평균 집계(Edge-Conditioned Mean Aggregation) 메커니즘을 구비한 2-layer 메시지 패싱 그래프 신경망(GNN)을 활용하여 전역적인 노드간 다자간 토폴로지 관계를 인코딩한다:
\[
\mathbf{h}_v^{(l+1)} = \sigma \left( \mathbf{W}_v \mathbf{h}_v^{(l)} + \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} f_e (\mathbf{e}_{uv}) \mathbf{h}_u^{(l)} \right)
\]
여기서 $\mathbf{h}_v^{(l)}$는 $l$번째 레이어의 노드 $v$의 임베딩이며, $f_e$는 에지 특징 벡터 $\mathbf{e}_{uv}$(노드 간 정규화 거리 및 연결 가용성)를 가중치 행렬로 맵핑하는 소형 신경망이다. 
최종 출력 레이어는 현재 패킷 포워딩을 수행하려는 노드의 1-hop 유효 이웃들과 드롭 행동만을 고르는 구조적 마스킹(Action Masking)을 반영한 카테고리컬 액터(Masked Categorical Actor)를 사용하여 포워딩 행동 분포 $\pi_T(\cdot \mid s_t)$를 결정한다. 학습은 완전 에피소드 리턴 및 클리핑 기반 PPO(Proximal Policy Optimization)[8]와 집중형 가치 네트워크(Centralized Value Network)를 연계해 CPU/CUDA 환경에서 오프라인으로 수렴될 때까지 수행된다.

\subsection{지역 학생 정책 (Local Student Policy) 및 순열 불변성}
실제 UAV 군집에 분산 탑재될 학생 정책 $\pi_S$는 전역 그래프 상태 정보 $s_t$ 대신, 노드의 고유 ID를 배제하고 오직 1-hop 인접 이웃들의 상태 피처 벡터만으로 구성된 로컬 Ego-graph 입력값 $o_{i,t}$를 활용한다. 이웃 노드들의 입력 순서에 무관하게 항상 동일한 라우팅 의사결정을 일관적으로 보장하도록, 학생 모델은 공유 파라미터 다층 퍼셉트론(Shared-parameter MLP)과 순열 불변 평균 풀링(Permutation-invariant Mean Pooling)으로 인코더를 구성한다:
\[
\mathbf{z}_{i,t} = \mathrm{MeanPooling} \left( \{ \mathrm{MLP}_{\mathrm{enc}}(o_{k,t}) \mid k \in \mathcal{N}(i) \} \right)
\]
도출된 로컬 컨텍스트 벡터 $\mathbf{z}_{i,t}$는 각 포워딩 후보 노드들의 logit scorer와 별도의 드롭($\dropact$) scorer를 통과한 후, 마스크드 소프트맥스(Masked Softmax) 레이어를 통해 최종 이산 행동 분포 $\pi_S(\cdot \mid o_{i,t})$를 출력한다.

\subsection{정책 증류 (Policy Distillation)}
오프라인 단계에서 교사 정책이 성공적으로 훈련되면, 교사의 탐색 궤적 데이터셋 $\mathcal{D}$를 생성하고 이를 기반으로 교사와 학생의 행동 분포 간 쿨백-라이블러 발산(Kullback-Leibler Divergence)을 최소화하도록 오프라인 정책 증류를 전개한다. 증류 손실 함수는 다음과 같다:
\[
\mathcal{L}_{\mathrm{KD}} = \mathbb{E}_{(s, o) \sim \mathcal{D}} \left[ D_{\mathrm{KL}} \left( \pi_T(\cdot \mid s) \,\|\, \pi_S(\cdot \mid o) \right) \right]
\]
KL Divergence 타겟으로 최적화할 때, 온도 매개변수 $\tau$를 적용한 마스킹 분포를 사용하여 불확실한 행동 분포 영역의 경향성까지 학생 모델이 세밀하게 복제하도록 지원한다.

\subsection{예측적 위험 스위칭 (Predictive Risk Switching; PRS)}
\method\ 모델의 핵심 혁신 요소는 온라인 배포 환경에서 통신 및 추론 오버헤드를 극적으로 통제하는 예측적 위험 스위칭(Predictive Risk Switching; PRS) 모듈이다. 평상시 안전한 상태에서 모든 노드는 기계학습 모델의 연산 없이 1-hop 거리와 이동 잠재력만 보는 가벼운 지리적 라우팅(Greedy Geographic Progress)을 기본 노미널(Nominal) 정책으로 동작시킨다. 
그러나 노드가 다음과 같은 다차원 위험 지표인 Danger Score ($DS_i$) 임계를 초과하는 이상 징후 상황에 직면하면, 즉각적으로 스위치 게이트를 활성화하여 증류된 학생 정책 모델을 구동하는 예측(Predictive) 모듈로 동작 분기를 전환한다.

위험 판단 점수 $DS_i$는 무선 수신 강도 마진(RSSI Margin), 버퍼 큐 길이, 그리고 칼만 필터 기반의 예측 링크 수명($L_i$) 및 다홉 안전성 평가지표(Onward Stability)의 복합적 평치로 도출된다:
\[
DS_i = w_Q \cdot \frac{q_i}{q_{\mathrm{max}}} + w_L \cdot \sigma_L(L_{\mathrm{threshold}} - L_i) + w_E \cdot p_{i,\mathrm{loss}}
\]
여기서 $p_{i,\mathrm{loss}}$는 다자간 페이딩 상태를 필터링하는 무선 패킷 손실 확률이며, $\sigma_L$은 활성 스케일링용 로지스틱 함수이다. 
위험도 임계값 $\eta$에 대한 스위칭 함수 $\delta_i$는 다음과 같이 동작한다:
\[
\text{Selected Policy} = \begin{cases} 
\text{Nominal Geographic Routing}, & \text{if } DS_i \le \eta \\ 
\pi_S(\cdot \mid o_{i,t}), & \text{if } DS_i > \eta 
\end{cases}
\]

추가적으로, 복수의 이웃이 동일한 안전 유틸리티 점수를 기록하는 동률 상황(Tie-breaking)이 발생하면 잔여 에너지가 가장 많은 노드에 높은 가중치를 인가하는 에너지 우대 토큰(Energy-Aware Token)과, 패킷 폐기 결정을 지연 차단하는 드롭抑制(Drop Suppression) 모듈을 결합하여 종단 안정성을 비약적으로 높이도록 고안하였다.
