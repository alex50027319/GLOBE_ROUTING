# Lite-GLOBE 설계 가정

## 환경 가정

1. UAV는 정사각형 2차원 영역 안에서 Random Waypoint 모델로 이동한다.
   고도와 3차원 안테나 효과를 제외하므로 실제 FANET보다 단순하지만, 위치와
   링크가 시간에 따라 변하는 핵심 비정상성은 유지한다.
2. 링크는 기본적으로 통신 반경 안에서 대칭적으로 연결된다. 선택적 확률 손실은
   독립적인 링크 소거로 처리한다. 간섭, SINR, MAC 충돌을 아직 모델링하지
   않으므로 Phase 1 결과를 실제 처리량으로 직접 해석하면 안 된다.
3. 한 에피소드에는 패킷 하나만 존재하고 한 번에 다음 홉 하나를 결정한다.
   큐 길이는 관측 특징으로 제공하지만, 다중 패킷 큐잉 동역학은 후속 단계에서
   추가해야 한다.
4. 목적지는 별도 지상국이 아니라 활성 UAV 중 하나다. source와 destination은
   항상 서로 다르다.

## 정책 가정

1. 정책은 현재 UAV의 상태, 1-hop 이웃 상태, 링크 거리, 목적지 방향, TTL,
   구조적 action mask만 관측한다.
2. 노드 배열은 전역 UAV ID로 패딩한다. 구현과 디버깅은 단순해지지만, ID 순서
   불변성을 자동 보장하지 않는다. 학습 정책 단계에서는 permutation 처리 또는
   명시적 데이터 증강이 필요하다.
3. 유효한 이웃이 없을 때의 행동은 암묵적 no-op이 아니라 명시적 `DROP`이다.
   따라서 단절 상태에서 대기 후 재연결하는 전략은 현재 평가하지 않는다.
4. GPSR은 목적지에 가장 가까운 관측 이웃을 선택하는 greedy 모드만 구현한다.
   planarization과 perimeter recovery가 없으므로 local minimum이나 이동성에
   의해 루프 또는 실패가 발생할 수 있다.

## 보상 가정

보상은 다음 세 항만 사용한다.

\[
r_t =
\mathbb{1}_{delivery}R_{delivery}
- R_{delay}
- \mathbb{1}_{failure}R_{failure}
\]

- \(R_{delivery}\): 목적지 전달 시 한 번 지급하는 양의 보상
- \(R_{delay}\): 매 라우팅 결정마다 부과하는 비용
- \(R_{failure}\): 드롭, 루프, TTL 만료 등 terminal failure 비용

에너지, 링크 안정성, 혼잡도, 전역 잠재함수는 보상에 넣지 않는다. 이는
Lite-GLOBE의 병목 원인을 분리하기 위한 의도적인 제한이며, 추가 항은 독립적인
ablation으로만 도입해야 한다.

## 재현성과 실행 환경

- 모든 확률성은 Gymnasium의 episode seed 또는 정책 seed에서 파생한다.
- 로컬 환경은 단위 테스트와 소규모 CPU 스모크 평가용이다.
- 대규모 학습, fine-tuning, seed sweep은 향후 Colab GPU 노트북에서 수행한다.
- raw 논문과 기존 Obsidian 문서는 구현 과정에서 수정하지 않는다.

## Phase 2 Local Student 가정

1. 후보 이웃의 행 순서에는 의미가 없다고 가정한다. 공유 encoder와 scorer,
   유효 이웃 mean pooling을 사용하여 후보 순열에 대해 정책을 equivariant하게
   만들었다. 후보에 연결된 action ID까지 함께 순열했을 때 해당 확률도 같은
   방식으로 이동한다.
2. `DROP`은 이웃 후보와 다른 의미를 가지므로 별도 scorer를 사용한다. 이웃이
   하나도 없으면 구조적 mask에 의해 `DROP` 확률은 정확히 1이 된다.
3. source와 destination의 정규화된 전역 ID가 packet feature에 남아 있다.
   이는 Phase 1 관측 계약을 유지하기 위한 선택이지만, 보지 못한 node ID에 대한
   일반화를 저해할 수 있다. OOD 실험 전에는 ID feature 제거 ablation이 필요하다.
4. 무작위 초기화 Student 평가는 아키텍처 smoke test일 뿐 학습 성능이 아니다.
   Phase 2 결과를 GPSR 또는 향후 Teacher와의 성능 비교 근거로 사용하지 않는다.
5. `hidden_dim`은 사전 명세대로 32 또는 64만 허용한다. 더 큰 모델은 계산량과
   성능의 별도 ablation 없이 기본 모델로 채택하지 않는다.

## Phase 3 Global Teacher 가정

1. Teacher는 학습과 dataset 생성에만 사용하는
   `privileged reference policy`다. optimal policy나 성능 upper bound로
   부르지 않는다.
2. 전역 관측에는 모든 UAV의 현재 위치·속도·queue, 전체 adjacency와 거리,
   source/current/destination/visited 표시가 포함된다. Student 입력이나 배포
   adapter에는 이 API를 전달하지 않는다.
3. message passing은 2-layer로 제한한다. 전체 graph를 입력받아도 각 node
   embedding의 receptive field는 두 홉이며, 과도한 깊이로 toy topology를
   암기하는 것을 피한다.
4. `DROP`은 유효한 구조적 행동이지만, 기본 초기화에서는 즉시 실패가 긴 탐색
   실패보다 덜 불리해 PPO가 `DROP`에 조기 붕괴했다. 정답 경로나 geographic
   score를 넣지 않고 Teacher의 초기 `DROP` bias만 -2로 설정해 forwarding
   탐색을 확보한다. 이 값은 향후 민감도 분석 대상이다.
5. Phase 3 gate의 routing-hole topology는 GPSR local minimum을 재현하는
   결정적 학습 검증 장치다. 해당 PDR은 일반 환경 결과가 아니며 논문 성능
   표에 사용하면 안 된다.
6. centralized value network는 Teacher PPO 학습에만 사용한다. Teacher actor
   자체도 배포 대상이 아니며, Phase 4 dataset에는 value나 global embedding을
   Student 입력으로 저장하지 않는다.

## Phase 4 Offline Distillation 가정

1. Phase 4 dataset은 Teacher가 생성한 deterministic trajectory를 사용한다.
   이는 offline teacher occupancy에서의 모방 가능성만 검증하며 Student가
   스스로 방문하는 상태의 covariate shift는 Phase 5 이후 문제로 남는다.
2. split 누출은 sample 단위가 아니라 `scenario_id + episode_seed` episode
   group 단위로 차단한다. 동일 episode의 서로 다른 hop은 반드시 같은 split에
   속한다.
3. Student 입력에는 global state를 저장하지 않는다. Teacher raw logits은
   temperature 재계산용 target이고, masked probability는 dataset 무결성 검사와
   분석용이다.
4. 기본 증류는 forward KL
   \(D_{KL}(\pi_T^\tau\|\pi_S^\tau)\)만 사용한다. hard-label cross entropy,
   latent matching, RL reward를 함께 섞지 않는다.
5. routing-hole에서는 로컬 관측만으로 Teacher 행동을 구분할 수 있어 100%
   agreement가 가능하다. 더 다양한 전역 graph가 같은 로컬 관측으로 투영되면
   비가역적인 information gap이 발생할 수 있으며, Phase 4 toy 결과는 그 격차를
   측정하지 않는다.
6. checkpoint가 없을 때 실행기는 Phase 3 Teacher를 재학습하지만, 기준선 gate를
   통과하지 못하면 dataset 생성 전에 중단한다. 실패한 Teacher target을
   자동으로 수용하지 않는다.

## Phase 5 Local PPO Fine-tuning 가정

1. 기본 fine-tuning 목적함수는 PPO policy/value loss와 entropy 항만 사용한다.
   Teacher 모델, global observation, global value는 실행 및 학습 인터페이스에
   전달하지 않는다.
2. 선택적 KD 모드는 Phase 4에 저장된 offline Teacher logits만 사용한다.
   \(\lambda_{KD}=0\)이 기본값이며, 양수일 때 update에 따라 지수 감소한다.
3. local critic은 self, packet, 유효 1-hop neighbor/edge의 mean feature만
   사용한다. actor보다 많은 전역 정보를 critic에 제공하지 않는다.
4. routing-hole의 Phase 4 Student는 이미 PDR 100%이므로 Phase 5 toy gate는
   개선이 아니라 catastrophic degradation이 없는지만 검증한다. PPO의 실질적
   이득은 Phase 6의 더 다양한 동적 FANET 시나리오에서 판단해야 한다.
5. 현재 toy run의 value loss는 policy 성능과 달리 충분히 낮게 수렴하지 않았다.
   일반 환경에서는 value normalization, GAE, critic learning-rate 분리 여부를
   ablation해야 한다.
6. Phase 5 출력 checkpoint는 배포용 actor만 저장한다. optimizer와 rollout
   상태를 포함한 중단 재개 기능은 최종 Colab orchestration 단계에서 추가해야
   한다.
