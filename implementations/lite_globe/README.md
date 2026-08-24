# Lite-GLOBE 구현

Lite-GLOBE는 GLOBE++의 복잡한 전역-지역 정책 증류를 도입하기 전에, FANET
라우팅 문제의 가장 작은 재현 가능한 기준선을 확립하는 구현이다.

## 완료된 범위

### Phase 1: 환경과 기준선

- 2차원 Random Waypoint UAV 이동성
- 통신 반경 기반 양방향 링크와 선택적 확률 손실
- 단일 패킷의 순차적 hop-by-hop 전달
- TTL, 경로 루프, 잘못된 행동, 명시적 `DROP` 처리
- 패딩된 1-hop 지역 관측과 구조적 action mask
- 전달 보상, 단계 지연 비용, 실패 비용만 사용하는 최소 보상
- Random 및 greedy GPSR 기준선
- PDR, 지연, 홉 수, 처리율, 루프 드롭률 평가

### Phase 2: Local Student

- 후보 이웃마다 파라미터를 공유하는 2-layer MLP encoder
- 유효한 이웃만 포함하는 permutation-invariant mean pooling
- 후보별 공유 logit scorer와 별도 `DROP` scorer
- invalid action 확률을 정확히 0으로 만드는 masked softmax
- 단일 관측과 batch 관측 지원
- 결정적 또는 sampling 실행을 위한 환경 adapter

### Phase 3: Global Teacher

- 전체 FANET graph를 입력받는 privileged reference policy
- edge-conditioned mean aggregation을 사용하는 2-layer message-passing GNN
- 현재 forwarding UAV의 1-hop action support에 대한 masked categorical actor
- global/current/destination embedding을 사용하는 centralized value network
- complete-episode return과 clipped PPO를 사용하는 CPU/CUDA/MPS 학습기
- Teacher, Random, GPSR, 무학습 Local Student를 동일 seed에서 비교하는 gate
- 모델 state와 실행 metadata 체크포인트 저장·복원

### Phase 4: Offline Policy Distillation

- gated Teacher의 deterministic rollout으로 local-only dataset 생성
- Teacher raw logits, masked probability, 선택 행동과 scenario metadata 저장
- `scenario_id + episode_seed` 단위 train/validation/test group split
- temperature를 지원하는 masked forward KL `KL(Teacher || Student)`
- batch KL, Teacher entropy, action agreement 평가
- Student checkpoint와 dataset NPZ 저장
- test KL·agreement·환경 PDR을 함께 검사하는 Phase 4 gate

### Phase 5: Local PPO Fine-tuning

- Phase 4 gated Student actor를 그대로 로드
- self·1-hop neighbor·edge·packet만 사용하는 local value network
- Teacher와 global graph를 받지 않는 on-policy rollout 인터페이스
- 기본 순수 clipped PPO fine-tuning
- 선택적 감소형 offline KD:
  \(\lambda_{KD}(k)=\lambda_0\exp(-\eta k)\)
- KD-only 전후 동일 seed 평가와 성능 보존 gate
- 배포용 Student policy checkpoint 저장

### Phase 6: Multi-seed Evaluation and Reporting

- Random, GPSR, 무학습 Student, PPO-only, KD-only, KD+PPO, Teacher 비교
- 학습 seed와 평가 seed를 분리한 episode-level raw CSV
- seed 평균, 표본 표준편차, Student-t 95% 신뢰구간
- link loss, mobility, density shift를 포함한 OOD 평가
- 파라미터 수, 입력 byte, 추론 지연, Python peak allocation 측정
- Markdown·LaTeX 표와 PNG·PDF·SVG 그래프 자동 생성
- 완료된 seed를 재사용하는 `--resume` checkpoint

### Phase 7: Generalization and Validity Hardening

- 매 episode 새로 생성되는 동적 topology와 무작위 source/destination
- 최소 2-hop 연결 endpoint pair를 사용하는 학습 curriculum
- easy, medium, hard topology family 순차 학습
- held-out seed와 link loss, fast mobility, sparse density, 10-node OOD
- 전체 PDR과 초기 연결 pair 조건부 PDR 분리
- dynamic shortest-path Oracle과 path stretch
- node ID를 제거한 Local Student 입력
- 거리 potential 차이를 사용하는 progress reward

### Phase 8: Geo-Residual Performance Optimization

- 방문 노드 action masking으로 routing loop를 구조적으로 차단
- GPSR geographic progress 위에 학습 residual을 더하는 Local Student
- DRAMA식 shortest-path auxiliary target과 Teacher forward-KL 결합
- Teacher rollout과 Oracle rollout의 상태 분포를 함께 사용
- 최소 2-hop forwardability beacon feature로 greedy dead end 식별
- 검증 PDR 저하 허용치 안에서 residual 강도를 자동 보정
- 평균·p95 지연, path stretch, ETX·energy·link-lifetime proxy,
  local input bytes, 실패 원인 분해
- 회전된 structural routing-hole과 link-loss hole 평가

Phase 3과 Phase 4의 routing-hole은 학습 가능성 검증용 toy scenario다. 여기서
얻은 수치를 일반 FANET 성능으로 해석하지 않는다. Student PPO fine-tuning은
toy 환경에서 검증했으며, 일반 FANET multi-seed 평가는 Phase 6 범위다.

## 설치

저장소 루트에서 실행한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lite-globe.txt
python -m pip install -e .
```

## 검증

```bash
python -m pytest tests/lite_globe
python -m implementations.lite_globe.run_phase1 --episodes 20 --seed 42
python -m implementations.lite_globe.run_phase2 --episodes 20 --seed 42
python -m implementations.lite_globe.run_phase3 \
  --evaluation-episodes 100 \
  --checkpoint artifacts/lite_globe/teacher_phase3.pt
python -m implementations.lite_globe.run_phase4
python -m implementations.lite_globe.run_phase5
python -m implementations.lite_globe.run_phase6 --smoke
python -m implementations.lite_globe.run_phase7 --smoke
python -m implementations.lite_globe.run_phase8 --smoke
```

결과를 파일로 남길 때만 `--output`을 사용한다. 생성 결과는 Git에서 제외되는
`artifacts/` 아래에 저장하는 것을 권장한다.

```bash
python -m implementations.lite_globe.run_phase1 \
  --episodes 100 \
  --seed 42 \
  --output artifacts/lite_globe/phase1_baselines.json
```

## 인터페이스

행동 `0..max_nodes-1`은 전역 UAV ID로 표현한 다음 홉이고,
`max_nodes`는 명시적 `DROP` 행동이다. `action_mask`는 현재 통신 가능한 실제
이웃과 `DROP`만 허용한다. GPSR도 환경 내부 상태가 아니라 정책에 제공된 지역
관측만 사용한다.

### Local Student 입력 shape

| 입력 | 단일 관측 shape | 의미 |
| --- | --- | --- |
| `self_features` | `(6,)` | 현재 위치·속도·queue·목적지 여부 |
| `neighbor_features` | `(max_nodes, 7)` | 상대 위치·속도·queue·목적지 여부·존재 표시 |
| `edge_features` | `(max_nodes, 2)` | 정규화 거리·현재 link 표시 |
| `packet_features` | `(6,)` | 목적지 방향·TTL·홉 수·source/destination ID |
| `action_mask` | `(max_nodes + 1,)` | 구조적으로 가능한 이웃과 `DROP` |

batch 입력은 각 shape 앞에 batch dimension을 추가한다. Student는 후보 행마다
동일한 encoder와 scorer를 적용하고 유효 후보의 평균만 context로 사용한다.
따라서 후보 행을 순열하면 후보별 출력도 동일하게 순열되며 `DROP` 확률은
변하지 않는다.

### Global Teacher 입력 shape

| 입력 | 단일 관측 shape | 의미 |
| --- | --- | --- |
| `node_features` | `(max_nodes, 9)` | 전역 위치·속도·queue·current/destination/source/visited |
| `adjacency` | `(max_nodes, max_nodes)` | 현재 전체 graph 연결성 |
| `edge_features` | `(max_nodes, max_nodes, 2)` | 정규화 거리와 link 표시 |
| `node_mask` | `(max_nodes,)` | 실제 존재하는 UAV |
| `packet_features` | `(2,)` | TTL과 hop count |
| `action_mask` | `(max_nodes + 1,)` | 현재 UAV의 1-hop 이웃과 `DROP` |

Teacher가 전체 graph를 보더라도 다음 홉으로 선택 가능한 행동은 Student와
동일하다. 전역 정보는 후보를 평가할 때만 사용하며 연결되지 않은 UAV를 직접
선택할 수 없다.

### Offline dataset 형식

압축 NPZ 파일에는 다음 배열만 저장한다.

- Student 입력: `self_features`, `neighbor_features`, `edge_features`,
  `packet_features`, `action_mask`
- Teacher target: `teacher_logits`, `teacher_probabilities`,
  `selected_actions`
- split metadata: `scenario_ids`, `episode_seeds`, `episode_steps`

`teacher_logits`는 temperature를 다시 적용할 수 있도록 mask 이전 finite raw
logit으로 저장한다. `teacher_probabilities`는 structural mask 적용 후 합이 1인
분포이며 invalid action은 정확히 0이다. 전체 adjacency, global node feature,
Teacher value와 global embedding은 저장하지 않는다.

Google Colab에서는 다음 노트북을 사용한다.

- `implementations/lite_globe/colab/phase3_teacher.ipynb`
- `implementations/lite_globe/colab/phase4_distillation.ipynb`
- `implementations/lite_globe/colab/phase5_finetune.ipynb`
- `implementations/lite_globe/colab/phase6_evaluation.ipynb`
- `implementations/lite_globe/colab/phase7_generalization.ipynb`
- `implementations/lite_globe/colab/phase8_optimization.ipynb`
- `implementations/lite_globe/colab/phase9_risk_aware.ipynb`

Phase 5 기본값은 순수 PPO다. 감소형 offline KD를 사용할 때만 Phase 4 dataset을
지정한다.

```bash
python -m implementations.lite_globe.run_phase5 \
  --kd-lambda 0.5 \
  --dataset artifacts/lite_globe/distillation_dataset.npz
```

Phase 6의 로컬 통합 검증과 Colab 전체 실행:

```bash
python -m implementations.lite_globe.run_phase6 --smoke
python -m implementations.lite_globe.run_phase6 \
  --device auto \
  --resume \
  --output-dir artifacts/lite_globe/phase6
```

`--resume`은 한 학습 seed의 모든 모델이 저장된 경우 해당 seed 학습을
건너뛴다. 현재 resume 단위는 PPO update가 아니라 완료된 seed이다.

Phase 8 전체 실행:

```bash
python -m implementations.lite_globe.run_phase8 \
  --device auto \
  --resume \
  --phase7-checkpoint-dir artifacts/lite_globe/phase7/checkpoints \
  --output-dir artifacts/lite_globe/phase8
```

Phase 8은 Phase 7의 `global_teacher.pt`와 `kd_only_student.pt`만 필요하다.
따라서 Colab에는 전체 workspace 대신 코드, 설정, 테스트와 약 3.5 MB의
필수 checkpoint만 옮겨도 된다.

최소 Colab bundle 생성:

```bash
python scripts/package_phase8_colab.py
```

생성 파일:
`artifacts/lite_globe/phase8_colab_bundle.zip`

Phase 9 전체 실행:

```bash
python -m implementations.lite_globe.run_phase9 \
  --device auto \
  --resume \
  --phase7-checkpoint-dir artifacts/lite_globe/phase7/checkpoints \
  --phase8-checkpoint-dir artifacts/lite_globe/phase8/checkpoints \
  --output-dir artifacts/lite_globe/phase9
```

Phase 9은 risk-aware link prediction, 강한 Predictive Geographic baseline,
Risk-aware Oracle, component ablation, severe link loss, 16·24 UAV,
predictive-break stress를 포함한다.

```bash
python scripts/package_phase9_colab.py
```

생성 파일:
`artifacts/lite_globe/phase9_colab_bundle.zip`

설계 가정과 일반화 영향은 [assumptions.md](assumptions.md)에 기록한다.
