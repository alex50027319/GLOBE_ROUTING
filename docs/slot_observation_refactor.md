# E-2: Local-Slot Observation and Size-Agnostic Policy

이 문서는 `LIMITATIONS_AND_EXTENSIONS_REVIEW.md`의 **F-11**(액션 공간이 32노드에 하드 고정,
`policy_input_bytes` 부풀림)에 대응하는 리팩터의 설계·근거·검증을 기술한다.

## 1. 문제

환경은 이웃 특징을 **전역 노드 ID로 인덱싱된** `(max_nodes, F)` 배열에 배치하고,
액션 공간도 같은 전역 ID 위의 `Discrete(max_nodes + 1)`이다.

```python
# env/fanet_env.py:38-39
self.drop_action = self.config.max_nodes                        # 32
self.action_space = spaces.Discrete(self.config.max_nodes + 1)  # 33

# env/observation.py:66  — node는 전역 인덱스
neighbor_features[node] = np.array([...])   # shape (32, 7)
```

결과:

1. **`max_nodes`를 넘는 스웜에 재학습 없이 배포 불가.** 라우팅 결정은 순수 로컬인데도.
2. **실제 차수와 무관하게 매 결정마다 32행을 읽는다.** `policy_input_bytes`가 부풀려지고,
   실제 이웃만 다루는 프로토콜(AODV 143 B)과 비교가 성립하지 않는다.
3. `include_node_ids=True`일 때 `packet_features`에 정규화된 **전역** source/destination ID가
   들어가 permutation 불변성과 일반화를 깨뜨린다.

## 2. 핵심 관찰: 재학습이 필요 없다

`LocalStudentPolicy`와 모든 서브클래스는 **공유 per-candidate 인코더**로 후보를 점수화하고
유효 후보에 대해서만 컨텍스트를 평균 풀링한다 (`student_policy.py:98-117`).

```python
expanded_shared = shared.unsqueeze(1).expand(-1, self.max_nodes, -1)
encoded = self.neighbor_encoder(candidate_input)     # 슬롯별 공유 가중치
valid = candidate_mask.unsqueeze(-1)
context = (encoded * valid).sum(1) / valid.sum(1)    # 유효 후보만 풀링
```

따라서 **`max_nodes` 차원을 갖는 파라미터 텐서가 하나도 없다.** 검증:

```
$ pytest tests/lite_globe/test_slot_observation.py::test_no_parameter_depends_on_max_nodes
keys identical: True
shape differences: NONE
cross-size load_state_dict: OK
```

두 가지가 따라온다:

- 모델은 후보 슬롯에 대해 **정확히 permutation-equivariant**하다.
- `max_nodes=32`로 학습한 체크포인트가 `max_nodes=8` 모델에 **그대로 로드된다**.

그러므로 로컬 슬롯으로의 압축은 **semantics-preserving re-indexing**이다. 선택되는 다음 홉은
변하지 않으면서 정책이 읽는 텐서만 실제 차수 크기로 줄어든다. **재학습이 필요 없다.**

이는 논문의 "semantics-preserving computation reuse" 논리를 forward 횟수가 아니라
**관측 레이아웃**에 적용한 것이다.

## 3. 구현

`implementations/lite_globe/models/slot_observation.py`

```python
def compact_observation(observation, *, max_slots, zero_node_ids=True)
    -> tuple[dict, SlotMapping]

class SlotCompactedPolicy:   # evaluator의 reset/act 프로토콜을 그대로 구현
    def __init__(self, model, *, env_drop_action, max_slots, ...)
```

동작:

1. `action_mask`에서 유효 후보를 전역 ID 오름차순으로 선행 슬롯에 압축한다.
2. 4개 candidate 배열(`neighbor_features`, `edge_features`, `candidate_forwardability`,
   `candidate_risk_features`)을 동일 순서로 재배치한다.
3. `SlotMapping`이 슬롯 → 전역 ID 역매핑을 보관하여 액션을 환경 액션으로 되돌린다.
4. 차수가 `max_slots`를 넘으면 **정규화 링크 거리 기준 가까운 순으로** 유지하고,
   잘린 개수를 `mapping.truncated`에 기록한다 — truncation은 절대 조용히 일어나지 않는다.
5. `zero_node_ids=True`(기본)가 `packet_features[4:6]`의 전역 ID를 0으로 만든다.

### 범위와 한계

정책 정의는 전혀 바뀌지 않는다. 유일한 동작상 주의점은 **argmax 동점 처리**다.
`torch.argmax`는 동점에서 최소 인덱스를 고르는데 압축이 후보 번호를 바꾸므로, 로짓이 정확히
동점이면 다른 전역 노드가 선택될 수 있다. 따라서 "동점을 제외하고 액션 보존"으로 문서화한다.

## 4. 검증

`tests/lite_globe/test_slot_observation.py` (20개 테스트, 전부 통과)

| 테스트 | 보장 |
| --- | --- |
| `test_no_parameter_depends_on_max_nodes` | 3개 정책 클래스 모두 state_dict 형상 동일, 교차 로드 가능 |
| `test_compacted_logits_match_per_candidate` | **후보별 로짓이 1e-4 이내로 동일** (argmax뿐 아니라) |
| `test_compaction_preserves_the_selected_next_hop` | 5개 시드에서 선택된 전역 노드 동일 |
| `test_permutation_of_slots_does_not_change_candidate_logits` | 슬롯 순열 하에서 equivariance |
| `test_compaction_reduces_policy_input_bytes` | 바이트 감소 |
| `test_truncation_is_recorded_and_keeps_nearest_candidates` | truncation 기록 + 가까운 후보 유지 |
| `test_zero_node_ids_suppresses_global_identifiers` | 전역 ID 억제 |
| `test_dead_end_returns_environment_drop_action` | 막다른 골목에서 환경 DROP |

핵심은 두 번째 항목이다. argmax만 같은 것이 아니라 **모든 후보의 로짓이 수치적으로 동일**하므로
압축은 근사가 아니라 정확한 재색인이다.

## 5. 측정 결과

14개 평가 시나리오, seed 0–149, 총 **6,450개 결정** 샘플:

```
relay degree: mean=1.96   p95=5   max=10
환경은 32개 후보 슬롯을 할당 -> 모든 관측의 93.9%가 padding
```

| K | 결정당 바이트 | 감소 | truncated 결정 |
| --- | --- | --- | --- |
| 8 | 537 B | **3.73×** | 7 / 6450 (0.11%) |
| **12** | **781 B** | **2.56×** | **0 / 6450 (0.00%)** ← 무손실 |
| 16 | 1025 B | 1.95× | 0 |
| 32 (현행) | 2001 B | 1.00× | 0 |

**권장: K = 12.** 평가 스위트 전체에서 무손실이면서 2.56× 절감.

### 논문 수치에 대한 함의

논문 Table 2의 `DiSwitch policy input bytes = 4821`은 **에피소드당** 값이며
(≈ 2.4 결정/에피소드 × 2001 B), 그중 **약 94%가 padding**이다.

로컬 슬롯 레이아웃(K=12)을 쓰면 정직한 값은 에피소드당 약 **1,880 B**가 된다.
AODV(143 B)와의 비교는 여전히 AODV에 유리하지만, 현재 표가 시사하는 **34배 차이가 아니라
13배**이며, §F-5(제안기법의 control_bytes = 0.0)를 함께 교정하면 방향 자체가 뒤집힌다.

### 부수 관찰: 결정 문제가 매우 작다

평균 릴레이 차수 **1.96**이다. 즉 전형적인 "라우팅 결정"은 후보가 2개다. 검토 문서 §5.1의
평균 홉 수 2.2–3.4와 결합하면, 현재 시뮬레이터의 의사결정 문제 규모가 매우 작다는 점이
정량적으로 확인된다.

## 6. 사용법

```python
from implementations.lite_globe.models.slot_observation import SlotCompactedPolicy

# max_nodes=32로 학습된 체크포인트를 K=12 모델에 그대로 로드
narrow = GeographicResidualStudentPolicy(12, hidden_dim=64)
narrow.load_state_dict(torch.load("geo_residual_kd.pt")["model"])

policy = SlotCompactedPolicy(narrow, env_drop_action=32, max_slots=12)
# 이후 기존 evaluator에 그대로 투입 가능 (reset/act 프로토콜 동일)
```

`policy.last_input_bytes`와 `policy.total_truncated`로 결정당 실제 footprint와 truncation을
감사할 수 있다.

## 7. 남은 작업

이 리팩터는 **관측 레이아웃과 액션 색인만** 바꾼다. 완전한 크기 무관 배포를 위해 남은 항목:

1. **환경 액션 공간 자체를 슬롯 인덱스로 전환.** 현재는 어댑터가 전역 ID로 되돌려주므로
   환경은 그대로다. 환경까지 바꾸면 `max_nodes` 개념이 완전히 사라진다.
2. **32노드 초과 시나리오에서의 검증.** 현행 평가 스위트의 최대 노드 수는 24이므로
   크기 무관성의 실질적 이득(예: 100노드)은 아직 측정되지 않았다. 검토 문서 §5.1(E-11)과
   함께 수행해야 한다.
3. ~~`SwitchGlobePolicy` 래핑~~ — **검증 완료.** 14개 시나리오의 실제 궤적에서
   **1,033개 결정 전부**(100.00%) 선택된 다음 홉과 **switch flag**가 32-wide 기준과 일치했다.
   `_switch_mask`가 `risk[batch, action]`으로 접근하므로 압축된 관측에서 자동으로 일관된다.
   `test_switchglobe_decision_survives_compaction`으로 고정했다.
4. ~~재보정 여부 판단~~ — **불필요.** 로짓과 switch 판정이 모두 동일하므로 gate 재보정
   (`switch_threshold`, `margin_gate`, `lifetime_gate`, `onward_gate`)이 필요 없다.

## 8. 관련 파일

| 경로 | 역할 |
| --- | --- |
| `implementations/lite_globe/models/slot_observation.py` | 압축, 역매핑, 정책 래퍼 |
| `tests/lite_globe/test_slot_observation.py` | 20개 계약 테스트 |
