# DiSwitch / SwitchGLOBE — 한계 및 확장성 전문가 검토

**작성일**: 2026-09-09
**대상**: `outputs/refactor/ieee_access/originals/DiSwitch_IEEEAccess_Manuscript_2026-09-07.pdf`
**브랜치**: `refactor/globev2` (HEAD `5edff2e`)
**검토 방식**: 원고 텍스트가 아니라 `implementations/`의 실행 코드와 `artifacts/`의 full-run raw
아티팩트를 직접 재집계하여 검증. 모든 수치는 저장소 파일에서 재현 가능하며, 재현 명령은
§9에 수록.

> 본 문서는 `source/internal_peer_review.md`(6개 항목)를 대체하지 않고 확장한다. 기존 문서가
> 제기하지 않은 **구조적 결함**이 §1에 있으며, 이는 제출 전 반드시 해소해야 한다.

## 후속 작업 현황 (2026-09-09 갱신)

| 항목 | 상태 | 산출물 |
| --- | --- | --- |
| **재프레이밍 초안 (옵션 A)** | 작성 완료 | `REFRAMING_DRAFT_OPTION_A.md` |
| **E-1 beacon 열화 실험** | 구현·테스트·smoke 검증 완료. full run은 체크포인트 필요 | `docs/beacon_degradation_study.md`, `env/beacon.py`, `scenarios/degradation_suite.py`, `scripts/run_beacon_degradation_study.py`, 테스트 18개 |
| **E-2 local-slot 관측** | 구현·검증 완료. **재학습 불필요** | `docs/slot_observation_refactor.md`, `models/slot_observation.py`, 테스트 20개 |

이 작업 중 **F-14**(Phase 8/11/12 체크포인트 부재)와 **F-15**(reporting `None` 가드 결함)를
추가로 발견했다. 아래 표와 §7에 반영했다. 테스트 스위트는 127개 → **166개** (전부 통과).

---

## 목차

- [0. Executive Summary](#0-executive-summary)
- [1. 치명적 결함: 제안기법의 두 축이 효과 0으로 측정됨](#1-치명적-결함-제안기법의-두-축이-효과-0으로-측정됨)
- [2. Deployment contract 주장과 구현의 불일치](#2-deployment-contract-주장과-구현의-불일치)
- [3. 시뮬레이터 충실도 한계](#3-시뮬레이터-충실도-한계)
- [4. 평가 설계 및 통계의 취약점](#4-평가-설계-및-통계의-취약점)
- [5. 확장성 한계](#5-확장성-한계)
- [6. 정책 내부 구조에서 발견된 숨은 동작](#6-정책-내부-구조에서-발견된-숨은-동작)
- [7. 코드 및 아티팩트 위생](#7-코드-및-아티팩트-위생)
- [8. 우선순위 실행 계획](#8-우선순위-실행-계획)
- [9. 재현 명령](#9-재현-명령)
- [부록 A. 검증된 수치 요약표](#부록-a-검증된-수치-요약표)

---

## 0. Executive Summary

| # | 발견 | 심각도 | 근거 |
|---|---|---|---|
| **F-1** | `Predictive Prior Only` ablation이 DiSwitch와 connected-pair PDR **완전 동일**(paired diff = 0.00000000, 95% CI ±0.083pp)하면서 **4.19× 빠름** | 🔴 **치명적** | `artifacts/final_paper_simulation/full/ablation/raw/seed_summaries.csv` |
| **F-2** | 논문 제목의 risk switch가 flagship 시나리오 4개 전부에서 효과 **정확히 0.00pp** | 🔴 **치명적** | 위 동일 |
| **F-3** | Predictive branch에 신경망이 없음 (`residual_weight=0` → MLP 우회). 학습 스칼라 ~10개짜리 닫힌 형태 함수 | 🔴 **치명적** | `models/student_policy.py:626, 477-488` |
| **F-4** | "strict 1-hop" 주장과 달리 실제로는 **2-hop kinematic 관측** 필요 | 🟠 높음 | `env/observation.py:85-123` |
| **F-5** | 제안기법만 `control_bytes = 0.0`, baseline은 최대 2107 B 과금 | 🟠 높음 | `artifacts/external_comparison_analysis/seed_summaries_all.csv` |
| **F-6** | Risk feature가 ground-truth. beacon 지연·잡음 모델이 **코드에 존재하지 않음** | 🟠 높음 | `env/`, `scenarios/` 전체 grep |
| **F-7** | Flagship 시나리오 2개가 유효 표본 n=1 (200 에피소드 전부 동일 결과) | 🟠 높음 | `episodes_all.csv` |
| **F-8** | "OOD"가 동일 지오메트리의 **회전각** 차이일 뿐 | 🟠 높음 | `scenarios/predictive_traps.py`, `structural_holes.py` |
| **F-13** | 14개 시나리오의 유효 독립 설정은 9–10개. `heldout_medium`은 학습 config와 동일, `ood_sparse`≡`unconditional_sparse` | 🟠 높음 | `scenarios/generalization_suite.py:81, 103-123` |
| **F-14** | Phase 8/11/12 체크포인트가 저장소에 **없어** 헤드라인 결과를 재생성할 수 없음. `docs`가 안내하는 `artifacts/switchglobe/final/checkpoints`도 부재 | 🟠 높음 | `find . -name "*.pt"` (Fast-SwitchGLOBE와 baseline만 존재) |
| **F-15** | 6개 reporting 모듈에서 `float(row[metric])`에 `None` 가드 부재 → smoke 파이프라인 중단. allowlist가 캠페인마다 상이 | 🟡 중간 | `phase7/8/11/12_reporting.py`, `reporting.py`, `baseline_reporting.py` (§7.5) |
| **F-16** | RWP 이동성이 구간별 등속이고 `predicted_link_lifetime_steps`도 등속 가정 → **`l_v`가 링크 만료의 완벽한 oracle**. 1-step stale beacon 오차 = 0.0000 | 🟠 높음 | `docs/beacon_degradation_study.md` §8 |
| **F-9** | 헤드라인 aggregate가 baseline 함정 시나리오에서 발생. node-count 축에서는 **AODV가 DiSwitch를 이김** | 🟠 높음 | 시나리오별 재집계 |
| **F-10** | 큐가 reset 시 난수로 고정, step에서 갱신되지 않음 → `q_v` 정보량 0 | 🟡 중간 | `env/fanet_env.py:173-177` |
| **F-11** | 액션 공간이 전역 노드 ID(`Discrete(33)`)에 하드 고정 → 32노드 초과 확장 불가 | 🟡 중간 | `env/fanet_env.py:38-39` |
| **F-12** | Eq.(3)의 `λ_L`, `λ_E` 항이 실제 보상 함수에 없음 | 🟡 중간 | `env/reward.py:24-29` |

**핵심 판단**: 현행 프레이밍(2-branch + risk switch가 기여)은 저자 자신의 5-seed full run이
지지하지 않는다. 그러나 데이터는 **더 강하고 더 방어 가능한 다른 논문**을 지지한다 — "증류로
학습한 해석 가능한 예측형 스코어 함수가 2-branch 스위칭 정책과 통계적으로 동등하면서 4배
빠르다"는 단순화/negative-result 논문. §1.5에 재프레이밍 옵션을 제시한다.

---

## 1. 치명적 결함: 제안기법의 두 축이 효과 0으로 측정됨

### 1.1 관측 사실

`artifacts/final_paper_simulation/full/ablation/raw/seed_summaries.csv`
(5 seed × 14 scenario × 200 episode = 14,000 episode/method, full run) 재집계 결과:

| 방법 | 42 | 77 | 123 | 314 | 2718 | **macro cpPDR** | **p95 latency** |
|---|---|---|---|---|---|---|---|
| Geo-Residual Student | 0.8032 | 0.8032 | 0.8014 | 0.8035 | 0.8039 | 0.803027 | — |
| **Predictive Prior Only** | 0.9062 | 0.9055 | 0.9044 | 0.9059 | 0.9044 | **0.90528954** | **1.487 ms** |
| Predictive Student (No Switch) | 0.9062 | 0.9051 | 0.9051 | 0.9059 | **0.8269** | 0.889864 | — |
| **SwitchGLOBE Exact (= DiSwitch)** | 0.9062 | 0.9059 | 0.9034 | 0.9059 | 0.9051 | **0.90528954** | **6.228 ms** |

`0.90529`는 논문 초록의 헤드라인 수치 **0.9053**이다.

### 1.2 Seed-paired 통계 (DiSwitch − Prior Only)

논문 §V-C의 프로토콜(seed-paired, 5쌍, two-sided 95% Student-t)을 그대로 적용:

| 지표 | seed별 paired diff | 평균 | 95% t-CI | 판정 |
|---|---|---|---|---|
| **connected-pair PDR** | `[0.0, +0.000357, −0.001071, 0.0, +0.000714]` | **+0.00000000** | **[−0.000829, +0.000829]** | 완전 동등 |
| deadline delivery ratio | `[−0.000357, −0.000357, −0.000714, +0.000714, +0.001071]` | +0.00007143 | [−0.000890, +0.001033] | 동등 |
| overall PDR | `[−0.000357, 0.0, −0.001429, 0.0, +0.000357]` | −0.00028571 | [−0.001139, +0.000567] | 동등 |
| **p95 decision latency** | `[4.636, 4.685, 4.814, 4.738, 4.836]` | **+4.74183 ms** | **[+4.637, +4.847]** | DiSwitch가 **일방적으로 느림** |

- connected-pair PDR의 paired difference는 **정확히 0**이고, 등가 구간은 **±0.083pp**로
  논문 자신의 non-inferiority margin **±0.5pp(Eq. 5)** 안쪽에 여유 있게 들어간다.
- latency 차이는 5 seed 전부 같은 방향, CI가 0에서 멀리 떨어져 있다.

즉 이것은 "비슷해 보인다"가 아니라 **형식적 등가성(equivalence) + 일방적 열세(inferiority)**
결과다.

### 1.3 시나리오별 분해 — flagship 시나리오에서 효과가 정확히 0

| 시나리오 | Prior Only | DiSwitch | 차이 (pp) |
|---|---|---|---|
| heldout_medium | 0.9850 | 0.9850 | **±0.00** |
| ood_nodes_10 | 0.9800 | 0.9800 | **±0.00** |
| ood_nodes_24 | 0.9380 | 0.9380 | **±0.00** |
| ood_sparse | 0.9800 | 0.9800 | **±0.00** |
| unconditional_sparse | 0.9870 | 0.9870 | **±0.00** |
| **predictive_break_45** | 1.0000 | 1.0000 | **±0.00** |
| **predictive_break_225_link_loss** | 0.5116 | 0.5116 | **±0.00** |
| **structural_hole_45** | 1.0000 | 1.0000 | **±0.00** |
| **structural_hole_225_link_loss** | 0.6984 | 0.6984 | **±0.00** |
| ood_extreme_mobility | 0.9660 | 0.9670 | +0.10 |
| ood_link_loss | 0.9100 | 0.9110 | +0.10 |
| ood_nodes_16 | 0.9730 | 0.9720 | −0.10 |
| ood_fast_mobility | 0.9610 | 0.9660 | +0.50 |
| ood_link_loss_30 | 0.7840 | 0.7780 | **−0.60** |

14개 중 **9개가 완전히 동일**하고, risk switch를 입증하려고 설계한 **4개 flagship 시나리오가
전부 정확히 동일**하다. 나머지 5개는 부호가 갈리는 ±0.6pp 노이즈다.

### 1.4 왜 이렇게 되는가 — 구현 추적

#### (a) `Predictive Prior Only`는 DiSwitch 내부의 predictive branch 그 자체다

```python
# implementations/lite_globe/experiments/phase12_campaign.py:334-340
predictive_only = LiteGlobePStudentPolicy(max_nodes, hidden_dim=config.hidden_dim)
predictive_only.load_state_dict(phase11.state_dict())
predictive_only.set_residual_weight(0.0)
risk_switch = _risk_switch_policy(phase8, phase11)   # ← 같은 phase11 가중치
```

```python
# implementations/lite_globe/models/student_policy.py:626
class RiskSwitchLiteGlobePStudentPolicy(nn.Module):
    def __init__(self, normal_policy, predictive_policy, ...):
        ...
        self.predictive_policy.set_residual_weight(0.0)   # ← DiSwitch도 동일하게 0
```

두 방법의 predictive branch는 **동일 가중치, 동일 설정**이다. 차이는 DiSwitch에 normal branch와
switch가 추가된 것뿐이며, 그 추가분의 순효과가 0이다.

#### (b) Predictive branch에는 신경망이 없다

```python
# implementations/lite_globe/models/student_policy.py:289-295
def set_residual_weight(self, weight: float) -> None:
    self.residual_weight.fill_(weight)
    self._residual_enabled = weight > 0.0        # ← 0.0이면 False

# implementations/lite_globe/models/student_policy.py:477-488
if self._residual_enabled:
    residual = LocalStudentPolicy.forward(self, observation)   # ← MLP. 호출되지 않음
    residual_logits = residual.logits
else:
    residual_logits = torch.zeros(...)           # ← 항상 이 경로
```

따라서 배포 시 predictive branch의 실효 스코어 함수는

```
z_v = α·Δd_v  +  w_f·f_v  +  Σ_{k∈{m,ℓ,q,o}} s_k·r_v[k]  −  P·[γ_m−m_v]₊+[γ_ℓ−ℓ_v]₊+[γ_o−o_v]₊
```

학습 가능한 **스칼라 파라미터 약 10개**(`log_prior_strength` 1, `log_forwardability_strength` 2,
`log_predictive_strength` 4, `log_break_penalty` 1, `log_residual_bound` 1)로 이루어진
**닫힌 형태의 선형 스코어 함수**다. Hidden layer도, 비선형 변환도 없다.

> **논문 서술과의 충돌**: 초록·§IV-B는 "its masked preferences are distilled into one-hop
> **normal and predictive student branches**"라고 쓰지만, predictive branch는 증류된 신경망이
> 아니라 **증류 과정에서 스칼라 계수만 학습한 해석 가능 휴리스틱**이다. §IV-D가
> "the learned residual contribution is set to zero after checkpoint loading"으로 언급은
> 하지만, 그 결과 신경망이 완전히 사라진다는 사실은 명시하지 않는다.

#### (c) Ablation 표가 이 사실을 가리는 구조

`source/tables/ablation_component_effects.csv`의 "Risk switching" 행:

```
Risk switching, Predictive Student (No Switch), DiSwitch, connected_pair_pdr,
  mean=+1.6621 pp, ci95=[-3.0345, +6.3587]
```

- base가 **`Predictive Student (No Switch)`** (residual MLP를 **켠** 변종, macro 0.8899)다.
- 그런데 그 base는 **seed 2718에서만 0.8269로 붕괴**하고 나머지 4개 seed는 0.905 대다.
- 따라서 "+1.66pp"는 **오로지 seed 2718 하나**에서 나오며, 95% CI가 **0을 포함**한다.
- 올바른 대조군인 `Predictive Prior Only`(0.90529)는 **같은 표 안에 이미 있으며**, 그것과의
  차이는 0이다.

**정직한 인과 서술**은 이렇다:

> 학습된 residual MLP는 seed에 따라 불안정하게 붕괴한다(2718: −7.8pp). Residual을 0으로 두면
> 안정화된다. Risk switch는 그 위에 얹은 **중복된 두 번째 안전장치**이며, residual이 이미 꺼져
> 있으므로 추가 이득이 없다.

#### (d) Normal branch가 유일한 MLP다 — 그리고 필요 없다

DiSwitch의 두 forward 중 실제 연산 비용은 거의 전부 normal(geo-residual) branch의 MLP에서
나온다. §1.2가 보여주듯 그 branch를 제거해도 신뢰도가 변하지 않는다. 즉:

- 논문의 **systems contribution 전체**(중복 forward 제거, 10.004 → 5.836 ms)는 **필요 없는
  branch를 두 번 대신 한 번 돌린 것**이다.
- 그 branch를 아예 제거하면 6.228 → 1.487 ms로, 중복 제거보다 훨씬 큰 개선을 얻는다.

### 1.5 논문 자신의 acceptance gate가 Prior Only를 선택한다

논문 Eq.(5):

> `LCB₀.₉₅(Δ_C) ≥ −0.005` 이고 `LCB₀.₉₅(Δ_D) ≥ −0.005` 이며 delay·energy에 실질적 열화가 없을 것

`Prior Only`를 후보로 놓고 평가하면:

| 조건 | 값 | 통과 |
|---|---|---|
| `LCB₀.₉₅(Δ_C)` | **−0.000829** ≥ −0.005 | ✅ |
| `LCB₀.₉₅(Δ_D)` | **−0.001033** ≥ −0.005 | ✅ |
| p95 delay | 4.334 vs 4.264 steps (+0.07 step) | ✅ (정수 분위수 해상도 내) |
| energy proxy | 2.2234 vs 2.2279 (−0.005) | ✅ (오히려 개선) |
| **p95 decision latency** | **1.487 vs 6.228 ms (4.19× 빠름)** | ✅✅ |

논문 §VI-F는 Fast-DiSwitch를 gate 실패로 기각한다(`Δ_C = −0.01809`, CI 하단 −0.03606).
그런데 **gate를 여유 있게 통과하면서 더 빠른 후보가 이미 자기 ablation 표 안에 있다.**
논문 자신의 결정 규칙을 일관되게 적용하면 제안기법은 DiSwitch가 아니라 Prior Only여야 한다.

### 1.6 대응 옵션

| 옵션 | 내용 | 장단점 | 권고 |
|---|---|---|---|
| **A. 재프레이밍** | 제안기법을 Prior-Only 계열로 전환. 논문을 "privileged distillation으로 학습한 **해석 가능한 예측형 스코어 함수**가 2-branch 스위칭 정책과 통계적으로 동등하면서 4× 빠르다"는 단순화/negative-result 논문으로 재구성 | 데이터가 그대로 지지. 방어 가능. IEEE Access의 "실용적 검증" 성향에 부합. 해석 가능성·경량성 서사가 오히려 강해짐 | ⭐ **권장** |
| **B. Switch 구제** | switch가 실제로 필요한 조건을 새로 설계·재실험 (§2.4의 beacon 열화 실험이 유일한 현실적 경로) | 성공 시 원래 서사 유지. 그러나 `docs/method_history.md` 기준 이미 4개 개입이 실패 | 조건부 |
| **C. 현행 유지** | — | **불가**. `source/tables/ablation_overall_metrics.csv`가 원고 소스에 **이미 동봉**되어 있어 심사위원이 `Predictive Prior Only = 0.9116116504854368`과 `DiSwitch = 0.9116116504854368`을 나란히 보게 됨 | ❌ |

> 옵션 A를 택하더라도 normal branch와 switch는 **ablation으로 반드시 남겨야** 한다. "왜 더
> 복잡한 구조가 도움이 되지 않는가"가 이 논문의 실질적 기여가 되기 때문이다.

---

## 2. Deployment contract 주장과 구현의 불일치

### 2.1 F-4: "strict 1-hop"이 아니라 2-hop이다

```python
# implementations/lite_globe/env/observation.py:85-123
onward = adjacency[node].copy()          # node = 1-hop 이웃 → adjacency[node]는 2-hop 인접성
onward[current] = False
for visited in packet.path:
    onward[visited] = False
onward_count = int(np.count_nonzero(onward))          # → candidate_forwardability f_v

for onward_node in np.flatnonzero(onward):
    onward_lifetimes.append(
        predicted_link_lifetime_steps(
            positions[int(onward_node)] - positions[node],       # 2-hop 이웃의 위치
            velocities[int(onward_node)] - velocities[node],     # 2-hop 이웃의 속도
            ...))
best_onward_lifetime = max(onward_lifetimes)          # → risk feature o_v
```

릴레이 `u`가 `o_v`(best-onward-link lifetime)와 `f_v`(forwardability)를 계산하려면 **이웃의
이웃의 위치와 속도**를 알아야 한다.

논문의 서술:
- Abstract: "distilled into **one-hop** normal and predictive student branches"
- §I: "restricts online forwarding to **relay-local and one-hop** information"
- §III-A: "The deployment contract excludes the full adjacency matrix, remote hidden states, and teacher queries"
- Fig. 2: "Strict-local input — relay + **one-hop neighbors**"

`o_v`는 본질적으로 2-hop 양이다. FANET 라우팅 심사위원이 가장 먼저 잡을 지점이며,
§1의 결론(predictive prior가 방법의 전부)과 결합하면 **방법의 핵심 신호가 2-hop 정보**라는
뜻이 되므로 더 중요하다.

**조치**:
1. 관측 계약을 "1-hop topology + **2-hop kinematic summary**"로 정확히 재기술.
2. §III-A 식 (2)의 `o_v` 정의에 2-hop 의존성을 명시.
3. Fig. 2의 "Strict-local input" 박스를 수정.
4. 2-hop 상태 유지에 필요한 beacon 주기·바이트를 §VII-C에 정량 제시
   (현재는 "require beacons or local prediction"이라는 정성 언급만 있음).

### 2.2 F-5: 제안기법만 control overhead가 0으로 기록된다

`artifacts/external_comparison_analysis/seed_summaries_all.csv` 재집계
(5 seed × 14 scenario 평균, episode당):

| Method | `mean_control_bytes` | `mean_control_messages` | `mean_policy_input_bytes` |
|---|---|---|---|
| OLSR | **2107.5** | 80.47 | 138 |
| AODV | **1014.0** | 43.15 | 143 |
| Evo-QGeo (Adapted) | **102.6** | 6.41 | 6452 |
| Greedy Geographic | 0.0 | 0.00 | 2284 |
| RDQN-HERP (Adapted) | 0.0 | 0.00 | 4977 |
| GAT-GRU-DDQN | 0.0 | 0.00 | 4451 |
| **SwitchGLOBE** | **0.0** | **0.00** | 4821 |

Evo-QGeo는 **동일한** `candidate_risk_features`를 사용하면서 candidate당 16 B를 정직하게
과금한다:

```python
# implementations/lite_globe/baselines/evo_qgeo.py:31-34
self.control_messages += int(candidates.size)
# node id, link estimate, expected duration and scalar Q value
self.control_bytes += int(candidates.size) * 16
```

SwitchGLOBE는 같은(그리고 2-hop까지 더 넓은) 필드를 쓰면서 0이다.

논문은 `control_bytes` 컬럼을 **보고하지 않고** `policy_input_bytes`만 Table 2·Fig. 6에 싣는다.
그 결과 AODV가 143 B로 "가벼워 보이는" 그림이 되는데, **실제 무선 오버헤드 축에서는 정확히
반대 방향**이다. Fig. 6 캡션의 "Small input size for AODV/OLSR is not wireless routing
overhead"는 방향을 명시하지 않아 오히려 오해를 굳힌다.

또한 `docs/simulation_protocol.md` §5는

> 해당 control plane이 구현되지 않은 결과에는 `routing overhead: not modeled`를 명시한다

라고 규정하지만, CSV에는 `not modeled`가 아니라 **`0.0`**이 들어 있어 하류 도구가 "오버헤드
없음"으로 읽을 수 있다.

**조치**:
1. Table 2에 `control_bytes` 열 추가, 또는 미구현 method 전부에 `not modeled` 문자열 기록.
2. `evaluation/` 집계 코드에서 미측정 필드를 `0.0`이 아니라 `NaN`/`None`으로 기록.
3. SwitchGLOBE의 관측 필드를 유지하는 데 필요한 beacon 바이트를 Evo-QGeo와 **동일한 회계
   규칙**으로 산정해 보고 (candidate당 위치 2 × float + 속도 2 × float + queue 1 = 최소 20 B,
   2-hop 요약 포함 시 더 큼).

### 2.3 F-2b: 제안기법이 fairness registry 바깥에 있다

```python
# implementations/lite_globe/baselines/registry.py
@dataclass(frozen=True)
class MethodSpec:
    name, slug, source, fidelity, trainable, control_plane,
    observation_fields, hop_radius, privileged_information, builder

METHOD_REGISTRY = {...}          # 6개 baseline만 등록
PROPOSED_METHOD = "SwitchGLOBE"  # ← registry 바깥
COMPARISON_METHODS = (*EXTERNAL_METHODS, PROPOSED_METHOD)
```

모든 baseline은 `hop_radius`, `control_plane`, `observation_fields`,
`privileged_information`을 매니페스트에 강제로 남긴다. 예:

```
Evo-QGeo (Adapted): hop_radius="1-hop plus beacon Q summary", control_plane=True
Greedy Geographic:  fidelity="partial: perimeter recovery not implemented"
GAT-GRU-DDQN:       fidelity="inspired architecture control; not SRRGD-DQN"
```

제안기법만 이 계약의 적용을 받지 않는다. §2.1·§2.2와 결합하면, **가장 검증이 필요한 방법이
가장 적게 검증되는** 구조다.

**조치**: `SwitchGLOBE`를 동일한 `MethodSpec`으로 등록하고
(`hop_radius="1-hop topology + 2-hop kinematic"`, `control_plane=False (not modeled)`),
매니페스트에 찍히게 한다. 이것만으로 §2.1·§2.2가 자동으로 문서화된다.

### 2.4 F-6: Risk feature가 ground-truth이며 열화 실험이 전무하다 ⭐

`implementations/lite_globe/env/`와 `implementations/lite_globe/scenarios/` 전체에
`beacon`, `stale`, `noise`, `observation_noise`, `beacon_period` 관련 코드가 **한 줄도 없다**.

```
$ grep -rn "stale|observation_noise|beacon_period|noisy" implementations/lite_globe/env/ \
    implementations/lite_globe/scenarios/
(결과 없음)
```

`m_v`(link margin), `ℓ_v`(link lifetime), `o_v`(onward lifetime)는 시뮬레이터 내부의 **진짜
위치·속도**에서 **오차 0, 지연 0, 손실 0**으로 계산된다.

§1에서 확인했듯 이 세 신호가 **사실상 방법의 전부**다. 그런데 신호가 낡거나 틀렸을 때의
민감도 분석이 전혀 없다. 이것이 현재 가장 시급하고 **논문 가치가 가장 높은** 추가 실험이다.

**제안 실험 설계 (E-1)**

| 축 | 값 | 근거 |
|---|---|---|
| beacon 주기 `T_b` | 1, 2, 5, 10 step | `heldout_medium`의 링크 수명이 ~10–20 step이므로 이 범위에서 열화가 드러남 |
| beacon 손실률 | 0.0, 0.1, 0.3 | `ood_link_loss`, `ood_link_loss_30`과 동일 스케일 |
| 위치/속도 추정 잡음 σ | 0, 5%, 15% (of R, of v_max) | GPS/IMU 급 오차 범위 |

구현은 `env/observation.py`에 **관측 열화 레이어**를 추가하는 것으로 충분하다 —
마지막 beacon 시점의 상태를 캐싱하고 등속 외삽 + 가우시안 잡음을 적용.

**기대 결과와 논문적 가치**:
- Prior-Only가 열화에 취약하고 switch가 그 지점에서 복구를 제공한다면 → **옵션 B가 살아나고
  논문이 훨씬 강해진다**. Switch의 존재 이유가 비로소 물리적으로 정당화된다.
- 둘 다 동일하게 열화한다면 → 옵션 A의 근거가 강화되며, "예측형 라우팅의 실질 한계는 모델이
  아니라 신호 신선도"라는 강한 메시지를 얻는다.

어느 쪽이든 지금보다 나은 논문이 된다.

---

## 3. 시뮬레이터 충실도 한계

| 항목 | 현재 구현 | 위치 | 결핍 |
|---|---|---|---|
| **채널** | `distances <= R` unit-disk (binary) | `env/link_model.py:54` | Path loss, SNR, Rician/Nakagami 페이딩, A2A LoS 확률, 안테나 패턴, 지향성 없음 |
| **패킷 손실** | i.i.d. Bernoulli edge drop | `env/link_model.py:56-60` | 공간·시간 **상관 없음**. 실제 페이딩/차폐는 강한 상관 구조. "link loss 30%"의 물리적 해석 불가 |
| **간섭·MAC** | 없음 | — | 경합, CSMA backoff, 재전송, hidden terminal, capture effect 전부 부재 |
| **이동성** | 2D Random Waypoint, pause 없음 | `env/mobility.py` | **고도(3D) 없음**. UAV 표준인 Gauss-Markov / semi-random circular / Paparazzi / group mobility 미사용. RWP의 speed-decay·border effect 미보정. **그리고 §3.3의 예측기 공모 문제** |
| **트래픽** | 에피소드당 **단일 패킷** | `env/packet.py`, `fanet_env.py:31` | 혼잡·큐잉·throughput·goodput 주장 원천 불가 |
| **큐** | reset 시 random 정수, step에서 **갱신 안 됨** | `env/fanet_env.py:173-177` | `q_v` 정보량 0 (아래 §3.1) |
| **에너지** | `(d/R)²` 누적 | `env/fanet_env.py:343` | 전력 제어, 회로 전력, 호버링 전력, 수신 전력 없음. Joule 아님 (문서에 명시되어 있음 — 유지 필요) |
| **전송 성공** | mask에 있으면 **결정론적 성공** | `fanet_env.py:222-234` | 전송 시점 PHY 실패·재전송 없음. 손실은 topology refresh에서만 적용 |

### 3.1 F-10: 큐가 정적이다

```python
# implementations/lite_globe/env/fanet_env.py:173-177  (reset 안)
self.queues = self.np_random.integers(
    0, self.config.max_queue_size + 1, size=self.config.num_nodes
).astype(np.float32)
```

`step()` 어디에서도 `self.queues`가 갱신되지 않는다. 즉 `q_v`(queue headroom)는 에피소드 내내
**변하지 않는 난수 상수**다.

이는 논문 §IV-D의 다음 서술을 재해석하게 만든다:

> "Queue headroom is available to the predictive prior but is intentionally not part of the
> switch danger in the verified implementation."

이것은 설계 선택처럼 읽히지만, 실제로는 **그 feature가 어떤 동적 정보도 담고 있지 않기
때문**에 danger에서 빠진 것으로 보아야 한다. `cumulative_queue_delay_proxy`
(`fanet_env.py:344-348`)도 같은 정적 난수에서 계산되므로 congestion 관련 해석이 불가능하다.

**조치 (택일)**:
- (a) 큐 동역학 구현: multi-flow 트래픽 도입 + arrival/service 프로세스. §5.3의 확장과 함께
  수행하면 자연스럽다.
- (b) `q_v`를 관측에서 제거하고, `initial_predictive_strength`의 3번째 성분(0.25)을 삭제한 뒤
  재보정. 논문에서 `r_v`를 3차원으로 재정의.

어느 쪽이든 현행 서술(§IV-D)은 수정이 필요하다.

### 3.2 "Dynamic FANET"이 대부분의 시나리오에서 준정적이다

주 평가 시나리오의 파라미터 (`scenarios/generalization_suite.py`):

| 시나리오 | N | area | R | max_speed | `v/R` per step | 12 step 최대 이동 |
|---|---|---|---|---|---|---|
| `_base` (phase7 stage 0) | 8 | 10.0 | 4.5 | 0.05 | **1.1%** | 0.13 R |
| `medium` (stage 1, `heldout_medium`의 모체) | 8 | 10.0 | 4.0 | 0.20 | 5.0% | 0.60 R |
| stage 2 | 8 | 10.0 | 3.5 | 0.35 | 10% | 1.2 R |
| `ood_extreme_mobility` | 8 | 10.0 | 4.0 | 1.20 | **30%** | 3.6 R |
| **`structural_hole_*`** | 8 | 10.0 | 1.85 | **0.00** | **0%** | **완전 정지** |

관측되는 switch activation rate가 이 구조를 그대로 반영한다:

| 시나리오 | switch activation |
|---|---|
| `heldout_medium` | 3.24% |
| `ood_sparse` | 4.21% |
| `ood_link_loss` | 4.32% |
| `ood_nodes_24` | 5.92% |
| `ood_extreme_mobility` | **13.92%** |
| `predictive_break_45` | 16.67% (= 정확히 1/6) |
| `predictive_break_225_link_loss` | 18.64% |

**`structural_hole_*` 시나리오는 `max_speed=0.0`으로 노드가 전혀 움직이지 않는다.** 정적
그래프에서는 링크 수명 `ℓ_v`가 항상 horizon으로 포화되므로, **예측 기반 메커니즘이 원리적으로
아무것도 할 수 없다.** 그럼에도 논문 §VI-B는

> "The predictive recovery mechanism is most useful in **structural-hole** and link-break settings"

라고 서술한다. §1.3의 측정치(structural_hole 두 시나리오 모두 차이 0.00pp)와 이 물리적 사실이
서로를 확인한다. **이 문장은 삭제하거나 반대로 서술해야 한다.**

또한 커버리지 비율이 매우 높다: `medium`은 π·4.0²/10² = **50.3%**, `_base`는 **63.6%**.
기대 차수 ≈ 4.5–4.8로 유지되며, 이는 §5.1의 얕은 hop 수로 직결된다.

---

### 3.3 F-16: 이동성 모델이 예측기와 같은 가정을 쓴다 ⭐

`link_model.py:13-37`의 `predicted_link_lifetime_steps`는 **등속 상대 운동**을 가정하고
2차 방정식으로 링크 만료 시각을 푼다. 그런데 `mobility.py`의 Random Waypoint는
**구간별 등속**이다. 두 모델의 가정이 정확히 일치한다.

측정 결과 (`test_one_step_dead_reckoning_is_exact_under_random_waypoint`):

```
step 1: dead-reckon 위치 오차  mean=0.0000  max=0.0000   (R=4.0)
step 2:                        mean=0.0964  max=0.7715
step 3:                        mean=0.3368  max=1.5430
step 5:                        mean=0.9184  max=3.4538
step 6:                        mean=1.4936  max=4.3966
```

**1-step stale beacon이 참 위치를 부동소수점 정밀도로 재현한다.** 오차는 노드가 waypoint에
도달해 속도를 바꿀 때부터 비로소 누적된다.

즉 `l_v`는 단순한 ground-truth 측정값이 아니라 **링크 만료에 대한 완벽한 oracle**이다.
predictive prior가 왜 그렇게 강해 보이는지에 대한 **구조적 설명**이며, 동시에 F-6의
beacon 열화 실험이 현재 RWP 위에서는 staleness 축에서 구조적으로 무력함을 뜻한다.

Gauss-Markov(상관 가속), Paparazzi, semi-random circular 등 실제 UAV 궤적 모델에서는 동일한
staleness가 실질적 오차를 낳는다. **따라서 E-1의 완전한 형태는 E-12(3D Gauss-Markov)와
함께 수행해야 한다.**

**조치**: (a) 논문 §VII-D에 이 공모 관계를 명시 — 우리가 먼저 밝히는 것이 유리하다.
(b) E-12 이동성 확장을 E-1의 전제조건으로 승격.

---

## 4. 평가 설계 및 통계의 취약점

### 4.1 F-7: Flagship 시나리오 2개는 유효 표본 n=1

`scenarios/predictive_traps.py:49-63`, `scenarios/structural_holes.py:44-56`은 **고정 좌표**
그래프를 사용한다:

- `predictive_break_*`: 9노드, area 10.0, R 2.1, **node 2만** `[0, 1.5]` 속도로 이동,
  나머지 속도 0, `source=0`, `destination=8`
- `structural_hole_*`: 8노드, area 10.0, R 1.85, **`max_speed=0.0`(전원 정지)**,
  `source=0`, `destination=5`

`predictive_break_45`와 `structural_hole_45`는 `stochastic_link_loss=0.0`이므로 **완전
결정론적**이다. 검증 결과:

```
('predictive_break_45', seed 42..2718): n=200, unique outcomes = {delivered}
('structural_hole_45',  seed 42..2718): n=200, unique outcomes = {delivered}
('heldout_medium',      seed 42..2718): n=200, unique outcomes = {delivered, dropped}   ← 정상
```

추가 확인:
- `predictive_break_45`: hop count가 **정확히 6.00** (최소·최대 동일), path stretch **정확히 2.000**
- `structural_hole_45`: hop count **정확히 4.00**, stretch **정확히 1.000**

즉 seed당 200 에피소드는 통계 표본이 아니라 **같은 계산의 200회 반복**이다. 그런데 이 두 셀이
scenario-macro 평균에서 각각 1/14 가중치를 온전히 받아 **헤드라인의 14.3%**를 구성한다.

**조치**:
- 이 두 시나리오를 "deterministic behavioural test"로 재분류하고 별도 표로 분리.
- macro 평균에서 제외한 버전을 병기 (12-scenario macro).
- 통계적 시나리오로 쓰려면 노드 위치에 지터를 넣거나 `stochastic_link_loss > 0`을 부여.

### 4.2 F-8: "OOD"가 동일 지오메트리의 회전이다

| 시나리오군 | training | calibration | **evaluation (OOD로 표기)** | 지오메트리 |
|---|---|---|---|---|
| predictive break | 0°, 90°, 180° | 270°, 135° | **45°, 225°** | 동일 9노드 고정 좌표 |
| structural hole | 0°, 90°, 180° | 270°, 315° | **45°, 225°** | 동일 8노드 고정 좌표 |

노드 수, 상대 위치, 소스(0)→목적지, 끊어지는 링크(node 2)가 **전부 동일**하고 회전각만 다르다.

```python
# scenarios/predictive_traps.py:72-75
rotated_positions = (positions - geometry_center) @ rotation.T + area_center
rotated_velocities = velocities @ rotation.T
```

관측 feature가 Cartesian 성분(회전 불변이 아님)이므로 완전한 항등은 아니지만, 이는
**회전 데이터 증강**이지 분포 외 일반화가 아니다. `EvaluationScenario`의 distribution 태그가
`"ood_predictive_break"`, `"ood_structural_hole"`인 것과 논문의 "14 **held-out or stress**
scenarios" 서술은 방어하기 어렵다.

**조치**: distribution 태그를 `rotational_holdout_*`로 변경하고, 논문에서 "OOD" 대신 "held-out
rotation of a fixed adversarial topology"로 정확히 기술. 진짜 OOD는 `ood_nodes_*` 3개와
`ood_link_loss_30`, `ood_extreme_mobility` 정도다.

### 4.3 F-9: 헤드라인 aggregate가 baseline 함정에서 나온다

시나리오별 connected-pair PDR (`seed_summaries_all.csv` 재집계):

| 시나리오 | DiSwitch | Evo-QGeo | AODV | 비고 |
|---|---|---|---|---|
| `ood_nodes_10` | 0.9800 | 0.9330 | **0.9800** | AODV 동률 |
| `ood_nodes_16` | 0.9720 | 0.9040 | **0.9900** | **AODV 승 (+1.8pp)** |
| `ood_nodes_24` | 0.9380 | 0.8530 | **0.9450** | **AODV 승 (+0.7pp)** |
| `ood_link_loss_30` | 0.7780 | 0.7370 | 0.7350 | DiSwitch 승 |
| `structural_hole_45` | 1.0000 | 1.0000 | 1.0000 | 3자 동률 |
| `structural_hole_225_link_loss` | 0.6984 | 0.6984 | 0.6905 | 거의 동률 |
| **`predictive_break_45`** | 1.0000 | 1.0000 | **0.0000** | ← 함정 |
| **`predictive_break_225_link_loss`** | 0.5116 | **0.5581** | 0.0872 | ← 함정, **Evo-QGeo 승** |
| **macro** | **0.9053** | 0.8869 | 0.7982 | |

두 가지가 동시에 드러난다:

1. **Evo-QGeo 대비 이득(+1.84pp)의 대부분은 node-count 시나리오에서 나오는데, 바로 그
   시나리오들에서 평범한 AODV가 DiSwitch를 이기거나 동률이다.** "가장 강한 adapted baseline
   대비"라는 프레이밍이 이 사실을 가린다. AODV는 macro에서 3위지만 **9개 랜덤 토폴로지
   시나리오 중 3개에서 1위**다.
2. **AODV의 macro(0.7982)가 무너지는 이유는 `predictive_break_45`에서 0.0000을 받기
   때문이다.** 이 시나리오는 비예측형 라우팅을 깨뜨리도록 저자가 직접 설계한 결정론적
   함정이며(§4.1), macro 평균에서 1/14 가중치를 온전히 받는다.

**조치**:
- (a) 시나리오군별(random topology / node-count / adversarial trap) 가중치를 분리 보고.
- (b) 각 지표에 대해 **시나리오별 최고 baseline**을 표기 (단일 "strongest baseline" 프레이밍 금지).
- (c) trap 시나리오를 제외한 macro를 병기.
- (d) `predictive_break_225_link_loss`에서 Evo-QGeo에 −4.65pp 뒤진다는 사실은 이미 §VI-B에
  정직하게 서술되어 있음 — 유지하되, 그것이 **유일하게 통계적 신호가 있는 predictive
  시나리오**임을 명시.

### 4.4 Switch calibration의 winner's curse와 튜닝 예산 비대칭

```python
# implementations/lite_globe/config/switchglobe.yaml
calibration:
  calibration_episodes_per_stage: 80
  switch_thresholds: [0.01, 0.05, 0.10, 0.20]     # 4
  margin_gates:     [0.00, 0.04, 0.08]            # 3
  lifetime_gates:   [0.10, 0.20, 0.30]            # 3
  onward_gates:     [0.10, 0.20, 0.30]            # 3
```

```python
# implementations/lite_globe/experiments/phase12_campaign.py:259-270
best = max(feasible, key=lambda item: (
    item["predictive_pdr"],      # ← 1차 정렬 기준
    item["generic_pdr"], item["hole_pdr"], item["deadline_delivery_ratio"],
    -item["mean_success_delay"], -item["mean_energy"], -item["mean_policy_input_bytes"]))
```

**문제 1 — 선택 편향**: 4×3×3×3 = **108개 후보**를 stage당 **80 에피소드** 추정치로 평가한 뒤
최댓값을 사전식으로 선택한다. 80 에피소드에서 PDR의 표준오차는 약 √(0.9·0.1/80) ≈ 3.4pp이며,
108개 중 최댓값을 고르면 기대 상향 편향이 수 pp에 달한다. 선택된 값에 대한 **재추정(nested
validation)이 없다**.

**문제 2 — 분포 수준 선택**: 1차 정렬 기준인 `predictive_pdr`은
`calibration_predictive_break_270/135`에서 측정되는데, 이는 평가 시나리오
`predictive_break_45/225`와 **같은 9노드 고정 지오메트리의 다른 회전각**이다(§4.2). 에피소드
시드(`seed+1000`, `+2000`)는 분리되어 있어 `experiment-contract`의 문구는 만족하지만,
**분포 수준의 선택은 분리되지 않았다**.

**문제 3 — 튜닝 예산 비대칭**: AODV, OLSR, Greedy, Evo-QGeo, RDQN-HERP, GAT-GRU-DDQN 어느
것에도 대응하는 시나리오별 하이퍼파라미터 그리드가 주어지지 않았다.

**조치**:
- 선택된 gate 값을 **완전히 새로운 seed 집합**에서 재평가하거나, nested CV로 편향을 보고.
- 5 seed 각각이 어떤 gate를 선택했는지 공개 (논문은 γ_m=0.04, γ_ℓ=0.20, γ_o=0.20, η=0.05를
  단일 값처럼 제시 — seed별로 다르다면 그 사실이 중요하다).
- Baseline에도 동등한 탐색 예산 부여, 또는 "제안기법만 보정됨"을 threats to validity에 명시.

### 4.5 Switch 진단 지표의 내부 모순

`seed_summaries_all.csv` 재집계 (SwitchGLOBE, 전 시나리오 평균):

| 지표 | 값 | 해석 |
|---|---|---|
| `switch_activation_rate` | **8.73%** | 결정의 **91.3%는 순수 normal branch** |
| `false_switch_rate` | **0.0000** | Eq.(12)의 세 번째 disjunct가 실질적으로 미발화 (§4.5.1) |
| `missed_risk_rate` | **5.13%** (최악 28.4%) | 위험한데 스위치가 안 걸림 (§4.5.2) |
| `branch_disagreement_rate` | 0.0 (AODV 등) / SwitchGLOBE만 유의미 | — |

#### 4.5.1 "demonstrably safer alternative" 항이 죽어 있다

```python
# models/student_policy.py:759-765
high_risk = normal_danger > self.switch_threshold
safer_predictive = ((predictive_action != normal_action)
                    & (safety_gain > 0.10)
                    & (predictive_danger < normal_danger))
return has_candidate & (normal_is_drop | high_risk | safer_predictive)
```

`false_switch_steps = switch & (normal_danger <= 0)` (line 814)가 **전 시나리오에서 0**이라는
것은, `normal_danger == 0`인 상태에서 `safer_predictive`만으로 스위치가 걸린 사례가 **한 번도
없었다**는 뜻이다. 이 항은 `0 < danger ≤ η` 구간에서만 단독 발화 가능하며, 그 구간의 기여는
측정되지 않았다.

Eq.(12)에서 가장 "새로워 보이는" 항이 실질적으로 기여하지 않는다.

**조치**: switch trigger를 **disjunct별로 분해 계측**
(`drop_trigger_steps`, `high_risk_trigger_steps`, `safer_trigger_steps`)하여 보고. 3번째 항이
0이면 Eq.(12)에서 제거하고 식을 단순화 — 이는 §1의 재프레이밍과도 일관된다.

#### 4.5.2 스위치가 가장 필요한 곳에서 정의상 비활성화된다

`missed_risk_steps = (~switch) & (normal_danger > switch_threshold)` (line 815).
`switch = has_candidate & (...)` 이므로, `missed_risk`는 **`has_candidate == False`**, 즉
유효 이웃이 하나도 없는 **막다른 릴레이**에서 발생한다.

| 시나리오 | `missed_risk_rate` |
|---|---|
| `unconditional_sparse` | **28.36%** |
| `structural_hole_225_link_loss` | **12.52%** |
| `predictive_break_225_link_loss` | **10.95%** |
| `ood_link_loss_30` | 8.75% |
| `heldout_medium` | 0.64% |

**회복이 가장 필요한 상황에서 스위치가 정의상 꺼진다.** 그리고 그 상황에서 정책이 취할 수
있는 행동은 DROP뿐이다 — store-carry-forward, 백트래킹, 버퍼링, 대기가 모두 없다.

**조치**: 이는 단순 threshold 문제가 아니라 **행동 공간의 한계**다. §5.4의 확장 항목으로
연결된다. 논문 §VII-D "Threats to validity"에 명시 필요.

### 4.6 p95 delay는 소수 정수의 분위수다

`p95_success_delay`는 hop 수가 2–6인 분포의 0.95 분위수이므로 실제로는 {3, 4, 5, 6}만 취한다.
`ablation_overall_metrics.csv`에서:

```
Geo-Residual Student, p95_success_delay_steps: mean=4.0, sd=0.0, ci=[4.0, 4.0]
Predictive Prior Only / No-Switch / DiSwitch / Fast: mean=6.0, sd=0.0, ci=[6.0, 6.0]
"Predictive/risk prior" 효과: mean=-2.0, sd=0.0, ci95=[-2.0, -2.0]
```

SD가 정확히 0.0이고 CI 폭이 0인 것은 **정밀도가 아니라 이산성의 아티팩트**다. 논문 §VI-A의
"0.2929 fewer p95 successful-delay steps (0.1503–0.4354)"는 이 정수들의 시나리오-macro
평균이며, 실질 해상도가 매우 낮다.

**조치**: (a) p95를 "steps"로 명시하고 이산성 한계를 각주로 기술, (b) 정밀한 지연 주장을 하려면
연속 시간 지연 모델(전송 시간 + 큐잉 + 전파)이 필요함을 threats에 추가.

### 4.7 Drop reason이 사실상 단일하다

`episodes_all.csv` 재집계:

| Method | 총 drop | `agent_drop` | `ttl_expired` | `routing_loop` | `invalid_action` | `time_limit` |
|---|---|---|---|---|---|---|
| SwitchGLOBE | 2,041 | **2,034 (99.7%)** | 7 (0.3%) | 0 | 0 | 0 |
| Evo-QGeo | 2,207 | 2,198 (99.6%) | 9 | 0 | 0 | 0 |
| AODV | 3,700 | 3,700 (100%) | 0 | 0 | 0 | 0 |
| RDQN-HERP | 4,450 | 4,270 (96.0%) | 180 (4.0%) | 0 | 0 | 0 |
| GAT-GRU-DDQN | 5,128 | 5,093 (99.3%) | 35 | 0 | 0 | 0 |

`docs/simulation_protocol.md` §3은 8종 outcome 분해를 요구하지만, 실제로는 **거의 전부가
`agent_drop` 한 종류**다. 그리고 §6.1에서 보듯 SwitchGLOBE의 `agent_drop`은 자발적 포기가
아니라 **막다른 골목 도달**이다. 즉 실패 유형 분석이 실질적으로 "dead-end 도달률" 하나로
환원되며, 이는 §4.5.2의 `missed_risk_rate`와 동일한 현상이다.

`routing_loop = 0`은 지리적 진행 prior 때문이며(`mask_visited_actions`는 주 시나리오에서
`False`), 루프 안전성 주장의 근거로는 약하다.

---

## 5. 확장성 한계

### 5.1 라우팅 문제 자체가 얕다

전달된 패킷의 hop 수 직접 계산 (`episodes_all.csv`, SwitchGLOBE, delivered only):

| 시나리오 | N | 평균 hop | 최대 hop | 평균 shortest-path hop |
|---|---|---|---|---|
| `heldout_medium` | 8 | 2.32 | 5 | 2.32 |
| `ood_sparse` | 8 | 2.21 | 4 | 2.21 |
| `ood_nodes_10` | 10 | 2.57 | 6 | 2.59 |
| `ood_nodes_16` | 16 | 2.84 | 8 | 2.90 |
| `ood_nodes_24` | 24 | 3.43 | 10 | 3.40 |
| `unconditional_sparse` | 8 | 1.56 | 5 | 1.49 |
| **전체 (delivered)** | — | **3.12** (중앙값 3) | 10 | — |

패킷당 라우팅 결정이 **평균 2–3회**다. 비교 대상으로 인용된 RoutePPO, RLFR, Predictive-Q 급
연구는 수십–수백 노드, 5–15 hop 경로를 다룬다.

밀도 유지 스케일링 자체는 합리적이다 (N=8/area=10/R=4.0 → 기대 차수 4.45; N=24/area=17/R=4.4
→ 기대 차수 4.83). 그러나 diameter가 √N으로만 자라므로 24노드에서도 3.4 hop에 그친다.

논문 §VI-B가 "limited generalization test, not a scalability proof"라고 정직하게 밝히고 있으나,
**24노드 / 3.4 hop은 FANET 라우팅 논문의 통상 기준선에 미치지 못한다**는 점을 threats에
더 강하게 써야 한다.

### 5.2 F-11: 아키텍처가 32노드에 하드 고정

```python
# env/fanet_env.py:38-39
self.drop_action = self.config.max_nodes                    # = 32
self.action_space = spaces.Discrete(self.config.max_nodes + 1)   # = 33

# env/observation.py:66  — 전역 node id로 인덱싱
neighbor_features[node] = np.array([...])   # node는 전역 인덱스, shape (32, 7)
```

결과:

1. 액션이 **전역 노드 ID**이므로 정책이 permutation-equivariant도, size-agnostic도 아니다.
2. `include_node_ids=True`(기본값)에서는 `packet_features`에 source/destination의 정규화된
   전역 ID까지 들어간다 (`observation.py:166-176`). 주 평가 시나리오는 `False`지만 기본값이
   `True`인 점은 학습 파이프라인 감사 시 확인 필요.
3. **32노드 초과 네트워크로 재학습 없이 확장 불가.**
4. `policy_input_bytes`가 실제 이웃 수와 무관하게 `max_nodes`에 비례해 부풀려진다 →
   AODV(143 B)와의 비교가 성립하지 않는다 (§2.2와 결합해 이중으로 왜곡).

**조치 (E-2) — 구현 및 검증 완료.** `docs/slot_observation_refactor.md` 참조.

핵심 발견: `LocalStudentPolicy`와 모든 서브클래스는 이미 **공유 per-candidate 인코더 +
유효 후보 풀링** 구조이며, **`max_nodes` 차원을 갖는 파라미터 텐서가 하나도 없다.** 따라서

- 모델은 후보 슬롯에 대해 **정확히 permutation-equivariant**하고,
- `max_nodes=32` 체크포인트가 `max_nodes=12` 모델에 **그대로 로드**된다.

즉 로컬 슬롯 압축은 **재학습이 필요 없는 semantics-preserving re-indexing**이다.
검증(6,450개 결정 샘플, seed 0–149, 14개 시나리오):

```
relay degree: mean=1.96   p95=5   max=10
환경은 32개 후보 슬롯 할당 -> 모든 관측의 93.9%가 padding
```

| K | 결정당 바이트 | 감소 | truncated |
| --- | --- | --- | --- |
| 8 | 537 B | 3.73× | 7/6450 (0.11%) |
| **12** | **781 B** | **2.56×** | **0/6450 (무손실)** |
| 32 (현행) | 2001 B | 1.00× | 0 |

정확성 검증: 후보별 **로짓이 1e-4 이내로 동일**하며(argmax뿐 아니라), 2-branch
`SwitchGlobePolicy` 전체에서도 **1,033개 결정 전부**(100.00%) 다음 홉과 switch flag가
일치했다. switch gate 재보정이 불필요하다.

**논문 수치에 대한 함의**: Table 2의 `policy input bytes = 4821`(에피소드당)은 **약 94%가
padding**이다. K=12 무손실 레이아웃에서 정직한 값은 약 **1,880 B**이며, AODV(143 B) 대비
차이가 34배가 아니라 13배로 줄어든다. F-5(control_bytes = 0.0)를 함께 교정하면 방향 자체가
뒤집힌다.

**부수 관찰**: 평균 릴레이 차수가 **1.96**이다. 전형적 "라우팅 결정"의 후보가 2개이며,
§5.1의 평균 홉 2.2–3.4와 결합하면 의사결정 문제 규모가 매우 작다는 점이 정량 확인된다.

### 5.3 트래픽 모델 확장

단일 패킷 에피소드 구조는 다음을 원천 차단한다:

- throughput / goodput (현재 `delivery_rate_proxy`로만 보고 — 문서 규정 준수 중)
- queue overflow, congestion-aware routing
- 다중 flow 간 간섭 및 공정성
- `q_v` feature의 실질 의미 (§3.1)

**조치 (E-3)**: Poisson arrival 다중 flow + 노드별 FIFO 큐 + 서비스율 모델 도입. 이는
§3.1(b) 큐 동역학과 함께 수행하면 하나의 확장으로 두 결함을 해소한다.

### 5.4 행동 공간 확장

§4.5.2에서 확인한 dead-end 실패(전체 drop의 99.7%)를 다루려면 현재 행동 공간
`A_u = N_u ∪ {DROP}`으로는 불가능하다.

**조치 (E-4)**: 다음 중 검증 가능한 것부터:
- `WAIT` / store-carry-forward 행동 추가 (DTN 계열 표준 대응)
- 방문 노드로의 백트래킹 허용 (현재 `onward` 계산에서 `packet.path`를 제외 —
  `observation.py:87-88`)
- Top-2 백업 다음 홉 (이미 `Fast-DiSwitch + Top-2`로 시도했으나 효과 정확히 0.0 —
  `ablation_component_effects.csv`. 원인 분석 필요)

### 5.5 F-12: Objective와 실제 학습 목적의 불일치

논문 Eq.(3):

```
max  E[PDR_C] + λ_D·E[DDR] − λ_L·E[L95] − λ_E·E[E_proxy]
```

실제 보상 함수:

```python
# env/reward.py:24-29
reward = -self.delay
reward += self.progress * normalized_progress
if delivered: reward += self.delivery
if failed:    reward -= self.failure
```

**`λ_L`(tail delay) 항도, `λ_E`(energy) 항도 존재하지 않는다.** `-self.delay`는 per-step 상수
페널티(0.1)로 평균 지연에만 대응하며 tail과는 무관하다.

논문이 "training uses staged PPO and distillation losses rather than a single scalar solution of
the constrained program"으로 방어하고 있으나, Eq.(3)에 구현되지 않은 항을 명시하는 것은
불필요한 공격면이다.

**조치**: Eq.(3)을 실제 최적화 대상만 남기도록 축소하거나, 두 항을 실제로 구현하고 λ를 보고.

---

## 6. 정책 내부 구조에서 발견된 숨은 동작

### 6.1 DiSwitch는 이웃이 있으면 구조적으로 DROP할 수 없다

```python
# models/student_policy.py:563-567  (LiteGlobePStudentPolicy.forward)
drop_logits = (-20.0
               + self.residual_weight * torch.tanh(residual_logits[:, self.max_nodes:]))
# residual 비활성 → residual_logits = 0 → tanh(0) = 0 → drop_logit ≡ -20.0
```

한편 switch 규칙(line 765)은 `normal_is_drop`일 때 반드시 발화한다. 따라서:

| normal branch 판단 | switch | 최종 행동 |
|---|---|---|
| 유효 이웃 선택 & danger ≤ η | 미발화 | normal의 선택 |
| 유효 이웃 선택 & danger > η | 발화 | predictive의 선택 (drop logit −20이므로 **항상 이웃**) |
| **DROP** & `has_candidate` | **발화** | predictive의 선택 → **항상 이웃** |
| DROP & `not has_candidate` | 미발화 | **DROP** (유일한 유효 행동) |

predictive branch의 candidate logit 범위를 추정하면 (α≈8, Δd ∈ [−0.35, 0.35] → ±2.8;
predictive_bonus = Σ s·r, s=(0.75, 3.00, 0.25, 6.00), r ∈ [0,1]⁴ → 최대 10;
gate_penalty 최대 ≈ 18 × 0.44 ≈ 7.9) 대략 [−8, +13]이므로, **drop logit −20이 선택되는 일은
사실상 불가능**하다.

즉 **DiSwitch는 이웃이 하나라도 있으면 절대 자발적으로 패킷을 버리지 않는다.** 이는:

1. `agent_drop`이 전체 drop의 99.7%인데 그것이 **전부 dead-end 도달**임을 설명한다(§4.7).
2. 논문 §III-A의 "The explicit DROP action prevents an invalid padded index from being
   interpreted as a relay"는 맞지만, **배포 정책이 DROP을 선호로 선택하는 일은 없다**는 사실이
   빠져 있다.
3. **PDR 이득과 에너지 비용 일부를 설명할 수 있는 교란 요인**이다. 포기하지 않으면 배달될
   확률이 오르지만 전송 시도도 는다 (`mean_transmission_attempts`: Geo-Residual 2.41 →
   DiSwitch 2.96).
4. `docs/method_history.md`는 Phase 13의 **명시적** DROP suppression을 "검증 이득 불충분"으로
   제외했는데, **구조적 형태의 DROP suppression은 최종안에 남아 있다.**

**조치**: 논문에 명시하고, "drop logit을 학습 가능하게 둔 변종" vs "현행"의 분리 측정을
ablation에 추가. §1의 재프레이밍에서 이것이 실제 기여 메커니즘의 후보다.

### 6.2 `residual_weight` 상태의 이중 표현

```python
# models/student_policy.py:203-204
self.register_buffer("residual_weight", torch.tensor(1.0))   # ← state_dict에 저장됨
self._residual_enabled = True                                 # ← 순수 파이썬 속성

# models/student_policy.py:289-295
def set_residual_weight(self, weight: float) -> None:
    self.residual_weight.fill_(weight)
    self._residual_enabled = weight > 0.0
```

`load_state_dict()`는 buffer만 복원하고 `_residual_enabled`는 갱신하지 않는다. 따라서 체크포인트
로드 순서에 따라 두 값이 어긋날 수 있다.

현재 파이프라인(`phase12_campaign.py:338-339`: load → `set_residual_weight(0.0)`)에서는
결과적으로 동작이 일치한다. 그러나:

- 논문의 **핵심 설계 결정**(residual 무력화, §1.4b)이 이 순서에 의존한다.
- `RiskSwitchLiteGlobePStudentPolicy.__init__`(line 626)은 생성 시점에 0으로 설정하는데,
  이후 `load_checkpoint(checkpoint, risk_switch, ...)`(line 360)가 buffer를 1.0으로 되돌릴 수
  있다. `_residual_enabled`가 False로 남아 있어 **결과는 같지만**, 이는 우연이지 설계가 아니다.

**조치**: 단일 소스로 통합 (`_residual_enabled`를 property로 만들어 buffer에서 파생) + 회귀
테스트 추가 (`assert policy.predictive_policy.residual_weight.item() == 0.0` 및 forward count
검증). `tests/lite_globe/test_switchglobe.py`에 이미 forward-count 테스트가 있으므로 확장이
용이하다.

---

## 7. 코드 및 아티팩트 위생

### 7.1 신뢰도 캠페인과 지연 벤치마크의 코드 버전이 다르다

`seed_summaries_all.csv`의 `decision_latency_p95_ms`는 전 시나리오에서 **≈10.0–10.6 ms**다:

| 시나리오 | p95 latency |
|---|---|
| `heldout_medium` | 10.319 |
| `structural_hole_45` | 10.650 |
| `unconditional_sparse` | 9.973 |

이는 논문 Table 3의 **legacy repeated CUDA 값 10.004 ms**와 일치한다. 즉:

- **신뢰도 캠페인(외부 비교 8-method)은 computation-reuse 이전 코드로 실행**
- **지연 벤치마크(Table 3)만 신규 코드로 실행**

논문의 semantic-equivalence 논증(§IV-F)으로 방어 가능하지만, 다음이 필요하다:

1. 두 캠페인의 코드 버전(commit)을 매니페스트에 명시.
2. 이 CSV 컬럼은 **원고에 인용하지 말 것** (현재 인용하지 않음 — 유지).
3. 동치성 회귀 테스트 결과(`artifacts/switchglobe_latency_optimization/step1_review/
   exact_replay_verification.json`)를 supplementary에 명시적으로 링크.

### 7.2 미측정 값이 `0.0`으로 기록된다

§2.2에서 지적한 `control_bytes = 0.0` 외에도, `ablation_overall_metrics.csv`는 **모든**
ablation 방법의 `mean_control_bytes`와 `mean_control_messages`를 0.0으로 기록한다.
`docs/simulation_protocol.md` §5의 `not modeled` 규정과 충돌한다.

**조치**: 집계 레이어에서 미측정 필드를 `NaN`으로 기록하고, 표 생성기가 `N/A`로 렌더링하도록
변경. (논문 §V-B가 이미 "A missing-delivery cell is reported as N/A, never as zero"라는 원칙을
세워 두었으므로, 같은 원칙을 오버헤드에도 적용하면 된다.)

### 7.3 Provenance

`source/internal_peer_review.md` 항목 2가 이미 지적한 대로 synthesis manifest가 dirty-file
목록을 기록한다. 추가로:

- 101,000행 통합 아카이브(98,000 = 7 method + 14,000 = Fast-DiSwitch)의 **병합 시 두 캠페인의
  코드 버전이 동일함을 보장하는 기록이 없다.** row-count 검증만 존재.
- clean tag에서 전체 재생성 + SHA-256 공개 필요.

### 7.4 F-14: 헤드라인 결과를 재생성할 체크포인트가 없다

저장소 전체에서 `.pt` 파일을 검색하면 다음만 존재한다.

- `artifacts/final_paper_simulation/full/ablation/fast_training/checkpoints/seed_*/fast_switchglobe.pt` (5개)
- 외부 baseline 체크포인트 (`aodv.pt`, `olsr.pt`, `evo_qgeo_adapted.pt` 등)
- `artifacts/evo_globe/` 실험 변종

**Phase 8(`geo_residual_kd.pt`), Phase 11(`lite_globe_p.pt`), Phase 12(`switchglobe.pt`)가
하나도 없다.** `docs/`와 `.claude/context/baseline-contract.md`가 실행 시 지정하라고 안내하는
`artifacts/switchglobe/final/checkpoints` 디렉터리 자체가 존재하지 않는다.

결과적으로 **논문의 헤드라인 0.9053을 이 저장소만으로 재생성할 수 없다.** Data Availability
절이 "source code, configuration files, raw episode archives, ... in the project repository"를
약속하므로, 공개 릴리스 전에 체크포인트 배포 경로를 확정해야 한다.

**조치**: (a) 체크포인트를 저장소 또는 Zenodo 등 영구 아카이브에 포함, (b) SHA-256 공개,
(c) `docs/`의 경로 안내를 실제 위치로 정정.

### 7.5 F-15: reporting 모듈의 `None` 가드 결함과 allowlist 불일치

**증상**: `delivered == 0`인 셀에서 전달 조건부 지표가 `None`이 되는데
`float(row[metric])`에 가드가 없어 `TypeError`로 파이프라인이 중단된다.
`git stash`로 본 세션의 변경분을 제거한 상태에서 재현하여 **기존 결함임을 확인**했다.

**영향**: `phase7_reporting.py`, `phase8_reporting.py`, `phase11_reporting.py`,
`phase12_reporting.py`, `reporting.py`, `baseline_reporting.py` (6개 파일)

**부수 발견 — allowlist가 캠페인마다 다르다**:

| 파일 | "전달 조건부"로 취급하는 지표 |
| --- | --- |
| phase7 / phase8 / baseline | `mean_success_delay`, `p95_success_delay`, `mean_path_stretch` |
| phase11 / phase12 | 위 3개 **+ `energy_per_delivered_packet`, `energy_per_on_time_delivery`** |

**동일 지표가 캠페인에 따라 다르게 집계된다.** 논문 §V-C가 언급한 "pooled full-analysis
table whose weighting differs from the scenario-macro synthesis"의 한 원인일 수 있다.

**조치**: `None` 가드는 추가했다 (이전에 성공하던 실행의 결과는 바뀌지 않는다 — 이전에는
crash였음). **allowlist 불일치는 아직 통일하지 않았다** — 통일하면 기존 full-run 결과의
재현성이 깨지므로 별도 결정이 필요하다. 최소한 논문에 어느 캠페인이 어느 규칙을 썼는지
명시해야 한다.

### 7.6 문서 간 서술 불일치

`docs/method_history.md`:

> SwitchGLOBE Exact(Phase 12)는 ... 6개 baseline 전부를 6개 핵심 지표에서 통계적으로 유의하게
> 앞선다

이는 **scenario-macro 수준**에서는 맞지만, §4.3에서 보듯 시나리오 수준에서는 AODV가 3개
시나리오에서 동률 이상이다. 문서에 "macro 수준" 한정을 명시할 것.

---

## 8. 우선순위 실행 계획

### 8.1 즉시 (제출 전 필수 — 이것 없이는 제출 불가)

| # | 작업 | 관련 발견 | 예상 공수 |
|---|---|---|---|
| I-1 | §1 결과 독립 재현 확인 → **논문 프레이밍 결정** (옵션 A 권장) | F-1, F-2, F-3 | 0.5일 (재현) + 프레이밍 결정 |
| I-2 | Fig. 8 / Table ablation에 `Predictive Prior Only vs DiSwitch` 대조 **추가** (숨기면 안 됨) | F-1, F-2 | 0.5일 |
| I-3 | `control_bytes` 열을 Table 2에 추가하거나 전 method `not modeled` 명시 | F-5 | 0.5일 |
| I-4 | "one-hop / strict-local" → "1-hop topology + 2-hop kinematic" 정정 (Abstract, §I, §III-A, Fig. 2) | F-4 | 0.5일 |
| I-5 | `predictive_break_45` / `structural_hole_45`를 결정론적 behavioural test로 재분류, 12-scenario macro 병기 | F-7 | 1일 |
| I-9 | 시나리오 스위트의 유효 다양성 명시 (`heldout_medium` = 학습 config, `ood_sparse` ≡ `unconditional_sparse`), config-family 2단계 집계 병기 | F-13 | 0.5일 |
| I-6 | §VI-B의 "predictive recovery is most useful in structural-hole ... settings" 삭제/수정 (structural hole은 `max_speed=0.0`) | F-2, §3.2 | 0.2일 |
| I-7 | §IV-D의 queue headroom 서술 수정 (`q_v`가 정적 난수임) | F-10 | 0.2일 |
| I-8 | Eq.(3)에서 미구현 `λ_L`, `λ_E` 항 제거 또는 구현 | F-12 | 0.3일 |

### 8.2 단기 (major revision 1회분)

| # | 작업 | 관련 발견 | 가치 |
|---|---|---|---|
| **E-1** | **Beacon staleness / 추정 잡음 민감도 그리드** (§2.4) | F-6 | ⭐⭐⭐ 최고. switch를 구제할 유일한 경로이자 독립적 논문 기여 |
| E-2 | 액션 공간을 로컬 슬롯 인덱스로 전환, permutation-invariant 인코더 | F-11 | ⭐⭐⭐ 확장성 + 정직한 byte 측정 동시 해결 |
| E-5 | GPSR **perimeter mode** 구현 (structural hole 주장의 정당성) | §2.3 | ⭐⭐ baseline 공정성 |
| E-6 | Baseline에 동등한 튜닝 예산 부여 + calibration/evaluation 지오메트리 분리 | §4.4 | ⭐⭐ |
| E-7 | Switch trigger의 disjunct별 계측, Eq.(12) 3항 유지/제거 결정 | §4.5.1 | ⭐⭐ |
| E-8 | Drop logit 학습 가능 변종 ablation (구조적 DROP 억제의 기여 분리) | §6.1 | ⭐⭐ |
| E-9 | `residual_weight` 상태 이중 표현 통합 + 회귀 테스트 | §6.2 | ⭐ 위생 |
| E-10 | Clean tag 재생성 + SHA-256 + 캠페인별 commit 매니페스트 | §7.1, §7.3 | ⭐⭐ |

### 8.3 중기 (다음 논문 / 본격 확장)

| # | 작업 | 관련 발견 |
|---|---|---|
| E-3 | Multi-flow Poisson 트래픽 + 실제 큐 동역학 | F-10, §5.3 |
| E-4 | 행동 공간 확장 (WAIT / store-carry-forward / 백트래킹) | §4.5.2, §5.4 |
| E-11 | 50–100 노드, 5–15 hop 규모 확장 | §5.1 |
| E-12 | 3D Gauss-Markov 이동성 + path-loss/페이딩 채널 + 간섭 기반 MAC | §3 |
| E-13 | 10 seed clean rerun + randomized runtime block order | `internal_peer_review.md` 항목 3 |
| E-14 | Jetson Orin / RPi 급 타깃 디바이스에서 전력·온도 동시 측정 | `internal_peer_review.md` 항목 1 |

---

## 9. 재현 명령

모든 수치는 저장소 루트에서 아래 명령으로 재현된다 (외부 의존성 없음, Python 표준 라이브러리만 사용).

### 9.1 F-1 / F-2: Prior-Only ≡ DiSwitch 등가성

```bash
cd /Users/alex/Documents/GLOBE_ROUTING
python3 - <<'EOF'
import csv, statistics as st, math
from collections import defaultdict
rows = list(csv.DictReader(open(
    'artifacts/final_paper_simulation/full/ablation/raw/seed_summaries.csv')))
def macro(m, k):
    by = defaultdict(list)
    for r in rows:
        if r['method'] == m:
            by[r['training_seed']].append(float(r[k]))
    return {s: st.mean(v) for s, v in by.items()}
seeds = ['42', '77', '123', '314', '2718']
for k in ['connected_pair_pdr', 'deadline_delivery_ratio',
          'overall_pdr', 'decision_latency_p95_ms']:
    a, b = macro('Predictive Prior Only', k), macro('SwitchGLOBE Exact', k)
    d = [b[s] - a[s] for s in seeds]
    m, se, t = st.mean(d), st.stdev(d) / math.sqrt(5), 2.776
    print(f"{k}\n  PriorOnly={st.mean(a.values()):.8f}  "
          f"DiSwitch={st.mean(b.values()):.8f}\n"
          f"  diff={m:+.8f}  95% CI=[{m-t*se:+.6f}, {m+t*se:+.6f}]\n")
EOF
```

### 9.2 F-2: 시나리오별 차이

```bash
python3 - <<'EOF'
import csv, statistics as st
rows = list(csv.DictReader(open(
    'artifacts/final_paper_simulation/full/ablation/raw/seed_summaries.csv')))
A, B = 'Predictive Prior Only', 'SwitchGLOBE Exact'
for s in sorted({r['scenario'] for r in rows}):
    f = lambda m: st.mean(float(r['connected_pair_pdr'])
                          for r in rows if r['method'] == m and r['scenario'] == s)
    print(f"{s:34s} {f(A):.4f} {f(B):.4f} {100*(f(B)-f(A)):+6.2f} pp")
EOF
```

### 9.3 F-5: control overhead 비대칭

```bash
python3 - <<'EOF'
import csv, statistics as st
rows = list(csv.DictReader(open(
    'artifacts/external_comparison_analysis/seed_summaries_all.csv')))
for m in sorted({r['method'] for r in rows}):
    rr = [r for r in rows if r['method'] == m]
    f = lambda k: st.mean(float(x[k]) for x in rr)
    print(f"{m:26s} ctrl_B={f('mean_control_bytes'):8.1f} "
          f"msgs={f('mean_control_messages'):6.2f} "
          f"policy_B={f('mean_policy_input_bytes'):8.1f}")
EOF
```

### 9.4 F-7: 결정론적 셀 확인

```bash
python3 - <<'EOF'
import csv
from collections import defaultdict
ep = defaultdict(list)
with open('artifacts/external_comparison_analysis/episodes_all.csv') as fh:
    for r in csv.DictReader(fh):
        if r['method'] == 'SwitchGLOBE' and r['scenario'] in (
                'predictive_break_45', 'structural_hole_45', 'heldout_medium'):
            ep[(r['scenario'], r['training_seed'])].append(r['delivered'])
for k in sorted(ep):
    print(f"{k}: n={len(ep[k])} unique={set(ep[k])}")
EOF
```

### 9.5 F-6: 열화 모델 부재 확인

```bash
grep -rn "stale\|observation_noise\|beacon_period\|noisy\|beacon" \
    implementations/lite_globe/env/ implementations/lite_globe/scenarios/
# 기대 결과: 출력 없음
```

### 9.6 §5.1: hop 수 분포

```bash
python3 - <<'EOF'
import csv, statistics as st
from collections import defaultdict
hop = defaultdict(list)
with open('artifacts/external_comparison_analysis/episodes_all.csv') as f:
    for r in csv.DictReader(f):
        if r['method'] == 'SwitchGLOBE' and r['delivered'] in ('True', 'true', '1'):
            hop[r['scenario']].append(float(r['hop_count']))
for s in sorted(hop):
    print(f"{s:34s} mean={st.mean(hop[s]):5.2f} max={max(hop[s]):3.0f}")
allh = [x for v in hop.values() for x in v]
print(f"\nALL: mean={st.mean(allh):.2f} median={st.median(allh):.1f} max={max(allh):.0f}")
EOF
```

---

## 부록 A. 검증된 수치 요약표

### A.1 Ablation full run (5 seed × 14 scenario × 200 ep)

출처: `artifacts/final_paper_simulation/full/ablation/raw/seed_summaries.csv`

| 방법 | cpPDR (macro) | deadline | overall PDR | p95 latency (ms) | policy bytes | switch rate |
|---|---|---|---|---|---|---|
| Geo-Residual Student | 0.803027 | 0.7400 | 0.7557 | — | 3956 | 0.000 |
| **Predictive Prior Only** | **0.90528954** | 0.837571 | 0.85450 | **1.487** | 5424 | 0.000 |
| Predictive Student (No Switch) | 0.889864 | 0.822286 | 0.83914 | — | 5775 | 0.000 |
| **SwitchGLOBE Exact** | **0.90528954** | 0.837643 | 0.85421 | **6.228** | 4821 | 0.0873 |
| Fast-DiSwitch | 0.887204 | 0.815000 | 0.83750 | — | 6437 | 0.000 |
| Fast-DiSwitch + Top-2 | 0.887204 | 0.815000 | 0.83750 | — | 6437 | 0.000 |

### A.2 External comparison full run (5 seed × 14 scenario × 200 ep, 7 method)

출처: `artifacts/external_comparison_analysis/seed_summaries_all.csv` (98,000 episode)

| Method | cpPDR | avail | stretch | p95 delay | switch | false switch | missed risk | ctrl bytes |
|---|---|---|---|---|---|---|---|---|
| **SwitchGLOBE** | **0.9053** | 0.9196 | 1.140 | 4.26 | 0.0873 | **0.0000** | 0.0513 | **0.0** |
| Evo-QGeo (Adapted) | 0.8869 | 0.9196 | 1.182 | 4.56 | — | — | — | 102.6 |
| AODV | 0.7982 | 0.9196 | 0.942 | 3.86 | — | — | — | 1014.0 |
| OLSR | 0.7657 | 0.9196 | 0.846 | 3.14 | — | — | — | 2107.5 |
| RDQN-HERP (Adapted) | 0.7251 | 0.9196 | 1.526 | 6.16 | — | — | — | 0.0 |
| Greedy Geographic | 0.6832 | 0.9196 | 0.864 | 3.57 | — | — | — | 0.0 |
| GAT-GRU-DDQN | 0.6758 | 0.9196 | 1.366 | 5.49 | — | — | — | 0.0 |

### A.3 시나리오 파라미터 요약

기준 설정 `medium` = `phase7_curriculum(seed)[1]` = `train_medium`
(N=8, area=10.0, R=4.0, speed 0.05–0.20, loss=0.0, `max_episode_steps`=12, `packet_ttl`=12).
아래 대부분의 시나리오는 이 설정에서 **한 파라미터만** 바꾼 것이다.

| 시나리오 | N | area | R | speed | loss | `require_connected` | 결정론적 | 비고 |
|---|---|---|---|---|---|---|---|---|
| `heldout_medium` | 8 | 10.0 | 4.0 | 0.05–0.20 | 0.0 | ✓ | ✗ | **학습 stage 2와 동일 config** (§A.3.1) |
| `ood_link_loss` | 8 | 10.0 | 4.0 | 0.05–0.20 | 0.15 | ✓ | ✗ | |
| `ood_link_loss_30` | 8 | 10.0 | **4.0** | 0.05–0.20 | 0.30 | ✓ | ✗ | |
| `ood_fast_mobility` | 8 | 10.0 | 4.0 | 0.30–0.70 | 0.0 | ✓ | ✗ | |
| `ood_extreme_mobility` | 8 | 10.0 | 4.0 | 0.60–1.20 | 0.0 | ✓ | ✗ | switch 13.9% |
| `ood_sparse` | 8 | 10.0 | 3.0 | 0.05–0.20 | 0.0 | ✓ | ✗ | ↓와 **동일 config** |
| `unconditional_sparse` | 8 | 10.0 | 3.0 | 0.05–0.20 | 0.0 | **✗ (None)** | ✗ | ↑와 동일, avail 0.385 |
| `ood_nodes_10` | 10 | 11.0 | 3.8 | 0.05–0.20 | 0.0 | ✓ | ✗ | |
| `ood_nodes_16` | 16 | 14.0 | 4.2 | 0.05–0.20 | 0.0 | ✓ | ✗ | AODV 승 |
| `ood_nodes_24` | 24 | 17.0 | 4.4 | 0.05–0.20 | 0.0 | ✓ | ✗ | AODV 승 |
| **`predictive_break_45`** | 9 | 10.0 | 2.1 | node 2만 1.5 | 0.0 | 고정 좌표 | **✓** | **n_eff = 1** |
| `predictive_break_225_link_loss` | 9 | 10.0 | 2.1 | node 2만 1.5 | 0.10 | 고정 좌표 | ✗ | **Evo-QGeo 승** |
| **`structural_hole_45`** | 8 | 10.0 | 1.85 | **0.00** | 0.0 | 고정 좌표 | **✓** | **n_eff = 1**, 전원 정지 |
| `structural_hole_225_link_loss` | 8 | 10.0 | 1.85 | **0.00** | 0.10 | 고정 좌표 | ✗ | 전원 정지, 3자 동률 |

#### A.3.1 시나리오 스위트의 유효 다양성이 14보다 작다

위 표에서 세 가지가 드러난다.

1. **`heldout_medium`은 분포 외 held-out이 아니다.**
   ```python
   # generalization_suite.py:81, 84-89
   medium = phase7_curriculum(seed)[1].config     # ← 학습 curriculum stage 2의 config 객체 그 자체
   EvaluationScenario("heldout_medium", medium, connected, "in_distribution_heldout")
   ```
   distribution 태그가 `in_distribution_heldout`으로 코드에서는 정직하지만, 논문의
   "14 **held-out or stress** scenarios"(§V-A)는 이를 흐린다. 에피소드 시드 수준의
   held-out일 뿐 설정은 학습과 동일하다.

2. **`ood_sparse`와 `unconditional_sparse`는 완전히 동일한 config**(`medium` + R=3.0)이며,
   차이는 `reset_options`가 `_connected_options()`인지 `None`인지뿐이다
   (`generalization_suite.py:103-107, 118-123`). 즉 하나의 토폴로지 설정이 macro 평균에서
   **2/14 가중치**를 받는다. `unconditional_sparse`의 `endpoint_availability`가 0.385인 이유도
   여기 있다 — 연결된 endpoint를 요구하지 않으므로 61.5%의 에피소드가 애초에 배달 불가다.

3. 결과적으로 **14개 셀의 유효 독립 설정 수는 대략 9–10개**다:
   - `medium` 파생 6종 (heldout / link_loss / link_loss_30 / fast / extreme / sparse×2)
   - node-count 3종
   - 고정 지오메트리 트랩 2계열 × 2 (회전각만 다름, §4.2)

   scenario-macro 평균이 "14개 독립 시나리오"의 평균이라는 인상을 주지만 실제로는
   상관이 높은 셀들의 평균이다. §4.1(n_eff=1 셀 2개)과 결합하면 실효 표본은 더 줄어든다.

**조치**: 시나리오군(config family)별로 먼저 평균한 뒤 군 간 평균하는 2단계 집계로 변경하거나,
최소한 어떤 셀들이 동일/유사 설정인지 논문 Table에 명시.

### A.4 검증에 사용한 주요 소스 파일

| 파일 | 라인 | 검증 내용 |
|---|---|---|
| `implementations/lite_globe/models/student_policy.py` | 289–295, 477–488, 626 | residual 무력화 → MLP 우회 (F-3) |
| " | 563–567, 759–765 | 구조적 DROP 억제 (§6.1) |
| " | 713–775 | switch 규칙 및 danger score |
| `implementations/lite_globe/env/observation.py` | 85–123 | 2-hop 관측 의존성 (F-4) |
| `implementations/lite_globe/env/fanet_env.py` | 38–39 | 전역 ID 액션 공간 (F-11) |
| " | 173–177 | 정적 큐 (F-10) |
| " | 337–348 | 에너지 proxy |
| `implementations/lite_globe/env/link_model.py` | 40–61 | unit-disk + i.i.d. 손실 (§3) |
| `implementations/lite_globe/env/mobility.py` | 24–96 | 2D RWP (§3) |
| `implementations/lite_globe/env/reward.py` | 17–30 | Eq.(3) 불일치 (F-12) |
| `implementations/lite_globe/experiments/phase12_campaign.py` | 145–282, 334–340 | calibration 그리드, Prior-Only 구성 (§4.4) |
| `implementations/lite_globe/scenarios/predictive_traps.py` | 14–81 | 고정 지오메트리 회전 (F-7, F-8) |
| `implementations/lite_globe/scenarios/structural_holes.py` | 14–68 | `max_speed=0.0` (§3.2) |
| `implementations/lite_globe/scenarios/generalization_suite.py` | 19–50, 128–265 | 시나리오 파라미터 (§3.2, §5.1) |
| `implementations/lite_globe/baselines/registry.py` | 전체 | fidelity 계약, 제안기법 제외 (§2.3) |
| `implementations/lite_globe/baselines/evo_qgeo.py` | 25–40 | control byte 회계 대조 (F-5) |
| `implementations/lite_globe/config/switchglobe.yaml` | 전체 | 108-후보 grid (§4.4) |

---

## 맺음말

가장 중요한 한 가지만 남긴다면:

> **`outputs/refactor/ieee_access/source/tables/ablation_overall_metrics.csv`는 이미 원고 소스와
> 함께 배포된다.** 심사위원이 이 파일을 열면
> `Predictive Prior Only, connected_pair_pdr, ..., 0.9116116504854368`과
> `DiSwitch, connected_pair_pdr, ..., 0.9116116504854368`을 **나란히** 보게 된다.
>
> 우리가 먼저 설명하는 편이 압도적으로 낫다.

> **집계 방식에 관한 주석**: 원고 동봉 표(`ablation_overall_metrics.csv`)는 0.9116116504854368,
> raw full run(`artifacts/.../ablation/raw/seed_summaries.csv`)의 scenario-macro는 0.90528954로
> 값이 다르다 — 두 파일의 가중 방식이 다르기 때문이며, 논문 §V-C가 "The repository also
> contains a pooled full-analysis table whose weighting differs from the scenario-macro
> synthesis"로 이미 언급한 사안이다. **중요한 것은 두 집계 방식 모두에서 `Predictive Prior
> Only`와 `DiSwitch`가 마지막 자리까지 동일하다는 점**이다. 즉 이 등가성은 집계 방식의
> 아티팩트가 아니다.

그리고 이 발견은 논문을 **약화시키지 않는다**. 다음 세 가지 주장은 여전히 유효하고, 오히려
더 선명해진다:

1. Privileged global training → strict(ish)-local deployment 계약은 유효하며, Geo-Residual
   Student 대비 **+10.2pp**(0.803 → 0.905)라는 큰 이득을 만든다.
2. 그 이득의 원천은 복잡한 아키텍처가 아니라 **link margin / lifetime / onward-connectivity라는
   세 개의 물리적으로 해석 가능한 신호**다.
3. 해석 가능한 스칼라 스코어 함수가 2-branch 스위칭 정책과 **통계적으로 동등하면서 4.19배
   빠르다** — 이것은 배포 지향 연구로서 강한 결론이며, negative result를 정직하게 보고하는
   논문은 심사에서 높이 평가된다.

바꿔야 할 것은 **결과가 아니라 서사**다.
