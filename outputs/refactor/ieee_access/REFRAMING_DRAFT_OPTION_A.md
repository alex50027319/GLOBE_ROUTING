# 재프레이밍 초안 (옵션 A) — DiSwitch → 단순화/Negative-Result 논문

**작성일**: 2026-09-09
**전제**: `LIMITATIONS_AND_EXTENSIONS_REVIEW.md` §1 (F-1/F-2/F-3)
**대상 원고**: `outputs/refactor/ieee_access/source/main.tex`

> 이 문서는 **초안이며 원고를 직접 수정하지 않는다.** 채택 여부를 결정한 뒤 반영한다.

---

## 0. 왜 재프레이밍인가

현행 서사는 "2-branch + risk switch가 신뢰도를 만든다"이다. 그러나 저자 자신의 5-seed
full run이 이를 지지하지 않는다.

| | Predictive Prior Only | DiSwitch | paired diff (95% CI) |
| --- | --- | --- | --- |
| connected-pair PDR | 0.90528954 | 0.90528954 | **+0.00000000** [−0.000829, +0.000829] |
| deadline ratio | 0.837571 | 0.837643 | +0.00007 [−0.000890, +0.001033] |
| p95 decision latency | **1.487 ms** | 6.228 ms | **+4.742 ms** [+4.637, +4.847] |

14개 시나리오 중 9개가 완전 동일, flagship 4개(predictive_break ×2, structural_hole ×2)가
**전부 정확히 0.00pp 차이**다.

**그러나 데이터는 더 강한 논문을 지지한다.** 다음 세 가지가 모두 사실이기 때문이다.

1. Privileged global training → local deployment 계약이 **+10.2pp**를 만든다
   (Geo-Residual 0.803 → Prior Only 0.905).
2. 그 이득의 원천은 신경망이 아니라 **물리적으로 해석 가능한 세 신호**
   (link margin, link lifetime, onward connectivity)다.
3. 해석 가능한 스칼라 스코어 함수가 2-branch 스위칭 정책과 **통계적으로 동등하면서
   4.19× 빠르다**.

배포 지향 연구에서 (3)은 약점이 아니라 결론이다. **바꿔야 할 것은 결과가 아니라 서사다.**

---

## 1. 새 제목 후보

| # | 제목 | 강조점 |
| --- | --- | --- |
| **A1** | *What Actually Carries the Gain in Learned UAV Routing? A Distilled Predictive Prior Matches a Two-Branch Switched Policy at One Quarter the Latency* | 질문 주도, negative result 정면 |
| **A2** | *Distilled Predictive Priors for Decentralized UAV Routing: Interpretable, Size-Agnostic, and Faster Than the Switched Policy It Replaces* | 제안기법 주도 |
| **A3** | *Privileged Distillation Without the Network: A Ten-Parameter Local Routing Score Matching a Two-Branch Deep Policy in FANETs* | 극단적 단순화 강조 |

**권장: A2.** IEEE Access는 "제안기법 + 검증" 구조를 선호하며, A2는 negative result를
부제에 담으면서도 제안기법이 명확하다. A1은 심사에서 "기여가 무엇인가" 질문을 받기 쉽다.

**제안기법 이름**: `DiSwitch` → **`GeoRisk-P`** 또는 **`DiPrior`**
(distilled prior). Switch가 최종안에서 빠지므로 "Switch"를 이름에 남기면 안 된다.
아래에서는 **`DiPrior`**로 표기한다.

---

## 2. 새 Abstract 초안

> Flying ad hoc networks require reliable packet-forwarding decisions under rapid
> topology change and constrained onboard computation. Global graph policies learn
> strong routing preferences but cannot be executed at a relay, while compact
> approximations reduce inference time at the cost of delivery. This article asks
> which component of a privileged-training, local-execution routing pipeline
> actually carries the reliability gain, and answers it with a five-seed ablation
> on a common simulator.
>
> We train a proximal-policy-optimization graph teacher offline and distil its
> masked preferences into local student policies. The resulting deployed method,
> **DiPrior**, scores each candidate next hop with a closed-form function of
> geographic progress, link margin, estimated link lifetime, and
> onward-connectivity evidence, whose ten coefficients are learned during
> distillation. It contains no neural network at execution time.
>
> Across eight methods, five training seeds, 14 held-out or stress scenarios, and
> 200 episodes per seed–scenario pair, DiPrior obtains a connected-pair packet
> delivery ratio of 0.9053 and a deadline-delivery ratio of 0.8376, improving on
> the strongest adapted baseline by 1.84 and 2.23 percentage points.
>
> **We further report a negative result that we consider the paper's main
> contribution.** A two-branch policy that adds a distilled geographic-residual
> network and a calibrated risk switch on top of DiPrior is *statistically
> equivalent* on every reliability endpoint — the seed-paired difference in
> connected-pair PDR is 0.00000 with a 95% confidence interval of ±0.083
> percentage points, inside the predeclared ±0.5-point non-inferiority margin —
> while being **4.19x slower** (6.228 vs 1.487 ms batch-one p95). Nine of fourteen
> scenarios, including all four adversarial scenarios designed to exercise
> predictive recovery, are bit-identical.
>
> We additionally show that the local observation is permutation-equivariant and
> that no policy parameter depends on the swarm size, so the observation can be
> compacted from 32 global-id slots to 12 local slots with identical logits, a
> 2.56x reduction in policy input footprint. The results argue that in this
> deployment regime the reliability gain comes from interpretable local risk
> evidence, not from architectural depth or conditional branching.

**INDEX TERMS**: Decentralized systems, flying ad hoc networks, inference latency,
interpretable policies, policy distillation, reinforcement learning, UAV routing.

---

## 3. 절별 변경 계획

| 절 | 현행 | 변경 | 근거 |
| --- | --- | --- | --- |
| **I. Introduction** | RQ 4개 (switch·reuse 중심) | RQ를 **"무엇이 이득을 만드는가"** 중심으로 재작성 (§4) | F-1 |
| **II. Related Work** | 유지 | Table 1 novelty matrix에서 "risk-switched recovery" 행을 **"component attribution"** 으로 교체 | F-2 |
| **III. System Model** | Eq.(1)–(4) | Eq.(3)의 `λ_L`, `λ_E` 항 **삭제** (구현되지 않음). 관측 계약을 **"1-hop topology + 2-hop kinematic"** 으로 정정 | F-12, F-4 |
| **IV-A/B. Teacher, KD** | 유지 | 그대로. 이 부분은 검증되었고 +10.2pp의 원천 | — |
| **IV-C. Normal branch** | 제안기법 구성요소 | **ablation 대상으로 강등** | F-1 |
| **IV-D. Predictive branch** | 보조 브랜치 | **제안기법 본체로 승격.** 닫힌 형태 수식과 10개 계수를 전면에 | F-3 |
| **IV-E. Risk switch** | 핵심 기여 | **§VI-C 이후의 negative result로 이동** | F-2 |
| **IV-F. Execution reuse** | systems contribution | **축소.** 불필요한 branch를 두 번 대신 한 번 돌린 것이며, 그 branch 자체가 불필요 | F-1 |
| **IV-G (신설)** | — | **Local-slot observation** (E-2): permutation equivariance, size-agnosticism, 2.56× | F-11 |
| **V. Methodology** | 유지 | §V-A에 시나리오 스위트의 유효 다양성(9–10개) 명시. §V-C에 정수 분위수 한계 | F-13, §4.6 |
| **VI-A. Overall** | 유지 | 시나리오별 최고 baseline 병기 (AODV가 node-count 3개에서 우위) | F-9 |
| **VI-B. Robustness** | "structural-hole에서 유용" | **삭제.** structural_hole은 `max_speed=0.0`인 정적 그래프 | §3.2 |
| **VI-C. Ablation** | switch +1.66pp | **전면 재작성.** Prior Only 대조를 주 결과로 | F-1, F-2 |
| **VI-D~G. Latency** | reuse 41.66% | Prior Only 대비 4.19×를 **주 수치로** | F-1 |
| **VI-H (신설)** | — | **Beacon degradation sensitivity** (E-1) | F-6 |
| **VII. Discussion** | RQ 답변 | 재작성 | — |
| **VII-D. Threats** | 6항목 | 시뮬레이터 충실도 항목 대폭 확대 (§5) | §3 |

---

## 4. 새 연구 질문

현행 4개를 다음 4개로 교체한다.

1. **Does privileged global-to-local distillation improve delivery reliability over
   conventional, value-based, and graph-RL references in a common simulator?**
   → 유지. `+1.84pp` vs Evo-QGeo, `+10.2pp` vs Geo-Residual. 답: 예.

2. **Which component actually carries that gain?**
   → **신규이자 핵심.** 답: 해석 가능한 predictive prior. 신경망 residual도, 2-branch
   구조도, risk switch도 아니다.

3. **Does architectural complexity buy anything measurable at deployment?**
   → **신규.** 답: 아니오. 등가성 구간 ±0.083pp, 대가는 4.19× latency.

4. **How far can the local observation be compacted without changing the policy?**
   → **신규.** 답: 32 → 12 슬롯, 로짓 동일, 2.56× 절감, 재학습 불필요.

---

## 5. 새 기여 목록

1. 오프라인 학습에는 privileged graph 정보를 허용하되 온라인 forwarding은 relay-local과
   **1-hop topology + 2-hop kinematic** 정보로 제한하는 배포 계약을 정식화한다.
2. 그 계약 하에서 **10개 학습 계수만 갖는 닫힌 형태 로컬 스코어 함수(DiPrior)**를 제시하고,
   8개 method·5 seed·14 시나리오·셀당 200 에피소드로 검증한다.
3. **주 결과(negative)**: DiPrior 위에 distilled residual network와 calibrated risk switch를
   얹은 2-branch 정책이 모든 신뢰도 지표에서 **통계적으로 동등**(paired ΔPDR = 0.00000,
   95% CI ±0.083pp)하면서 **4.19× 느리다**. 14개 중 9개 시나리오가 완전 동일하며,
   predictive recovery를 입증하려 설계한 4개 시나리오는 **전부** 차이가 없다.
4. 로컬 관측이 **permutation-equivariant**이고 어떤 정책 파라미터도 스웜 크기에 의존하지
   않음을 보이고, 32개 전역-ID 슬롯을 12개 로컬 슬롯으로 압축하여 **로짓을 보존한 채**
   입력 footprint를 2.56× 줄인다. 재학습이 필요 없다.
5. Risk feature를 **주기적·손실성·잡음 beacon**에서 획득하도록 하는 열화 민감도 연구를
   설계·구현하고, 신호 획득 비용을 통제한 조건에서 위 결론이 유지되는지 보고한다.
6. 실패 유형과 비용 지표를 함께 보고한다: dead-end 실패가 전체 drop의 99.7%이며,
   `predictive_break_225_link_loss`에서 Evo-QGeo에 4.65pp 뒤진다.

---

## 6. Ablation 절(§VI-C) 재작성 초안

> **C. Component Attribution**
>
> Table X reports the five-seed scenario-macro ablation. The privileged
> distillation pipeline produces a large, unambiguous gain: replacing the
> geographic-residual student (0.8030) with the predictive prior (0.9053) improves
> connected-pair PDR by 10.23 percentage points (95% CI 4.20–13.74).
>
> The remaining components do not. Adding the learned residual back into the
> predictive branch *reduces* reliability by 1.66 points (CI −6.30 to +2.98) and
> introduces seed-dependent instability: seed 2718 collapses from 0.9044 to
> 0.8269 while the other four seeds move by less than 0.1 points. Adding the
> calibrated risk switch on top recovers exactly that collapse, which is why a
> naive base-of-comparison makes the switch appear to contribute +1.66 points.
>
> The correct comparison is against the predictive prior itself, which is already
> stable across all five seeds. Against that reference the complete two-branch
> switched policy has a seed-paired connected-pair PDR difference of **0.00000**
> (95% CI −0.00083 to +0.00083), a deadline-ratio difference of +0.00007, and an
> overall-PDR difference of −0.00029. Nine of fourteen scenarios are identical to
> four decimal places, including `predictive_break_45`,
> `predictive_break_225_link_loss`, `structural_hole_45` and
> `structural_hole_225_link_loss` — the four scenarios constructed specifically to
> exercise predictive recovery. The five that differ move by between −0.60 and
> +0.50 points with inconsistent sign.
>
> We therefore report that, in this deployment regime, the switch and the second
> branch are redundant. The switch's causal role is to mask the instability that
> the learned residual introduces; disabling the residual removes the instability
> directly, at one quarter of the latency.

---

## 7. Latency 절 재작성 방향

현행 §VI-D는 legacy(10.004 ms) → DiSwitch(5.836 ms) 41.66% 감소를 systems contribution으로
제시한다. 재프레이밍 후:

- **주 수치**: DiPrior 1.487 ms vs 2-branch 6.228 ms — **4.19×**, 동일 하네스, seed-paired
  CI [+4.637, +4.847] ms.
- **부차**: 중복 forward 제거(41.66%)는 2-branch를 유지할 경우의 구현 개선으로 강등.
  "필요 없는 브랜치를 두 번 대신 한 번 실행한 것"이라고 정직하게 기술.
- **CPU > CUDA 관찰은 유지.** batch-one 워크로드에서 launch/sync 비용이 지배한다는 분석은
  여전히 유효하며 DiPrior에도 적용된다.
- **Early Exit negative result 강화**: 논문은 `.item()` 동기화 비용으로만 설명하지만,
  더 근본적인 이유는 **predictive branch에 MLP가 없어 `T_P ≈ 0`**이라는 점이다.
  Eq.(14)의 `T_g < p_skip·T_P` 조건은 구조적으로 만족될 수 없었다. 이 설명을 추가하면
  negative result가 "미스터리"에서 "예측 가능한 결과"로 격상된다.
- **Fast-DiSwitch 논의는 유지하되 재배치**: 이제 비교 기준이 DiPrior(1.487 ms)이므로
  Fast-DiSwitch(2.005 ms CUDA)의 latency 이점이 사라진다. 즉 **근사 압축은 이제 아무런
  이유가 없다** — 정확한 방법이 더 빠르고 더 정확하다. 이것은 강한 결론이다.

---

## 8. 그림 변경

| 그림 | 조치 |
| --- | --- |
| Fig. 1 (privileged training) | 유지 |
| Fig. 2 (deployment) | **재작성.** 2-branch + switch → 단일 predictive prior. "1-hop" → "1-hop topology + 2-hop kinematic" |
| Fig. 4 (legacy vs reuse) | **강등** (supplementary로 이동) |
| Fig. 8 (ablation) | **재작성.** `Predictive Prior Only vs DiSwitch` 대조를 추가하고 0 효과를 명시 |
| Fig. 10 (paired latency) | **재작성.** 기준선을 DiPrior로 |
| Fig. 12 (Pareto) | **재작성.** DiPrior가 좌상단 단독 점유 (1.487 ms, 0.9053) |
| Fig. 13 (decision map) | **재작성.** 수용된 후보가 DiPrior |
| **신규** | Beacon degradation slope (E-1) |
| **신규** | Slot compaction byte reduction + degree 분포 (E-2) |

---

## 9. 예상 심사 반론과 대응

| 반론 | 대응 |
| --- | --- |
| "신경망이 없으면 RL 논문이 아니다" | 계수는 PPO teacher로부터의 **distillation으로 학습**된다. Geo-Residual(0.803) → Prior(0.905)의 +10.2pp가 그 증거다. 학습된 것은 파라미터의 개수가 아니라 값이다. |
| "그냥 휴리스틱 아닌가" | 그렇다면 **왜 8개 baseline 중 어느 것도 이 성능에 도달하지 못하는가**를 설명해야 한다. Evo-QGeo는 동일한 risk feature에 접근하면서 0.8869에 그친다. 계수 학습이 기여한다. |
| "negative result가 주 기여인 논문은 약하다" | 등가성 구간이 **±0.083pp**로 매우 좁다. 이는 "차이를 못 찾았다"가 아니라 "차이가 없음을 정밀하게 보였다"이다. 게다가 4.19× latency라는 실용적 결론이 따라온다. |
| "왜 처음부터 단순한 방법을 시도하지 않았나" | 계보(`docs/method_history.md`)가 개발 순서를 정직하게 기록한다. 복잡한 방법을 먼저 만들고 ablation으로 단순화한 것은 정상적인 연구 과정이다. |
| "시뮬레이터가 너무 단순하다" | 정당한 지적. §VII-D를 대폭 확대하고 E-1(beacon 열화), E-12(3D Gauss-Markov)를 future work로 명시한다. RWP에서 dead-reckoning이 정확하다는 점(`docs/beacon_degradation_study.md` §8)을 **우리가 먼저** 밝힌다. |

---

## 10. 반영 순서 (권장)

1. §VI-C ablation 재작성 + Fig. 8 갱신 — **가장 먼저**. 다른 모든 변경이 여기서 파생된다.
2. Abstract, §I RQ/기여 재작성
3. §IV-D 승격, §IV-E 강등, §IV-F 축소
4. §VI-D~G latency 재작성 + Fig. 10/12/13
5. §III Eq.(3) 정정, 관측 계약 정정
6. §V-A/§V-C 방법론 보강, §VI-B 삭제
7. §VII-D threats 확대
8. E-1/E-2 결과 절 신설 (실행 후)

**1–3단계만으로도 제출 가능한 정직한 원고가 된다.** 4–8은 major revision 범위다.

---

## 11. 대안: 옵션 B가 살아날 조건

E-1(beacon degradation, `docs/beacon_degradation_study.md`) full run에서
`degradation_slopes.csv`의 `SwitchGLOBE − Predictive Prior Only` 효과가 **열화가 심해질수록
증가하고 CI가 0을 배제**하면, switch의 존재 이유가 물리적으로 정당화되고 원래 서사를
유지할 수 있다.

smoke 실행(1 seed × 10 ep, 검증 전용)에서 **`severe_topology` 셀에서만** SwitchGLOBE 0.4881
vs Prior Only 0.4667 (+2.1pp)이 관찰되었다. 이것이 full run에서 확인해야 할 **유일한 후보
지점**이다.

단, `docs/beacon_degradation_study.md` §8이 보이듯 현재 RWP 이동성에서는 staleness 축이
구조적으로 무력하므로, 옵션 B의 완전한 검증은 **3D Gauss-Markov 이동성(E-12)과 함께**
수행해야 한다. 그 전까지는 옵션 A가 데이터에 부합하는 유일한 서사다.
