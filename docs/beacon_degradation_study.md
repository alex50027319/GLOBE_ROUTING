# E-1: Beacon Degradation Sensitivity Study

이 문서는 `LIMITATIONS_AND_EXTENSIONS_REVIEW.md`의 **F-6**(risk feature가 ground-truth이며
열화 실험이 전무)에 대응하는 실험의 설계·구현·실행 방법을 기술한다.

## 1. 연구 질문

배포된 정책은 `m_v`(link margin), `l_v`(link lifetime), `o_v`(best onward lifetime)로
구동된다. 현재 시뮬레이터는 이 값들을 **오차 0, 지연 0, 손실 0**으로 내부 상태에서 직접
읽는다. 따라서 발표된 신뢰도 수치는 릴레이가 이웃과 **이웃의 이웃**의 운동 상태를 완벽하고
즉각적으로 안다고 가정한다.

이 실험은 그 가정을 완화했을 때 각 방법이 어떻게 열화하는지, 특히 **risk switch가 자신이
스위칭 기준으로 삼는 신호가 불확실해졌을 때 비로소 값어치를 하는지**를 측정한다.

이는 검토 문서 §1.6의 **옵션 B(Switch 구제)를 살릴 수 있는 유일한 현실적 경로**다.

## 2. 비교 대상

| 방법 | 구성 | 역할 |
| --- | --- | --- |
| `SwitchGLOBE` | normal branch + predictive branch + calibrated risk switch | 제안기법 |
| `Predictive Prior Only` | predictive branch 단독 (MLP 없음, switch 없음) | 정확 신호 하에서 SwitchGLOBE와 통계적으로 **동등**하며 4.19× 빠름 |
| `Geo-Residual Student` | normal branch 단독 | risk feature를 **전혀 쓰지 않으므로** 열화에 면역인 기준선 |

세 방법 모두 동일한 (cell, scenario, seed) 평가 시드 블록을 사용하여 episode 수준에서
paired 비교가 성립한다.

## 3. 열화 모델

`implementations/lite_globe/env/beacon.py`

```python
@dataclass(frozen=True)
class BeaconDegradation:
    beacon_period: int = 1          # beacon 간격 (step)
    beacon_loss: float = 0.0        # 개별 broadcast 미수신 확률
    position_noise_std: float = 0.0 # 통신 반경 대비 위치 오차 σ
    velocity_noise_std: float = 0.0 # max_speed 대비 속도 오차 σ
    dead_reckon: bool = True        # 마지막 속도로 외삽
    degrade_topology: bool = False  # action mask도 beacon 기반으로 산출
```

동작:

1. `reset`에서 모든 노드가 한 번 beacon을 보내 캐시를 초기화한다.
2. `step_index % beacon_period == 0`인 step에서 전 노드가 broadcast하고, 각각 `1 - beacon_loss`
   확률로 수신된다. 수신 시 (잡음이 섞인) 참 상태와 타임스탬프를 저장한다.
3. 관측 생성 시 캐시된 위치를 마지막 속도로 dead-reckoning하여 believed state를 만들고,
   그로부터 **거리를 재계산**하여 margin과 lifetime이 일관되게 열화하도록 한다.
4. 릴레이 자신의 상태는 정확하다 (`_believed_state`에서 `current` 인덱스를 참값으로 덮어씀).

### 모델링 범위 (과대 주장 방지)

- **에피소드당 공유 beacon 캐시 1개**, 관측 대상 노드로 인덱싱. 실제 프로토콜은 관측자별
  테이블을 갖는다. 공유 캐시는 주기적 네트워크 전역 beacon flood의 1차 근사이며 MAC
  시뮬레이션이 아니다.
- beacon 자체는 시뮬레이션된 airtime을 소비하지 않는다. 바이트 비용은
  `BeaconCache.control_accounting()`이 별도로 보고한다 (`BEACON_PAYLOAD_BYTES = 20`).
- `degrade_topology=True`일 때 believed adjacency로 mask를 만들고, 참 링크가 없으면
  전송이 실패하여 `drop_reason="link_failure"`가 된다. 에너지는 소비된 것으로 계상된다.

### 기본값은 정확한 no-op

`BeaconDegradation()`은 `enabled == False`이며 이때 환경은 beacon 경로를 **완전히 건너뛴다**.
- 관측이 비트 단위로 동일 (`test_disabled_degradation_is_bit_identical`)
- RNG 스트림을 소비하지 않음 (`test_disabled_degradation_does_not_consume_rng`)
- 따라서 모든 기존 체크포인트·아티팩트·전체 실행 결과가 그대로 재현된다.

## 4. 실험 격자

`implementations/lite_globe/scenarios/degradation_suite.py`

정확 설정을 기준점으로 하는 **one-factor-at-a-time** 민감도 스윕 + 소수의 조합 지점.
전 축 교차곱은 시나리오당 36셀이며 열화 기울기를 구하는 데 불필요하다.

| 축 | 값 | 근거 |
| --- | --- | --- |
| `staleness` | period ∈ {1, 2, 3, 5, 8, 10} | 에피소드 12–16 step, 링크 수명 10–20 step |
| `loss` | ∈ {0, 0.1, 0.2, 0.3} | `ood_link_loss`/`ood_link_loss_30`과 동일 스케일 |
| `noise` | σ ∈ {0, 0.05, 0.10, 0.15} | GPS/IMU 급 오차 |
| `combined` | moderate (T=2, L=0.1, σ=0.05), severe (T=5, L=0.3, σ=0.15) | 현실적 운용점 |
| `combined_topology` | 위 두 지점 + `degrade_topology=True` | 이웃 집합까지 모르는 경우 |

중복 제거 후 **16개 셀**. 14개 시나리오 × 3개 방법 × 5 seed × 200 episode
= 셀당 42,000 에피소드, 전체 **672,000 에피소드**.

## 5. 실행

### 전제: Phase 8 / Phase 11 체크포인트

**이 저장소에는 Phase 8/11/12 체크포인트가 없다.** `artifacts/` 안에는 Fast-SwitchGLOBE와
외부 baseline 체크포인트만 존재하며, `docs/`가 안내하는
`artifacts/switchglobe/final/checkpoints` 디렉터리도 없다. 이는 별도의 재현성 문제이며
검토 문서 §7에 기록되어 있다.

전체 실행에는 원래 학습에 사용한 체크포인트를 지정해야 한다:

```bash
python scripts/run_beacon_degradation_study.py \
  --phase8-checkpoint-dir  <path>/geo_residual/checkpoints \
  --phase11-checkpoint-dir <path>/predictive/checkpoints \
  --output-dir artifacts/beacon_degradation/full \
  --seeds 42 77 123 314 2718 \
  --episodes 200 \
  --device cpu
```

체크포인트가 없으면 스크립트가 경로와 재생성 방법을 명시하며 즉시 중단한다.

### Smoke (하네스 검증 전용)

```bash
# 1. smoke 체크포인트 생성 (Teacher → Geo-Residual → Predictive)
python scripts/train_switchglobe_pipeline.py --device cpu --smoke --resume \
  --artifacts-root artifacts/beacon_degradation/_smoke_checkpoints \
  --stop-after predictive

# 2. smoke 스터디
python scripts/run_beacon_degradation_study.py \
  --phase8-checkpoint-dir  artifacts/beacon_degradation/_smoke_checkpoints/training/geo_residual/checkpoints \
  --phase11-checkpoint-dir artifacts/beacon_degradation/_smoke_checkpoints/training/predictive/checkpoints \
  --smoke
```

> smoke는 1 seed × 10 episode이며 체크포인트가 undertrained다. **논문 수치로 사용하지 않으며
> full 결과와 절대 병합하지 않는다.**

## 6. 산출물

| 파일 | 내용 |
| --- | --- |
| `episodes.csv` | 에피소드당 1행 + 열화 파라미터 컬럼 |
| `seed_summaries.csv` | (method, cell, scenario, seed)당 1행 |
| `degradation_slopes.csv` | 셀별 `SwitchGLOBE − Predictive Prior Only` seed-paired 효과와 95% t-CI |
| `manifest.json` | 설정, 체크포인트 SHA-256, row count |

`degradation_slopes.csv`가 핵심 산출물이다. 해석:

- 열화가 심해질수록 효과가 **커지면** → risk switch가 신호 불확실성 하에서 값어치를 한다.
  검토 문서의 옵션 B가 성립하고 논문의 원래 서사를 유지할 수 있다.
- 모든 운용점에서 효과가 **0에 머무르면** → switch는 모든 조건에서 중복이다.
  옵션 A(재프레이밍)의 근거가 결정적으로 강화된다.

어느 쪽이든 현재보다 강한 논문이 된다.

## 7. Smoke 실행에서 관찰된 하네스 동작

아래는 **하네스가 의도대로 동작하는지 확인한 결과이며 연구 결과가 아니다**
(1 seed × 10 episode, undertrained 체크포인트).

| cell | SwitchGLOBE | Prior Only | Geo-Residual |
| --- | --- | --- | --- |
| exact | 0.8939 | 0.8939 | 0.6816 |
| moderate | 0.8960 | 0.8960 | 0.6857 |
| **moderate_topology** | **0.6690** | **0.6690** | 0.5571 |
| severe | 0.7821 | 0.7821 | 0.6589 |
| **severe_topology** | **0.4881** | 0.4667 | 0.4286 |

세 가지가 드러난다.

1. **`degrade_topology`가 지배적 효과다.** moderate 0.8960 → 0.6690 (−22.7pp),
   severe 0.7821 → 0.4881 (−29.4pp). 불완전한 beacon의 실제 비용은 risk feature가
   부정확해지는 것이 아니라 **이웃이 누구인지 모르게 되는 것**이다.
2. **feature-only 열화는 거의 무료다.** staleness/loss/noise 축이 단조 추세 없이
   0.87–0.95에서 진동한다. 이는 §8의 dead-reckoning 정확성과 일치한다.
3. `severe_topology`에서 처음으로 SwitchGLOBE(0.4881) > Prior Only(0.4667)가 나타난다
   (+2.1pp). full run에서 확인해야 할 유일한 후보 지점이다.

## 8. 부수 발견: RWP에서 dead reckoning은 정확하다

`test_one_step_dead_reckoning_is_exact_under_random_waypoint`가 문서화하는 성질:

```
step 1: dead-reckon 위치 오차  mean=0.0000  max=0.0000   (R=4.0)
step 2:                        mean=0.0964  max=0.7715
step 3:                        mean=0.3368  max=1.5430
step 5:                        mean=0.9184  max=3.4538
step 6:                        mean=1.4936  max=4.3966
```

**Random Waypoint 이동성은 구간별 등속**이고, `predicted_link_lifetime_steps`도 **등속 상대
운동을 가정**한다. 두 모델이 정확히 일치하므로:

- 1-step stale beacon은 참 위치를 **부동소수점 정밀도로** 재현한다 (오차 0.0000).
- 오차는 노드가 waypoint에 도달해 속도를 바꿀 때부터 비로소 누적된다.
- 즉 `l_v`는 단순한 ground-truth 측정값이 아니라 **링크 만료에 대한 완벽한 oracle**이다.

이는 predictive prior가 왜 그렇게 강해 보이는지에 대한 **구조적 설명**이며, 검토 문서 §3
(이동성 모델 충실도)의 정량적 근거다. Gauss-Markov, Paparazzi 등 가속을 포함하는 UAV 이동성
모델에서는 동일한 staleness가 실질적 오차를 낳는다.

> 따라서 E-1의 완전한 형태는 **3D Gauss-Markov 이동성(E-12)과 함께 수행**해야 한다.
> 현재 RWP 위에서는 staleness 축이 구조적으로 무력하다.

## 9. 병행 수정된 기존 결함

이 작업 중 `aggregate_generalization` 계열 함수 5곳에서 기존 결함을 발견하고 수정했다.

**증상**: `delivered == 0`인 셀에서 전달 조건부 지표가 `None`이 되는데,
`float(row[metric])`에 `None` 가드가 없어 `TypeError`로 파이프라인이 중단된다.
`git stash`로 변경분을 제거한 상태에서 재현하여 **기존 결함임을 확인**했다.

**영향 파일**: `phase7_reporting.py`, `phase8_reporting.py`, `phase11_reporting.py`,
`phase12_reporting.py`, `reporting.py`, `baseline_reporting.py`

**부수 발견**: 각 파일의 "전달 조건부 지표" allowlist가 **서로 다르다**.

| 파일 | allowlist |
| --- | --- |
| phase7 / phase8 / baseline | `mean_success_delay`, `p95_success_delay`, `mean_path_stretch` |
| phase11 / phase12 | 위 3개 + `energy_per_delivered_packet`, `energy_per_on_time_delivery` |

동일 지표가 캠페인에 따라 다르게 집계된다는 뜻이며, 논문 §V-C가 언급한
"pooled full-analysis table whose weighting differs from the scenario-macro synthesis"의
한 원인일 수 있다. **이 불일치는 아직 통일하지 않았다** — 기존 full-run 결과의 재현성을
깨뜨리므로 별도 결정이 필요하다.

**수정 내용**: `None` 값을 건너뛰어 `count=0`, `mean=None` 분기로 떨어지게 했다.
`simulation_protocol.md`의 "missing cell은 N/A로 보고하며 0으로 대체하지 않는다" 규칙과
일치하며, 이전에 성공하던 실행의 결과는 바뀌지 않는다 (이전에는 crash였음).

## 10. 관련 파일

| 경로 | 역할 |
| --- | --- |
| `implementations/lite_globe/env/beacon.py` | 열화 모델 |
| `implementations/lite_globe/env/config.py` | `FanetConfig.beacon` 필드, YAML 로딩 |
| `implementations/lite_globe/env/fanet_env.py` | `_believed_state()`, beacon step, `link_failure` |
| `implementations/lite_globe/scenarios/degradation_suite.py` | 실험 격자 |
| `scripts/run_beacon_degradation_study.py` | 실행기 |
| `tests/lite_globe/test_beacon_degradation.py` | 18개 계약 테스트 |
