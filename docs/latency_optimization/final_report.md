# SwitchGLOBE decision-latency 최적화 연구 보고서

작성 기준일: 2026-09-05  
대상 브랜치: `refactor/globev2`  
검증 코드 기준: `92d17df3f4451e0412858a9927a898a2696023b3` (primary A100) / `29a8b4f768fc573cd273c7ea5dba463de17b80eb` (확장 측정 코드)  
GPU 실행: Google Colab CLI 0.6.0, NVIDIA A100-SXM4-40GB

## 1. 결론

현재 가장 안전한 결론은 **SwitchGLOBE Exact의 단일 fused forward를 유지하고, batch=1 온라인 라우팅은 CPU를 기본 배치 장치로 사용하는 것**이다. 이미 제거한 legacy 중복 추론과 비교하면 현재 Exact는 A100 CUDA p95 기준 10.004 ms에서 5.836 ms로 41.66% 감소한다. 동일 A100 런타임의 CPU에서는 Exact p95가 2.208 ms로 CUDA보다 62.16% 낮다. 작은 모델·batch 1에서는 GPU kernel launch와 동기화 비용이 계산 절감분보다 크기 때문이다.

FastSwitchGLOBE는 latency 자체는 가장 강하다. 현재 커밋에서 중복 진단 forward를 제거한 단일-pass Fast는 A100 CUDA p95 2.005 ms로 Exact 대비 65.65% 빠르며, CPU p95 0.847 ms다. 그러나 5 seeds × 14 scenarios × 200 episodes의 full reliability 검증에서 connected-pair PDR이 0.90529에서 0.88720으로 1.81 percentage points 하락했고, paired 95% CI `[-3.606, -0.011] pp`가 0을 넘지 않았다. 사전 정의한 허용 열화 0.5 pp를 통과하지 못하므로 Exact의 기본 대체안으로 채택할 수 없다.

Calibrated Early Exit는 43,467개 decision 중 32,285개(74.27%)에서 predictive branch를 건너뛰면서 현재 체크포인트·시나리오에서 episode trajectory 불일치가 0건이었다. 그러나 A100 CUDA p95는 6.386 ms로 Exact보다 9.42% 느렸다. branch 판단을 위한 Python control flow와 CUDA scalar synchronization이 절약된 작은 predictive MLP 계산보다 비쌌다. 따라서 현재 구현은 **동작 보존에는 성공했지만 latency 최적화에는 실패**했다.

## 2. 무엇을 바꾸었는가

### 2.1 FastSwitchGLOBE 중복 추론 제거

기존 adapter 경로는 route action을 얻기 위한 model forward 뒤에 diagnostics를 다시 호출할 수 있어 한 decision에서 네트워크가 두 번 실행되는 회귀가 있었다. `FastStudentPolicyOutput`에 `switch_logit`을 함께 반환하도록 바꾸고 adapter가 첫 output을 재사용하게 했다. 회귀 테스트는 adapter 호출 한 번당 model forward가 정확히 한 번만 발생하는지 검사한다.

### 2.2 Calibrated Early Exit 구현과 guard 정합성 수정

Early Exit는 먼저 normal branch를 계산하고, 아래 안전 조건에서 predictive branch를 생략한다.

\[
g(x)=\mathbf{1}[\exists\,a_{\mathrm{live}}]\,
\mathbf{1}[a_N\neq a_{\mathrm{drop}}]\,
\mathbf{1}[d_N\le m]
\]

여기서 \(d_N\)은 normal branch가 계산한 danger score, \(m\)은 calibration margin이다. `g(x)=1`일 때 normal output을 그대로 사용하고, 아니면 predictive branch와 원래 switch gate를 실행한다. calibration counter도 실제 실행 guard와 동일하게 DROP action을 skip 대상으로 세지 않도록 수정했다.

### 2.3 재현 가능한 검증 도구

- `run_early_exit_validation.py`: 5×14×200 paired episode trajectory 검증
- `run_latency_benchmark.py --include-early-exit --include-fast`: 동일 프로세스·동일 observation에서 CPU/CUDA 비교
- `analyze_latency_optimization.py`: seed-paired t-CI, deterministic bootstrap CI, exact sign-flip test, Holm correction
- `generate_latency_optimization_figures.py`: 20개 figure를 PNG 240 dpi와 SVG로 동시 생성
- `scripts/colab/*`: Colab CLI 업로드, GPU 확인, 실행, 결과 패키징 절차

## 3. 실험 설계

### 3.1 통계 단위와 반복

- training seeds: `42, 77, 123, 314, 2718`
- latency warm-up: seed·variant·component별 50회
- latency repetitions: 2,000회
- batch size: 1
- primary latency statistic: 각 seed의 synchronized end-to-end p95
- 비교: 동일 seed끼리 paired reduction 계산
- CI: seed 평균에 대한 95% Student-t CI와 10,000회 bootstrap percentile CI
- 비모수 확인: 가능한 `2^5=32`개 부호 조합을 모두 열거한 two-sided exact sign-flip test
- 다중 비교: device별 Holm correction
- full routing validation: 5 seeds × 14 scenarios × 200 episodes = 14,000 episodes

5개 seed의 two-sided exact sign-flip 검정은 최솟값이 0.0625다. 따라서 모든 seed의 방향이 일치해도 전통적인 0.05를 넘는다. 이 보고서는 p-value 하나로 결론을 정하지 않고 effect size, CI, reliability acceptance gate를 함께 사용한다.

### 3.2 latency 분해

의사결정 시간은 다음과 같이 본다.

\[
T_{dec}=T_{pre}+T_{model}+T_{gate}+T_{extract}+T_{sync}+T_{runtime}
\]

각 component benchmark는 병목 위치를 찾는 독립 타이머이며 합산값을 end-to-end로 간주하지 않는다. 최종 비교는 observation 입력부터 action과 metadata 반환까지를 한 번에 재는 synchronized end-to-end 값이다.

### 3.3 품질 acceptance gate

속도 후보는 다음을 모두 만족해야 기본 Exact를 대체할 수 있다.

- connected-pair PDR 열화의 95% CI 하한이 `-0.005` 이상
- deadline delivery ratio 열화의 95% CI 하한이 `-0.005` 이상
- success-delay와 energy-per-delivered-packet의 방향성 악화가 없거나 사전 정의 허용범위 이내
- NaN/Inf, invalid action, mask violation, episode count 누락 없음
- latency 개선이 동일 장치·동일 세션·동일 checkpoint에서 관찰됨

## 4. A100 결과

장치: `NVIDIA A100-SXM4-40GB`, CUDA runtime 12.8, PyTorch 2.11.0+cu128, Python 3.13.15. 원격 bundle·Exact checkpoint·Fast checkpoint archive의 SHA-256은 로컬 값과 일치했다.

| Variant | Device | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Exact 대비 paired p95 |
|---|---:|---:|---:|---:|---:|---:|
| Exact eager | CPU | 2.168 | 2.161 | 2.208 | 2.344 | 기준 |
| Early Exit | CPU | 2.366 | 2.356 | 2.427 | 2.581 | -9.90% |
| Fast single-pass | CPU | 0.812 | 0.814 | 0.847 | 0.927 | +61.66% |
| Fast + Top-2 | CPU | 0.853 | 0.852 | 0.881 | 0.933 | +60.11% |
| Buffered Exact | CPU | 2.187 | 2.179 | 2.244 | 2.370 | -1.62% |
| Legacy repeated | CPU | 3.698 | 3.687 | 3.775 | 3.935 | -70.97% |
| Exact eager | CUDA | 5.671 | 5.648 | 5.836 | 6.101 | 기준 |
| Early Exit | CUDA | 6.198 | 6.172 | 6.386 | 6.663 | -9.42% |
| Fast single-pass | CUDA | 1.957 | 1.950 | 2.005 | 2.120 | +65.65% |
| Fast + Top-2 | CUDA | 2.040 | 2.032 | 2.092 | 2.183 | +64.16% |
| Buffered Exact | CUDA | 5.626 | 5.602 | 5.788 | 6.020 | +0.83% |
| Legacy repeated | CUDA | 9.741 | 9.706 | 10.004 | 10.376 | -71.43% |

주요 paired 95% t-CI는 다음과 같다.

| Candidate | Device | p95 감소율 | 95% t-CI | bootstrap 95% CI | 판정 |
|---|---|---:|---:|---:|---|
| Early Exit | CUDA | -9.42% | [-10.62, -8.22] | [-10.22, -8.61] | reject |
| Buffered Exact | CUDA | +0.83% | [+0.21, +1.44] | [+0.48, +1.25] | 효과가 작아 hold |
| Fast | CUDA | +65.65% | [+65.06, +66.24] | [+65.24, +65.95] | latency pass, reliability fail |
| Fast + Top-2 | CUDA | +64.16% | [+63.65, +64.66] | [+63.84, +64.50] | latency pass, reliability 미해결 |

CUDA가 CPU보다 느리다는 결과는 오류가 아니다. Exact batch-1 모델은 CPU p95 2.208 ms, CUDA p95 5.836 ms다. 각 call의 작은 연산량에 비해 kernel launch, host-device tensor 구성, action extraction의 scalar synchronization 비용이 지배적이다. UAV onboard deployment의 실제 선택은 해당 SoC/CPU/GPU에서 다시 측정해야 하지만, 현재 A100 환경에서는 CPU 경로가 합리적이다.

## 5. 로컬 결과

현재 Mac CPU 동일 조건(5 seeds, warm-up 50, repeats 500)에서 seed-mean p95는 다음과 같다.

| Variant | P95 (ms) | Exact 대비 paired 감소율 | 95% t-CI |
|---|---:|---:|---:|
| Exact eager | 0.918 | 기준 | — |
| Early Exit | 1.020 | -11.71% | [-29.63, +6.20] |
| Fast single-pass | 0.271 | +70.19% | [+64.73, +75.64] |
| Fast + Top-2 | 0.326 | +65.09% | [+56.17, +74.00] |
| Buffered Exact | 0.988 | -7.90% | [-34.50, +18.69] |
| Legacy repeated | 1.677 | -83.77% | [-149.53, -18.01] |

로컬과 A100 모두 Fast의 큰 latency 감소, Early Exit의 runtime 악화, buffer reuse의 작거나 불안정한 효과라는 방향이 일치한다.

확장 측정 코드에서는 latency summary에 `p90_ms`, `max_ms`, `std_ms`, `coefficient_of_variation`을 추가하고 CUDA peak device memory를 `deployment_costs.csv`에 기록한다. 이 확장 코드의 confirmatory A100 benchmark는 backend 세션 회수로 결과 CSV가 생성되지 않았으므로, 확장 컬럼의 A100 수치를 primary 결과에 소급해 채우지 않았다.

## 6. Early Exit 검증

margin sweep에서 `m=0`만 43,467 decision 전수 관찰에서 action divergence 0이었다. `m=0.01`부터 all-step divergence 0.0437%, skipped-step divergence 0.0579%가 나타났다. 더 큰 margin은 skip률을 높이지만 divergence도 증가했다.

실제 `m=0` 정책을 14,000 episode에 실행한 결과:

- decisions: 43,467
- early-exit decisions: 32,285 (74.2747%)
- predictive-branch decisions: 11,182
- Exact와 delivered/dropped/reason/steps/hops/transmission-attempts/deadline mismatch: 0
- 단, 이것은 현재 checkpoint와 평가 시나리오에 대한 경험적 동치이며 모든 미관측 상태에 대한 형식적 증명은 아니다.

기대시간은 다음과 같다.

\[
E[T_{EE}]=T_N+(1-p_{skip})T_P+T_g
\]

따라서 Early Exit가 이기려면 \(T_g<p_{skip}T_P\)여야 한다. 측정 결과는 현재 Python/CUDA 구현에서 이 부등식이 성립하지 않음을 보여준다. 다음 시도는 branch를 Python이 아니라 하나의 compiled graph 또는 tensor-only predication으로 내려야 한다.

## 7. FastSwitchGLOBE 품질 결과

| Metric | Exact | Fast | 방향성 차이 | paired 95% CI | Gate |
|---|---:|---:|---:|---:|---|
| connected-pair PDR | 0.90529 | 0.88720 | -0.01809 | [-0.03606, -0.00011] | fail |
| deadline delivery ratio | 0.83764 | 0.81500 | -0.02264 | [-0.04625, +0.00096] | fail |
| p95 success delay | 4.2643 | 4.4864 | -0.2221 (lower-better 방향) | [-0.4208, -0.0235] | fail |
| energy/delivered packet | 2.2279 | 2.3694 | -0.1415 (lower-better 방향) | [-0.2799, -0.0031] | fail |

여기서 energy는 프로젝트 simulator의 energy proxy이며 실제 Joule 측정이 아니다. Fast+Top2는 표준 seed-42 평가에서 Fast와 동일 outcome을 유지하고 stale-primary 합성 실험 463/463건에서 backup resolution에 성공했으나, Fast 자체의 full reliability 손실을 회복하는 기법은 아니다.

## 8. 후보별 최종 판정

| 후보 | 핵심 아이디어 | 결과 | 판정 |
|---|---|---|---|
| Fused Exact | 중복 diagnostics/branch forward 제거 | A100 CUDA legacy 대비 p95 41.66% 감소, 동작 동일 | 채택·현재 기본 |
| CPU device placement | batch-1을 CPU에서 실행 | A100 host CPU p95 2.208 ms vs CUDA 5.836 ms | 현재 환경 채택 |
| Calibrated Early Exit | 안전 상태에서 predictive branch 생략 | skip 74.27%, trajectory mismatch 0, CUDA p95 9.42% 악화 | reject |
| Fast single-pass | 작은 shared representation과 1회 forward | CUDA p95 65.65% 감소, PDR gate fail | 연구 후보/기본 대체 불가 |
| Fast + Top-2 | Fast primary와 backup을 한 번에 산출 | CUDA p95 64.16% 감소, stale-primary 복구 | 조건부 failover 후보 |
| Freshness cache | 동일 observation/fingerprint 재사용 | 표준 workload hit 0%, 반복-query 합성에서만 효과 | 조건부 전용 |
| Tensor buffer reuse | 입력 tensor storage 재사용 | A100 CUDA +0.83%, CPU -1.62%; 효과 작음 | hold |
| `torch.compile` | graph compilation | 구 동일세션 검증에서 CPU/CUDA 모두 개선 없음 | reject |

## 8.1 A100 operator profile

Primary latency timer와 분리한 PyTorch profiler를 A100에서 seed 42, `heldout_medium`, warm-up 50, profiled calls 30으로 수행했다. Exact에서 상위 device time은 boolean reduction kernel, `aten::all`, `aten::_local_scalar_dense`, DtoH memcpy, `aten::any`였다. `_local_scalar_dense`/DtoH가 1,050회씩 발생한 것은 GPU scalar를 Python 조건·action extraction에서 읽어오는 동기화 비용의 직접 증거다. Early Exit는 해당 패턴이 1,140회로 늘어났고, Fast는 300회 수준으로 감소했다.

이 결과는 다음 최적화 우선순위를 지지한다.

1. gate/action extraction을 host scalar로 materialize하지 않고 tensor-only로 유지한다.
2. Early Exit의 Python branch를 `torch.cond` 또는 deployment runtime conditional graph로 바꾼다.
3. Fast의 단일 forward 경로는 kernel 수와 DtoH sync를 줄이지만 reliability gate를 별도로 통과해야 한다.

프로파일 trace·operator CSV·table은 `artifacts/switchglobe_latency_optimization/globev2_colab_cli_20260905/profile/results/`에 있다. profiler의 30회 이벤트 합계는 primary p95 latency가 아니며, 병목 attribution용으로만 사용한다.

## 9. 문헌에서 얻은 설계 원칙

Notion 데이터베이스의 접근 가능한 45편 전수를 검토하고 latency 직접성이 높은 논문을 우선순위화했다.

- Predictive-Q의 dual-path/Q-table은 per-packet 1.00 ms 수준의 경량 경로와 backup 준비의 가치를 보여준다. SwitchGLOBE에는 Fast+Top2로 대응했지만 quality gate가 남았다.
- RLFR/DRLFR의 local neighbor sharing과 Raspberry Pi testbed는 전역 graph 재구성보다 로컬 feature·경량 action head·실기기 검증이 중요함을 보여준다.
- RoutePPO의 eBPF/P4 path update는 추론뿐 아니라 forwarding-plane 반영시간을 분리 측정해야 함을 보여준다.
- KG-DDRL의 KAN-GNN sparse/local encoding과 DFS-PPO의 hard pruning/action shielding은 모델을 줄이기 전에 후보 action 수를 구조적으로 줄이는 방향을 지지한다.
- Q-learning/fuzzy 계열은 작은 inference 자체는 유리하지만 destination별 table과 HELLO overhead가 노드 수에 따라 커질 수 있으므로 compute latency만으로 비교하면 안 된다.

상세 문헌 행렬은 `literature_matrix.md`, 설계 가설은 `candidate_designs.md`에 기록했다.

## 10. 한계

1. latency raw decision 2,000개 자체는 CSV에 저장하지 않고 seed·component별 집계값만 저장했다. 다음 strict run은 raw timing과 randomized block order를 남겨야 한다.
2. A100은 onboard UAV hardware를 대표하지 않는다. Jetson Orin 계열 또는 목표 비행 컴퓨터에서 CPU/GPU/accelerator를 다시 비교해야 한다.
3. Early Exit의 0 mismatch는 현재 14개 시나리오·현재 checkpoint에 대한 관찰 결과다.
4. Fast와 Fast+Top2 reliability는 동일 Fast 정책 가중치를 사용한다. Top-2가 stale action에는 대응하지만 policy approximation error를 수정하지 않는다.
5. 5개 seed는 effect-size 추정에는 사용 가능하지만 exact two-sided sign test에서 0.05를 달성할 수 없다. 후속 confirmatory run은 seed 수를 최소 8–10으로 늘리는 것이 바람직하다.
6. `policy_input_bytes`는 in-memory tensor 크기이며 무선 routing control bytes가 아니다.
7. simulator energy proxy를 Joule로 표현해서는 안 된다.

## 11. 향후 우선순위

1. **Exact CPU production path 고정**: 중복 forward 회귀 테스트와 p95 budget CI에 포함.
2. **Tensor-only early exit**: Python `.item()`/조건 분기를 제거하고 normal branch, gate, conditional predictive branch를 `torch.cond` 또는 deployment runtime의 native conditional graph로 묶어 재측정.
3. **Action-space pruning**: hard mask로 live/risk-feasible top-k 후보만 남긴 뒤 Exact head를 실행한다. 정책 의미를 바꾸지 않는 conservative shield부터 시작한다.
4. **Selective Fast escalation**: Fast가 높은 confidence와 low-risk를 동시에 만족하는 상태에서만 사용하고, 그 외 Exact로 되돌리는 cascading policy를 새로 학습·calibration한다.
5. **Distillation 개선**: route KL만이 아니라 switch logit, normal/predictive branch logit, pairwise ranking, worst-scenario reweighting을 포함한다.
6. **실기기 검증**: Jetson/Raspberry Pi 계열에서 model time, end-to-end decision, routing-table update, 무선 control overhead, 전력(Joule)을 분리 측정한다.
7. **confirmatory statistics**: 10 seeds, randomized block, raw timings, hierarchical bootstrap과 scenario-family worst-case gate를 사전 등록한다.

## 12. 재현 자료

- A100 결과 ZIP: `artifacts/switchglobe_latency_optimization/globev2_colab_cli_20260905/switchglobe_globev2_a100_20260905.zip`
- A100 operator profile ZIP: `artifacts/switchglobe_latency_optimization/globev2_colab_cli_20260905/profile/switchglobe_globev2_a100_profile_20260905.zip`
- A100 session log: `artifacts/switchglobe_latency_optimization/globev2_colab_cli_20260905/colab_cli_session.jsonl`
- 실행 notebook: `notebooks/switchglobe_latency_a100_colab_cli.ipynb`
- paired statistics: `artifacts/switchglobe_latency_optimization/globev2_final_20260905/metrics/paired_latency_statistics.csv`
- 20 figures: `artifacts/switchglobe_latency_optimization/globev2_final_20260905/figures/`
- figure index: `artifacts/switchglobe_latency_optimization/globev2_final_20260905/FIGURE_INDEX.md`

원격 결과 ZIP SHA-256: `07bbd3abb9991ed6293d8326926ea2f154453db47cc76c1923da709688f12b41`.
operator profile ZIP SHA-256: `8d4212add8cdaa537da668b37cbce6d9904bc58637d5e21120cdff8e44cf9252`.
