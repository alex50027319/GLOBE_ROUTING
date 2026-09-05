# SwitchGLOBE latency 후보 설계와 가설 카드

## 1. 문제 정의

SwitchGLOBE Exact는 local observation \(x_t\)에서 normal policy \(\pi_N\), predictive policy \(\pi_P\), switch gate \(s\)를 계산한다.

\[
z_N=f_N(x_t),\quad z_P=f_P(x_t),\quad
s=\mathbf{1}[\sigma(h(x_t,z_N,z_P))>\tau]
\]

\[
\pi(a\mid x_t)=(1-s)\operatorname{softmax}_M(z_N)
+s\operatorname{softmax}_M(z_P)
\]

여기서 \(M\)은 invalid action을 제거하는 action mask다. 최적화 목적은 단순 평균시간이 아니라 tail과 품질을 함께 포함한다.

\[
\min_\theta\;E[T_{dec}(\theta)]
+\lambda_{95}\operatorname{CVaR}_{0.95}(T_{dec})
+\lambda_C C(\theta)
\]

subject to

\[
\Delta\mathrm{PDR}_{L95}\ge-0.005,\quad
\Delta\mathrm{DDR}_{L95}\ge-0.005,
\]

mask violation = 0, NaN/Inf = 0, episode accounting complete.

## 2. 병목 가설

프로파일링 전 가설은 predictive branch의 중복 계산이 주 병목이라는 것이었다. 실험 결과 병목은 계층적으로 나뉜다.

1. 과거 legacy adapter에는 diagnostics와 action 계산을 위한 중복 forward가 있었다.
2. Exact는 normal과 predictive branch를 모두 실행한다.
3. batch=1 CUDA에서는 연산 자체보다 kernel launch와 host synchronization이 크다.
4. Early Exit도 gate가 Python control flow와 CUDA scalar readback을 만들면 계산을 줄이고도 느려질 수 있다.
5. 작은 Fast model은 계산량을 크게 줄이지만 approximation error가 reliability를 훼손한다.

## 3. 후보 1 — Fused Exact

### 가설

같은 logits와 diagnostics를 얻기 위해 동일 network를 반복 호출하면 latency만 증가하고 policy 의미는 바뀌지 않는다. 모든 필요한 값을 한 forward output에 담으면 exact equivalence를 유지하면서 시간을 줄일 수 있다.

### 구현

- `RiskSwitchLiteGlobePStudentPolicy.decide()`의 fused output 사용
- Fast output에도 `switch_logit`을 포함해 adapter가 diagnostics를 재호출하지 않게 수정
- forward-call count regression test 추가

### 검증과 판정

- A100 CUDA legacy repeated p95 10.004 ms
- current Exact p95 5.836 ms
- 감소율 41.66%
- 정책 수식, checkpoint, action semantics 불변

**판정: 채택.** 가장 강한 low-risk 최적화다.

## 4. 후보 2 — Device-aware CPU 배치

### 가설

작은 batch-1 MLP/GNN에서 \(T_{launch}+T_{sync}\)가 GPU 계산 이득보다 크면 CPU가 더 빠르다.

\[
T_{CUDA}=T_{H2D}+\sum_k T_{launch,k}+T_{kernel}+T_{sync}+T_{D2H}
\]

### 결과

- A100 session host CPU Exact p95: 2.208 ms
- A100 CUDA Exact p95: 5.836 ms
- CPU가 62.16% 낮음

### 주의

이는 A100 자체가 느리다는 의미가 아니라 현재 작은 model과 호출 방식에 대한 결과다. batching 또는 persistent graph에서는 순위가 달라질 수 있다.

**판정: 현재 batch-1 배치 기본값으로 채택.** 목표 onboard hardware에서 재검증 필요.

## 5. 후보 3 — Calibrated Early Exit

### 가설

normal branch가 명백히 안전하고 DROP이 아니면 predictive branch를 생략한다.

\[
p_{skip}=P(\exists a_{live},a_N\neq DROP,d_N\le m)
\]

\[
E[T_{EE}]=T_N+(1-p_{skip})T_P+T_g
\]

개선 조건은 \(T_g<p_{skip}T_P\)다.

### calibration

| Margin | Skip rate | All-step divergence | Divergence among skipped |
|---:|---:|---:|---:|
| 0.00 | 74.2747% | 0 | 0 |
| 0.01 | 75.5056% | 0.0437% | 0.0579% |
| 0.02 | 76.7272% | 증가 | 0.1169% |
| 0.05 | 83.2448% | 증가 | 0.3068% |
| 0.10 | 88.9893% | 증가 | 3.0790% |
| 0.20 | 93.0752% | 증가 | 3.9894% |

### full execution

- 14,000 episodes, 43,467 decisions
- actual skip 32,285, predictive execution 11,182
- paired trajectory mismatch 0
- A100 CUDA p95 6.386 ms vs Exact 5.836 ms: 9.42% 악화

### 원인

`danger <= margin` 판단 과정의 scalar synchronization과 Python branch dispatch가 predictive MLP 절약보다 크다. CPU에서도 9.90% 악화했다.

**판정: 현재 구현 reject.** tensor-only/compiled conditional graph 형태로만 재시도.

## 6. 후보 4 — FastSwitchGLOBE single-pass

### 가설

큰 teacher/Exact branch를 작은 student \(q_\phi\)로 압축하면 parameter와 compute를 줄일 수 있다.

\[
\mathcal{L}_{KD}=T^2\,KL(\pi_E^T\Vert\pi_F^T)
+\alpha\,BCE(s_E,s_F)
+\beta\,\mathcal{L}_{rank}
\]

현재 checkpoint는 작은 Fast policy를 사용하며 parameter count는 약 7,011이다.

### 결과

- local CPU p95 0.271 ms, Exact 대비 70.19% 감소
- A100 CPU p95 0.847 ms, Exact 대비 61.66% 감소
- A100 CUDA p95 2.005 ms, Exact 대비 65.65% 감소
- connected-pair PDR `-1.8086 pp`, CI `[-3.6064, -0.0108] pp`
- deadline delivery ratio `-2.2643 pp`, CI `[-4.6248, +0.0962] pp`

**판정: latency pass, reliability fail.** 기본 대체 불가.

## 7. 후보 5 — Fast + Top-2 hybrid

### 가설

한 forward에서 primary와 backup action을 함께 얻으면 stale primary가 invalid가 되었을 때 재추론 없이 backup으로 바꿀 수 있다.

\[
a_t=\begin{cases}
a_1,&M_t(a_1)=1\\
a_2,&M_t(a_1)=0\land M_t(a_2)=1\\
DROP,&\text{otherwise}
\end{cases}
\]

### 결과

- A100 CUDA p95 2.092 ms; Exact 대비 64.16% 감소
- Fast 단독보다 p95 약 4.35% 증가
- stale-primary 합성 463/463건에서 backup resolution 성공
- 표준 seed-42 결과는 Fast와 동일

### 한계

Top-2는 action freshness를 보완하지만 student의 policy approximation error를 보정하지 않는다.

**판정: 조건부 failover 후보.** Fast reliability gate가 해결되기 전 기본 배치 금지.

## 8. 후보 6 — Observation freshness cache

### 가설

동일 노드에서 observation fingerprint가 변하지 않으면 최근 decision을 재사용한다.

\[
k_t=H(node, destination, action\ mask, quantized\ features, epoch)
\]

cache hit에서 \(T_{dec}\approx T_{hash}+T_{lookup}\)가 된다.

### 결과

- 표준 episode workload: hit rate 0%, p95 0.3335→0.3767 ms 악화
- repeated-query 합성 workload: p95 약 90.75% 개선

**판정: 일반 해법 reject, event-driven 재조회가 실제로 반복되는 시스템에서만 조건부 채택.** TTL뿐 아니라 mask/topology epoch를 key에 포함해야 한다.

## 9. 후보 7 — Tensor buffer reuse

### 가설

매 decision의 tensor allocation/copy를 줄이면 \(T_{pre}\)와 allocator jitter가 감소한다.

### 결과

- local CPU: Exact 대비 p95 -7.90%, CI가 0을 가로지름
- A100 host CPU: -1.62%
- A100 CUDA: +0.83%, t-CI `[+0.21,+1.44]`

### 해석

CUDA에서는 작은 양의 효과가 일관되지만 절대 개선은 약 0.048 ms다. 복잡성과 thread-safety 비용, observation copy 정확성을 감안하면 독립 핵심 기법으로는 약하다.

**판정: hold.** production integration 전에 multi-thread correctness와 실제 onboard allocator를 재검증.

## 10. 후보 8 — `torch.compile`

### 가설

operator fusion과 CUDA graph가 작은 batch overhead를 줄일 수 있다.

### 결과

구 동일세션 candidate-1/2 검증에서 compiled Exact는 eager Exact보다 CPU와 CUDA 모두 빨라지지 않았다. dynamic shape, graph break, compile mode, 작은 반복 graph가 원인 후보다.

**판정: 현재 형태 reject.** static tensor-only graph와 충분한 amortization이 확보된 경우에만 재평가.

## 11. 문헌 기반 미구현 후보

### 11.1 Conservative action-space pruning

DFS-PPO, visual-risk Q-learning, KG-DDRL의 action masking에서 가져온 방향이다. learned policy 전에 물리적으로 불가능하거나 명백히 위험한 후보를 제거한다.

\[
\mathcal{A}'_t=\{a\in\mathcal{A}_t:M_t(a)=1,\;\hat{L}_{link}(a)>L_{min}\}
\]

정확성 보존을 위해 먼저 기존 Exact가 절대로 고르지 않는 후보만 제거하는 conservative shield로 시작해야 한다.

### 11.2 Confidence-routed cascade

Fast가 매우 확실한 상태만 처리하고 나머지는 Exact로 escalation한다.

\[
\pi=\begin{cases}
\pi_F,&\max_a p_F(a)>c,\;d_F<m,\;agreement\ predictor>q\\
\pi_E,&\text{otherwise}
\end{cases}
\]

calibration 목표는 confidence 자체가 아니라 **Exact와의 action/outcome disagreement risk**다. worst-scenario conditional coverage를 함께 제한해야 한다.

### 11.3 Sparse local encoder

KG-DDRL의 local K-neighbor encoder와 대규모 routing 문헌을 참고해 dense node tensor를 K-neighbor set으로 바꾼다.

\[
O(N^2d)\rightarrow O(NKd),\quad K\ll N
\]

그러나 현재 SwitchGLOBE observation semantics와 checkpoint 입력 차원을 바꾸므로 재학습과 full reliability 검증이 필요한 중기 과제다.

## 12. 다음 실험의 우선순위

1. Exact CPU path를 배치 기준선으로 고정한다.
2. `torch.cond`/TensorRT conditional execution 또는 branchless predication으로 Early Exit를 다시 구현한다.
3. conservative mask pruning을 Exact와 action-equivalence 기준으로 먼저 검증한다.
4. Fast-Exact disagreement predictor를 별도 calibration하고 selective coverage–risk curve를 만든다.
5. 10 seeds와 raw randomized-block timings으로 confirmatory A100/onboard benchmark를 수행한다.
