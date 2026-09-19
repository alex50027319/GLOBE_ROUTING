# Metric definitions and interpretation boundaries

## Network-level outcomes

### Connected-pair packet delivery ratio

For episode-level delivery indicator `Y_e` and initial source–destination connectivity indicator `C_e`,

`PDR_C = sum_e(Y_e C_e) / sum_e(C_e)`.

This is the primary reliability measure because it removes episodes in which delivery is structurally impossible at initialization. It does not guarantee that a connected pair remains connected throughout the episode.

### Overall packet delivery ratio

`PDR_all = |E|^{-1} sum_e Y_e`.

This includes initially disconnected pairs and is retained as a secondary stress measure.

### Deadline delivery ratio

`DDR = |E|^{-1} sum_e D_e`, where `D_e=1` only when the packet arrives before the predefined deadline. This metric jointly penalizes non-delivery and late delivery.

### p95 successful end-to-end delay

The 0.95 quantile of simulator forwarding steps among successfully delivered packets. It is conditional on success. Cells with no successful delivery are N/A, not zero. It is not neural-network decision latency and does not isolate queueing, MAC access, retransmission, or hop count.

### Energy per delivered packet

A simulator proxy based on modeled transmission cost and a failure penalty. It is useful for consistent within-simulator comparison but is not a physical Joule measurement from an onboard radio.

## Policy/system outcomes

### Decision latency

Wall-clock time from prepared local policy observation to returned routing action for a batch-one inference call. CUDA measurements synchronize around the timed region. The primary statistic is the mean of five checkpoint-specific p95 values after 50 warm-up calls and 2,000 timed repetitions per checkpoint. It excludes radio transmission, queueing, route discovery, and simulator end-to-end forwarding.

### Cold-start latency

The first invocation before steady-state warm-up. It is stored separately and never mixed into the steady-state p95 statistic.

### Operator profile

Thirty post-warm-up decisions from seed 42 in the held-out-medium scenario. Event totals diagnose kernel-launch and scalar-synchronization overhead; they are not substituted for the 2,000-repeat confirmatory latency benchmark.

### Policy-input bytes

The in-memory tensor footprint passed to the routing policy. This is not wireless control-plane overhead and cannot be compared directly with protocol header bytes.

### Model footprint

Serialized parameter or checkpoint size under the stated representation. It is a storage measure, not peak runtime memory.

## Aggregation and uncertainty

Episode outcomes are aggregated within each seed–scenario cell and then macro-averaged over 14 scenarios. The inferential unit is the training seed (`n=5`). Paired method differences are formed seed by seed. Two-sided 95% Student-t intervals, deterministic bootstrap checks, and exact sign-flip checks are retained. With only five pairs, the minimum two-sided exact sign-flip p-value is 0.0625; therefore effect direction, magnitude, and interval are emphasized over a binary `p<0.05` label.

## Approximate-model acceptance gate

An approximate candidate is eligible only if the lower 95% confidence bounds of its seed-paired changes relative to Exact satisfy:

- connected-pair PDR: at least `-0.005`;
- deadline delivery ratio: at least `-0.005`;
- conditional delay and energy: no material directional degradation.

Fused Exact is exempt from this approximation gate because it preserves the exact policy semantics; its acceptance instead requires replay/action equivalence and branch-forward-count regression tests.
