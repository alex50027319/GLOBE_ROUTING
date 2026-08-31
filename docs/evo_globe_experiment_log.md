# EVO+GLOBE experiment log

This log records gated development results.  A candidate advances to a full
five-seed Colab run only after its frozen seed-42 smoke configuration changes
routing outcomes without violating the latency gate.

## Baseline integrity: corrected Evo-QGeo TD continuation

- Invalid forwarding actions use `-1e6` strictly as a selection sentinel.
- A next state with no legal forwarding neighbor now has continuation value
  zero instead of bootstrapping that sentinel into the Q table.
- Regression coverage confirms the learned update remains bounded.

A seed-42 smoke retraining used 50 episodes for each of the 11 training stages
(550 episodes, 1,568 updates).  All 1,088 learned Q entries were finite and
ranged from -0.2601 to 26.4092; the old invalid-sentinel checkpoints contained
values near -850,000.  The 280-episode evaluation smoke produced connected-pair
PDR 0.8008 and deadline ratio 0.7286.  These smoke metrics are not comparable to
the 3,850-training-episode full baseline and are recorded only as an integrity
check.

## Candidate 1: risk-conditioned bounded residual fusion

Implementation: `EvoFusionSwitchGlobePolicy`.

The existing SwitchGLOBE predictive prior is preserved.  A bounded Phase-11
residual is added with a weight that decreases as the selected normal action's
danger exceeds the calibrated switch threshold.  The explicit DROP logit is
never modified.  Fusion weight zero is exactly equivalent to SwitchGLOBE.

### Seed-42 calibration smoke

Calibration used only the existing generic curriculum, held-out calibration
rotations, and the separate predictive-link-loss calibration rotation.  Each
stage used 50 paired episodes.

| Maximum fusion weight | Aggregate PDR | Deadline ratio | Predictive-loss PDR |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.8622 | 0.8489 | 0.440 |
| 0.25 | 0.8644 | 0.8511 | 0.440 |
| 0.50 | 0.8511 | 0.8378 | 0.320 |

The frozen smoke choice was maximum weight 0.25 and danger confidence scale
0.20.

### Seed-42 held-out smoke

The held-out suite used all 14 evaluation scenarios with 50 paired episodes
per scenario.

| Method | Connected-pair PDR | Deadline ratio | Energy/delivered | CPU P95 decision latency |
| --- | ---: | ---: | ---: | ---: |
| SwitchGLOBE Exact | 0.91938 | 0.84714 | 2.02374 | 1.1239 ms |
| EvoFusion 0.25 | 0.91938 | 0.84714 | 2.02548 | 1.6645 ms |

All 14 scenario-level PDR and deadline differences were zero.  CPU P95
latency increased by 48.1% because the disabled predictive residual network
must execute as an additional forward pass.

### Gate decision

Candidate 1 does **not** advance to a five-seed full run in its current form.
It changed no held-out routing outcomes and exceeded the planned 10% latency
increase limit.  The result supports moving the learned long-horizon signal to
training-time auxiliary distillation rather than executing another residual
network at deployment.

## Candidate 2: deployment-neutral cost-to-go distillation

Implementation: return-guided fine-tuning of the exact `SwitchGlobePolicy`.

Reference, oracle, and risk-oracle rollouts provide discounted multi-step
return and rollout-action targets.  Training combines the existing reference
KL objective with a return-weighted rollout-action cross-entropy term.  The
deployed policy receives no global graph, auxiliary head, parameter, state key,
or additional inference branch.  The resulting model therefore retains the
exact 37,776-parameter SwitchGLOBE architecture.

### Seed-42 calibration smoke

Four calibration distributions used 50 paired evaluation episodes each.  The
dataset used 30 rollout episodes per training scenario and the coefficient was
the only swept return-loss setting.

| Return-action coefficient | Aggregate PDR | Deadline ratio | Energy/delivered |
| ---: | ---: | ---: | ---: |
| 0.00 (Exact) | 0.90499 | 0.84667 | 2.28909 |
| 0.02 | 0.90499 | 0.84667 | 2.28922 |
| 0.05 | 0.90499 | 0.84667 | 2.28922 |
| 0.10 | 0.90499 | 0.84667 | 2.28922 |
| 0.20 | 0.90736 | 0.84889 | 2.28324 |

Coefficient 0.20 was frozen before held-out evaluation.  It completed six
epochs with validation KL `1.8931e-05`, reference action agreement 1.0, and
rollout-action agreement 0.95625.

### Seed-42 held-out smoke

The independent suite used all 14 evaluation scenarios with 50 paired
episodes per scenario (700 episodes per method).

| Method | Connected-pair PDR | Deadline ratio | Energy/delivered | Median CPU P95 latency |
| --- | ---: | ---: | ---: | ---: |
| SwitchGLOBE Exact | 0.91938 | 0.84714 | 2.023742 | 0.5872 ms |
| Cost-to-go 0.20 | 0.91938 | 0.84714 | 2.023868 | 0.5855 ms |

All 14 scenario-level PDR and deadline differences were zero.  Only the
extreme-mobility scenario changed energy, increasing it by 0.001559 per
delivered packet.  The candidate/reference median-P95 latency ratio was 0.997,
which is measurement noise consistent with the identical inference graph.

Reproducible outputs are written under
`artifacts/evo_globe/candidate2_gate/seed_42` by
`python -m implementations.lite_globe.run_evo_globe_candidate2`.

### Gate decision

Candidate 2 does **not** advance to a five-seed full run in its current form.
Its calibration gain did not reproduce on held-out scenarios: routing outcomes
were identical and energy was marginally worse.  This rejects cost-to-go
fine-tuning as the isolated primary improvement while retaining the result as
evidence that a training-only objective can preserve deployment cost.
