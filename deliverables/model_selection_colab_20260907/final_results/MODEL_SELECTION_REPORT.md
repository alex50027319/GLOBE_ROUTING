# SwitchGLOBE confirmatory ablation and model selection

- Protocol: 5 training seeds × 14 scenarios × 200 episodes × 10 methods = 140,000 episode rows.
- All outcome contrasts use the training seed as the inferential unit and two-sided 95% Student-t intervals.
- Positive paired effects mean that the named variant is better than SwitchGLOBE Exact.
- Energy is a simulator transmission-energy proxy, not measured Joules.
- The Phase-7 KD comparison is a historical matched-architecture audit, not a causal retraining ablation of the final architecture.

## Primary predictive-only vs risk-switch result

- Connected PDR effect: -0.000000 (95% CI -0.000902, +0.000902).
- Deadline-delivery effect: -0.000071 (95% CI -0.001033, +0.000890).
- p95-success-delay direction-aligned effect: +0.000000 steps (95% CI +0.000000, +0.000000); positive favors predictive-only.
- CPU p95 latency: predictive-only 0.5664 ms vs SwitchGLOBE 2.3474 ms.

## Predeclared selection rule

Reliability-first rule: predictive-only is selected only when its lower 95% CI is no worse than -0.5 percentage points for both connected PDR and deadline delivery, and it reduces CPU p95 latency; otherwise SwitchGLOBE Exact is retained.

**Selected model under this rule: Predictive Prior Only.**

The selection is an engineering conclusion under the stated margin, not proof of statistical equivalence. Review the worst-scenario table before publication.
