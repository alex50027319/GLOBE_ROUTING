# Predictive Prior policy-distillation causal ablation

## 실험 설계

- 5 training seeds × 14 evaluation scenarios × 200 episodes × 7 methods = 98,000 raw episodes.
- 모든 신규 모델은 같은 초기값, residual=0, 동일 oracle/risk-oracle rollout 상태, Adam, batch 512, 120 epochs를 사용했다.
- 차이는 학습 label/objective뿐이다. Teacher가 no-teacher 모델의 rollout 경로를 결정하지 않았다.
- 통계 단위는 training seed이며 paired effect에 양측 95% Student-t CI를 적용했다.
- 양의 direction-aligned effect는 variant가 baseline보다 우수함을 뜻한다.

## 핵심 질문: teacher distillation이 필요한가?

- Full connected PDR: 0.911379 [0.910516, 0.912241]
- No-teacher connected PDR: 0.911456 [0.911456, 0.911456]
- Teacher 추가 paired PDR 효과: -0.000078 [-0.000940, +0.000785]
- Teacher 추가 deadline 효과: -0.000071 [-0.000865, +0.000722]
- CPU p95 latency: Full 0.5163 ms, No-teacher 0.5209 ms.

**Teacher의 독립적 신뢰성 향상 판정: not_supported.**

95% CI가 0을 포함하면 teacher가 쓸모없다는 증명이 아니라, 이 프로토콜에서 독립적 개선을 입증하지 못했다는 뜻이다. 반대로 CI 하한이 0보다 크면 동일구조·동일상태·동일예산 조건에서 teacher supervision의 추가 가치를 지지한다.

## 해석 원칙

- Fixed Manual은 학습 없는 hand-set coefficient 기준이다.
- Risk-only 및 Shortest-only는 supervision source의 단독 효과를 보여준다.
- Shortest+Risk (No Teacher) 대 Full이 distillation novelty의 1차 인과 비교다.
- Existing Phase11은 historical reference이며 상태분포와 학습 절차가 달라 1차 causal contrast가 아니다.
- energy는 simulator proxy이지 실제 Joule 측정값이 아니다.
- 최종 논문 주장은 paired CI와 worst-scenario 표를 함께 보고 결정해야 한다.
