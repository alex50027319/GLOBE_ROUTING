---
paths:
  - "implementations/lite_globe/evaluation/**"
  - "artifacts/**"
  - "docs/simulation_protocol.md"
---

# Evaluation

- metric denominator와 disconnected endpoint 처리 방식을 확인한다.
- expected row count를 seeds × scenarios × methods × episodes로 계산한다.
- 누락 seed, 중복 row, 누락 scenario-method cell을 검사한다.
- primary comparison은 seed-paired estimate와 95% interval을 사용한다.
- PDR와 deadline, p95 delay, energy, latency, input bytes, drop reason을 함께 본다.
- CSV, table, figure, manuscript 숫자의 일치를 검증한다.
- reward는 디버깅 지표이며 QoS 우월성 근거로 사용하지 않는다.
