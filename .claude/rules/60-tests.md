---
paths:
  - "tests/**"
---

# Tests

- test는 현재 public contract와 재현성 invariant를 검증한다.
- smoke test에 full checkpoint나 네트워크가 필요하지 않게 한다.
- environment 변경에는 deterministic reset, action mask, outcome accounting을 확인한다.
- evaluation 변경에는 denominator, expected rows, duplicate detection을 확인한다.
- baseline 변경에는 registry, method contract, paired seed behavior를 확인한다.
