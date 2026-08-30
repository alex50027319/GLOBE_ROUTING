---
paths:
  - "implementations/**/*.py"
  - "tests/**/*.py"
  - "scripts/**/*.py"
---

# Python Engineering

- Python 3.11+와 기존 typing/dataclass 패턴을 따른다.
- 난수는 기존 seed 및 RNG 흐름을 유지하고 전역 난수를 새로 섞지 않는다.
- CSV, JSON, YAML은 구조화 parser로 처리한다.
- 좁은 변경은 관련 test, 공용 env/evaluation 변경은 전체 suite로 검증한다.
- experiment 의미를 바꾸는 default에는 config·test·문서를 함께 갱신한다.
- 관련 없는 리팩터링과 artifact churn을 만들지 않는다.
