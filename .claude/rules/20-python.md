---
paths:
  - "ResearchAIWorkspace/**/*.py"
  - "ResearchAIWorkspace/tests/**/*.py"
---

# Python Rules

- Python 3.11+와 기존 type hint 스타일을 따른다.
- 작은 변경에는 관련 test file을, 공용 환경·평가 모듈 변경에는 전체 suite를 실행한다.
- 난수 생성은 명시적 seed 또는 기존 RNG 객체를 사용한다.
- CSV/JSON/YAML은 표준 parser를 사용하고 문자열 분해로 구조를 해석하지 않는다.
- 실험 의미를 바꾸는 default 변경에는 테스트와 문서 근거를 함께 갱신한다.
- 관련 없는 리팩터링과 formatting churn을 피한다.
