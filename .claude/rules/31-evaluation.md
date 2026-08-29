---
paths:
  - "ResearchAIWorkspace/implementations/lite_globe/evaluation/**"
  - "ResearchAIWorkspace/artifacts/lite_globe/**"
---

# Evaluation Rules

- metric denominator와 성공 episode 필터를 확인한다.
- p95 delay가 성공 패킷 기준인지 명시한다.
- 평균뿐 아니라 seed 분산, paired effect, 실패 유형을 확인한다.
- 중복 row, 누락 seed, 누락 scenario-method 조합을 검사한다.
- SVG/PNG figure의 수치가 원본 summary와 일치하는지 확인한다.
- Phase 10 외부 baseline과 Phase 12/13 내부 ablation을 같은 campaign처럼 쓰지 않는다.
