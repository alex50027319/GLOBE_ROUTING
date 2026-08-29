---
paths:
  - "ResearchAIWorkspace/artifacts/**"
---

# Artifact Rules

- artifact는 생성 결과이며 수동 편집하지 않는다.
- seed ZIP 병합 전 각 입력의 `raw/episodes.csv`, `seed_summaries.csv`,
  `training_metrics.csv` 존재를 확인한다.
- 병합 후 manifest의 training seed 집합과 요청 seed 집합을 비교한다.
- 오래된 로그의 마지막 행을 현재 프로세스 상태로 해석하지 않는다.
- 결과를 재생성할 때 기존 디렉터리 대신 명시적 새 output directory를 선호한다.
