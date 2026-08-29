---
paths:
  - "ResearchAIWorkspace/implementations/lite_globe/experiments/**"
  - "ResearchAIWorkspace/implementations/lite_globe/config/**"
  - "ResearchAIWorkspace/implementations/lite_globe/run_phase*.py"
---

# Experiment Rules

- 실험 전 config, upstream checkpoint, output directory, seed를 출력하거나 기록한다.
- smoke와 full 결과를 명확히 구분한다.
- 기존 output을 덮어쓰지 않도록 `--resume` 또는 새 경로를 사용한다.
- scenario/method 수를 변경하면 기대 row 수와 reporting test를 갱신한다.
- source/destination 고정 여부는 topology stress 의도와 함께 검증한다.
- full run 완료는 process 종료뿐 아니라 manifest와 raw 파일 존재로 확인한다.
