---
paths:
  - "implementations/lite_globe/experiments/**"
  - "implementations/lite_globe/config/**"
  - "implementations/lite_globe/run_*.py"
  - "scripts/train_switchglobe_pipeline.py"
---

# SwitchGLOBE Experiments

- 학습 계보는 teacher → geo_residual → predictive → switchglobe 순서다.
- `--start-at` 사용 전 필요한 upstream checkpoint를 확인한다.
- smoke와 full output directory를 섞지 않는다.
- 기존 결과가 있으면 config와 completion marker를 확인한 뒤 `--resume`을 사용한다.
- full run 완료는 5개 seed checkpoint와 manifest로 검증한다.
- historical Phase 12 파일명은 구현 호환용이며 출력 명칭은 SwitchGLOBE를 사용한다.
