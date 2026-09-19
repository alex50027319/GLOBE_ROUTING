---
paths:
  - "implementations/lite_globe/baselines/**"
  - "implementations/lite_globe/experiments/*external*"
  - "implementations/lite_globe/evaluation/*external*"
  - "implementations/lite_globe/run_external_comparison.py"
  - "README_BASELINES_COLAB.md"
---

# External Baselines

- 실제 method 집합은 registry를 읽어 확인한다.
- SwitchGLOBE checkpoint는 read-only proposed method로 사용한다.
- 실행 시 legacy default 대신 현재 checkpoint directory를 명시한다.
- 모든 method에 동일 evaluation seeds와 reset options를 적용한다.
- adapted method에는 원 논문에서 유지·변경·생략한 요소를 기록한다.
- external campaign 결과를 SwitchGLOBE calibration ablation과 한 표본처럼 병합하지 않는다.
