# External Baseline Contract

최종 external comparison의 proposed method는 read-only SwitchGLOBE checkpoint다.

주요 비교군:

- AODV
- OLSR
- Greedy Geographic
- Evo-QGeo (Adapted)
- RDQN-HERP (Adapted)
- GAT-GRU-DDQN
- SwitchGLOBE

IQMR과 DRAMA-inspired Graph-DQN은 optional legacy이며 최종 비교군에 포함되었다고
자동 가정하지 않는다. 실제 목록은 `baselines/registry.py`의 `COMPARISON_METHODS`로
확인한다.

- baseline 학습은 SwitchGLOBE weight나 calibration을 변경하지 않는다.
- adapted baseline은 원 논문과 현재 simulator의 차이를 명시한다.
- control overhead는 구현된 protocol accounting과 실제 PHY/MAC overhead를 구분한다.
- `run_external_comparison.py`에는 legacy checkpoint default가 남아 있으므로 실행할 때
  `--switchglobe-checkpoint-dir artifacts/switchglobe/final/checkpoints`를 명시한다.
