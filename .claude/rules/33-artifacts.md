---
paths:
  - "artifacts/**"
---

# Generated Artifacts

- artifact 파일을 수동으로 고쳐 결과를 맞추지 않는다.
- manifest의 mode, complete, config, seed set, checkpoint paths를 먼저 확인한다.
- 오래된 로그 마지막 행만으로 현재 process/session liveness를 판단하지 않는다.
- partial seed 결과는 완료된 seed만 명시하고 full이라고 부르지 않는다.
- rerun은 별도 output path 또는 검증된 `--resume`을 사용한다.
