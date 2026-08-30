# Current Algorithm: SwitchGLOBE

SwitchGLOBE는 Global-to-Local policy distillation과 predictive risk switching을 결합한
분산 FANET 라우팅 알고리즘이다.

- PPO Global Teacher는 학습 중 전체 그래프를 관측한다.
- Geo-Residual Student는 1-hop local observation과 geographic prior를 사용한다.
- Predictive Student는 margin, link lifetime, queue headroom, onward lifetime을 사용한다.
- 최종 SwitchGLOBE는 normal branch와 predictive branch를 위험 기준으로 선택한다.
- 배포 시 Global Teacher나 전체 그래프는 사용하지 않는다.
- 최종 단계는 PPO update가 아니라 PDR 제약 risk-switch calibration이다.
- 두 Student가 PPO Teacher로부터 증류되었으므로 전체 방법은 RL-derived다.

역사적 Phase 12가 최종 SwitchGLOBE 구현이다. Phase 13/P+의 redundancy, loss-keep,
energy tie, DROP suppression은 검증 이득이 충분하지 않아 최종 정의에서 제외한다.
