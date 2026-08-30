# SwitchGLOBE Fact Cases

## Final algorithm

- Question: 현재 최종 제안기법은 Phase 13/P+인가?
- Expected: 아니다. 역사적 Phase 12 Risk-Switch가 SwitchGLOBE 최종안이고 Phase 13/P+는 제외됐다.

## RL training

- Question: SwitchGLOBE 최종 단계도 PPO로 학습하는가?
- Expected: 최종 단계는 calibration이다. 다만 두 Student가 PPO Teacher로부터 증류돼 RL-derived다.

## Deployment information

- Question: 실제 UAV 배포 시 Global Teacher와 전체 그래프가 필요한가?
- Expected: 아니다. 배포 정책은 1-hop local observation만 사용한다.

## Metrics

- Question: policy input bytes가 routing overhead인가?
- Expected: 아니다. tensor input cost이며 control-plane overhead는 별도 계측이 필요하다.

- Question: energy 결과를 Joule로 써도 되는가?
- Expected: 아니다. 현재 값은 simulator-level transmission energy proxy다.

## Results

- Question: seed 42 결과만 있으면 full 결과인가?
- Expected: 아니다. full claim은 5개 training seed와 complete manifest를 요구한다.

## Baselines

- Question: IQMR과 DRAMA가 최종 external comparison의 필수 비교군인가?
- Expected: optional legacy다. 실제 목록은 baseline registry에서 확인한다.
