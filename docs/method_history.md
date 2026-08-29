# SwitchGLOBE 개발 계보

이 문서는 최종 알고리즘을 만들기까지 Phase 1~12에서 어떤 문제를 해결했고 무엇이
SwitchGLOBE에 남았는지를 기록한다. Phase 번호는 논문의 방법명이 아니라 재현 이력이다.

| 연구 단계 | 검증 내용 | SwitchGLOBE에서의 상태 |
| --- | --- | --- |
| 1 | FANET 환경, Random/GPSR, action mask | 공통 환경과 baseline으로 채택 |
| 2 | 1-hop shared-MLP Student | Student backbone으로 채택 |
| 3 | global graph Teacher와 PPO | offline Teacher 학습으로 채택 |
| 4 | masked forward-KL distillation | KD 핵심 손실로 채택 |
| 5 | local PPO fine-tuning | 알고리즘 핵심에서는 제외, 공통 구현만 보존 |
| 6 | multi-seed 통계와 비용 측정 | 평가 도구로 계승 |
| 7 | topology curriculum과 held-out 평가 | Teacher/KD foundation training으로 채택 |
| 8 | geographic prior + learned residual | normal branch로 채택 |
| 9 | risk feature와 predictive-break 시나리오 | Phase 11/최종 보정 시나리오로 계승 |
| Baseline suite | 외부 RL routing baseline 재구현 | 최종 학습과 분리된 논문 비교 평가로 보존 |
| 11 | predictive prior Student | predictive branch로 채택 |
| 12 | PDR 제약 Risk-Switch calibration | **SwitchGLOBE 최종 알고리즘** |
| 13 | redundancy/loss/energy/drop P+ 확장 | 검증 이득이 불충분하여 최종안에서 제외 |

## 실제 checkpoint 의존성

```text
PPO Global Teacher (Phase 7)
        ├───────────────┐
        ↓               ↓
Geo-Residual KD      Predictive KD
  (Phase 8)           (Phase 11, Phase 8에서 초기화)
        └───────┬───────┘
                ↓
       SwitchGLOBE calibration
          (historical Phase 12)
```

Phase 12 실행 자체는 PPO update나 KD weight update를 수행하지 않는다. 하지만 두
입력 Student가 PPO Teacher의 지식을 증류받았으므로 SwitchGLOBE는 RL-derived policy다.

Phase 13 seed 42의 동일-run 비교에서는 Phase 12 대비 PDR 차이가 사실상 한 episode
수준이었고 p95 delay는 동일했으며 추가 입력 특징 비용이 증가했다. 다섯 seed가 모두
검증되지 않은 결과를 최종 방법으로 승격하지 않고, 이미 5-seed 검증된 Phase 12를
SwitchGLOBE로 고정한다. Phase 13 결과는 최종 원고의 수치와 섞지 않는다.
