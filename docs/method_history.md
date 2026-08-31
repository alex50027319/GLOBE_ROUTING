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
| FastSwitchGLOBE | SwitchGLOBE Exact의 single-pass 증류(CPU 배포 지연시간 최적화) | 부제안(지연시간-성능 트레이드오프)으로 보존, 최종 방법 대체 아님 |

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

## FastSwitchGLOBE와 외부 baseline 비교 이후의 최종 방법 확정

5-seed 외부 baseline 비교(AODV, OLSR, Greedy Geographic, Evo-QGeo (Adapted),
RDQN-HERP (Adapted), GAT-GRU-DDQN)와 FastSwitchGLOBE ablation을 결합한 결과,
**SwitchGLOBE Exact(Phase 12)는 가장 강한 baseline인 Evo-QGeo (Adapted)를 포함해
6개 baseline 전부를 6개 핵심 지표(connected-pair PDR, overall PDR, deadline
delivery ratio, p95 success delay, energy per delivered packet, policy input
bytes)에서 통계적으로 유의하게 앞선다**(paired 5-seed 95% CI 기준, 모두 0을
포함하지 않음). 이 결과가 SwitchGLOBE를 최종 제안기법으로 유지하는 일차 근거다.

FastSwitchGLOBE(단일 순전파 증류판)는 Evo-QGeo (Adapted)와 6개 지표 전부에서
통계적으로 동률이며(CI가 0을 포함), Exact 대비 §11 acceptance gate(연결-pair
PDR·deadline ratio 저하가 -0.5pp 이내)를 통과하지 못한다. 저하는 대부분 밀도
OOD 시나리오(`ood_nodes_16`, `ood_nodes_24`)와 `predictive_break_225_link_loss`
두 축에 집중돼 있었고, 이를 좁히려 다음 개입을 시도했다.

| 시도 | 대상 문제 | 결과 |
| --- | --- | --- |
| hidden_dim 32→48 (용량 증가) | 밀도 OOD 저하 | 악화 (`ood_nodes_24` connected-pair PDR -5.3pp) — 좁은 학습 분포에 과적합한 것으로 추정 |
| KD 학습 시나리오에 고밀도(12/20-노드) 추가 | 밀도 OOD 저하 | `ood_nodes_16`(+6.7pp), `ood_nodes_24`(+9.1pp) 유의한 개선. 다만 predictive_break 계열에서 5-seed 중 2개(77, 123)가 새로 붕괴해 분산이 크게 늘었고, Exact 대비 macro acceptance gate 결론은 바뀌지 않았다(여전히 fail) |
| Phase 12 risk-switch 재보정(calibration에 stochastic link loss 시나리오 추가) | `predictive_break_225_link_loss`에서 Evo-QGeo에 뒤처짐 | 무효과 — switch_steps는 줄었지만 최종 PDR은 소수점까지 동일. 이 시나리오의 실패는 switch threshold가 아니라 하위 네트워크 자체의 행동임을 시사 |
| Phase 11 Predictive Student 재학습(학습 시나리오에 동일 stochastic link loss 노출) | 위와 동일 | 무효과 — 완전히 새로 학습된 checkpoint(dataset_samples 21,082)에서도 동일 시나리오의 실패 episode가 재보정 실험과 소수점까지 일치. SwitchGLOBE Exact 자체도 이 시나리오에서 Evo-QGeo에 유의하게 뒤처지므로(-4.65pp), 이는 증류·보정 품질이 아니라 방법론 자체의 한계로 판단된다 |

Evo-QGeo 계열 아이디어를 SwitchGLOBE에 접목하는 별도 실험(`refactor/EVO+GLOBE`
브랜치)도 유의미한 개선을 보이지 못했다(**needs-verification**: 세부 수치는 이
문서 작성 세션에서 직접 재현·검증하지 않았음).

결론: 여러 독립적 개입이 모두 명확한 개선을 만들지 못했으므로, **SwitchGLOBE
Exact(Phase 12)를 최종 제안기법으로 확정**한다. FastSwitchGLOBE는 대체재가
아니라 "지연시간-성능 트레이드오프" 부제안으로만 보고하며, Evo-QGeo 대비 우위나
Exact와의 동등성을 주장하지 않는다. 밀도-보강 학습 데이터로 개선한 FastSwitchGLOBE
결과를 최종본으로 쓸 경우, predictive_break 계열의 seed별 분산 증가를 실패
유형으로 함께 보고한다.
