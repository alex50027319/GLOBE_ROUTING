# Project Audit for Professor Presentation

작성일: 2026-06-29

## 1. Current Directory

현재 작업 디렉토리는 `/Users/alex/Documents/GLOBE_ROUTING`이다. 실제 연구 구현과 산출물의 중심은 `ResearchAIWorkspace/` 아래에 있다.

## 2. Found Research Notes

확인된 주요 연구 노트는 다음과 같다.

| Path | 내용 |
| --- | --- |
| `ResearchAIWorkspace/vault/06_Experiments/Lite-GLOBE/` | Phase 1부터 Phase 13까지의 구현 기록, 결과 분석, 비교 기법 설명 |
| `ResearchAIWorkspace/vault/01_Sources/Papers/` | Evo-QGeo, DRAMA, GLo-MAPPO, IQMR 관련 논문 요약 |
| `Lite-GLOBE Implementation Overview.md` | 초기 구현 개요 |
| `GLOBE++ Novelty Claims.md` | novelty 주장 정리 |
| `Dec-POMDP.md` | 문제 정식화 관련 노트 |

## 3. Found Source Code

핵심 구현은 `ResearchAIWorkspace/implementations/lite_globe/`에 있다.

| Module | 역할 |
| --- | --- |
| `env/` | FANET routing 환경, mobility, link model, observation, reward |
| `models/teacher_gnn.py` | privileged global teacher actor-critic |
| `algorithms/ppo.py` | clipped PPO 업데이트 |
| `algorithms/distillation.py` | masked forward-KL policy distillation |
| `models/student_policy.py` | Local Student, Geo-Residual, Lite-GLOBE-P, Risk-Switch P/P+ |
| `baselines/` | GPSR, Predictive Geographic, Evo-QGeo, IQMR, DRAMA 등 baseline |
| `experiments/phase6_campaign.py` ~ `phase13_campaign.py` | phase별 실험 campaign |
| `evaluation/` | metric 집계, 통계, report/figure 생성 |

## 4. Found Simulation Environment

시뮬레이터는 `FanetRoutingEnv` 기반의 hop-by-hop packet routing 환경이다. 기본 환경 변수는 `FanetConfig`에 정의되어 있다.

| 항목 | 기본값 |
| --- | --- |
| UAV 수 | 12 |
| 최대 노드 수 | 32 |
| 시뮬레이션 영역 | 1000 x 1000 |
| 통신 반경 | 350 |
| 이동성 | 2D Random Waypoint |
| 속도 범위 | 2.0 ~ 12.0 |
| packet TTL | 20 |
| 최대 episode step | 64 |

## 5. Found Model Implementations

| 구성 | 현재 구현 상태 | 근거 |
| --- | --- | --- |
| Global teacher | Done | `models/teacher_gnn.py`, `algorithms/teacher_trainer.py` |
| PPO actor-critic | Done | `algorithms/ppo.py` |
| Local student | Done | `models/student_policy.py` |
| Policy distillation | Done | `algorithms/distillation.py` |
| Risk-Switch Lite-GLOBE-P | Done | `RiskSwitchLiteGlobePStudentPolicy` |
| Risk-Switch Lite-GLOBE-P+ | Partial/In Progress | `RiskSwitchLiteGlobePPlusStudentPolicy`, Phase 13 smoke/full run pending |
| Local GNN student | Not current main implementation | 현재 student는 후보별 MLP encoder와 pooling 기반이다. Local GNN은 향후 확장 항목이다. |
| MAPPO | Not current main implementation | 현재 teacher 학습은 PPO actor-critic이며 MAPPO 전체 multi-agent framework는 아니다. |

## 6. Found Experimental Results

| Phase | 결과 상태 | 주요 산출물 |
| --- | --- | --- |
| Phase 6 | multi-seed 평가 | `artifacts/lite_globe/phase6/` |
| Phase 7 | 일반화/validity 강화 | `artifacts/lite_globe/phase7/` |
| Phase 8 | Geo-Residual KD 최적화 | `artifacts/lite_globe/phase8/` |
| Phase 9 | risk-aware 및 논문용 figure catalog | `artifacts/lite_globe/phase9/` |
| Phase 10 | Evo-QGeo, IQMR, DRAMA 등 external RL baseline | `artifacts/lite_globe/phase10/` |
| Phase 11 | Lite-GLOBE-P | `artifacts/lite_globe/phase11/` |
| Phase 12 | Risk-Switch Lite-GLOBE-P full 결과 | `artifacts/lite_globe/phase12/` |
| Phase 13 | Risk-Switch Lite-GLOBE-P+ smoke 및 Colab bundle | `artifacts/lite_globe/phase13_smoke/`, `README_PHASE13_COLAB.md` |

## 7. Found Figures

발표에 우선 사용할 그림은 `professor_presentation/figures/`로 복사했다.

| Figure | 의미 |
| --- | --- |
| `lite_globe_full_method_infographic.png` | 전체 방법 개념도 |
| `phase12_risk_switch_pdr.png` | Phase 12 PDR 결과 |
| `phase12_risk_switch_delay_p95.png` | Phase 12 delay p95 결과 |
| `phase12_risk_switch_input_bytes.png` | Phase 12 입력 byte proxy 결과 |
| `phase10_external_rl_pdr.png` | external RL baseline과 PDR 비교 |
| `phase9_scalability_uav_count.png` | UAV 수 증가 scalability 분석 |
| `phase9_component_ablation.png` | 구성요소 ablation |
| `phase13_smoke_switch_steps.png` | Phase 13 smoke switch 동작 진단 |

## 8. Found Draft Papers

| Path | 상태 |
| --- | --- |
| `paper/main.pdf` | main draft |
| `paper/main_final.pdf` | final main draft candidate |
| `paper/main_with_teacher_rl.pdf` | teacher RL 설명이 추가된 draft |
| `paper/kci/kci_submission_lite_globe_p.pdf` | KCI 스타일 제출용 draft |
| `paper/figures/method_overview_pub.pdf` | 논문용 방법 개요 그림 |

## 9. Currently Verified Claims

| Claim | Verified? | Evidence |
| --- | ---: | --- |
| FANET simulator exists | Yes | `env/fanet_env.py`, tests |
| Global teacher PPO implementation exists | Yes | `algorithms/ppo.py`, `teacher_trainer.py` |
| Policy distillation is implemented | Yes | `algorithms/distillation.py` |
| Risk-Switch Lite-GLOBE-P is implemented | Yes | `student_policy.py`, Phase 12 artifacts |
| Phase 12 full multi-seed results exist | Yes | `phase12/raw`, `phase12/summaries`, `phase12/tables` |
| External baselines exist in the common simulator | Yes | `baselines/external_rl.py`, Phase 10 artifacts |
| Phase 13 P+ is implemented | Partially | code and smoke artifacts exist |
| Phase 13 full multi-seed result is final | No | smoke and Colab bundle found; full merged result not confirmed in audit |

## 10. Claims Not Yet Verified

| Claim | Reason for caution |
| --- | --- |
| SCI급으로 충분한 성능 우월성 | Phase 13 full multi-seed, stronger ablations, packet-level realism 검증 필요 |
| 실제 UAV 적용 가능성 | 실제 MAC/PHY, beacon overhead, GPS/communication delay 모델이 단순화되어 있음 |
| Local GNN student novelty | 현재 main student는 MLP 기반 후보 스코어러이므로 GNN student claim은 향후 확장으로 둬야 함 |
| MAPPO 기반 제안기법 | 현재 구현은 PPO teacher + distillation이며 MAPPO 전체 구현은 아님 |
| 실제 energy 절감 | 현재 값은 simulator-level proxy이며 실제 Joule 모델이 아님 |

## 11. Missing Materials for Professor Presentation

| Missing Material | Priority | Comment |
| --- | ---: | --- |
| Phase 13 full multi-seed merged result | High | P+ 최종 주장 전 필요 |
| policy-KD vs latent-KD ablation | High | novelty 직접 검증 |
| MLP student vs local GNN student comparison | High | GNN claim을 쓰려면 필요 |
| metric definition/caption 정리 | Medium | 논문 표/그림 해석 안정화 |
| ns-3 또는 packet-level validation 계획 | Medium | SCI급 확장 논의용 |

