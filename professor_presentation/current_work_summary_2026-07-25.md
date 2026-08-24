# Current Work Summary

작성일: 2026-07-25

## 1. 프로젝트 개요

본 프로젝트는 UAV Flying Ad Hoc Network(FANET)에서 동적인 링크 단절, routing hole, queue 지연 문제를 줄이기 위한 분산형 라우팅 기법을 연구한다. 핵심 제안기법은 **GLOBE++ / Risk-Switch Lite-GLOBE-P**이며, 전역 정보를 활용한 offline teacher 학습과 local student 실행, 그리고 위험 상황에서만 활성화되는 predictive risk switch를 결합한다.

## 2. 주요 제안기법

- **Global Teacher**: 학습 단계에서 전체 topology, 위치, 속도, queue, destination 정보를 사용해 PPO 기반 actor-critic routing policy를 학습한다.
- **Local Student**: 실제 UAV 실행 단계에서 1-hop local observation만 사용한다.
- **Policy Distillation**: teacher의 soft action distribution을 forward-KL 기반 policy distillation으로 student에 전달한다.
- **Predictive Risk Switch**: normal student가 선택한 next-hop이 위험하면 predictive branch로 전환한다. 위험 판단에는 link margin, predicted link lifetime, queue headroom, onward stability 등이 사용된다.
- **Phase 13 P+ 확장**: link-loss-aware danger, top-k onward stability, energy-aware tie-breaking, drop suppression을 추가한 확장 버전이다.

## 3. 구현된 내용

- Lite-GLOBE FANET routing simulator 구현
- Random/GPSR 및 외부 RL 성격 baseline 구현
- Global teacher PPO 학습 코드 구현
- Local student 및 Geo-Residual KD 계열 구현
- Risk-Switch Lite-GLOBE-P 구현
- Phase 13 Lite-GLOBE-P+ 코드 및 smoke 검증 산출물 생성
- 평가/통계/figure/report 생성 파이프라인 구축

## 4. 실험 및 검증 현황

- **Phase 12**: 14개 scenario, 5개 random seed 기반 full multi-seed 평가 결과가 존재한다.
- Phase 12 결과는 PDR, deadline delivery, p95 delay, input-byte footprint 비교에 사용할 수 있는 현재의 가장 안정적인 근거다.
- GPSR, Predictive Geographic, Evo-QGeo, IQMR Q(lambda), DRAMA 등과 비교 가능한 결과 테이블과 그림이 정리되어 있다.
- **Phase 13 P+**: 구현과 smoke test는 확인되었지만, full multi-seed 결과와 ablation은 아직 최종 확정 전이다.

## 5. 작성된 산출물

- 교수님 보고용 요약/감사 문서: `professor_presentation/`
- IEEE 스타일 논문 draft: `paper/`
- KCI 제출 스타일 draft: `kci_paper/`
- 실험 결과 및 figure: `ResearchAIWorkspace/artifacts/lite_globe/`
- 연구 wiki 및 문헌/실험 노트: `ResearchAIWorkspace/vault/`

## 6. 현재 주장 가능한 핵심 메시지

현재 가장 방어 가능한 주장은 다음과 같다.

> 제안기법은 전역 teacher가 학습한 routing 지식을 local student에 증류하여, 실제 실행 시 online global state나 multi-hop message passing 없이도 안정적인 next-hop routing을 수행한다. Phase 12 full evaluation 기준으로 routing reliability와 deadline delivery에서 강한 성능 근거가 있으며, predictive risk switch는 routing hole과 link break 상황에서 local policy의 취약점을 보완한다.

## 7. 주의해야 할 부분

- Phase 13 P+의 full multi-seed superiority는 아직 최종 주장하면 안 된다.
- 현재 main student는 MLP 기반 candidate scorer이며, local GNN student를 핵심 구현으로 주장하면 안 된다.
- 현재 energy metric은 실제 Joule 측정이 아니라 simulator-level proxy다.
- 실제 UAV 적용성을 강하게 주장하려면 ns-3 또는 MAC/PHY 계층 검증이 추가로 필요하다.
- DRAMA 대비 communication overhead 주장은 input-byte proxy 중심으로 제한해서 표현해야 한다.

## 8. 다음 작업

1. Phase 13 P+ full multi-seed evaluation 실행
2. Phase 13 component ablation 정리
3. explicit control-byte overhead metric 추가
4. BibTeX 및 참고문헌 metadata 보완
5. ns-3 또는 packet-level validation 계획 구체화
