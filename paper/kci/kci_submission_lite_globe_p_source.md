# KCI 제출용 원고: Lite-GLOBE-P

이 파일은 `kci_submission_lite_globe_p.pdf` 생성을 위한 한국어 원고 소스입니다. 특정 학회/저널 템플릿이 제공되지 않았기 때문에 KCI 일반 투고 원고에 맞춘 보수적 1단 조판으로 작성했습니다.

## 핵심 주장

- 제안기법 Risk-Switch Lite-GLOBE-P는 offline global teacher, global-to-local distillation, online local student, predictive risk-switch를 결합한다.
- Phase 12 전체 14개 시나리오, 5개 seed, 84,000 episode 평가에서 PDR과 deadline delivery가 비교군 중 최고였다.
- Predictive Geographic, Evo-QGeo, DRAMA 대비 낮은 p95 지연과 낮은 입력 정보량을 보였다.
- GPSR 대비 신뢰성은 크게 개선되지만, 에너지 및 입력 정보량은 증가하므로 해당 부분은 한계로 명시한다.

## 수식 구성

1. Global teacher 상태: s_t = (G_t, p_t, v_t, q_t, d)
2. PPO 목적함수와 clipped surrogate loss
3. Teacher-student KD loss
4. Danger score, safety utility, risk-switch decision
5. 최종 next-hop 선택식
