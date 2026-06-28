# Risk-Switch Lite-GLOBE-P+ 교수님 보고 발표자료

- PPTX: /Users/alex/Documents/GLOBE_ROUTING/ResearchAIWorkspace/outputs/presentations/globe_lite_globe_professor_report_2026-06-28.pptx
- Render preview: /Users/alex/Documents/GLOBE_ROUTING/ResearchAIWorkspace/outputs/presentations/globe_professor_report_rendered/deck-montage.webp
- 생성일: 2026-06-28

## 발표 핵심 메시지

이 연구는 UAV/FANET 라우팅에서 전역 그래프를 계속 공유하는 방식이 아니라, 학습 단계에서만 전역 교사를 사용하고 실제 실행 단계에서는 각 UAV가 1-hop 지역 정보만으로 next-hop을 선택하도록 만드는 경량 위험회피 라우팅 기법이다.

## 슬라이드별 발표 노트

### 1. Title

첫 장에서는 연구를 한 문장으로 정리합니다. 이 연구는 UAV 군집망에서 전역 정보가 없어도 지역 노드가 안정적으로 다음 홉을 선택하도록 만드는 경량 라우팅 알고리즘입니다.

### 2. Executive Summary

교수님께는 결론부터 제시하는 것이 좋습니다. Phase12 기준으로 제안기법은 GPSR보다 확실히 개선되었고, 외부 baseline과 비교해도 평균 PDR과 deadline에서 경쟁력이 있습니다. 다만 Phase13은 아직 전체 검증 전이라는 점을 투명하게 말합니다.

### 3. Problem

이 슬라이드는 문제 정의입니다. UAV는 움직임 때문에 네트워크 토폴로지가 계속 바뀌고, 각 노드는 전체 네트워크가 아니라 주변 이웃만 압니다. 그래서 가장 가까운 노드로만 보내는 정책은 보기에는 단순하지만 routing-hole과 링크 단절에 취약합니다.

### 4. Prior Art

기존 연구와 겹치지 않는 언어가 중요합니다. GPSR은 baseline, Evo-QGeo는 예측 링크 baseline, DRAMA는 실행 중 통신 baseline으로 두고, 우리 차별점은 학습 시 전역 지식과 실행 시 지역 정책의 분리라고 설명합니다.

### 5. Novelty

이 장은 novelty 문장입니다. 기존 MARL이나 CTDE와 겹치지 않게, 전역 정보는 학습에만 쓰고 실제 라우팅은 지역 feature만 쓴다는 점을 분명히 해야 합니다.

### 6. System Model

환경 설명은 짧고 명확하게 갑니다. 이 연구는 통신 반경 기반 UAV 네트워크에서 next-hop을 고르는 문제이며, 성능은 전달률뿐 아니라 deadline, delay, energy, 입력 바이트까지 같이 봅니다.

### 7. Method Evolution

진화 과정을 한눈에 보여줍니다. Phase8은 구조적으로 강했고, Phase9는 위험 feature의 필요성을 보여줬지만 불안정했습니다. Phase11/12에서 risk-switch 구조로 정리했고, Phase13은 그 약점을 더 보완하는 단계입니다.

### 8. Core Algorithm

이 장에서는 교사-학생 구조를 설명합니다. 교사는 전체 그래프를 보고 좋은 next-hop 분포를 만들고, 학생은 그 판단을 학습하지만 실제 deployment에서는 주변 이웃 정보만 사용합니다.

### 9. Formal Problem

문제 정의 슬라이드입니다. 그래프 G_t는 시간에 따라 링크가 바뀌고, 현재 패킷을 가진 노드 u_t는 1-hop 이웃 또는 DROP 중 하나를 선택합니다. 핵심 제약은 실행 시 전체 그래프를 보지 않는다는 점입니다.

### 10. Learning Objective

이 슬라이드는 학습 목적함수입니다. KL divergence로 교사의 후보별 선호 분포를 학생에게 전달합니다. 이것은 단순 imitation보다 풍부합니다. 후보 A가 가장 좋고 B가 두 번째라는 상대적 정보를 함께 배웁니다.

### 11. Candidate Score

후보 점수식 설명입니다. S_PG는 GPSR의 장점인 목적지 방향성을 살리면서, forwarding 가능성, 링크 margin, lifetime, queue, onward lifetime을 합친 prior입니다. 그 위에 학습 residual을 더하되 tanh로 영향력을 제한하고, 위험 gate penalty로 곧 끊길 링크를 강하게 낮춥니다.

### 12. Risk Switch Details

Risk-Switch의 상세 슬라이드입니다. 핵심은 predictive branch가 다르다고 무조건 바꾸는 것이 아니라, normal 후보가 위험하거나 predictive 후보가 충분히 더 안전할 때만 바꾸는 것입니다.

### 13. Algorithm Flow

이 슬라이드는 pseudo-code보다 더 발표 친화적인 알고리즘 흐름입니다. 후보 생성, feature 계산, normal 후보, predictive 후보, switch 결정의 5단계로 설명하면 교수님이 전체 구조를 빠르게 이해할 수 있습니다.

### 14. Executable Algorithm

실제 실행 알고리즘입니다. 각 hop마다 후보 이웃의 진행도, 링크 안정성, 큐, onward 안정성, 에너지 효율을 계산하고 normal branch와 predictive branch 중 안전한 선택을 고릅니다. 복잡도는 후보 수에 선형이며 실행 시 전역 그래프가 필요 없습니다.

### 15. Variable Dictionary

이 장은 질문 대비용입니다. 수식에 등장하는 변수의 의미를 한 번에 보여줍니다. 특히 각 값이 높을수록 어떤 의미인지 직관적으로 설명하면 교수님이 알고리즘을 빠르게 이해할 수 있습니다.

### 16. Mathematical Features

수식은 어렵게 보이지 않게 변수 의미를 같이 풀어줍니다. m은 링크 여유, l은 링크 수명, o는 다음 홉 이후의 길, rho는 후보 경로의 중복성입니다.

### 17. Switch Rule

Risk-switch는 핵심입니다. 현재 후보가 DROP이거나 위험 점수가 높거나, 예측 후보가 충분히 더 안전할 때만 바꿉니다. 이렇게 해야 무조건 복잡한 정책보다 안정적입니다.

### 18. P+ Algorithm

Phase13은 교수님께 '최종 후보'라고 말해야 합니다. 아직 full run 전이지만, 설계 방향은 명확합니다. 예측 단절을 피하면서도 불필요한 switch와 energy 낭비를 줄이는 것입니다.

### 19. Phase12 Results Chart

수치 설명은 간결하게 합니다. Phase12에서 Risk-Switch는 GPSR보다 확실히 좋고, Predictive Geographic/Evo-QGeo/DRAMA와 비교해도 평균 PDR이 근소하게 더 높습니다.

### 20. Metric Table

표에서는 강점과 약점을 같이 말합니다. Risk-Switch는 PDR과 deadline이 좋고 DRAMA보다 제어 바이트가 낮지만, energy는 DRAMA보다 약간 좋지 않으므로 P+에서 개선 중이라고 연결합니다.

### 21. Scenario Findings

교수님이 물을 가능성이 높은 부분입니다. 전체 평균만 좋다고 하지 말고, routing-hole에서는 강하고 predictive-break 일부 조건에서는 아직 Evo-QGeo가 강한 케이스가 있다고 말해야 합니다. 이 약점이 Phase13의 동기입니다.

### 22. External Baselines

외부 baseline의 역할을 설명합니다. GPSR만 이기면 약하므로 Evo-QGeo, IQMR, DRAMA를 넣었습니다. 다만 원 논문을 완전 재현했다고 과장하지 않는 것이 중요합니다.

### 23. Evidence Figures

이 슬라이드는 기존 결과 그래프를 그대로 보여주는 장입니다. 말로만 설명하는 것보다 Phase12 결과가 이미 시각화되어 있다는 점을 보여줄 수 있습니다.

### 24. Claim Boundaries

이 장은 교수님께 신뢰를 주는 장입니다. 강한 주장은 명확히 하되, 실제 장비나 MAC 계층까지 검증한 것은 아니라고 선을 긋습니다.

### 25. Next Plan

다음 계획은 실험의 우선순위를 제시합니다. Phase13 full run, ablation, overhead, stress test, manuscript 반영 순서로 가면 교수님께도 연구가 체계적으로 진행 중이라는 인상을 줄 수 있습니다.

### 26. Closing

마지막은 한 문장 메시지로 마무리합니다. 교수님께는 이 연구가 단순 성능 튜닝이 아니라, 지역 실행 가능한 위험회피 라우팅 알고리즘으로 정리되고 있다고 보고하면 됩니다.
