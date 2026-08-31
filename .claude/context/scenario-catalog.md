# Scenario Catalog

## 일반 및 OOD

- heldout medium: 학습과 유사하지만 평가용으로 분리된 중간 난도
- OOD link loss: 학습 분포보다 높은 패킷·링크 손실
- OOD fast/extreme mobility: UAV 이동 속도가 더 빠른 환경
- OOD sparse: 통신 반경 대비 노드 밀도가 낮아 경로 연결이 어려운 환경
- OOD nodes: 학습 노드 수와 다른 10/16/24 UAV 확장성 평가

## 구조적 홀

`structural_hole_45`, `structural_hole_225_link_loss`는 목적지에 가까워 보이는
이웃이 실제로는 다음 경로가 없는 막다른 길로 이어지는지 시험한다. 45와 225는
훈련 방향과 다른 회전 변형이며 전체 네트워크가 계속 회전한다는 뜻은 아니다.

## 예측 단절

`predictive_break_45`, `predictive_break_225_link_loss`는 지금 연결되어 있지만
상대 이동으로 곧 끊길 링크를 미리 피할 수 있는지 시험한다. 현재 link quality만
좋은 경로와 미래 lifetime까지 좋은 경로를 구분하는 것이 핵심이다.

정확한 scenario 목록과 파라미터는 해당 phase campaign/config에서 다시 확인한다.
