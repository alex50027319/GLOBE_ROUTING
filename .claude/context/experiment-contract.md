# Experiment Contract

## 용어

- scenario: 이동성, 노드 수, 링크 손실, 구조적 홀 등 하나의 네트워크 조건
- method: 동일한 scenario에서 비교하는 라우팅 정책
- training seed: 학습 초기화와 stochastic process를 재현하는 난수 기준
- episode: 한 source-destination packet routing 시행

## Episode 설정

- 일반/OOD/node 시나리오는 source와 destination을 유효 후보에서 무작위 선택한다.
- structural hole은 의도된 막다른 길을 재현하기 위해 source/destination을 고정한다.
- predictive break도 미래 링크 단절 경로를 재현하기 위해 endpoint를 고정한다.
- queue occupancy는 노드별로 환경 설정 범위에서 생성된다.
- link margin은 현재 거리와 통신 반경의 여유를 나타낸다.
- link lifetime은 상대 위치·속도로 현재 링크가 유지될 미래 시간을 예측한다.

## 재현성

- 결과에는 phase, config, seed, mode, device, checkpoint, output path를 기록한다.
- 여러 seed를 병합할 때 raw row를 보존하고 중복 행을 검사한다.
- full과 smoke 결과를 같은 표본처럼 합치지 않는다.
