# Metrics Glossary

- PDR (Packet Delivery Ratio): 생성된 패킷 중 목적지에 도착한 비율
- mean delay: 성공적으로 전달된 패킷의 평균 종단간 지연
- p95 delay: 성공 패킷 지연 분포의 95번째 백분위수
- throughput: 단위 시간 동안 성공적으로 전달된 데이터량
- input bytes: 정책 판단을 위해 전달·관측한 입력 정보량의 근사
- energy: 통신·이동·계산 모델에서 소비된 에너지 지표
- deadline miss: 제한 시간 안에 도착하지 못한 비율
- agent drop rate: 정책이 전달 포기/drop 행동을 선택한 비율
- link margin: `max(0, 1 - distance / communication_radius)` 형태의 거리 여유도
- queue headroom: `1 - queue_occupancy / max_queue_size`
- link lifetime: 상대 운동을 고려한 예상 링크 유지 시간

PDR만으로 방법의 우수성을 결론 내리지 않고 delay, deadline, energy, overhead,
drop과 함께 해석한다. 시뮬레이션 단위를 meter로 바꿀 때는 변환 가정을 명시한다.
