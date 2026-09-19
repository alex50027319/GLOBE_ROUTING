# Metrics Glossary

- `overall_pdr`: 모든 episode 중 전달 성공 비율
- `connected_pair_pdr`: 초기 endpoint가 연결된 episode 중 전달 성공 비율
- `endpoint_availability`: 초기 source-destination 연결 가능 비율
- `deadline_delivery_ratio`: deadline 안에 전달된 비율
- `p95_success_delay`: 성공 packet delay의 95th percentile
- `energy_per_delivered_packet`: 전송 energy proxy 합을 전달 성공 수로 나눈 값
- `policy_input_bytes`: policy가 읽은 tensor byte 수
- `decision_latency_p95`: warm-up 후 policy `act` wall time의 95th percentile
- `delivery_rate_proxy`: 단일-packet simulator의 전달/step 비율
- `switch_activation_rate`: forwarding decision 중 predictive branch 전환 비율

Energy는 Joule이 아니며, policy input bytes는 routing-control overhead가 아니다.
현재 simulator의 delivery rate proxy를 Mbps throughput 또는 goodput으로 표현하지 않는다.
