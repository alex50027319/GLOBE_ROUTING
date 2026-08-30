# External baseline fidelity map

이 디렉터리의 main comparison은 동일한 1-hop data-forwarding simulator contract에
맞춘 구현이다. `Adapted`와 `Inspired` 표시는 원 논문의 simulator, state, reward 또는
communication model을 변경했다는 뜻이며 원 논문의 절대 성능을 재현한다는 뜻이 아니다.

| Method | Paper/standard | State | Action | Reward | Required messages | Training | Fidelity | Simulator mapping / known deviations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AODV | RFC 3561 | destination sequence number, reverse/forward route, expiry | route-table next hop | 없음 | RREQ/RREP/RERR | 없음 | common-contract adaptation | expanding-ring discovery, duplicate suppression, link invalidation과 byte accounting을 구현했다. IP/UDP headers, precursor별 local repair와 wireless collision은 생략한다. Physical graph는 control-message delivery에만 쓰며 forwarding은 route table만 읽는다. |
| OLSR | RFC 3626 | 1/2-hop neighbor, MPR, expiring topology tuple | disseminated route next hop | 없음 | HELLO/TC | 없음 | common-contract adaptation | periodic HELLO/TC와 stale tuple을 구현했다. MID/HNA, jitter, packet loss와 다중 interface는 생략한다. 현재 graph 직접 최단경로는 사용하지 않는다. |
| Greedy Geographic | GPSR의 greedy mode | neighbor relative position, destination delta | closest observed neighbor | 없음 | 위치 beacon은 별도 모델링하지 않음 | 없음 | partial | 기존 코드에는 planarization/perimeter recovery가 없어 `GPSR`로 표시하지 않는다. |
| Evo-QGeo (Adapted) | DOI 10.3390/drones10020150 | progress, link proxy, expected duration, forwardability, local Q | 1-hop neighbor | delivery/drop 및 fused link-state proxy | neighbor Q beacon | decentralized tabular Q update | common-contract adaptation | 출판사 원문의 future link evolution과 routing-hole bypass를 simulator lifetime/forwardability로 매핑한다. 3D PRR window, MAC ACK와 원 논문 PHY는 재현하지 않는다. |
| RDQN-HERP (Adapted) | DOI 10.1109/TVT.2026.3668740 | strict-local candidate feature | 1-hop neighbor/drop | simulator routing reward | 없음 | Double DQN, dueling NoisyNet, PER, n-step, target network | common-contract adaptation | 공개 원문에서 확인되지 않은 hidden size, tier threshold와 schedule은 config assumption이다. HERP는 success/high-instability/ordinary의 3-tier priority proxy다. |
| GAT-GRU-DDQN | architecture reference DOI 10.1109/WCSP68525.2025.1010249 | 1-hop candidate graph와 episode temporal state | masked 1-hop neighbor/drop | simulator routing reward | learned inter-node message 없음 | GAT, GRU, DDQN, PER | inspired architecture control | global graph, trust/anomaly, MOH/MoE/GhostConv를 사용하지 않으므로 SRRGD-DQN이라 부르지 않는다. |
| SwitchGLOBE | 이 저장소의 최종 checkpoint | 선택적 1-hop local/risk fields | 1-hop neighbor/drop | 평가 시 학습 없음 | 없음 | 기존 PPO teacher→KD students; checkpoint read-only | proposed method | baseline 디렉터리에서 재학습하지 않는다. |

IQMR Q(lambda)는 `legacy/iqmr.py`, 기존 DRAMA 유사 구조는
`legacy/drama_inspired.py`에 compatibility import로 남지만 main comparison에서 제외한다.

## 정보 예산

method registry가 실제 observation field, hop radius, privileged-information flag,
control-plane requirement, source와 fidelity label을 manifest에 기록한다. AODV/OLSR에
전달되는 topology snapshot은 control packet을 전달하기 위한 network medium이며,
정책의 next-hop 계산은 메시지로 생성된 table만 사용한다. `policy_input_bytes`와
`control_bytes`는 별도 열이며 서로 대체하지 않는다.

## 확인된 근거와 가정

- RFC 3561/3626은 RFC Editor 원문을 기준으로 했다.
- Evo-QGeo는 MDPI 공개 전문을 기준으로 했다.
- RDQN-HERP와 graph-temporal reference는 2026-08-30 조회 시 공개 publisher/author
  전문에서 모든 hyperparameter를 확인할 수 없었다. 따라서 정확 reproduction 주장을
  하지 않으며 위 표에 없는 세부값은 `external_comparison.yaml`의 명시적 가정이다.
