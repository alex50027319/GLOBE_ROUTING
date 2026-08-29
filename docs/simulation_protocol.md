# SwitchGLOBE Simulation Protocol

이 문서는 외부 baseline 비교와 SwitchGLOBE ablation이 공유해야 할 평가 계약을
정의한다. 수치와 정의는 실행 전에 고정하며, smoke 결과와 full 결과를 섞지 않는다.

## 1. 근거와 평가 원칙

Notion의 `UAV/FANET RL 데일리 논문 분석` 데이터베이스에서 직접 packet-routing으로
분류된 18편을 집계하면 Delay 15편, PDR 13편, Packet Loss 8편, Energy 7편,
Throughput 7편, Routing Overhead 5편이 해당 지표를 사용한다. 따라서 PDR와 delay를
공통 성능축으로 두되, SwitchGLOBE의 주장인 deadline reliability, tail latency,
strict-local deployment cost와 selective switching을 별도 축으로 검증한다.

주요 참고:

- [UAV/FANET RL 논문 데이터베이스](https://app.notion.com/p/0a291669e68449f6918ddca01538d040)
- [Traffic-Adaptive Per-Hop Multipath Routing](https://app.notion.com/p/3caad79990e68174ad2ed9abe277f6bd)
- [Adaptive Intelligent Routing in Ultra-High-Speed FANETs](https://app.notion.com/p/3c7ad79990e6811da0e3ca071599b67b)
- [RoutePPO](https://app.notion.com/p/3cbad79990e68170b83fd5fae294030a)
- [Predictive-Q Dual-Path Routing](https://app.notion.com/p/3c6ad79990e681e999fdc7b22c9cefab)

## 2. Primary endpoints

Primary endpoint는 원고의 우월성 주장과 통계 검정에 직접 사용한다.

| Metric | 정의 | 방향 | 이유 |
| --- | --- | --- | --- |
| Connected-pair PDR | 초기 source-destination path가 존재한 episode 중 전달 성공 비율 | 높을수록 좋음 | topology 자체의 불가능성과 routing 실패를 분리 |
| Deadline delivery ratio | 전체 평가 packet 중 deadline 이내 전달된 비율 | 높을수록 좋음 | 최근 deadline-aware routing의 on-time PDR에 대응 |
| P95 success delay | 전달 성공 packet의 end-to-end step 수 95 percentile | 낮을수록 좋음 | 평균이 숨기는 tail delay와 recovery 지연 측정 |
| Energy per delivered packet | 전체 transmission-energy proxy 합 / 전달 성공 packet 수 | 낮을수록 좋음 | 조기 drop 방식이 낮은 energy로 보이는 오류 방지 |
| P95 decision latency | 고정 CPU, batch 1에서 policy `act`의 warm-up 후 p95 wall time | 낮을수록 좋음 | strict-local 경량 배치 주장을 직접 검증 |
| Policy input bytes per decision | 실제 policy가 읽은 tensor의 byte 합 / decision 수 | 낮을수록 좋음 | global-to-local 정보 비용을 정량화 |

`overall_pdr`도 항상 보고하지만 endpoint-disconnected episode의 비율에 영향을 받으므로
primary routing comparison은 `connected_pair_pdr`로 수행한다. `endpoint_availability`를
같이 제시해 denominator를 감사할 수 있게 한다.

## 3. Delivery outcome accounting

모든 episode는 다음 outcome 중 하나로 분해한다.

1. deadline 내 전달
2. deadline 이후 전달
3. routing loop drop
4. TTL expiration
5. explicit agent DROP
6. invalid action
7. link/queue/environment failure
8. 초기 endpoint disconnected

다음을 추가 집계한다.

- `late_delivery_ratio_all = delivered_after_deadline / all_episodes`
- `late_delivery_ratio_delivered = delivered_after_deadline / delivered`
- `total_drop_rate = dropped / all_episodes`
- drop-reason별 비율

Deadline-aware 논문에서는 late-but-delivered packet이 on-time PDR에도 packet loss에도
속하지 않을 수 있다. 따라서 `packet_loss = 1 - PDR`로 단순 계산하지 않고 실제 drop
event에서만 loss를 집계한다.

## 4. Secondary performance metrics

| 범주 | Metric | 해석 |
| --- | --- | --- |
| Delay | mean success delay | 기존 FANET 논문과 직접 비교 가능한 평균 E2E delay |
| Delay | mean per-hop delay | 성공 packet의 `steps / hop_count`; RDQN-HERP 대응 |
| Deadline | deadline slack | `deadline_steps - arrival_steps`; 음수는 deadline miss |
| Route quality | hop count, path stretch | reliability gain이 과도한 detour에서 나온 것인지 확인 |
| Link efficiency | attempts, expected transmissions proxy | 재전송과 불안정 link 선택 비용 |
| Energy | energy per generated packet | network가 모든 요청에 소비한 평균 transmission proxy |
| Energy | energy per on-time delivery | deadline을 만족한 유효 전달 한 건당 energy proxy |
| Queue | cumulative queue-delay proxy | congestion-aware baseline과 비교하기 위한 보조 지표 |
| Robustness | minimum link lifetime and margin | imminently failing link 회피 여부 |
| Reward | episode reward | debugging 전용; 실제 QoS 우월성 근거로 사용하지 않음 |

Energy는 Joule이 아니라 simulator-level transmission proxy다. 평균 episode energy만
사용하면 packet을 일찍 drop한 정책이 효율적으로 보일 수 있으므로 generated-packet,
delivered-packet, on-time-delivery 세 denominator를 함께 보고한다.

## 5. Throughput and routing overhead

현재 simulator는 episode당 단일 packet 전달 문제이므로 `delivered / total_steps`는
`delivery_rate_proxy`로 부르고 Mbps throughput이라고 부르지 않는다. 실제 throughput,
goodput, queue overflow를 주장하려면 multi-flow traffic, packet size, simulation time,
link capacity와 contention을 포함하는 별도 traffic experiment가 필요하다.

`policy_input_bytes`는 policy tensor 크기이며 실제 routing-control overhead가 아니다.
AODV/OLSR, CQMR, graph message-passing 방식의 overhead를 비교하려면 다음을 별도
계측해야 한다.

- beacon/HELLO, route request/reply, topology update의 bytes와 packet 수
- learned inter-node message의 dimension, quantization, frequency
- data packet 대비 control-byte ratio
- stale/lost control message 조건에서의 성능

해당 control plane이 구현되지 않은 결과에는 `routing overhead: not modeled`를 명시한다.

## 6. Deployment-cost metrics

모든 neural policy를 동일 hardware와 runtime 설정에서 측정한다.

- trainable parameter count
- serialized checkpoint bytes
- peak inference RAM
- decision latency p50/p95/p99
- policy input bytes per decision
- observation hop radius와 필요한 field 목록
- 필요하면 multiply-accumulate count 또는 profiler operation count
- 학습 wall-clock, environment interactions, peak accelerator memory

CPU timing은 warm-up 후 single-thread, batch 1 조건을 primary로 하고 CUDA timing은
보조 자료로 분리한다. I/O와 environment step 시간을 policy inference 시간에 섞지 않는다.

## 7. SwitchGLOBE-specific ablation diagnostics

다음은 외부 baseline 우월성보다 switching mechanism의 인과적 설명에 사용한다.

- switch activation rate per forwarding decision
- normal/predictive branch action disagreement rate
- switch 시 선택 action의 danger 감소량
- switch 후 link survival과 delivery 성공률
- false-switch rate: normal action도 안전했는데 불필요하게 전환한 비율
- missed-risk rate: 위험한 normal action에서 switch하지 않은 비율
- calibration constraint violation: normal/structural-hole PDR tolerance 위반 여부
- teacher-student action KL과 top-1 agreement

공개 ablation 이름은 `Geo-Residual Student`, `Predictive Prior Only`,
`Predictive Student (No Switch)`, `SwitchGLOBE`를 사용한다. Phase 번호는 checkpoint
호환을 위한 내부 이름으로만 유지한다.

## 8. Robustness and scaling summaries

평균 성능 하나 외에 다음을 보고한다.

- mobility, link loss, node count 증가에 따른 degradation slope
- unseen node-count와 unseen-speed 성능
- scenario별 최악 seed와 전체 worst-scenario 결과
- structural-hole과 predictive-break scenario를 분리한 결과
- PDR--p95 delay--energy--decision latency Pareto plot

서로 다른 scenario의 단위를 무리하게 하나의 scalar score로 합치지 않는다.

## 9. Statistics

- training seeds: `42, 77, 123, 314, 2718`
- scenario당 evaluation episode: 200
- 모든 방법에 동일 evaluation seed와 reset option 사용
- primary statistical unit: training seed
- method difference는 seed-paired estimate로 계산
- mean difference, relative difference, 95% confidence interval 보고
- episode-level bootstrap을 사용할 경우 training seed를 outer cluster로 유지
- primary endpoint와 contrast는 full run 전에 고정
- 누락 seed, 중복 scenario-method-seed, calibration/evaluation overlap을 manifest에서 차단

외부 baseline campaign과 ablation campaign은 별도 manifest와 결과 디렉터리를 사용한다.
두 campaign의 simulator contract가 동일하더라도 결과표의 연구 질문은 분리한다.

## 10. Implementation status

현재 evaluator가 이미 제공하는 지표:

- overall/connected-pair PDR, endpoint availability
- deadline delivery, mean/p95 success delay
- path stretch, ETX와 energy proxy
- drop reason, queue-delay proxy, link lifetime/margin
- policy/local-observation bytes와 기본 switch diagnostics

추가 구현이 필요한 지표:

- late-delivery 분해
- mean per-hop delay와 deadline slack distribution
- delivered/on-time-delivery당 energy
- p50/p95/p99 decision latency와 model/RAM cost
- switch false-positive/missed-risk 진단
- 실제 control-message overhead
- multi-flow throughput/goodput 실험
