# Claude / Claude Cowork용 SwitchGLOBE 최종 시뮬레이션 마스터 프롬프트

아래 내용을 그대로 복사하여 Claude Code 또는 Claude Cowork에 전달한다.

---

당신은 FANET/UAV routing, reinforcement learning evaluation, 통계 분석,
재현 가능한 시뮬레이션 실험을 담당하는 선임 연구 엔지니어다. 다음 저장소에서
SwitchGLOBE 논문의 최종 baseline comparison, ablation, latency benchmark, 통계,
table, figure를 생성하라.

## 0. 프로젝트와 절대 규칙

- 프로젝트 루트: `/Users/alex/Documents/GLOBE_ROUTING`
- 작업 브랜치: `refactor/GLOBE`
- Python 환경 우선순위:
  `/Users/alex/Documents/GLOBE_ROUTING/ResearchAIWorkspace/.venv/bin/python`
- 실행 구조: 위 local venv는 코드 검사, job 준비, 결과 검증용 controller로 사용하고,
  장시간 학습/시뮬레이션은 프로젝트에서 이미 사용한 `colab-cli -> Google Colab
  A100` workflow로 실행한다. 현재 셸에서 launcher 명령이 보이지 않는다는 이유만으로
  과거 Colab 실행 결과가 없다고 판단하지 마라. 기존 실행 script/config/history를 먼저
  찾아 동일한 workflow를 복원하라.
- 기존 Exact SwitchGLOBE checkpoint 후보:
  `/Users/alex/Documents/GLOBE_ROUTING/ResearchAIWorkspace/artifacts/lite_globe/phase12/checkpoints`
- 검증이 필요한 기존 Colab A100 결과:
  - external comparison seed별 ZIP:
    `/Users/alex/Documents/GLOBE_ROUTING/artifacts/external_comparison_colab_results/seeds_{42,77,123,314,2718}.zip`
  - Exact latency ZIP:
    `/Users/alex/Documents/GLOBE_ROUTING/artifacts/switchglobe_latency_optimization/colab_a100.zip`
- 최종 출력 루트:
  `/Users/alex/Documents/GLOBE_ROUTING/artifacts/final_paper_simulation`

절대 규칙:

1. 실제로 실행되지 않은 결과를 만들거나 추정하지 마라.
2. seed 42 pilot 결과를 5-seed 최종 결과로 재사용하거나 표현하지 마라.
3. smoke 결과와 full 결과를 절대 합치지 마라.
4. 기존 dirty worktree를 보존하라. `git reset --hard`, `git checkout --`, 임의
   파일 삭제, 기존 결과 덮어쓰기를 하지 마라.
5. commit, push, merge는 사용자가 별도로 승인하기 전에는 하지 마라.
6. eBPF는 코드, 실험, figure, 논문 주장에 포함하지 마라.
7. freshness cache를 일반 routing 성능 결과에 섞지 마라.
8. Energy는 Joule이 아니라 simulator transmission-energy proxy라고 명시하라.
9. `policy_input_bytes`를 실제 routing control overhead라고 부르지 마라.
10. 현재 simulator의 `delivered / steps`를 Mbps throughput이라고 부르지 마라.
11. Adapted 또는 inspired baseline을 원 논문의 완전한 재현이라고 표현하지 마라.
12. 기존 public method name과 checkpoint 호환성을 깨지 마라.
13. 평가 결과가 불리하더라도 숨기거나 method/scenario를 임의로 제외하지 마라.
14. 오류나 acceptance gate 실패를 수치 조정으로 해결하지 말고 원인을 보고하라.
15. 장시간 full run 전에 반드시 preflight와 smoke gate에서 사용자 승인을 받아라.
16. complete manifest가 있는 기존 Colab 결과를 무조건 다시 실행하지 마라. config,
    checkpoint hash, scenario/seed, metric schema, expected row count를 감사한 뒤 호환되면
    재사용하고, 불완전하거나 비호환인 seed/variant만 다시 실행하라.
17. 서로 다른 Colab VM에서 측정한 latency를 직접적인 속도 향상 비율의 primary
    근거로 사용하지 마라. Exact와 Fast variant의 최종 latency 비교는 동일한 A100
    session에서 함께 재측정하고, CPU primary latency는 동일한 물리 CPU session에서
    함께 측정하라.

## 1. 먼저 해야 할 preflight audit

아직 full simulation을 시작하지 말고 다음을 감사하라.

1. `git status --short --branch`, 현재 commit hash, 변경 파일 목록을 기록한다.
2. 저장소의 `CLAUDE.md`, `implementations/lite_globe/CLAUDE.md`,
   `tests/lite_globe/CLAUDE.md`, `docs/simulation_protocol.md`를 읽고 준수한다.
3. 다음 코드 경로를 실제로 확인한다.
   - `implementations/lite_globe/experiments/external_comparison_campaign.py`
   - `implementations/lite_globe/experiments/phase12_campaign.py`
   - `implementations/lite_globe/experiments/latency_optimization_campaign.py`
   - `implementations/lite_globe/evaluation/evaluator.py`
   - `implementations/lite_globe/evaluation/generalization.py`
   - `implementations/lite_globe/evaluation/external_comparison_reporting.py`
   - `implementations/lite_globe/evaluation/statistics.py`
   - `implementations/lite_globe/baselines/registry.py`
   - `implementations/lite_globe/scenarios/generalization_suite.py`
4. Exact SwitchGLOBE checkpoint가 seed `42, 77, 123, 314, 2718`에 모두 존재하며
   각 checkpoint가 해당 seed와 model shape에 맞게 load되는지 확인한다.
5. FastSwitchGLOBE checkpoint 현황을 확인한다. 현재 seed 42만 있을 가능성이
   높으므로 나머지 seed를 기존 Exact teacher와 동일한 protocol로 새로 학습해야 한다.
6. training/calibration/evaluation seed와 scenario가 겹치지 않는지 확인한다.
7. `phase9_evaluation_scenarios(seed)`가 실제로 14개이며 모든 seed에서 scenario
   name/order가 동일한지 확인한다.
8. 전체 `tests/lite_globe`를 실행하여 시작 상태를 기록한다.
9. 기존 Colab 결과 ZIP 6개를 안전한 임시 디렉터리에 풀어 manifest, raw CSV,
   checksum, duplicate, NaN/Inf, method/scenario/seed coverage를 감사한다.
   - external comparison ZIP 5개는 각각 `complete=true`, 19,600 episode rows,
     98 seed-summary rows여야 한다.
   - 다섯 seed를 합치면 external baseline episode rows는
     `7 methods × 5 seeds × 14 scenarios × 200 = 98,000`
     이다. 이는 새 실행 예정량이 아니라 먼저 검증할 기존 결과량이다.
   - 합친 external baseline summary rows:
     `7 × 5 × 14 = 490`
   - Exact latency ZIP은 `suite=switchglobe_exact_latency`,
     `training_seeds=[42,77,123,314,2718]`, runtime rows 130,
     deployment-cost rows 30인지 확인한다.
10. FastSwitchGLOBE의 seed별 checkpoint와 결과를 별도로 감사한다. seed 42 결과가
    있더라도 나머지 seed까지 완료됐다고 추정하지 마라. 새로운 Fast routing 평가의
    최소 예상 episode rows는 두 variant를 모두 새로 평가할 경우
    `2 × 5 × 14 × 200 = 28,000`이다.
11. 다음 ablation row count는 기존 호환 결과를 얼마나 재사용할 수 있는지에 따라
    `reused`와 `newly_run`으로 나누어 validation contract에 기록한다.
   - ablation episode rows:
     `6 variants × 5 × 14 × 200 = 84,000`
   - ablation summary rows:
     `6 × 5 × 14 = 420`
12. preflight 결과를
    `artifacts/final_paper_simulation/preflight/preflight_report.md`와
    `preflight_manifest.json`에 저장한다.

Preflight가 끝나면 다음을 사용자에게 보고하고 승인을 기다려라.

- 누락 checkpoint
- 예상 학습/평가 작업량
- 수정이 필요한 코드
- 예상 output schema와 row count
- 발견된 공정성 또는 data leakage 위험

## 2. 논문 method 이름을 고정하라

외부 baseline:

- `AODV`
- `OLSR`
- `Greedy Geographic`
- `Evo-QGeo (Adapted)`
- `RDQN-HERP (Adapted)`
- `GAT-GRU-DDQN`

내부 ablation 및 제안기법:

- `Geo-Residual Student`
- `Predictive Prior Only`
- `Predictive Student (No Switch)`
- `SwitchGLOBE Exact`
- `FastSwitchGLOBE`
- `FastSwitchGLOBE + Top-2`

정확한 의미:

- `SwitchGLOBE Exact`: 알고리즘 의미를 보존한 reference implementation
- `FastSwitchGLOBE`: single-pass distilled low-latency variant이며 두 Fast variant
  중 계산적으로 가장 가볍다.
- `FastSwitchGLOBE + Top-2`: 같은 신경망을 한 번 실행한 뒤 backup 후보를 추가로
  고르는 reliability-enhanced low-latency variant다. Fast 단독보다 더 가볍다고
  표현하지 마라.
- `Legacy repeated SwitchGLOBE`: latency-only historical implementation이며 routing
  성능 표의 별도 method로 넣지 마라.
- `Freshness cache`: repeated-query supplementary runtime experiment 전용이다.

내부 historical Phase 이름은 checkpoint loading에만 사용하고 최종 table/figure에는
위 public name을 사용하라. 이름 mapping을 JSON으로 저장하라.

## 3. 고정 실험 protocol

- training seeds: `42, 77, 123, 314, 2718`
- evaluation episodes: scenario당 `200`
- evaluation scenarios: `phase9_evaluation_scenarios(seed)`의 14개
- 모든 method에 동일한 evaluation seed와 동일한 reset option 사용
- 현재 코드의 evaluation seed 공식 유지:
  `1_100_000 + scenario_index * 10_000 ... +199`
- primary statistical unit: training seed
- CPU latency primary device: CPU, batch size 1
- 학습은 가능한 device를 사용할 수 있으나 평가 seed와 결과는 device와 무관하게
  재현 가능한지 확인한다.
- calibration 결과로 evaluation set을 선택하거나 hyperparameter를 다시 조정하지 마라.

모든 run manifest에 다음을 저장하라.

- git commit hash와 dirty file 목록
- config 전체와 config SHA-256
- checkpoint 경로와 SHA-256
- Python, NumPy, PyTorch, Gymnasium 버전
- OS, CPU, device, thread count
- 시작/종료 시각과 wall time
- training seed, evaluation seed range
- scenario 목록
- expected/actual row count
- complete 여부와 실패 이유

## 4. Phase A: smoke simulation

사용자 preflight 승인 후 다음 smoke를 수행하라.

- training seed: 42
- 14 scenarios
- scenario당 3~10 episodes
- external baseline 전체
- ablation 전체
- Fast training/load
- latency variant 전체
- Top-2 synthetic stale-primary test

Smoke의 목적은 성능 결론을 내는 것이 아니라 다음을 검사하는 것이다.

- 모든 model/checkpoint load 성공
- row schema 완전성
- 동일 evaluation seed 사용
- invalid/NaN/Inf 없음
- method/scenario/seed duplicate 없음
- action mask 위반 없음
- manifest와 expected row count 일치
- figure/table generator가 headless 환경에서 동작
- `FastSwitchGLOBE`와 `FastSwitchGLOBE + Top-2`의 일반 routing action/outcome이
  동일함
- Top-2 failover resolve 과정의 model forward가 총 1회임

출력:

`artifacts/final_paper_simulation/smoke/`

Smoke 결과와 validation report를 사용자에게 보고하고 full run 승인을 기다려라.
Smoke 수치는 논문용 final CSV에 복사하지 마라.

## 5. Phase B: external baseline full comparison

비교 방법:

- AODV
- OLSR
- Greedy Geographic
- Evo-QGeo (Adapted)
- RDQN-HERP (Adapted)
- GAT-GRU-DDQN
- SwitchGLOBE Exact

연구 질문:

`SwitchGLOBE Exact가 전통 및 학습 기반 baseline보다 routing reliability와 deadline
성능을 개선하는가?`

이 Phase는 먼저
`artifacts/external_comparison_colab_results/seeds_{42,77,123,314,2718}.zip`의
기존 Colab A100 결과를 감사하고 병합하는 단계다. 다섯 ZIP이 위 protocol과 호환되고
각 manifest와 row-level validation을 통과하면 98,000 episodes를 다시 돌리지 마라.
실패하거나 누락된 seed만 동일한 `colab-cli -> Colab A100` workflow로 재실행하라.
병합 결과에는 각 row의 source archive와 원본 manifest/checksum provenance를 남겨라.

가능하면 기존 `run_external_comparison`과 reporting contract를 재사용하라. 필요한
수정은 최소화하고 test를 추가하라. baseline 학습 checkpoint는 seed별로 저장하고
`--resume`이 incomplete checkpoint를 complete로 오인하지 않도록 검사하라.

출력:

`artifacts/final_paper_simulation/full/baselines/`

필수 raw/summary 산출물:

- `raw/episodes.csv`
- `raw/training.csv`
- `raw/deployment_costs.csv`
- `summaries/seed_summaries.csv`
- `summaries/aggregate_statistics.csv`
- `summaries/paired_effects.csv`
- `method_contracts.json`
- `manifest.json`
- validation report

## 6. Phase C: SwitchGLOBE ablation full comparison

비교 variant:

- Geo-Residual Student
- Predictive Prior Only
- Predictive Student (No Switch)
- SwitchGLOBE Exact
- FastSwitchGLOBE
- FastSwitchGLOBE + Top-2

연구 질문:

1. predictive information이 routing 성능에 기여하는가?
2. selective switching이 no-switch보다 우수한가?
3. distillation이 Exact의 성능을 유지하며 latency를 줄이는가?
4. Top-2가 정상 routing 동작을 바꾸지 않고 failover 준비도를 높이는가?

FastSwitchGLOBE 학습 규칙:

- 각 training seed에 대응하는 Exact SwitchGLOBE를 teacher로 사용한다.
- teacher data는 기존 training scenarios에서만 수집한다.
- dataset split은 individual decision row가 아니라 episode/scenario group 단위로
  분리하여 leakage를 방지한다.
- 기본 config는 현재 `LatencyOptimizationConfig`를 사용한다.
- seed마다 training/validation/test sample 수, action agreement, switch accuracy,
  KL, best epoch를 저장한다.
- seed 42 기존 checkpoint는 config/hash가 정확히 일치할 때만 resume하고 그렇지
  않으면 별도 output 경로에 재학습한다.
- Top-2는 Fast model과 같은 weight를 사용하며 별도로 재학습하지 않는다.

Standard routing evaluation에서는 Top-2 resolver를 인위적으로 호출하지 마라.
backup 준비만으로 primary action이 바뀌면 구현 오류다. Fast와 Fast+Top-2 사이에
다음 값이 evaluation episode별로 동일해야 한다.

- delivered/dropped
- drop reason
- steps/hop count
- transmission attempts
- primary action sequence

SwitchGLOBE Exact 결과를 baseline campaign에서 재사용하려면 git/config/checkpoint/
scenario/evaluation seed가 완전히 같고 SHA-256 provenance가 기록되어야 한다. 조건이
하나라도 다르면 재평가하라.

출력:

`artifacts/final_paper_simulation/full/ablation/`

## 7. Phase D: Top-2 synthetic failover audit

현재 simulator는 policy inference와 `env.step()` 사이에 자연스러운 link-state race가
없으므로 standard PDR 결과로 Top-2 failover 효과를 주장하지 마라.

별도 audit에서만 다음을 수행하라.

1. Fast+Top-2 decision을 한 번 계산한다.
2. backup이 있는 decision만 eligible event로 센다.
3. live action mask에서 primary만 무효화한다.
4. backup은 유효하게 유지한다.
5. `resolve_decision()` 결과가 backup인지 확인한다.
6. forward hook으로 전체 neural model forward가 1회인지 확인한다.
7. primary와 backup이 모두 무효일 때 DROP으로 가는지도 검사한다.

보고 지표:

- backup availability rate
- eligible failover events
- successful backup resolutions
- failover success rate
- failover miss rate
- additional neural forwards
- resolver-only latency p50/p95/p99

이 결과는 `synthetic stale-primary audit`이라고 명확히 표기하라. 실제 무선 link
failure test라고 부르지 마라.

## 8. Phase E: latency benchmark

비교 variant:

- Legacy repeated SwitchGLOBE
- SwitchGLOBE Exact
- FastSwitchGLOBE
- FastSwitchGLOBE + Top-2

Freshness cache는 이 primary 비교에서 제외하고 다음 Phase F에 둔다.

`artifacts/switchglobe_latency_optimization/colab_a100.zip`은 다섯 seed에 대한 기존
Exact latency reference로 먼저 감사하라. 하지만 Fast variant가 그 ZIP과 동일한
Colab VM/session에서 함께 측정된 것이 아니라면 이 파일과 새 Fast 결과를 단순히 나눠
속도 향상률을 주장하지 마라. 최종 paper용 비교에서는 Legacy, Exact, Fast, Fast+Top-2를
한 번의 동일한 Colab A100 session에서 순서를 교차하거나 randomized block으로 함께
측정하라. GPU 결과는 secondary이며, CPU primary 비교도 네 variant를 동일한 CPU
session에서 함께 측정해야 한다.

공정한 latency 측정 조건:

- 동일한 물리 CPU와 power mode
- CPU primary result
- batch size 1
- `torch.inference_mode()`
- 동일 observation
- 동일 warm-up
- 가능하면 `torch.set_num_threads(1)` 및 별도 프로세스에서 thread 설정 고정
- preprocessing 포함/제외를 각각 분리
- GPU가 있으면 synchronized secondary result로만 보고
- environment step time과 policy inference time을 섞지 않음

seed별 최소 권장 측정:

- warm-up 50회 이상
- steady-state repeat 2,000회 이상
- cold start는 fresh process 반복으로 별도 측정
- preprocess, model, action extraction, end-to-end policy를 각각 측정

필수 지표:

- cold-start latency
- mean
- p50/p95/p99
- decisions per second
- parameter count
- serialized checkpoint bytes
- policy input bytes
- peak inference memory

Latency 결과는 raw repetition 또는 재계산 가능한 충분한 원시 timing 데이터를
보존하라. episode별 p95들의 p95와 전체 raw decision p95를 혼동하지 마라. 논문의
primary latency 값은 고정 benchmark의 raw decision distribution에서 계산하라.

출력:

`artifacts/final_paper_simulation/full/latency/`

## 9. Phase F: freshness cache supplementary benchmark

Freshness cache는 일반 routing simulator에서 cache hit가 거의 없으므로 main routing
comparison에서 기본 비활성화한다.

별도의 repeated-query 조건에서만 다음을 비교하라.

- FastSwitchGLOBE + Top-2, cache disabled
- FastSwitchGLOBE + Top-2, freshness cache enabled

두 workload를 모두 보고하라.

1. 일반 변화 관측 workload: cache hit가 낮고 overhead가 있음을 숨기지 않는다.
2. 동일 fresh observation 반복 workload: TTL 이내 반복 query의 이득을 측정한다.

필수 지표:

- hit/miss rate
- stale/state/capacity eviction rate
- mean/p50/p95/p99 latency
- action equality
- TTL 만료 후 재추론 여부
- action-mask 또는 neighbor-state 변경 후 재추론 여부

이 실험을 실제 network deployment 성능이라고 부르지 마라.

출력:

`artifacts/final_paper_simulation/supplementary/freshness_cache/`

## 10. Primary metrics와 denominator

논문 primary metrics를 다음 순서로 고정한다.

1. Connected-pair PDR — higher is better
2. Deadline delivery ratio — higher is better
3. P95 success delay — lower is better
4. Energy per delivered packet — lower is better
5. P95 decision latency — lower is better
6. Policy input bytes per decision — lower is better

항상 같이 보고할 보조 지표:

- endpoint availability
- overall PDR
- mean success delay
- mean per-hop delay
- path stretch
- expected transmissions proxy
- energy per generated packet
- energy per on-time delivery
- minimum link lifetime/margin
- queue delay proxy
- deadline slack
- late delivery ratio
- drop reason별 비율
- switch activation/disagreement/danger reduction
- false-switch/missed-risk rate
- backup availability/failover metrics

P95 success delay는 전달 성공 packet만 사용하되 delivered count를 반드시 같이 적는다.
Energy per delivered packet은 실패 episode를 포함한 전체 transmission energy를 성공
전달 수로 나눈다. 분모가 0이면 임의로 0을 넣지 말고 undefined로 처리하고 이유를
보고하라.

## 11. 사전 고정 acceptance gates

FastSwitchGLOBE를 Exact의 대체 최종 방법으로 부르기 전에 다음을 5-seed full 결과로
확인하라.

- mean connected-pair PDR degradation이 `-0.005`보다 나쁘지 않을 것
- mean deadline delivery ratio degradation이 `-0.005`보다 나쁘지 않을 것
- p95 success delay와 energy proxy의 악화를 숨기지 않고 CI와 함께 보고할 것
- CPU p95 decision latency가 Exact보다 최소 30% 감소할 것
- validation/test teacher action agreement와 switch accuracy를 seed별 보고할 것

추가 권장 audit:

- scenario family 중 Fast의 connected PDR 하락이 0.01을 초과하는 곳 표시
- worst seed와 worst scenario를 별도 표로 저장

Gate를 통과하지 못하면 Fast를 최종 대체 방법이라고 쓰지 말고 `latency-performance
trade-off variant`로 보고하라.

Top-2 acceptance gates:

- standard routing outcome mismatch: 0
- eligible synthetic stale-primary backup resolution success: 100%
- additional neural forward: 0
- latency overhead를 Fast 단독 대비 정확히 보고

## 12. 통계 분석

14,000개 episode를 서로 독립인 표본으로 취급하지 마라. primary independent unit은
training seed다.

1. 각 `(method, scenario, training_seed)`에 대해 200 episode를 집계한다.
2. 같은 training seed끼리 method contrast를 paired 계산한다.
3. 각 metric에 대해 다음을 저장한다.
   - seed count
   - mean
   - sample standard deviation
   - two-sided 95% t interval
   - paired mean difference
   - paired relative difference
   - paired 95% CI
4. 가능하면 fixed RNG seed를 사용한 hierarchical/cluster bootstrap을 추가한다.
   training seed를 outer cluster로 유지하고 bootstrap 설정을 manifest에 기록한다.
5. 여러 ablation contrast에 p-value를 추가한다면 paired test와 Holm correction을
   적용하되, `n=5`의 낮은 검정력을 명시한다. p-value만으로 결론을 내리지 마라.
6. higher-is-better와 lower-is-better 방향을 코드에서 명시적으로 관리한다.
7. CI가 불리하거나 0을 포함하는 결과도 그대로 보고한다.

## 13. Raw-data validation

통계와 figure를 만들기 전에 자동 validator를 작성하거나 기존 validator를 확장해
다음을 검사하라.

- expected episode/summary row count
- method/scenario/training-seed/evaluation-seed uniqueness
- 누락/중복 seed
- 모든 method가 동일 evaluation seed 집합 사용
- scenario order/name 동일
- primary metric NaN/Inf 검사
- delivered + dropped 또는 terminal outcome accounting 일관성
- drop reason 합계
- connected-pair denominator
- success-delay denominator
- energy denominator
- checkpoint/config hash 일치
- smoke/full 혼입 없음
- Fast/Top-2 standard outcome equality
- Exact fused/legacy semantic replay equality

검증 실패 시 figure와 final table 생성을 중단하고 실패 보고서를 작성하라.

## 14. 최종 figure

모든 figure는 colorblind-safe palette, 일관된 method color/order, 읽을 수 있는 font,
seed-level 95% CI를 사용하고 PNG 300 dpi와 PDF/SVG vector를 함께 저장하라.

필수 figure:

1. `fig1_connected_pdr_by_scenario`
   - 14 scenario 또는 scenario family facet
   - baseline과 SwitchGLOBE Exact
   - seed-level CI
2. `fig2_pdr_latency_energy_pareto`
   - x: CPU p95 decision latency
   - y: connected-pair PDR
   - color/size: energy per delivered packet
   - Exact, Fast, Fast+Top-2 포함
3. `fig3_success_delay_distribution`
   - 성공 packet만 사용
   - CDF 또는 box/violin
   - delivered sample count 표시
4. `fig4_ablation_four_panel`
   - connected PDR
   - deadline delivery ratio
   - p95 success delay
   - energy per delivered packet
5. `fig5_latency_decomposition`
   - legacy, exact, fast, fast+top2
   - preprocess/model/action extraction/end-to-end 구분

Freshness cache figure는 supplementary에만 저장하고 main Pareto에 넣지 마라.

## 15. 최종 table

CSV, Markdown, LaTeX 세 형식으로 생성하라.

Table 1 — external baseline comparison:

- Method
- Connected-pair PDR
- Overall PDR
- Deadline ratio
- P95 success delay
- Energy per delivered packet
- CPU P95 decision latency
- Policy input bytes

Table 2 — ablation:

- Variant
- Connected PDR
- Deadline ratio
- P95 delay
- Energy per delivered packet
- Switch rate
- Backup availability/failover rate

Table 3 — model/runtime cost:

- Variant
- Parameter count
- Checkpoint bytes
- Input bytes
- Mean/P50/P95/P99 latency
- Decisions per second

각 숫자 옆에는 적절한 seed-level uncertainty를 표시하고, best/second-best 강조는
metric 방향을 고려하여 자동 계산하라. 통계적으로 불분명한 결과를 단순 bold로
과장하지 마라.

## 16. 논문 수치 provenance

다음을 생성하라.

`artifacts/final_paper_simulation/synthesis/manuscript_number_map.csv`

각 row에는 다음을 넣는다.

- manuscript claim ID
- metric
- method/contrast
- scenario scope
- displayed value
- raw CSV path
- source columns
- aggregation function
- training seeds
- config/checkpoint hashes

모든 table cell과 figure point를 raw CSV로 역추적할 수 있어야 한다. 숫자를 Markdown에
직접 중복 입력하지 말고 가능한 한 CSV에서 자동 생성하라.

## 17. 최종 산출물 구조

```text
artifacts/final_paper_simulation/
  preflight/
  smoke/
  full/
    baselines/
    ablation/
    latency/
  supplementary/
    freshness_cache/
    top2_failover/
  synthesis/
    raw/
    summaries/
    statistics/
    tables/
    figures/
    validation/
    manuscript_number_map.csv
    final_results_report.md
    manifest.json
```

## 18. 완료 조건과 최종 보고

다음이 모두 충족되어야 complete로 표시하라.

- 5 training seeds 모두 존재
- 14 scenarios 모두 존재
- method별 200 episodes/scenario 완료
- expected row count 일치
- 모든 validator 통과
- paired statistics 생성
- final table/figure 생성
- raw-to-manuscript provenance 생성
- 전체 tests 통과
- incomplete/resumed run이 최종 결과에 섞이지 않음

최종 보고서에는 다음을 명확히 구분하라.

1. 검증된 사실
2. acceptance gate 통과/실패
3. Exact 대비 Fast의 성능/latency trade-off
4. Top-2의 표준 routing 동일성과 synthetic failover 결과
5. baseline 대비 효과와 CI
6. worst seed/scenario
7. simulator limitation
8. 논문에서 주장해도 되는 문장
9. 논문에서 주장하면 안 되는 문장
10. 재실행 명령

장시간 작업 중 30~60분 이상 출력이 없지 않도록 진행률을 seed/method/scenario 단위로
기록하라. 실패한 chunk는 다른 완료 결과를 덮어쓰지 말고 resume 가능한 상태로
보존하라.

우선 지금은 Phase 0 preflight audit까지만 수행하고 결과를 보고한 뒤, smoke simulation
실행 전 나의 승인을 기다려라.

---
