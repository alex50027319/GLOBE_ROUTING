# Figure index

> 모든 그림은 PNG(240 dpi)와 SVG를 함께 제공한다. Smoke, local full, legacy A100 evidence는 캡션에서 구분한다.

## 01_baseline_component_latency

각 구성요소는 독립 타이머라 합이 end-to-end와 정확히 같지 않다. batch=1, 5 seeds, local CPU.

- Data: `metrics/figure_01.csv`

## 02_latency_summary

동일 checkpoint와 5 seeds에서 mean/P50/P95/P99를 비교한다. Fast-DiSwitch 계열은 낮지만 reliability gate는 별도 판단한다.

- Data: `metrics/figure_02.csv`

## 03_latency_ecdf

개별 decision을 독립 표본으로 부풀리지 않고 seed별 p95 5개로 그린 ECDF다.

- Data: `metrics/figure_03.csv`

## 04_latency_forest

양수는 DiSwitch 대비 빠름. Early Exit과 buffering은 CI가 0을 가로질러 개선으로 판정하지 않는다.

- Data: `metrics/figure_04.csv`

## 05_pareto_connected_pair_pdr

A100 latency는 2026-09-05 colab-cli 동일 세션, reliability는 5×14×200 full 결과다.

- Data: `metrics/figure_05.csv`

## 06_pareto_deadline_delivery_ratio

A100 latency는 2026-09-05 colab-cli 동일 세션, reliability는 5×14×200 full 결과다.

- Data: `metrics/figure_06.csv`

## 07_reliability_heatmap

5-seed scenario 평균. Fast의 열세가 OOD 대규모 노드 등 특정 조건에 집중되는지 보여준다.

- Data: `metrics/figure_07.csv`

## 08_latency_heatmap

14,000 episode 전수 실행 결과이나 same-session randomized benchmark가 아니므로 진단용이다.

- Data: `metrics/figure_08.csv`

## 09_seed_variability

모든 후보를 동일 5개 training seed에서 비교한다.

- Data: `metrics/figure_09.csv`

## 10_switch_activation

Switch가 켜지는 빈도와 predictive branch 계산 필요 빈도는 동일하지 않다.

- Data: `metrics/figure_10.csv`

## 11_gate_score_distribution

sweep의 누적 skip count 차이로 재구성한 구간 질량이며 원시 continuous score histogram은 아니다.

- Data: `metrics/figure_11.csv`

## 12_gate_calibration

margin=0만 전수 관찰에서 divergence 0이다. 양의 margin은 divergence를 유발해 채택하지 않는다.

- Data: `metrics/figure_12.csv`

## 13_branch_usage_latency

높은 skip 비율만으로 실제 p95 개선이 보장되지 않으며 runtime control overhead가 중요함을 보여준다.

- Data: `metrics/figure_13.csv`

## 14_model_footprint

Early Exit은 DiSwitch 가중치를 그대로 사용해 크기가 같고 Fast-DiSwitch는 7,011 parameters다.

- Data: `metrics/figure_14.csv`

## 15_policy_input_bytes

정책 입력 바이트이며 무선 control overhead와 동일한 지표가 아니다.

- Data: `metrics/figure_15.csv`

## 16_ablation_contribution

SwitchGLOBE 성능을 만든 구성요소의 seed-level 95% CI다.

- Data: `metrics/figure_16.csv`

## 17_device_comparison

2026-09-05 colab-cli 결과는 commit 92d17df의 현재 구현이다. 2026-08-29 번들은 독립 재현성 참고값으로만 병기한다.

- Data: `metrics/figure_17.csv`

## 18_worst_case_scenario

Fast-DiSwitch 손실이 OOD node-scale 조건에서 크게 나타나 최종 대체안으로 부적합하다.

- Data: `metrics/figure_18.csv`

## 19_early_exit_paired_difference

5 seeds 중 일관된 개선이 아니며 평균적으로 악화했다.

- Data: `metrics/figure_19.csv`

## 20_failed_candidate_tradeoff

오른쪽 위가 바람직하다. DiSwitch는 동작 동일성을 보존하면서 legacy 중복-forward 대비 2026-09-05 A100에서 개선됐다. Fast-DiSwitch는 reliability gate를 통과하지 못했다.

- Data: `metrics/figure_20.csv`
