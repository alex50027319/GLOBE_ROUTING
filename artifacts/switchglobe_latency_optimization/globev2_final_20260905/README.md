# SwitchGLOBE globev2 latency evidence pack

이 디렉터리는 2026-09-05 decision-latency 최적화의 발표·논문용 figure와 파생 통계를 모은다.

## 핵심 결과

- Exact A100 CUDA p95: 5.836 ms
- Fast A100 CUDA p95: 2.005 ms, Exact 대비 65.65% 감소
- Fast+Top2 A100 CUDA p95: 2.092 ms, Exact 대비 64.16% 감소
- Early Exit A100 CUDA p95: 6.386 ms, Exact 대비 9.42% 악화
- Exact A100 host CPU p95: 2.208 ms
- Early Exit branch skip: 74.27%, full paired trajectory mismatch 0/14,000 episodes
- Fast connected-pair PDR: Exact 대비 -1.81 pp, reliability gate fail

## 구조

- `figures/`: 20개 PNG와 20개 SVG
- `metrics/figure_XX.csv`: 각 figure를 재생성할 수 있는 파생 데이터
- `metrics/paired_latency_statistics.csv`: seed-paired t-CI, bootstrap CI, exact sign-flip, Holm 결과
- `FIGURE_INDEX.md`: figure별 caption과 해석 제한
- `figure_manifest.json`: figure source artifact 목록

## 원천 artifact

- `../globev2_local_full_20260905/`: local full latency
- `../globev2_colab_cli_20260905/`: current-commit A100 결과 ZIP, 추출 결과, Colab CLI log, upload bundle
- `../../gated_switchglobe/calibration_guarded_20260905/`: Early Exit margin sweep
- `../../gated_switchglobe/early_exit_full_validation_20260905/`: 14,000-episode actual Early Exit execution
- `../../final_paper_simulation/full/ablation/`: Exact vs Fast full quality
- `../verified_candidate_1_2/`: buffering/compile 구 동일세션 검증

## 관련 문서

- `docs/latency_optimization/final_report.md`
- `docs/latency_optimization/literature_matrix.md`
- `docs/latency_optimization/candidate_designs.md`
- `docs/latency_optimization/experiment_protocol.md`
- `notebooks/switchglobe_latency_a100_colab_cli.ipynb`

## 해석 규칙

- smoke와 full을 혼합하지 않는다.
- A100과 local Mac을 직접 속도비로 일반화하지 않는다.
- Fast의 latency 개선은 reliability acceptance와 별개다.
- energy는 simulator proxy이며 Joule이 아니다.
- policy input bytes는 무선 control overhead가 아니다.
