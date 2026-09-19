# SwitchGLOBE latency 실험·재현 프로토콜

## 1. 범위

이 문서는 `refactor/globev2`에서 수행한 decision-latency 최적화의 재현 절차를 정의한다. smoke 결과와 full 결과를 혼합하지 않으며, 기존 full simulation artifact와 checkpoint를 덮어쓰지 않는다.

## 2. 고정 조건

- source commit: `92d17df3f4451e0412858a9927a898a2696023b3`
- training seeds: `42, 77, 123, 314, 2718`
- scenario observation: 각 seed의 `phase9_evaluation_scenarios(seed)[0]`, reset seed `1099999`
- batch size: 1
- torch threads: 1
- warm-up: 50
- measured repetitions: local 500, A100 2,000
- CUDA timing: 매 timed call 전후 `torch.cuda.synchronize`
- primary statistic: seed별 end-to-end p95의 paired difference
- auxiliary: mean, p50, p99, throughput, cold-start, component profile, parameter count, checkpoint bytes, input bytes

## 3. 데이터·체크포인트 무결성

Colab 전송 archive 해시:

| Artifact | SHA-256 |
|---|---|
| source ZIP | `b0aa2e386c7339e9d1f0b0097bc5bf9eee1247ee86084466da0eeb18d64041fe` |
| Exact phase12 checkpoints | `6e8c4edc9f435db8f4458c191e82bd41983e7976500e3dd02e1b39601a5a262c` |
| Fast checkpoints | `83e03ba9536a7cf2031d69d4456dfd53a05d84cf40cffba939d3f29dfc5ae324` |
| returned A100 result ZIP | `07bbd3abb9991ed6293d8326926ea2f154453db47cc76c1923da709688f12b41` |

원격 manifest는 실제로 읽은 Exact/Fast seed별 checkpoint SHA-256 10개를 별도로 기록한다.

## 4. 로컬 실행

```bash
.venv/bin/python -m pytest -q tests/lite_globe

MPLCONFIGDIR=/tmp/switchglobe-matplotlib \
.venv/bin/python -m implementations.lite_globe.run_latency_benchmark \
  --checkpoint-dir ResearchAIWorkspace/artifacts/lite_globe/phase12/checkpoints \
  --fast-checkpoint-dir artifacts/switchglobe_latency_optimization/fast_switchglobe/checkpoints \
  --output-dir artifacts/switchglobe_latency_optimization/globev2_local_full_20260905 \
  --include-fast \
  --include-early-exit \
  --warmup 50 \
  --repeats 500 \
  --zip-results
```

## 5. Early Exit calibration과 full trajectory 검증

Calibration은 Exact replay에서 margin `0, .01, .02, .05, .10, .20`을 sweep한다. 실제 채택 margin은 observed divergence가 0인 `0`으로 고정한다.

```bash
.venv/bin/python -m implementations.lite_globe.run_early_exit_validation \
  --checkpoint-dir ResearchAIWorkspace/artifacts/lite_globe/phase12/checkpoints \
  --exact-episodes artifacts/gated_switchglobe/calibration_guarded_20260905/raw/episodes.csv \
  --output-dir artifacts/gated_switchglobe/early_exit_full_validation_20260905 \
  --episodes 200
```

검증 비교 키:

- training seed, scenario, episode index
- delivered, dropped, terminal reason
- steps, hops, transmission attempts
- deadline outcome

## 6. Colab CLI 준비

공식 CLI는 Python 3.12 이상이 필요하다. 프로젝트 내부 전용 환경을 사용했다.

```bash
/path/to/python3.12 -m venv .colab-cli-venv
.colab-cli-venv/bin/python -m pip install google-colab-cli
.colab-cli-venv/bin/python -m pip install \
  --upgrade git+https://github.com/googlecolab/jupyter-kernel-client.git
.colab-cli-venv/bin/colab version
```

`google-colab-cli==0.6.0`의 표준 pip 해석은 PyPI의 동명 `jupyter-kernel-client==1.0.2`를 선택해 `KernelClient`가 없는 오류를 만들 수 있었다. 공식 CLI `pyproject.toml`이 지정한 GoogleColab Git 저장소 버전으로 교체해 해결했다.

## 7. Colab A100 실행

다른 세션과 충돌하지 않는 고유 이름을 사용한다.

```bash
.colab-cli-venv/bin/colab new \
  -s switchglobe-globev2-20260905-a100 --gpu A100

.colab-cli-venv/bin/colab status \
  -s switchglobe-globev2-20260905-a100
```

status의 `Hardware: A100`만으로 끝내지 않고 실행 스크립트가 `torch.cuda.get_device_name(0)`에 `A100`이 포함되는지 다시 assert한다.

```bash
.colab-cli-venv/bin/colab upload \
  -s switchglobe-globev2-20260905-a100 \
  artifacts/switchglobe_latency_optimization/globev2_colab_cli_20260905/_bundle/switchglobe_globev2_source.zip \
  /content/switchglobe_globev2_source.zip

.colab-cli-venv/bin/colab upload \
  -s switchglobe-globev2-20260905-a100 \
  artifacts/final_paper_simulation/full/final_latency_verified/_bundle/phase12_checkpoints.tar.gz \
  /content/phase12_checkpoints.tar.gz

.colab-cli-venv/bin/colab upload \
  -s switchglobe-globev2-20260905-a100 \
  artifacts/final_paper_simulation/full/final_latency_verified/_bundle/fast_checkpoints.tar.gz \
  /content/fast_checkpoints.tar.gz

.colab-cli-venv/bin/colab exec \
  -s switchglobe-globev2-20260905-a100 \
  -f scripts/colab/setup_switchglobe_a100.py \
  --timeout 600

.colab-cli-venv/bin/colab exec \
  -s switchglobe-globev2-20260905-a100 \
  -f scripts/colab/run_switchglobe_a100_latency.py \
  --timeout 1800
```

결과 회수와 종료:

```bash
.colab-cli-venv/bin/colab download \
  -s switchglobe-globev2-20260905-a100 \
  /content/switchglobe_globev2_a100_20260905.zip \
  artifacts/switchglobe_latency_optimization/globev2_colab_cli_20260905/switchglobe_globev2_a100_20260905.zip

.colab-cli-venv/bin/colab log \
  -s switchglobe-globev2-20260905-a100 \
  -o artifacts/switchglobe_latency_optimization/globev2_colab_cli_20260905/colab_cli_session.jsonl

.colab-cli-venv/bin/colab log \
  -s switchglobe-globev2-20260905-a100 \
  -o notebooks/switchglobe_latency_a100_colab_cli.ipynb

.colab-cli-venv/bin/colab stop \
  -s switchglobe-globev2-20260905-a100
```

세션은 결과와 로그를 내려받은 즉시 종료한다. 현재 실험의 원격 실행 시간은 약 775.94초였다.

## 8. 통계 분석

```bash
.venv/bin/python -m implementations.lite_globe.analyze_latency_optimization \
  --local-csv artifacts/switchglobe_latency_optimization/globev2_local_full_20260905/runtime_benchmarks.csv \
  --a100-csv artifacts/switchglobe_latency_optimization/globev2_colab_cli_20260905/results/runtime_benchmarks.csv \
  --output-csv artifacts/switchglobe_latency_optimization/globev2_final_20260905/metrics/paired_latency_statistics.csv \
  --bootstrap-resamples 10000 \
  --random-seed 20260905
```

각 candidate의 seed별 감소율은 다음과 같다.

\[
r_i=100\frac{T_{Exact,i}-T_{Candidate,i}}{T_{Exact,i}}
\]

보고값은 \(\bar r\), Student-t 95% CI, seed bootstrap 95% percentile CI다. exact sign-flip은 latency 차이 \(d_i=T_{Candidate,i}-T_{Exact,i}\)의 가능한 32개 부호 조합을 전부 사용한다.

## 9. Figure 생성

```bash
MPLCONFIGDIR=/tmp/switchglobe-matplotlib \
.venv/bin/python -m implementations.lite_globe.generate_latency_optimization_figures \
  --output-dir artifacts/switchglobe_latency_optimization/globev2_final_20260905
```

20개 figure 각각에 PNG 240 dpi, SVG, figure-level CSV를 생성한다. `FIGURE_INDEX.md`에 데이터 범위와 해석 제한이 기록된다.

## 10. 완료 검증 체크리스트

- [x] A100 이름을 API status와 PyTorch 양쪽에서 확인
- [x] source/checkpoint archive SHA-256 일치
- [x] 5개 seed 모두 존재
- [x] warm-up 50, repetitions 2,000
- [x] CUDA synchronization 포함
- [x] runtime benchmark rows 220
- [x] deployment cost rows 50
- [x] returned ZIP SHA-256 일치
- [x] session JSONL과 notebook export
- [x] Colab 세션 종료
- [x] local full test suite 통과
- [x] 20 PNG + 20 SVG 생성 및 핵심 figure 시각 점검

## 11. 확장 측정 run의 실패 기록

`29a8b4f`에서 추가한 P90/max/std/CV/peak-memory benchmark는 confirmatory A100 세션에서 장시간 실행 중 backend가 세션을 회수하여 CSV/ZIP을 만들지 못했다. 이 negative result는 `artifacts/switchglobe_latency_optimization/globev2_colab_cli_20260905/confirmatory_failure_20260905.md`에 기록했다. primary A100 수치와 혼합하지 않으며, 확장 코드 자체는 local·향후 짧은 split run에서 재사용할 수 있다.

짧은 operator profile은 별도 A100 세션에서 성공했고, 결과는 `.../profile/results/` 아래에 보관했다. profile event 합계는 latency benchmark의 p95가 아니다.

## 12. 해석 금지 사항

- smoke 결과를 full 결과처럼 보고하지 않는다.
- 개별 2,000회 decision을 독립적인 statistical replicate로 취급하지 않는다.
- `energy_per_delivered_packet`을 실측 Joule이라고 부르지 않는다.
- `policy_input_bytes`를 무선 control overhead라고 부르지 않는다.
- A100 latency를 onboard UAV latency로 일반화하지 않는다.
- Fast의 latency 개선만으로 최종 기법 채택을 주장하지 않는다.
- Early Exit의 관찰된 0 mismatch를 모든 가능한 상태의 수학적 동치 증명으로 표현하지 않는다.
