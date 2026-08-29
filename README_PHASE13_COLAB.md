# Phase 13 Colab Upload Package

이 패키지는 Risk-Switch Lite-GLOBE-P+를 Colab에서 calibration, full evaluation,
ablation까지 실행하기 위한 최소 구성입니다.

## 포함된 것

- `implementations/lite_globe/`: Phase 13 코드
- `tests/lite_globe/`: smoke 검증용 테스트
- `artifacts/lite_globe/phase8/checkpoints/`: Phase 8 normal branch checkpoint
- `artifacts/lite_globe/phase11/checkpoints/`: Phase 11 predictive branch checkpoint
- `artifacts/lite_globe/phase12/checkpoints/`: Phase 12 calibrated Risk-Switch checkpoint
- `requirements-lite-globe.txt`
- `pyproject.toml`

## Colab 실행 순서

Google Drive의 `MyDrive` 아래에서 zip을 풀면 다음 구조가 됩니다.

```text
/content/drive/MyDrive/GLOBE_ROUTING
```

노트북으로 실행하려면 다음 파일을 Colab에서 열면 됩니다.

```text
implementations/lite_globe/colab/phase13_risk_switch_lite_globe_p_plus.ipynb
```

압축 해제:

```bash
%cd /content/drive/MyDrive
!rm -rf GLOBE_ROUTING
!mkdir -p GLOBE_ROUTING
!unzip -q phase13_colab_bundle.zip -d GLOBE_ROUTING
```

설치:

```bash
%cd /content/drive/MyDrive/GLOBE_ROUTING
!python -m pip install -q -r requirements-lite-globe.txt
!python -m pip install -q -e .
```

먼저 smoke:

```bash
!python -m implementations.lite_globe.run_phase13 \
  --smoke \
  --device cuda \
  --resume \
  --output-dir artifacts/lite_globe/phase13_smoke
```

문제 없으면 full:

```bash
!python -m implementations.lite_globe.run_phase13 \
  --device cuda \
  --resume \
  --output-dir artifacts/lite_globe/phase13
```

## 짧고 복구 가능한 Colab 세션으로 실행 (권장)

Phase 13 full 설정은 seed마다 calibration 약 61,860 episodes와 evaluation
약 28,000 episodes를 계산한다. 기존 `--resume`은 calibration이 모두 끝난 뒤에만
checkpoint를 남겼으므로 세션이 중간에 사라지면 긴 구간을 다시 실행해야 했다.

기본 seed queue는 이제 다음 단위를 결과 ZIP에 저장한다.

- calibration candidate 1개
- evaluation의 scenario-method 조합 1개
- config, method 순서, scenario 순서를 확인하는 run signature

각 짧은 세션이 끝나면 partial ZIP을 로컬에 다운로드하고 세션을 종료한다. 다음
세션은 그 ZIP을 복원한 후 완료되지 않은 단위만 계산한다. candidate grid, seed,
episode 수, method, 평가 seed, 후보 선택 tie-break는 기존 full 실행과 동일하다.

전체 five-seed 실행:

```bash
python scripts/run_phase13_seed_queue.py \
  --seeds 42,77,123,314,2718 \
  --gpu A100 \
  --calibration-candidates-per-chunk 32 \
  --evaluation-units-per-chunk 20 \
  --exec-timeout 7200 \
  --detach
```

기본값으로 candidate 32개 또는 scenario-method 20개까지만 새로 처리한 뒤
세션을 반납한다. seed당 약 10개 chunk가 필요하다. 실제 세션 시간이 여전히
길면 각각 `16`, `10` 또는 `8`, `5`로 낮출 수 있다. 반대로 세션이 안정적이면
`64`, `40`으로 높여 업로드 횟수를 줄일 수 있다.

중단된 queue를 같은 명령으로 다시 실행하면 `phase13_seed_runs`의 마지막 유효
chunk ZIP에서 자동으로 이어간다. 각 seed가 끝나면 호환성을 위해 다음 최종 ZIP을
만든다.

```text
artifacts/lite_globe/phase13_seeds_42_results.zip
artifacts/lite_globe/phase13_seeds_77_results.zip
...
```

기존의 한 seed당 한 장시간 세션 방식이 꼭 필요하면 `--single-session`을 추가한다.

## Seed 단위 장시간 실행 (legacy)

이미 일부 seed 결과가 있으면 나머지 seed만 실행할 수도 있다. 아래 방식은 한
세션을 오래 점유하므로 현재는 권장하지 않는다.

로컬 `colab-cli`에서 실행:

```bash
python scripts/colab_run.py \
  --phase 13 \
  --gpu A100 \
  --session globe-phase13-seeds-123-314-2718 \
  --seeds 123,314,2718 \
  --exec-timeout 86400
```

위 명령의 결과 zip은 다음 경로에 저장된다.

```text
artifacts/lite_globe/phase13_seeds_123_314_2718_results.zip
```

여러 seed chunk를 합쳐 최종 Phase13 리포트를 다시 만들려면 다음 스크립트를 사용한다.

```bash
python scripts/merge_phase13_artifacts.py \
  --inputs \
    artifacts/lite_globe/phase13_seeds_42_77_results.zip \
    artifacts/lite_globe/phase13_seeds_123_314_2718_results.zip \
  --output-dir artifacts/lite_globe/phase13_merged
```

결과 압축:

```bash
!zip -r phase13_risk_switch_plus_results.zip artifacts/lite_globe/phase13 > /dev/null
!ls -lh phase13_risk_switch_plus_results.zip
```

## 결과 확인 위치

```text
artifacts/lite_globe/phase13/
├── raw/
├── summaries/
├── tables/
├── figures/
├── checkpoints/
└── manifest.json
```

논문용 주요 파일:

- `tables/risk_switch_plus_results.md`
- `tables/risk_switch_plus_paired_effects.md`
- `figures/risk_switch_plus_pdr.svg`
- `figures/risk_switch_plus_delay_p95.svg`
- `figures/risk_switch_plus_input_bytes.svg`
- `figures/risk_switch_plus_switch_steps.svg`

## 해석 주의

Phase13은 PDR만 보는 실험이 아니다.
`summaries/paired_effects.csv`에서 PDR, deadline, delay, energy, input bytes,
agent drop rate를 함께 확인해야 한다.
