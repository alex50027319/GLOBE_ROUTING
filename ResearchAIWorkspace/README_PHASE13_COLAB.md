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
/content/drive/MyDrive/ResearchAIWorkspace
```

노트북으로 실행하려면 다음 파일을 Colab에서 열면 됩니다.

```text
implementations/lite_globe/colab/phase13_risk_switch_lite_globe_p_plus.ipynb
```

압축 해제:

```bash
%cd /content/drive/MyDrive
!unzip -q phase13_colab_upload.zip
```

설치:

```bash
%cd /content/drive/MyDrive/ResearchAIWorkspace
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
