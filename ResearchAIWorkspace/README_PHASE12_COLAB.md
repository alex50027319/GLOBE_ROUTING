# Phase 12 Colab Upload Package

이 패키지는 Risk-Switch Lite-GLOBE-P 최종 후보를 Colab에서 calibration 및 평가하기
위한 최소 구성입니다.

## 포함된 것

- `implementations/lite_globe/`: Phase 12 코드와 Colab notebook
- `tests/lite_globe/`: smoke 검증용 테스트
- `artifacts/lite_globe/phase8/checkpoints/`: Phase 8 normal branch checkpoint
- `artifacts/lite_globe/phase11/checkpoints/`: Phase 11 predictive branch checkpoint
- `requirements-lite-globe.txt`
- `pyproject.toml`

## Colab 실행 순서

Google Drive의 `MyDrive` 아래에서 zip을 풀면 다음 구조가 됩니다.

```text
/content/drive/MyDrive/ResearchAIWorkspace
```

압축 해제:

```bash
%cd /content/drive/MyDrive
!unzip -q phase12_colab_upload.zip
```

설치:

```bash
%cd /content/drive/MyDrive/ResearchAIWorkspace
!python -m pip install -q -r requirements-lite-globe.txt
!python -m pip install -q -e .
```

먼저 smoke:

```bash
!python -m implementations.lite_globe.run_phase12 \
  --smoke \
  --resume \
  --output-dir artifacts/lite_globe/phase12_smoke
```

문제 없으면 full:

```bash
!python -m implementations.lite_globe.run_phase12 \
  --resume \
  --output-dir artifacts/lite_globe/phase12
```

결과 압축:

```bash
!zip -r phase12_risk_switch_results.zip artifacts/lite_globe/phase12 > /dev/null
!ls -lh phase12_risk_switch_results.zip
```

## 결과 확인 위치

```text
artifacts/lite_globe/phase12/
├── raw/
├── summaries/
├── tables/
├── figures/
├── checkpoints/
└── manifest.json
```

논문용 주요 파일:

- `tables/risk_switch_results.md`
- `tables/risk_switch_paired_effects.md`
- `figures/risk_switch_pdr.svg`
- `figures/risk_switch_delay_p95.svg`
- `figures/risk_switch_input_bytes.svg`
