# Phase 11 Colab Upload Package

이 패키지는 Lite-GLOBE-P 최종 후보를 Colab에서 학습·검증하기 위한 최소 구성입니다.

## 포함된 것

- `implementations/lite_globe/`: Phase 11 코드와 Colab notebook
- `tests/lite_globe/`: smoke 검증용 테스트
- `artifacts/lite_globe/phase7/checkpoints/`: Global Teacher checkpoint
- `artifacts/lite_globe/phase8/checkpoints/`: Phase 8 Geo-Residual KD checkpoint
- `requirements-lite-globe.txt`
- `pyproject.toml`

## Colab 실행 순서

Google Drive의 `MyDrive` 아래에서 zip을 풀면 다음 구조가 됩니다.

```text
/content/drive/MyDrive/ResearchAIWorkspace
```

Colab 첫 셀:

```python
from google.colab import drive
drive.mount('/content/drive')
```

압축 해제:

```bash
%cd /content/drive/MyDrive
!unzip -q phase11_colab_upload.zip
```

설치:

```bash
%cd /content/drive/MyDrive/ResearchAIWorkspace
!python -m pip install -q -r requirements-lite-globe.txt
!python -m pip install -q -e .
```

먼저 smoke:

```bash
!python -m implementations.lite_globe.run_phase11 \
  --smoke \
  --resume \
  --output-dir artifacts/lite_globe/phase11_smoke
```

문제 없으면 full:

```bash
!python -m implementations.lite_globe.run_phase11 \
  --resume \
  --output-dir artifacts/lite_globe/phase11
```

결과 압축:

```bash
!zip -r phase11_lite_globe_p_results.zip artifacts/lite_globe/phase11 > /dev/null
!ls -lh phase11_lite_globe_p_results.zip
```

## 결과 확인 위치

```text
artifacts/lite_globe/phase11/
├── raw/
├── summaries/
├── tables/
├── figures/
├── checkpoints/
└── manifest.json
```

논문용 주요 파일:

- `tables/lite_globe_p_results.md`
- `tables/lite_globe_p_paired_effects.md`
- `figures/lite_globe_p_pdr.svg`
- `figures/lite_globe_p_delay_p95.svg`
- `figures/lite_globe_p_input_bytes.svg`

## 해석 주의

Phase 11의 최종 목표는 Phase 8의 일반 OOD 성능을 유지하면서 predictive break
성능을 회복하는 것입니다. Full run 후에는 Phase 10 external RL baseline 결과와
scenario별로 병합해 최종 우월성 주장을 판단해야 합니다.
