# Phase 10 Colab Upload Package

이 패키지는 Phase 10 external RL baseline 평가를 Colab에서 바로 실행하기 위한 최소 구성입니다.

## 포함된 것

- `implementations/lite_globe/`: Phase 10 코드와 Colab notebook
- `tests/lite_globe/`: smoke 검증용 테스트
- `artifacts/lite_globe/phase8/checkpoints/`: Phase 8 Geo-Residual KD 비교용 checkpoint
- `requirements-lite-globe.txt`
- `pyproject.toml`

## Colab 실행 순서

Google Drive의 `MyDrive` 아래에서 zip을 풀면 다음 구조가 됩니다.

```text
/content/drive/MyDrive/GLOBE_ROUTING
```

Colab 첫 셀:

```python
from google.colab import drive
drive.mount('/content/drive')
```

압축 해제:

```bash
%cd /content/drive/MyDrive
!unzip -q phase10_colab_upload.zip
```

설치:

```bash
%cd /content/drive/MyDrive/GLOBE_ROUTING
!python -m pip install -q -r requirements-lite-globe.txt
!python -m pip install -q -e .
```

먼저 smoke:

```bash
!python -m implementations.lite_globe.run_phase10 \
  --smoke \
  --resume \
  --output-dir artifacts/lite_globe/phase10_smoke
```

문제 없으면 full:

```bash
!python -m implementations.lite_globe.run_phase10 \
  --resume \
  --output-dir artifacts/lite_globe/phase10
```

결과 압축:

```bash
!zip -r phase10_external_rl_results.zip artifacts/lite_globe/phase10 > /dev/null
!ls -lh phase10_external_rl_results.zip
```

## 결과 확인 위치

```text
artifacts/lite_globe/phase10/
├── raw/
├── summaries/
├── tables/
├── figures/
├── checkpoints/
└── manifest.json
```

논문용 주요 파일:

- `tables/external_rl_main_results.md`
- `tables/phase8_improvement_over_external_rl.md`
- `figures/external_rl_pdr.svg`
- `figures/external_rl_delay_p95.svg`
- `figures/external_rl_energy.svg`
- `figures/external_rl_input_bytes.svg`
