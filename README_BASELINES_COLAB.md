# External Baseline Colab 실행

이 번들은 최종 **SwitchGLOBE**와 외부 routing baseline을 동일한 training seed,
scenario, evaluation seed에서 비교한다.

비교 대상은 GPSR, Predictive Geographic, Evo-QGeo, IQMR Q(lambda), DRAMA,
SwitchGLOBE다. Evo-QGeo, IQMR, DRAMA는 각 seed에서 학습하며, SwitchGLOBE는 검증된
최종 checkpoint를 읽기 전용으로 사용한다.

## 번들 생성

저장소 루트에서 다음을 실행한다.

```bash
python scripts/package_baselines_colab.py
```

생성물: `artifacts/baselines_colab_bundle.zip`

번들에는 소스, 테스트, 설정, notebook과 5개 SwitchGLOBE checkpoint가 포함된다.

## Colab 설치

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content
!unzip -q /content/drive/MyDrive/baselines_colab_bundle.zip -d SwitchGLOBE
%cd /content/SwitchGLOBE
!python -m pip install -q -r requirements-lite-globe.txt
!python -m pip install -q -e .
```

## 실행

먼저 smoke evaluation을 확인한다.

```bash
!python -m implementations.lite_globe.run_baselines \
  --device cuda \
  --smoke \
  --resume \
  --output-dir artifacts/baselines_smoke
```

그다음 5-seed full evaluation을 실행한다. 중단 후 같은 명령을 다시 실행하면 학습된
baseline checkpoint를 재사용한다.

```bash
!python -m implementations.lite_globe.run_baselines \
  --device cuda \
  --resume \
  --output-dir artifacts/baselines
```

## 결과 압축

```bash
!zip -r baseline_results.zip artifacts/baselines > /dev/null
!ls -lh baseline_results.zip
```

주요 논문용 파일:

- `tables/external_baseline_results.md`
- `tables/switchglobe_improvement_over_external_baselines.md`
- `summaries/statistics.csv`
- `summaries/paired_effects.csv`
- `figures/external_baseline_pdr.svg`
- `figures/external_baseline_delay_p95.svg`
- `figures/external_baseline_energy.svg`
- `manifest.json`

원고에는 5개 seed의 raw CSV, manifest, 집계 통계를 교차 검증한 결과만 반영한다.
