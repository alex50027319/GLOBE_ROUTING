# FastSwitchGLOBE vs external baselines: Colab full-run manual

이 절차는 기존 7-method full 결과를 다시 계산하지 않는다. 각 seed에 대해
FastSwitchGLOBE를 seed-matched SwitchGLOBE Exact에서 distill한 뒤, 기존 외부 baseline
실험과 동일한 14개 scenario, 동일한 200 evaluation seed/scenario로 2,800 episode를
평가한다. 마지막에 기존 ZIP과 결합하여 다음 8개 방법의 CSV, 통계, 표, figure를 만든다.

- AODV
- OLSR
- Greedy Geographic
- Evo-QGeo (Adapted)
- RDQN-HERP (Adapted)
- GAT-GRU-DDQN
- SwitchGLOBE
- FastSwitchGLOBE

`FastSwitchGLOBE`는 single-pass distilled 모델이며 Top-2 failover와 freshness cache를
사용하지 않는다. Smoke 결과는 full 결과와 병합할 수 없다.

## 0. 로컬: 사전 점검과 Colab 번들 생성

저장소 루트에서 실행한다.

```bash
cd /Users/alex/Documents/GLOBE_ROUTING
python -m pytest \
  tests/lite_globe/test_fast_external_comparison.py \
  tests/lite_globe/test_external_comparison.py
python scripts/package_external_comparison_colab.py \
  --output artifacts/fast_external_comparison_colab_bundle.zip
unzip -t artifacts/fast_external_comparison_colab_bundle.zip
```

생성된 `artifacts/fast_external_comparison_colab_bundle.zip`을 Google Drive의
`MyDrive/SwitchGLOBE/`에 업로드한다. 번들에는 소스와 SwitchGLOBE Exact checkpoint
5개가 들어 있다.

## 1. Colab: A100 runtime 확인

Colab에서 **Runtime → Change runtime type → A100 GPU**를 선택하고 다음 셀을 실행한다.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!nvidia-smi
```

`NVIDIA A100`이 표시되지 않으면 full run을 시작하지 않는다.

## 2. Colab: 소스 설치

새 세션마다 실행해도 안전하다.

```bash
%cd /content
!rm -rf /content/SwitchGLOBE_colab
!mkdir -p /content/SwitchGLOBE_colab
!unzip -q /content/drive/MyDrive/SwitchGLOBE/fast_external_comparison_colab_bundle.zip \
  -d /content/SwitchGLOBE_colab
%cd /content/SwitchGLOBE_colab
!python -m pip install -q -r requirements-lite-globe.txt
!python -m pip install -q -e .
```

`rm -rf` 대상은 반드시 위의 `/content/SwitchGLOBE_colab` 정확한 경로만 사용한다.

## 3. Colab: smoke gate

full과 분리된 Drive 디렉터리에 seed 42 smoke를 실행한다.

```bash
!python -m implementations.lite_globe.run_fast_external_comparison \
  --device cuda \
  --smoke \
  --seed 42 \
  --resume \
  --zip-results \
  --output-dir /content/drive/MyDrive/SwitchGLOBE/fast_external_comparison
```

Smoke manifest를 검사한다.

```python
import json
from pathlib import Path

p = Path('/content/drive/MyDrive/SwitchGLOBE/fast_external_comparison/smoke/seeds_42/manifest.json')
m = json.loads(p.read_text())
assert m['complete'] is True
assert m['mode'] == 'smoke'
assert m['methods'] == ['FastSwitchGLOBE']
assert m['training_seeds'] == [42]
assert m['episode_rows'] == m['expected_episode_rows'] == 42
assert m['seed_summary_rows'] == 14
print('SMOKE VERIFIED')
```

## 4. Colab: seed별 full run

한 번에 seed 하나만 실행한다. 결과와 checkpoint는 Drive에 직접 기록되므로 세션이
끊겨도 같은 명령에 `--resume`을 붙여 다시 실행하면 checkpoint부터 재사용한다.

### Seed 42

```bash
!python -m implementations.lite_globe.run_fast_external_comparison \
  --device cuda --seed 42 --resume --zip-results \
  --output-dir /content/drive/MyDrive/SwitchGLOBE/fast_external_comparison
```

### Seed 77

```bash
!python -m implementations.lite_globe.run_fast_external_comparison \
  --device cuda --seed 77 --resume --zip-results \
  --output-dir /content/drive/MyDrive/SwitchGLOBE/fast_external_comparison
```

### Seed 123

```bash
!python -m implementations.lite_globe.run_fast_external_comparison \
  --device cuda --seed 123 --resume --zip-results \
  --output-dir /content/drive/MyDrive/SwitchGLOBE/fast_external_comparison
```

### Seed 314

```bash
!python -m implementations.lite_globe.run_fast_external_comparison \
  --device cuda --seed 314 --resume --zip-results \
  --output-dir /content/drive/MyDrive/SwitchGLOBE/fast_external_comparison
```

### Seed 2718

```bash
!python -m implementations.lite_globe.run_fast_external_comparison \
  --device cuda --seed 2718 --resume --zip-results \
  --output-dir /content/drive/MyDrive/SwitchGLOBE/fast_external_comparison
```

각 명령은 다음 파일을 생성한다.

```text
/content/drive/MyDrive/SwitchGLOBE/fast_external_comparison/full/fast_seeds_<SEED>.zip
```

## 5. Colab: 5개 ZIP 일괄 검증

```python
import json, zipfile
from pathlib import Path

root = Path('/content/drive/MyDrive/SwitchGLOBE/fast_external_comparison/full')
seeds = [42, 77, 123, 314, 2718]
for seed in seeds:
    zpath = root / f'fast_seeds_{seed}.zip'
    assert zpath.is_file(), zpath
    with zipfile.ZipFile(zpath) as zf:
        assert zf.testzip() is None
        m = json.loads(zf.read('manifest.json'))
    assert m['complete'] is True
    assert m['mode'] == 'full'
    assert m['methods'] == ['FastSwitchGLOBE']
    assert m['training_seeds'] == [seed]
    assert m['episode_rows'] == m['expected_episode_rows'] == 2800
    assert m['seed_summary_rows'] == m['expected_seed_summary_rows'] == 14
    assert m['training_rows'] == 1
    assert m['deployment_cost_rows'] == 1
    print(seed, 'VERIFIED', zpath.stat().st_size)
```

## 6. 로컬: ZIP 배치

Drive에서 다음 5개 파일을 다운로드하여 로컬 디렉터리에 둔다.

```text
artifacts/fast_external_comparison_colab_results/fast_seeds_42.zip
artifacts/fast_external_comparison_colab_results/fast_seeds_77.zip
artifacts/fast_external_comparison_colab_results/fast_seeds_123.zip
artifacts/fast_external_comparison_colab_results/fast_seeds_314.zip
artifacts/fast_external_comparison_colab_results/fast_seeds_2718.zip
```

기존 7-method ZIP은 다음 위치에 있어야 한다.

```text
artifacts/external_comparison_colab_results/seeds_42.zip
artifacts/external_comparison_colab_results/seeds_77.zip
artifacts/external_comparison_colab_results/seeds_123.zip
artifacts/external_comparison_colab_results/seeds_314.zip
artifacts/external_comparison_colab_results/seeds_2718.zip
```

## 7. 로컬: 8-method 병합, 통계, figure 생성

```bash
cd /Users/alex/Documents/GLOBE_ROUTING
python scripts/merge_fast_external_comparison.py \
  --baseline-zip-dir artifacts/external_comparison_colab_results \
  --fast-zip-dir artifacts/fast_external_comparison_colab_results \
  --output-dir artifacts/final_paper_simulation/full/baselines_with_fast
```

병합기는 다음 조건 중 하나라도 어기면 실패한다.

- 10개 source ZIP 모두 `complete=true`, `mode=full`
- 각 seed Fast row가 정확히 2,800개
- 8개 method × 14 scenario × 5 seed × 200 episode = 112,000 rows
- duplicate episode key 0개
- FastSwitchGLOBE와 SwitchGLOBE의 scenario/training-seed/evaluation-seed key 완전 일치

최종 산출물:

```text
artifacts/final_paper_simulation/full/baselines_with_fast/
  manifest.json
  validation_report.md
  raw/episodes.csv
  raw/seed_summaries.csv
  raw/training.csv
  raw/deployment_costs.csv
  summaries/statistics.csv
  summaries/paired_effects.csv
  tables/external_comparison.md
  figures/*.png
  figures/*.pdf
  figures/*.svg
```

`paired_effects.csv`의 `proposed_method` 열로 SwitchGLOBE와 FastSwitchGLOBE의
각 외부 baseline 대비 paired 결과를 구분한다. 원고에는 병합 완료 후
`validation_report.md`와 `manifest.json`을 통과한 full 결과만 사용한다.
