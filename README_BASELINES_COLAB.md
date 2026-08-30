# External comparison: resumable Colab chunks

Main roster는 AODV, OLSR, Greedy Geographic, Evo-QGeo (Adapted),
RDQN-HERP (Adapted), GAT-GRU-DDQN, SwitchGLOBE다. SwitchGLOBE checkpoint는
읽기 전용으로 로드되며 baseline마다 독립적인 원자적 checkpoint가 생성된다.

## 번들 생성과 설치

```bash
python scripts/package_external_comparison_colab.py
```

생성된 `artifacts/external_comparison_colab_bundle.zip`을 Drive에 올린 뒤 Colab에서:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content
!unzip -q /content/drive/MyDrive/external_comparison_colab_bundle.zip -d SwitchGLOBE
%cd /content/SwitchGLOBE
!python -m pip install -q -r requirements-lite-globe.txt
!python -m pip install -q -e .
```

## Smoke와 seed별 full chunk

Smoke와 full 결과는 CLI가 서로 다른 하위 디렉터리에 강제로 분리한다.

```bash
!python -m implementations.lite_globe.run_external_comparison \
  --device cpu --smoke --seed 42 --resume --zip-results
```

GPU session 하나당 seed 하나만 실행한다. 예를 들어 seed 42:

```bash
!python -m implementations.lite_globe.run_external_comparison \
  --device cuda --seed 42 --resume --zip-results
```

나머지 session은 `--seed 77`, `123`, `314`, `2718`로 실행한다. 같은 명령을 다시
실행하면 complete checkpoint와 manifest를 검사해 완료 chunk를 재학습하지 않는다.
불완전 checkpoint 또는 `complete: false` manifest는 완료 산출물로 인정하지 않는다.

결과 ZIP은 `artifacts/external_comparison/{smoke|full}/seeds_<seed>.zip`이다.
병합 전 각 ZIP의 `manifest.json`에서 mode, method contracts, scenario, seed, row count를
검증한다. smoke 수치를 원고에 사용하지 않는다.
