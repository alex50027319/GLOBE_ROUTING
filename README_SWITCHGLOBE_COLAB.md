# SwitchGLOBE Colab 실행

이 번들은 최종 알고리즘 **SwitchGLOBE**를 처음부터 학습하거나 기존 Phase 8/11
체크포인트에서 calibration·평가하기 위한 소스 전용 패키지다.

## 번들 생성

```bash
python scripts/package_switchglobe_colab.py
```

생성물: `artifacts/switchglobe_colab_bundle.zip`

## Colab 설치

```bash
%cd /content
!unzip -q /content/drive/MyDrive/switchglobe_colab_bundle.zip -d SwitchGLOBE
%cd /content/SwitchGLOBE
!python -m pip install -q -r requirements-lite-globe.txt
!python -m pip install -q -e .
```

## 전체 학습 계보 실행

먼저 smoke로 네 단계를 확인한다.

```bash
!python scripts/train_switchglobe_pipeline.py --device cuda --smoke --resume
```

정상 확인 후 full run을 실행한다.

```bash
!python scripts/train_switchglobe_pipeline.py --device cuda --resume
```

실행 순서는 다음과 같다.

1. PPO Global Teacher와 기초 KD Student 학습(역사적 Phase 7)
2. Geo-Residual Student 학습(역사적 Phase 8)
3. Predictive Student 학습(역사적 Phase 11)
4. SwitchGLOBE 위험 전환 보정과 5-seed 평가(역사적 Phase 12)

## 기존 체크포인트에서 최종 단계만 실행

Phase 8/11 체크포인트를 아래 기본 경로에 배치하거나 명시적으로 전달한다.

```bash
!python -m implementations.lite_globe.run_switchglobe \
  --device cuda \
  --resume \
  --phase8-checkpoint-dir artifacts/switchglobe/training/geo_residual/checkpoints \
  --phase11-checkpoint-dir artifacts/switchglobe/training/predictive/checkpoints \
  --output-dir artifacts/switchglobe/final
```

최종 결과에는 raw episode, seed summary, 통계, paired effect, 표, 그림,
`switchglobe.pt` checkpoint 및 manifest가 포함된다. 논문 수치는 반드시 5개 seed가
모두 존재하는 full manifest와 raw CSV를 검증한 뒤 사용한다.
