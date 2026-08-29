# SwitchGLOBE

SwitchGLOBE는 동적 FANET에서 사용하는 **Global-to-Local policy distillation +
predictive risk switching** 라우팅 알고리즘이다. 연구 개발 당시 Phase 12로 불렸던
정책을 최종 알고리즘으로 고정하고, Phase 번호 대신 재현 가능한 학습 계보와 최종
실행 진입점을 중심으로 프로젝트를 정리했다.

Global Teacher는 전체 그래프를 관측하며 PPO로 학습된다. 배포 정책은 전역 그래프를
사용하지 않고 1-hop 관측만 사용한다. Teacher 지식은 Geo-Residual Student와
Predictive Student에 offline knowledge distillation으로 전달되며, SwitchGLOBE는 현재
후보의 margin·link lifetime·onward lifetime 위험에 따라 두 Student를 선택한다.

## 프로젝트 구조

```text
.
├── implementations/lite_globe/
│   ├── env/                    # FANET 환경, 이동성, 링크와 관측
│   ├── models/                 # PPO Teacher, local Students, SwitchGLOBE
│   ├── algorithms/             # PPO, KD, 선택적 local fine-tuning
│   ├── data/                   # Teacher rollout과 grouped split
│   ├── scenarios/              # 학습·보정·OOD·routing-hole 시나리오
│   ├── evaluation/             # PDR, delay, energy, 통계와 표/그림
│   ├── experiments/            # 필요한 학습 계보(7, 8, 11, 12)
│   ├── config/                 # 계보별 설정과 switchglobe.yaml
│   ├── colab/switchglobe.ipynb
│   └── run_switchglobe.py      # 최종 알고리즘 진입점
├── scripts/
│   ├── train_switchglobe_pipeline.py
│   └── package_switchglobe_colab.py
├── tests/lite_globe/           # 보존된 구현에 대응하는 테스트
├── docs/
│   ├── method_history.md       # Phase 1~12 개발 계보와 채택/제외 결정
│   └── paper_method_summary.md # 코드와 수식의 대응 관계
├── submission/ad_hoc_networks_overleaf/ # SwitchGLOBE 논문 초안
└── README_SWITCHGLOBE_COLAB.md
```

## 설치와 검증

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lite-globe.txt
python -m pip install -e .
pytest tests/lite_globe
```

## 처음부터 학습

```bash
python scripts/train_switchglobe_pipeline.py --device cpu --smoke --resume
python scripts/train_switchglobe_pipeline.py --device auto --resume
```

재현 계보는 PPO/KD foundation(역사적 Phase 7), Geo-Residual Student(Phase 8),
Predictive Student(Phase 11), SwitchGLOBE calibration/evaluation(Phase 12) 순서다.

## 기존 checkpoint로 최종 단계만 실행

```bash
python -m implementations.lite_globe.run_switchglobe \
  --device auto \
  --resume \
  --phase8-checkpoint-dir artifacts/switchglobe/training/geo_residual/checkpoints \
  --phase11-checkpoint-dir artifacts/switchglobe/training/predictive/checkpoints \
  --output-dir artifacts/switchglobe/final
```

Phase 12/11/8이라는 내부 이름은 기존 checkpoint와 연구 계보를 검증하기 위해 일부
모듈에 남겨 두었다. 공개 알고리즘명, 최종 checkpoint, 결과 표·그림에서는
`SwitchGLOBE`를 사용한다.

## 연구 무결성

- smoke 결과와 full 결과를 합치지 않는다.
- 5개 training seed와 동일한 paired evaluation seed를 확인한다.
- 원고의 수치는 raw CSV, manifest, 집계 통계가 일치할 때만 반영한다.
- Phase 13/P+는 최종 알고리즘에서 제외되었으며 이 브랜치의 주장을 뒷받침하지 않는다.
