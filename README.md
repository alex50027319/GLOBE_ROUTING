# Lite-GLOBE / GLOBE++ Routing

Decentralized FANET(Flying Ad-hoc Network) 라우팅을 위한 **Global-to-Local 정책 증류**
연구 구현체입니다. 전역 그래프 정보를 볼 수 있는 privileged teacher를 강화학습으로 먼저
학습하고, 1-hop 로컬 관측만 사용하는 경량 student로 지식을 증류한 뒤, 예측적 위험
지표로 안전 분기를 전환하는 Predictive Risk Switching(PRS)까지 단계적으로(Phase 1~13)
구현합니다.

> 이 브랜치(`feature/phase13`)는 리서치 노트/Obsidian vault/논문 LaTeX 빌드 등 실행에
> 불필요한 자료를 제거하고, 소스 코드와 재현에 필요한 최소한의 문서만 남긴 정리된
> 프로젝트 구조입니다. 원 자료는 `main` 브랜치에 남아 있습니다.

## 프로젝트 구조

```text
.
├── implementations/lite_globe/   # 핵심 구현 (env, models, algorithms, evaluation, run_phaseN.py)
├── tests/lite_globe/             # 결정적 pytest 테스트 스위트
├── scripts/                      # Google Colab 자동화 (패키징/실행/결과 병합)
├── docs/
│   └── paper_method_summary.md   # 문제 정식화(Dec-POMDP) 및 방법론 요약 (구현 참고용)
├── pyproject.toml                # 패키지 정의 (`implementations*`)
├── requirements-lite-globe.txt   # 실행/학습에 필요한 실제 의존성
├── requirements.txt              # 로컬 오케스트레이션용 (google-colab-cli)
├── colab_args.json               # colab_run.py가 생성/사용하는 원격 실행 설정
└── README_PHASE{10,11,12,13}_COLAB.md, README_COLAB_CLI.md
```

`implementations/lite_globe/`는 phase가 진행되며 공유 모듈(`env/`, `models/`,
`algorithms/`, `evaluation/`) 위에 각 단계 전용 `run_phaseN.py` / `config/phaseN.yaml`이
쌓이는 구조입니다. Phase별 구현 범위는 [`implementations/lite_globe/README.md`](implementations/lite_globe/README.md),
설계 가정은 [`implementations/lite_globe/assumptions.md`](implementations/lite_globe/assumptions.md)에
정리되어 있습니다.

연구 질문 정리부터 실험 계획, checkpoint 검증, multi-seed 실행·병합, 결과 감사,
원고와 투고 준비까지의 프로젝트 전용 절차는
[`docs/globe_routing_research_workflow_19_steps.md`](docs/globe_routing_research_workflow_19_steps.md)를 참고하세요.

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lite-globe.txt
python -m pip install -e .
```

## 로컬 실행

```bash
# 결정적 테스트
pytest

# Phase 1 CPU 기준선 스모크
python -m implementations.lite_globe.run_phase1 --episodes 5

# 최신 단계 (Phase 13: Risk-Switch Lite-GLOBE-P+) 스모크
python -m implementations.lite_globe.run_phase13 --smoke --device cpu --output-dir artifacts/lite_globe/phase13_smoke
```

각 `run_phaseN.py`의 인자와 필요한 이전 phase 체크포인트는 `implementations/lite_globe/README.md`를 참고하세요.

## 논문 방법론 참고

`docs/paper_method_summary.md`에 Dec-POMDP 문제 정식화와 Teacher PPO / Global-to-Local
Distillation / Predictive Risk Switching(P+) 수식을 구현 위치와 매핑해 정리했습니다. 전체
논문 원문(LaTeX, 그림, 참고문헌)은 이 브랜치에 포함하지 않습니다.

## Google Colab에서 실행

GPU/TPU가 필요한 학습·평가는 `google-colab-cli`를 통해 원격 Colab 인스턴스에서 실행할 수
있습니다. 전체 가이드는 [`README_COLAB_CLI.md`](README_COLAB_CLI.md), phase별 세부 가이드는
`README_PHASE10_COLAB.md` ~ `README_PHASE13_COLAB.md`를 참고하세요.

```bash
pip install -r requirements.txt   # google-colab-cli
colab auth login
python scripts/colab_run.py --phase 13 --gpu T4 --smoke
```

`scripts/package_phaseN_colab.py`는 코드/설정/테스트와 필요한 체크포인트만 모아 최소
번들을 만들고, `scripts/run_phase13_seed_queue.py` + `scripts/merge_phase13_artifacts.py`는
Phase 13을 seed 단위로 나눠 실행한 뒤 결과를 하나의 리포트로 병합합니다.
