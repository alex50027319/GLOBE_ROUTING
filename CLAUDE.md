# SwitchGLOBE Research Harness

이 저장소의 최종 제안기법은 동적 FANET을 위한 **SwitchGLOBE**다. 개발 중 사용한
Phase 번호는 재현 계보이며 공개 알고리즘명이 아니다. Phase 13/P+는 최종안에서
제외되었으므로 SwitchGLOBE 성능 주장이나 결과에 섞지 않는다.

## Source of Truth

1. `implementations/`, `tests/`, `scripts/`의 실행 코드와 YAML 설정
2. full-run raw CSV와 manifest
3. 자동 생성 통계·표·그림
4. `docs/`와 README
5. `submission/` 원고와 발표자료

충돌하면 상위 자료를 확인하고 추측으로 정합성을 만들지 않는다.

## Project Map

- `implementations/lite_globe/`: 환경, 모델, 학습, 시나리오, 평가, 실행 진입점
- `tests/lite_globe/`: 체크포인트 없이 실행 가능한 코드 검증
- `scripts/train_switchglobe_pipeline.py`: Teacher → Geo-Residual → Predictive → SwitchGLOBE
- `scripts/package_*_colab.py`: 소스·체크포인트 Colab 번들 생성
- `docs/`: 방법 계보, 수식-코드 대응, simulation protocol
- `submission/ad_hoc_networks_overleaf/`: 최종 논문 초안
- `artifacts/`: 생성 결과이며 소스가 아니다
- `ResearchAIWorkspace/`: 현재 브랜치에서 Git 제외된 legacy/local workspace

## Common Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lite-globe.txt
python -m pip install -e .
python -m pytest tests/lite_globe
python scripts/train_switchglobe_pipeline.py --device cpu --smoke --resume
```

## Non-Negotiable Rules

- smoke와 full 결과를 합치지 않는다.
- full 완료는 5개 seed `42, 77, 123, 314, 2718`, raw CSV, manifest로 검증한다.
- `overall_pdr`와 `connected_pair_pdr`의 denominator를 구분한다.
- energy는 simulator proxy이며 Joule로 부르지 않는다.
- `policy_input_bytes`는 routing-control overhead가 아니다.
- single-packet `delivery_rate_proxy`를 Mbps throughput으로 부르지 않는다.
- 외부 baseline campaign은 SwitchGLOBE 학습·calibration과 분리한다.
- `.env`, 기존 checkpoint, full-run artifact, legacy workspace를 수정하지 않는다.
- GPU/Colab full run은 사용자가 실행을 요청한 범위에서만 시작한다.

세부 규칙과 반복 워크플로는 `.claude/rules/`와 `.claude/skills/`를 따른다.

## Persistent Project Context

@.claude/context/project-map.md
@.claude/context/current-algorithm.md
@.claude/context/method-lineage.md
@.claude/context/experiment-contract.md
@.claude/context/metrics-glossary.md
@.claude/context/baseline-contract.md
