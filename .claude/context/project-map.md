# Project Map

## Tracked source

- `implementations/lite_globe/env/`: FANET 상태, 이동성, 링크, queue, episode
- `implementations/lite_globe/models/`: Global Teacher와 local Student/Switch 정책
- `implementations/lite_globe/algorithms/`: PPO, KD, fine-tuning
- `implementations/lite_globe/data/`: Teacher rollout과 grouped split
- `implementations/lite_globe/scenarios/`: curriculum, calibration, OOD, routing traps
- `implementations/lite_globe/evaluation/`: metric, statistics, report generation
- `implementations/lite_globe/baselines/`: 외부 routing baseline과 fidelity registry
- `implementations/lite_globe/experiments/`: 역사적 Phase 7/8/11/12 및 비교 campaign
- `tests/lite_globe/`: 단위·통합 계약 검증
- `scripts/`: 학습 계보, 패키징, 보조 실행기

## Research documents

- `docs/method_history.md`: Phase 1~13 채택·제외 결정
- `docs/paper_method_summary.md`: 수식과 구현 대응
- `docs/simulation_protocol.md`: 지표·통계·baseline 공정성 계약
- `submission/ad_hoc_networks_overleaf/`: 최종 원고

## Local or generated

- `artifacts/`, `runs/`, `checkpoints/`: 생성 산출물
- `ResearchAIWorkspace/`: 현재 브랜치에서 제외된 legacy/local workspace
- `paper/`, `presentations/`: 현재 브랜치에서 제외된 로컬 자료

현재 브랜치 작업은 tracked source를 기준으로 하며 legacy 경로를 새 코드 정본으로
사용하지 않는다.
