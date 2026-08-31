# GLOBE_ROUTING — Lite-GLOBE / GLOBE++ Routing

Decentralized FANET 라우팅을 위한 Global-to-Local 정책 증류 연구 구현체.
`feature/phase13` 브랜치는 소스 실행에 필요한 것만 남긴 정리된 구조이고,
논문 LaTeX 원본·발표자료·Obsidian 노트 등은 `main` 브랜치에만 있다.

## 설치 및 실행

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lite-globe.txt
.venv/bin/python -m pip install -e .

pytest                                              # 전체 테스트
python -m implementations.lite_globe.run_phaseN --smoke --device auto
```

## 구조

- `implementations/lite_globe/`가 실제 소스. `env/`, `models/`, `algorithms/`,
  `evaluation/`은 phase 공용 모듈이고, `run_phaseN.py` / `config/phaseN.yaml`이
  단계별 진입점이다.
- `tests/lite_globe/test_phaseN_*.py`는 체크포인트 없이 도는 코드 레벨 검증(smoke)이다.
- `run_phaseN.py --smoke`는 코드 검증이 아니라 **축소된 규모의 실제 학습/평가
  파이프라인**이다. Phase 8 이후는 이전 phase의 학습된 체크포인트
  (`artifacts/lite_globe/phaseN/checkpoints/seed_*/*.pt`)가 로컬에 있어야
  실행된다 — 없으면 `pytest tests/lite_globe/test_phaseN_*.py`로 코드만 검증한다.
- `scripts/`는 Google Colab 원격 실행 자동화(`colab_run.py`, `package_phaseN_colab.py`,
  `run_phase13_seed_queue.py`, `merge_phase13_artifacts.py`)만 포함한다.
- `docs/paper_method_summary.md`에 논문 Dec-POMDP 정식화 + 방법론(P+/PRS) 요약과
  구현 위치 매핑이 있다. 전체 논문 원문은 이 브랜치에 없다.

## 규칙

- `artifacts/`, `checkpoints/`, `.venv/`는 커밋하지 않는다 (`.gitignore` 대상).
- `paper/`, `kci_paper/`, `professor_presentation/` 등은 이 브랜치에 의도적으로
  없다 — 되살리지 말고 필요하면 `main` 브랜치를 참고한다.
- `scripts/package_phase{10,11,12,13}_colab.py`는 `README_PHASEN_COLAB.md`를
  레포 루트에서 상대 경로로 읽는다 — 이 README들을 `docs/`로 옮기지 않는다.
- Colab 원격 실행(`scripts/colab_run.py`)은 실제 GPU/TPU 과금이 발생하므로
  사용자 확인 없이 실행하지 않는다.
