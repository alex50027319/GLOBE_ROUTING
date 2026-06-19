# GLOBE++ Obsidian Research Wiki

This workspace builds a local, source-traceable research knowledge base for **GLOBE++: Partial Observability-Aware Global-to-Local Policy Distillation for Decentralized FANET Routing**. It organizes literature, concepts, reviewer critiques, claims, experiments, and manuscript notes without modifying original source files.

Vault 문서의 기본 작성 언어는 한국어다. 논문 원제, 저자명, 알고리즘명, 수식, 인용 메타데이터는 정확성을 위해 원문을 유지한다.

## Structure

- `raw/`: immutable source material grouped by papers, articles, memos, datasets, slides, code, and assets.
- `vault/`: the Obsidian vault containing generated and curated Markdown knowledge.
- `scripts/`: deterministic extraction, ingestion, indexing, linting, and matrix utilities.
- `implementations/lite_globe/`: reproducible Lite-GLOBE routing environment and baselines.
- `tests/lite_globe/`: deterministic environment, policy, and metric tests.
- `processed_index.json`: hash-based processing registry.

## Installation

```bash
cd ResearchAIWorkspace
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Add Raw Files

Place files in the matching `raw/` subfolder. Never edit, move, rename, or delete a raw file through the automation. Supported extraction formats are `.txt`, `.md`, `.pdf`, `.docx`, `.csv`, and `.xlsx`.

## Ingest

Preview changes:

```bash
python scripts/ingest.py --dry-run
```

Process new hashes:

```bash
python scripts/ingest.py
```

The deterministic ingester creates metadata-rich source summaries and paper-card placeholders. Sections that require scholarly judgment remain marked `TODO: Codex review required`.

## Open in Obsidian

Open this folder as a vault:

```text
ResearchAIWorkspace/vault
```

Dataview blocks are optional. The pages remain readable without the plugin.

## Maintenance

```bash
python scripts/update_index.py
python scripts/build_literature_matrix.py
python scripts/find_links.py
python scripts/lint_wiki.py
```

The lint report is written to `vault/00_Index/lint_report.md`.

## Notion 동기화

Obsidian을 원본으로 유지하고 전체 vault를 Notion의 `GLOBE++ Research Wiki`
데이터베이스에 단방향 동기화한다.

1. `.env.example`을 `.env`로 복사한다.
2. Notion Developer Portal에서 personal access token 또는 internal connection
   token을 만든다.
3. internal connection을 사용했다면 `GLOBE++ Research Wiki` 데이터베이스를
   해당 connection에 공유한다.
4. `.env`의 `NOTION_TOKEN`에 토큰을 입력한다.
5. 먼저 dry-run으로 대상을 확인한 뒤 실제 동기화를 실행한다.

```bash
cp .env.example .env
python3 scripts/sync_notion.py --dry-run
python3 scripts/sync_notion.py
```

새 raw 파일 처리와 Notion 동기화를 한 번에 실행할 수도 있다.

```bash
python3 scripts/ingest.py --sync-notion
```

`Obsidian 경로`와 `콘텐츠 Hash`를 사용하므로 이미 동기화된 문서는 중복 생성하지
않고, 변경된 문서만 갱신한다. `.env`와 `.notion_sync.json`은 Git에서 제외된다.

## processed_index.json

The registry uses SHA-256 as the primary key. A previously processed hash is skipped; the same filename with a new hash is treated as a new source. Each entry records source metadata, timestamps, created pages, updated pages, status, and notes.

## Literature Workflow

1. Add a source to `raw/`.
2. Run ingestion and inspect the generated source summary.
3. Complete evidence-backed paper-card sections.
4. Create or update concept pages and comparisons.
5. Rebuild the literature matrix and index.
6. Run lint and resolve warnings.

## GLOBE++ Workflow

Use literature evidence to update `04_GLOBE_PlusPlus/`, then connect claims to required theory and experiments. Keep paper claims separate from agent assessments. Move material into `07_Manuscript/` only when citations and evidence metadata exist.

## Lite-GLOBE Phase 1

Lite-GLOBE의 환경, Random/GPSR 기준선, 평가기는 별도 경량 의존성을 사용한다.

```bash
python -m pip install -r requirements-lite-globe.txt
python -m pip install -e .
python -m pytest tests/lite_globe
python -m implementations.lite_globe.run_phase1 --episodes 20 --seed 42
python -m implementations.lite_globe.run_phase2 --episodes 20 --seed 42
python -m implementations.lite_globe.run_phase3 \
  --checkpoint artifacts/lite_globe/teacher_phase3.pt
python -m implementations.lite_globe.run_phase4
python -m implementations.lite_globe.run_phase5
```

Phase 5의 Teacher-free local PPO fine-tuning까지 지원한다. Multi-seed 일반
환경 평가와 통계 보고는 Phase 6에서 분리한다. 모든 학습 단계는 로컬 CPU와
Google Colab GPU에서 같은 명령으로 실행할 수 있다. 자세한 범위와 가정은
`implementations/lite_globe/README.md`를 따른다.

Phase 9 risk-aware 전체 검증:

```bash
python -m implementations.lite_globe.run_phase9 --device auto --resume
```

Google Colab에는 `artifacts/lite_globe/phase9_colab_bundle.zip`과
`implementations/lite_globe/colab/phase9_risk_aware.ipynb`를 사용한다.

## Safety

- Raw files are immutable.
- Source material stays local.
- No external upload, cloud API, or remote LLM call is performed by these scripts.
- Do not invent author, venue, year, results, or theoretical support.
