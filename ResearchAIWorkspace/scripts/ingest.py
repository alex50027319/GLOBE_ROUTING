"""Hash-aware, local-only ingestion for immutable raw research sources."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from extract_text import extract
from utils import (
    RAW_DIR,
    ROOT,
    VAULT_DIR,
    append_log,
    load_registry,
    relative_to_root,
    save_registry,
    sha256_file,
    slugify,
    today,
    yaml_document,
)

SUPPORTED = {".txt", ".md", ".pdf", ".docx", ".csv", ".xlsx"}

SOURCE_DESTINATIONS = {
    "papers": VAULT_DIR / "01_Sources" / "Papers",
    "articles": VAULT_DIR / "01_Sources" / "Articles",
    "memos": VAULT_DIR / "01_Sources" / "Memos",
    "datasets": VAULT_DIR / "01_Sources" / "Datasets",
    "code": VAULT_DIR / "01_Sources" / "Code",
    "slides": VAULT_DIR / "01_Sources" / "Memos",
    "assets": VAULT_DIR / "01_Sources" / "Memos",
}

CONCEPT_TERMS = {
    "Dec-POMDP",
    "FANET",
    "UAV Routing",
    "CTDE",
    "MAPPO",
    "Graph Neural Network",
    "Knowledge Distillation",
    "Policy Distillation",
    "Latent Distillation",
    "Ego-Graph",
    "Partial Observability",
    "Teacher-Student Policy",
    "Routing Overhead",
    "Packet Delivery Ratio",
    "End-to-End Delay",
    "OOD Generalization",
    "Ablation Study",
    "Performance Difference Lemma",
    "KL Divergence",
    "Student-induced Distribution",
}

RELEVANCE_TERMS = {
    "fanet",
    "uav",
    "routing",
    "mappo",
    "multi-agent",
    "gnn",
    "graph neural",
    "distillation",
    "partial observability",
    "dec-pomdp",
}


def raw_files() -> list[Path]:
    return sorted(
        path
        for path in RAW_DIR.rglob("*")
        if path.is_file() and path.suffix.casefold() in SUPPORTED
    )


def source_kind(path: Path) -> str:
    try:
        return path.relative_to(RAW_DIR).parts[0]
    except (ValueError, IndexError):
        return "memos"


def unique_page_path(directory: Path, stem: str, source_hash: str) -> Path:
    candidate = directory / f"{stem}.md"
    if not candidate.exists():
        return candidate
    return directory / f"{stem}_{source_hash[:8]}.md"


def concept_candidates(text: str) -> list[str]:
    folded = text.casefold()
    return sorted(term for term in CONCEPT_TERMS if folded.count(term.casefold()) >= 2)


def globe_relevance(text: str) -> tuple[str, list[str]]:
    folded = text.casefold()
    matches = sorted(term for term in RELEVANCE_TERMS if term in folded)
    if len(matches) >= 4:
        return "high", matches
    if len(matches) >= 1:
        return "medium", matches
    return "low", matches


def source_summary(
    path: Path,
    source_hash: str,
    extracted: dict[str, Any],
    candidates: list[str],
    relevance: str,
    matches: list[str],
) -> str:
    title = f"Source - {path.stem} - {source_hash[:8]}"
    source_path = relative_to_root(path)
    metadata = {
        "title": title,
        "type": "source-summary",
        "created": today(),
        "updated": today(),
        "status": "draft",
        "tags": ["raw-summary"],
        "source_files": [path.name],
        "source_file": path.name,
        "source_path": source_path,
        "source_hash": source_hash,
        "source_type": extracted.get("kind", path.suffix.lstrip(".")),
        "processed_by": "Codex deterministic ingest",
        "related_concepts": candidates,
        "related_papers": [],
        "globe_relevance": relevance,
        "confidence": "medium",
    }
    concept_links = ", ".join(f"[[{name}]]" for name in candidates) or "None detected."
    extraction_note = ""
    if extracted.get("kind") in {"csv", "xlsx"}:
        extraction_note = (
            f"- Rows: {extracted.get('row_count')}\n"
            f"- Columns: {extracted.get('column_count')}\n"
            f"- Target candidates: {extracted.get('target_column_candidates')}\n"
        )
    elif extracted.get("kind") == "pdf":
        extraction_note = f"- Pages extracted: {extracted.get('page_count')}\n"

    body = f"""# {title}

## 1. 출처 메타데이터

| Field | Value |
| --- | --- |
| 원본 파일 | `{path.name}` |
| 원본 경로 | `{source_path}` |
| 원본 해시 | `{source_hash}` |
| 원본 유형 | `{extracted.get("kind", path.suffix.lstrip("."))}` |
| 처리 날짜 | {today()} |

## 2. 핵심 요약

TODO: Codex의 논문 검토가 필요하다. 결정론적 텍스트 추출만 완료했으며 학술적 주장은 아직 해석하지 않았다.

## 3. 핵심 개념

{concept_links}

## 4. 중요 세부사항

{extraction_note or "TODO: review extracted text and record evidence with page or section references."}

## 5. 추출된 주장

| Claim | Evidence | Page/Section | Confidence |
| --- | --- | --- | --- |
| TODO: Codex review required | Not inferred by deterministic ingestion | needs-verification | low |

## 6. 내 연구와의 관련성

### GLOBE++에 직접 유용

- Deterministic relevance: **{relevance}**
- Matched terms: {", ".join(matches) or "none"}
- TODO: distinguish source evidence from Agent Assessment.

### 간접적으로 유용

TODO

### 관련성이 낮거나 유용하지 않음

TODO

## 7. 생성 또는 갱신할 개념 페이지

{concept_links}

## 8. 생성 또는 갱신할 Paper Card

{"A placeholder paper card was created." if source_kind(path) == "papers" else "Not automatically created for this source category."}

## 9. 기존 Wiki와의 모순 또는 긴장

Unresolved contradiction: needs-verification.

## 10. 후속 질문

- Which claims are directly supported by the source?
- Which assumptions affect transfer to dynamic [[FANET]] routing?
- What evidence can support or challenge [[GLOBE++ Novelty Claims]]?

## 11. 인용 메모

- Source file: `{path.name}`
- Source path: `{source_path}`
- Source hash: `{source_hash}`
- Page/section references: TODO
- Evidence type: needs-verification
"""
    return yaml_document(metadata, body)


def paper_card(path: Path, source_hash: str, source_title: str) -> str:
    title = f"Paper - {path.stem} - {source_hash[:8]}"
    metadata = {
        "title": title,
        "type": "paper-card",
        "created": today(),
        "updated": today(),
        "status": "draft",
        "tags": ["paper", "literature"],
        "source_files": [path.name],
        "paper_title": "",
        "authors": [],
        "year": "",
        "venue": "",
        "doi": "",
        "url": "",
        "research_area": [],
        "methods": [],
        "baselines": [],
        "metrics": [],
        "datasets_or_simulators": [],
        "related_concepts": [],
        "related_papers": [],
        "globe_relevance": "",
        "confidence": "low",
        "uses_gnn": "needs-verification",
        "uses_marl": "needs-verification",
        "uses_kd": "needs-verification",
        "handles_partial_observability": "needs-verification",
        "simulator": "",
        "seed_count": "",
        "statistical_reporting": "needs-verification",
        "reproducibility": "needs-verification",
    }
    body = f"""# {title}

> Deterministic placeholder linked to [[{source_title}]]. Bibliographic and technical fields must be verified from the source.

## 1. 한 줄 요약

TODO: Codex의 논문 검토가 필요하다.

## 2. 문제 설정

TODO

## 3. 핵심 기여

### 논문 주장

TODO: cite source page or section.

### Agent 평가

TODO: reviewer-level assessment.

## 4. 방법

### 4.1 모델 / 알고리즘

TODO

### 4.2 학습 설정

TODO

### 4.3 실행 단계 가정

TODO

## 5. 실험 설정

| Item | Description |
| --- | --- |
| Simulator | TODO |
| Number of nodes | TODO |
| Mobility model | TODO |
| Traffic model | TODO |
| Baselines | TODO |
| Metrics | TODO |
| Seeds | TODO |
| Statistical reporting | TODO |

## 6. 주요 결과

TODO: no result should be recorded without evidence.

## 7. 이론적 주장

TODO

## 8. 가정

TODO

## 9. 한계

TODO

## 10. Reviewer 관점 비판

### 강점

TODO

### 약점

TODO

### 숨겨진 취약 가정

TODO

### 재현성 위험

TODO

## 11. GLOBE++와의 연결

| Aspect | Relevance |
| --- | --- |
| [[Dec-POMDP]] | TODO |
| [[CTDE]] | TODO |
| [[MAPPO]] | TODO |
| [[Graph Neural Network]] | TODO |
| [[Policy Distillation]] | TODO |
| [[Partial Observability]] | TODO |
| [[FANET]] Routing | TODO |
| [[GLOBE++ Experiment Plan]] | TODO |

## 12. 내 논문에서 뒷받침할 수 있는 내용

- Introduction: TODO
- Related Work: TODO
- Method: TODO
- Theory: TODO
- Experiments: TODO
- Discussion: TODO

## 13. 뒷받침할 수 없는 내용

TODO

## 14. 후속 질문

TODO

## 15. 출처 메모

- Source file: `{path.name}`
- Source path: `{relative_to_root(path)}`
- Source hash: `{source_hash}`
- Page/section references: TODO
"""
    return yaml_document(metadata, body)


def planned_actions(files: list[Path], registry: dict[str, Any]) -> list[tuple[Path, str]]:
    actions = []
    for path in files:
        source_hash = sha256_file(path)
        if source_hash not in registry["files"]:
            actions.append((path, source_hash))
    return actions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--sync-notion",
        action="store_true",
        help="ingest 완료 후 Obsidian vault를 Notion에 동기화한다.",
    )
    args = parser.parse_args()

    registry = load_registry()
    files = raw_files()
    actions = planned_actions(files, registry)
    print("Change plan:")
    print("- Scan raw/ read-only and calculate SHA-256 hashes.")
    print("- Create source summaries for new hashes.")
    print("- Create paper-card placeholders for raw/papers sources.")
    print("- Update index, append log, and update processed_index.json.")
    print("- raw/ files will not be modified, moved, renamed, or deleted.")
    print(f"Raw files found: {len(files)}; new hashes: {len(actions)}")
    for path, source_hash in actions:
        print(f"- CREATE from {relative_to_root(path)} [{source_hash[:12]}]")
    if args.dry_run:
        return 0

    created_pages: list[str] = []
    failures: list[str] = []
    for path, source_hash in actions:
        try:
            extracted = extract(path)
            text = str(extracted.get("text", ""))
            candidates = concept_candidates(text)
            relevance, matches = globe_relevance(text)
            kind = source_kind(path)
            destination = SOURCE_DESTINATIONS.get(kind, VAULT_DIR / "01_Sources" / "Memos")
            stem = slugify(f"{path.stem}_{source_hash[:8]}_source")
            summary_path = unique_page_path(destination, stem, source_hash)
            summary_title = f"Source - {path.stem} - {source_hash[:8]}"
            summary_path.write_text(
                source_summary(path, source_hash, extracted, candidates, relevance, matches),
                encoding="utf-8",
            )
            pages = [relative_to_root(summary_path)]
            if kind == "papers":
                card_path = unique_page_path(
                    VAULT_DIR / "03_Papers" / "Paper_Cards",
                    slugify(f"{path.stem}_{source_hash[:8]}_paper"),
                    source_hash,
                )
                card_path.write_text(
                    paper_card(path, source_hash, summary_title), encoding="utf-8"
                )
                pages.append(relative_to_root(card_path))
            registry["files"][source_hash] = {
                "source_file": path.name,
                "source_path": relative_to_root(path),
                "source_type": extracted.get("kind", path.suffix.lstrip(".")),
                "processed_at": today(),
                "created_pages": pages,
                "updated_pages": [],
                "status": "processed",
                "notes": "Deterministic extraction; scholarly review remains TODO.",
            }
            created_pages.extend(pages)
        except Exception as exc:  # Continue processing independent sources.
            failures.append(f"{relative_to_root(path)}: {exc}")

    save_registry(registry)
    update_script = ROOT / "scripts" / "update_index.py"
    subprocess.run([sys.executable, str(update_script)], check=True)
    append_log(
        "ingest",
        "raw/",
        {
            "created": ", ".join(created_pages) or "none",
            "updated": "processed_index.json, vault/00_Index/index.md",
            "linked": "source summaries and paper cards",
            "notes": f"Processed {len(created_pages)} generated pages.",
            "warnings": "; ".join(failures) or "none",
        },
    )
    print(f"Created pages: {len(created_pages)}")
    print(f"Failures: {len(failures)}")
    for failure in failures:
        print(f"- {failure}")
    if args.sync_notion and not failures:
        sync_script = ROOT / "scripts" / "sync_notion.py"
        subprocess.run([sys.executable, str(sync_script)], check=True)
    print("Recommended next command: python scripts/lint_wiki.py")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
