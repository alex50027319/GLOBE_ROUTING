"""Build a literature matrix from paper-card frontmatter and content."""

from __future__ import annotations

from utils import VAULT_DIR, read_markdown, today

CARD_DIR = VAULT_DIR / "03_Papers" / "Paper_Cards"
OUTPUT = VAULT_DIR / "03_Papers" / "Literature_Matrix" / "literature_matrix.md"


def cell(value: object) -> str:
    if isinstance(value, list):
        value = ", ".join(str(item) for item in value)
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def main() -> int:
    rows = []
    for path in sorted(CARD_DIR.glob("*.md")):
        metadata, body = read_markdown(path)
        title = metadata.get("paper_title") or metadata.get("title") or path.stem
        weakness = "TODO"
        if "### Weaknesses" in body:
            weakness = "See paper card"
        rows.append(
            [
                f"[[{metadata.get('title') or path.stem}]]",
                metadata.get("year", ""),
                metadata.get("venue", ""),
                cell(metadata.get("research_area", "")),
                cell(metadata.get("methods", "")),
                metadata.get("uses_gnn", ""),
                metadata.get("uses_marl", ""),
                metadata.get("uses_kd", ""),
                metadata.get("handles_partial_observability", ""),
                cell(metadata.get("baselines", "")),
                cell(metadata.get("metrics", "")),
                weakness,
                metadata.get("globe_relevance", ""),
            ]
        )
    headers = [
        "논문",
        "연도",
        "출판처",
        "문제",
        "방법",
        "GNN 사용",
        "MARL 사용",
        "KD 사용",
        "부분 관측 처리",
        "Baseline",
        "지표",
        "약점",
        "GLOBE++ 관련성",
    ]
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    table.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    if not rows:
        table.append("| _No paper cards_ | " + " | ".join("" for _ in headers[1:]) + " |")
    content = (
        "---\n"
        'title: "Literature Matrix"\n'
        'type: "comparison"\n'
        f'created: "{today()}"\n'
        f'updated: "{today()}"\n'
        'status: "active"\n'
        "tags: [literature, matrix]\n"
        "source_files: []\n"
        "related_concepts: []\n"
        "related_papers: []\n"
        'globe_relevance: "high"\n'
        'confidence: "medium"\n'
        'language: "ko"\n'
        "---\n\n"
        "# Literature Matrix\n\n"
        + "\n".join(table)
        + "\n"
    )
    OUTPUT.write_text(content, encoding="utf-8")
    print(f"Updated {OUTPUT.relative_to(VAULT_DIR.parent)} with {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
