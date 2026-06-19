"""Rebuild the content-oriented Obsidian page catalog."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from utils import VAULT_DIR, first_summary_line, markdown_files, read_markdown, today

INDEX_PATH = VAULT_DIR / "00_Index" / "index.md"

CATEGORIES = [
    ("출처", "01_Sources"),
    ("논문", "03_Papers"),
    ("개념", "02_Concepts"),
    ("GLOBE++ 연구 노트", "04_GLOBE_PlusPlus"),
    ("비교", "05_Comparisons"),
    ("실험", "06_Experiments"),
    ("원고", "07_Manuscript"),
    ("연구 아이디어", "08_Research_Ideas"),
    ("인덱스와 템플릿", ""),
]


def category_for(path: Path) -> str:
    relative = path.relative_to(VAULT_DIR).as_posix()
    for label, prefix in CATEGORIES[:-1]:
        if relative.startswith(prefix + "/"):
            return label
    return "인덱스와 템플릿"


def escape_cell(value: object) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def build_index() -> str:
    groups: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for path in markdown_files():
        if path == INDEX_PATH:
            continue
        metadata, body = read_markdown(path)
        title = str(metadata.get("title") or path.stem)
        page = f"[[{title}]]"
        summary = str(metadata.get("summary") or first_summary_line(body))
        groups[category_for(path)].append(
            (
                page,
                str(metadata.get("type", "")),
                summary,
                str(metadata.get("status", "")),
                str(metadata.get("updated", "")),
            )
        )

    sections = []
    for label, _ in CATEGORIES:
        rows = sorted(groups[label], key=lambda row: row[0].casefold())
        sections.extend(
            [
                f"## {label}",
                "",
                "| 페이지 | 유형 | 요약 | 상태 | 갱신일 |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        sections.extend(
            "| " + " | ".join(escape_cell(cell) for cell in row) + " |" for row in rows
        )
        if not rows:
            sections.append("| _None_ |  |  |  |  |")
        sections.append("")

    return (
        "---\n"
        'title: "Research Wiki Index"\n'
        'type: "index"\n'
        f'created: "{today()}"\n'
        f'updated: "{today()}"\n'
        'status: "active"\n'
        "tags: [index, research-wiki]\n"
        "source_files: []\n"
        "related_concepts: []\n"
        "related_papers: []\n"
        'globe_relevance: "high"\n'
        'confidence: "high"\n'
        'language: "ko"\n'
        "---\n\n"
        "# Research Wiki Index\n\n"
        "`scripts/update_index.py`가 재생성한 내용 중심 문서 목록이다.\n\n"
        + "\n".join(sections)
    )


def main() -> int:
    INDEX_PATH.write_text(build_index(), encoding="utf-8")
    print(f"Updated {INDEX_PATH.relative_to(VAULT_DIR.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
