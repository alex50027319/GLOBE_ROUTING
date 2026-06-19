"""Lint the research vault and write a Markdown report."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from utils import (
    RAW_DIR,
    VAULT_DIR,
    extract_links,
    load_registry,
    markdown_files,
    page_names,
    read_markdown,
    sha256_file,
    today,
)

REPORT = VAULT_DIR / "00_Index" / "lint_report.md"
INDEX = VAULT_DIR / "00_Index" / "index.md"
ISSUE_NAMES_KO = {
    "Missing frontmatter": "Frontmatter 누락",
    "Missing frontmatter fields": "Frontmatter 필드 누락",
    "H1 title mismatch": "H1 제목 불일치",
    "Broken links": "깨진 링크",
    "Unindexed pages": "인덱스에 없는 페이지",
    "Missing source metadata": "출처 메타데이터 누락",
    "Strong claims without citation metadata": "인용 메타데이터 없는 강한 주장",
    "Unresolved contradictions": "해결되지 않은 모순",
    "Stale drafts": "오래된 초안",
    "High-relevance paper cards without GLOBE++ link": "GLOBE++ 연결이 없는 고관련성 논문 카드",
    "Concept candidates not created": "생성되지 않은 개념 후보",
    "Duplicate titles": "중복 제목",
    "Orphan pages": "고립된 페이지",
    "Unprocessed raw files": "처리되지 않은 raw 파일",
    "Source hash/path mismatch": "출처 hash/path 불일치",
}
STRONG_CLAIM_RE = re.compile(
    r"\b(significantly|proves?|guarantees?|outperforms?|state-of-the-art|optimal)\b",
    re.IGNORECASE,
)
CONCEPT_CANDIDATE_RE = re.compile(r"concept candidate\s*:\s*([^\n]+)", re.IGNORECASE)


def relative(path: Path) -> str:
    return path.relative_to(VAULT_DIR).as_posix()


def issue(bucket: dict[str, list[str]], name: str, value: str) -> None:
    bucket[name].append(value)


def parse_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def collect_issues() -> dict[str, list[str]]:
    issues: dict[str, list[str]] = defaultdict(list)
    paths = [path for path in markdown_files() if path != REPORT]
    names = page_names()
    titles: Counter[str] = Counter()
    inbound: Counter[str] = Counter()
    aliases_by_path: dict[Path, set[str]] = {}
    indexed_text = INDEX.read_text(encoding="utf-8") if INDEX.exists() else ""
    registry = load_registry()

    for path in paths:
        metadata, body = read_markdown(path)
        rel = relative(path)
        if not metadata:
            issue(issues, "Missing frontmatter", rel)
            continue
        title = str(metadata.get("title") or "").strip()
        aliases_by_path[path] = {path.stem.casefold()}
        if not title:
            issue(issues, "Missing frontmatter fields", f"{rel}: title")
        else:
            titles[title.casefold()] += 1
            aliases_by_path[path].add(title.casefold())
        for required in ("type", "created", "updated", "status", "tags", "confidence"):
            if required not in metadata:
                issue(issues, "Missing frontmatter fields", f"{rel}: {required}")
        first_h1 = next(
            (line[2:].strip() for line in body.splitlines() if line.startswith("# ")),
            "",
        )
        if title and first_h1 and first_h1 != title:
            issue(issues, "H1 title mismatch", f"{rel}: `{first_h1}` != `{title}`")
        for target in extract_links(body):
            key = target.casefold()
            if key not in names:
                issue(issues, "Broken links", f"{rel}: [[{target}]]")
            else:
                inbound[key] += 1
        if path != INDEX and title and f"[[{title}]]" not in indexed_text:
            issue(issues, "Unindexed pages", rel)
        if (
            metadata.get("type") in {"source-summary", "paper-card"}
            and metadata.get("status") != "template"
        ):
            required_source = ("source_files",)
            for field in required_source:
                if not metadata.get(field):
                    issue(issues, "Missing source metadata", f"{rel}: {field}")
        if metadata.get("type") == "source-summary":
            for field in ("source_path", "source_hash", "source_file"):
                if not metadata.get(field):
                    issue(issues, "Missing source metadata", f"{rel}: {field}")
        strong_lines = [
            line
            for line in body.splitlines()
            if STRONG_CLAIM_RE.search(line)
            and "TODO" not in line
            and "not " not in line.casefold()
            and "do not" not in line.casefold()
        ]
        if strong_lines and "Source hash:" not in body:
            issue(issues, "Strong claims without citation metadata", rel)
        if re.search(r"unresolved contradiction", body, re.IGNORECASE):
            issue(issues, "Unresolved contradictions", rel)
        updated = parse_date(metadata.get("updated"))
        if metadata.get("status") == "draft" and updated:
            age = (date.today() - updated).days
            if age > 90:
                issue(issues, "Stale drafts", f"{rel}: {age} days")
        if (
            metadata.get("type") == "paper-card"
            and metadata.get("globe_relevance") == "high"
            and "GLOBE++" not in body
        ):
            issue(issues, "High-relevance paper cards without GLOBE++ link", rel)
        for match in CONCEPT_CANDIDATE_RE.finditer(body):
            for candidate in match.group(1).split(","):
                name = candidate.strip()
                if name and name.casefold() not in names:
                    issue(issues, "Concept candidates not created", f"{rel}: {name}")

    for title, count in titles.items():
        if count > 1:
            issue(issues, "Duplicate titles", f"{title}: {count}")

    exempt_orphans = {"research wiki index", "research dashboard", "research wiki log"}
    for path in paths:
        metadata, _ = read_markdown(path)
        aliases = aliases_by_path.get(path, {path.stem.casefold()})
        title_key = str(metadata.get("title") or path.stem).casefold()
        if (
            title_key in exempt_orphans
            or metadata.get("status") == "template"
            or any(inbound[alias] > 0 for alias in aliases)
        ):
            continue
        issue(issues, "Orphan pages", relative(path))

    supported = {".txt", ".md", ".pdf", ".docx", ".csv", ".xlsx"}
    for path in RAW_DIR.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in supported:
            continue
        source_hash = sha256_file(path)
        entry = registry["files"].get(source_hash)
        rel = path.relative_to(VAULT_DIR.parent).as_posix()
        if not entry:
            issue(issues, "Unprocessed raw files", rel)
        elif entry.get("source_path") != rel:
            issue(issues, "Source hash/path mismatch", rel)

    return issues


def render(issues: dict[str, list[str]]) -> str:
    total = sum(len(values) for values in issues.values())
    lines = [
        "---",
        'title: "Wiki Lint Report"',
        'type: "log"',
        f'created: "{today()}"',
        f'updated: "{today()}"',
        'status: "active"',
        "tags: [lint, quality]",
        "source_files: []",
        "related_concepts: []",
        "related_papers: []",
        'globe_relevance: "high"',
        'confidence: "high"',
        'language: "ko"',
        "---",
        "",
        "# Wiki Lint Report",
        "",
        f"- 생성 시각: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- 전체 문제 수: {total}",
        "",
    ]
    if not issues:
        lines.append("문제가 발견되지 않았다.")
    for name in sorted(issues):
        lines.extend([f"## {ISSUE_NAMES_KO.get(name, name)}", ""])
        lines.extend(f"- {value}" for value in sorted(set(issues[name])))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    issues = collect_issues()
    REPORT.write_text(render(issues), encoding="utf-8")
    total = sum(len(values) for values in issues.values())
    print(f"{REPORT.relative_to(VAULT_DIR.parent)} 작성 완료 (문제 {total}개)")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
