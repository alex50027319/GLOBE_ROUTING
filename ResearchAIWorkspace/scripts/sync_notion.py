"""Obsidian vault 문서를 Notion 데이터 소스로 단방향 동기화한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from utils import ROOT, VAULT_DIR, markdown_files, read_markdown
except ModuleNotFoundError:
    from scripts.utils import ROOT, VAULT_DIR, markdown_files, read_markdown

NOTION_VERSION = "2026-03-11"
DEFAULT_DATA_SOURCE_ID = "6d7160f7-0df3-4faf-b5e1-099b3540ae0e"
ENV_FILE = ROOT / ".env"
SYNC_STATE = ROOT / ".notion_sync.json"
NOTION_CODE_LANGUAGES = {
    "bash",
    "c",
    "c++",
    "c#",
    "css",
    "go",
    "html",
    "java",
    "javascript",
    "json",
    "julia",
    "kotlin",
    "lua",
    "markdown",
    "matlab",
    "mermaid",
    "objective-c",
    "perl",
    "php",
    "plain text",
    "powershell",
    "python",
    "r",
    "ruby",
    "rust",
    "scala",
    "shell",
    "sql",
    "swift",
    "typescript",
    "xml",
    "yaml",
}

TYPE_MAP = {
    "paper-card": "논문 카드",
    "source-summary": "출처 요약",
    "concept": "개념",
    "comparison": "비교",
    "experiment": "실험",
    "manuscript-section": "원고",
    "claim": "주장",
    "index": "인덱스",
    "research-idea": "연구 아이디어",
    "dashboard": "대시보드",
    "log": "로그",
}
STATUS_MAP = {
    "draft": "초안",
    "reviewed": "검토 완료",
    "in-progress": "진행 중",
    "planned": "계획",
    "active": "활성",
    "template": "템플릿",
}
RELEVANCE_MAP = {
    "low": "낮음",
    "low-medium": "중간",
    "medium": "중간",
    "medium-high": "높음",
    "high": "높음",
    "critical": "핵심",
}
CONFIDENCE_MAP = {"low": "낮음", "medium": "중간", "high": "높음"}
AREA_MAP = {
    "00_Index": "시작·인덱스",
    "01_Sources": "원본·출처",
    "02_Concepts": "개념 사전",
    "03_Papers": "문헌 연구",
    "04_GLOBE_PlusPlus": "GLOBE++ 설계",
    "05_Comparisons": "비교 분석",
    "06_Experiments": "실험",
    "07_Manuscript": "원고",
    "08_Research_Ideas": "연구 아이디어",
    "99_Templates": "템플릿",
}


def load_env() -> None:
    if not ENV_FILE.exists():
        return
    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_markdown(text: str) -> str:
    text = re.sub(r"!\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]|#]+)(?:#[^\]|]+)?\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]|#]+)(?:#[^\]]+)?\]\]", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    text = text.replace("**", "").replace("__", "")
    return text.strip()


def rich_text(text: str) -> list[dict[str, Any]]:
    value = clean_markdown(text)
    if not value:
        return []
    return [
        {"type": "text", "text": {"content": value[index : index + 1900]}}
        for index in range(0, len(value), 1900)
    ]


def plain_rich_text(text: str) -> list[dict[str, Any]]:
    value = str(text)
    if not value:
        return []
    return [
        {"type": "text", "text": {"content": value[index : index + 1900]}}
        for index in range(0, len(value), 1900)
    ]


def text_block(kind: str, text: str) -> dict[str, Any]:
    return {"object": "block", "type": kind, kind: {"rich_text": rich_text(text)}}


def markdown_to_blocks(body: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    paragraph: list[str] = []
    table: list[str] = []
    code: list[str] = []
    code_language = "plain text"
    in_code = False

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(text_block("paragraph", " ".join(paragraph)))
            paragraph.clear()

    def flush_table() -> None:
        if table:
            blocks.append(
                {
                    "object": "block",
                    "type": "code",
                    "code": {
                        "language": "plain text",
                        "rich_text": rich_text("\n".join(table)),
                    },
                }
            )
            table.clear()

    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_paragraph()
            flush_table()
            if in_code:
                blocks.append(
                    {
                        "object": "block",
                        "type": "code",
                        "code": {
                            "language": code_language,
                            "rich_text": rich_text("\n".join(code)),
                        },
                    }
                )
                code.clear()
                in_code = False
            else:
                requested_language = stripped[3:].strip().casefold() or "plain text"
                code_language = (
                    requested_language
                    if requested_language in NOTION_CODE_LANGUAGES
                    else "plain text"
                )
                in_code = True
            continue
        if in_code:
            code.append(line)
            continue
        if stripped.startswith("|"):
            flush_paragraph()
            table.append(line)
            continue
        flush_table()
        if not stripped:
            flush_paragraph()
            continue
        heading = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            blocks.append(
                text_block(f"heading_{len(heading.group(1))}", heading.group(2))
            )
            continue
        bullet = re.match(r"^[-*]\s+(.+)$", stripped)
        if bullet:
            flush_paragraph()
            blocks.append(text_block("bulleted_list_item", bullet.group(1)))
            continue
        numbered = re.match(r"^\d+\.\s+(.+)$", stripped)
        if numbered:
            flush_paragraph()
            blocks.append(text_block("numbered_list_item", numbered.group(1)))
            continue
        if stripped.startswith(">"):
            flush_paragraph()
            blocks.append(text_block("quote", stripped.lstrip("> ")))
            continue
        paragraph.append(stripped)

    flush_paragraph()
    flush_table()
    if code:
        blocks.append(
            {
                "object": "block",
                "type": "code",
                "code": {
                    "language": code_language,
                    "rich_text": rich_text("\n".join(code)),
                },
            }
        )
    return blocks


class NotionClient:
    def __init__(self, token: str) -> None:
        self.token = token
        self.last_request = 0.0

    def request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        wait = 0.36 - (time.monotonic() - self.last_request)
        if wait > 0:
            time.sleep(wait)
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"https://api.notion.com/v1{path}",
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Notion-Version": NOTION_VERSION,
                "Content-Type": "application/json",
            },
        )
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    self.last_request = time.monotonic()
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                self.last_request = time.monotonic()
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code in {429, 529} and attempt < 4:
                    time.sleep(float(exc.headers.get("Retry-After", 2)))
                    continue
                raise RuntimeError(f"Notion API {exc.code}: {detail}") from exc
        raise RuntimeError("Notion API 재시도 한도를 초과했다.")

    def existing_pages(self, data_source_id: str) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        cursor: str | None = None
        while True:
            payload: dict[str, Any] = {"page_size": 100}
            if cursor:
                payload["start_cursor"] = cursor
            response = self.request(
                "POST", f"/data_sources/{data_source_id}/query", payload
            )
            for page in response.get("results", []):
                properties = page.get("properties", {})
                path = property_text(properties.get("Obsidian 경로", {}))
                if path:
                    result[path] = {
                        "id": page["id"],
                        "hash": property_text(properties.get("콘텐츠 Hash", {})),
                    }
            if not response.get("has_more"):
                break
            cursor = response.get("next_cursor")
        return result

    def append_blocks(self, page_id: str, blocks: list[dict[str, Any]]) -> None:
        for index in range(0, len(blocks), 100):
            self.request(
                "PATCH",
                f"/blocks/{page_id}/children",
                {"children": blocks[index : index + 100]},
            )

    def clear_page(self, page_id: str) -> None:
        cursor: str | None = None
        block_ids: list[str] = []
        while True:
            suffix = f"?page_size=100{f'&start_cursor={cursor}' if cursor else ''}"
            response = self.request("GET", f"/blocks/{page_id}/children{suffix}")
            block_ids.extend(block["id"] for block in response.get("results", []))
            if not response.get("has_more"):
                break
            cursor = response.get("next_cursor")
        for block_id in block_ids:
            self.request("DELETE", f"/blocks/{block_id}")


def property_text(prop: dict[str, Any]) -> str:
    values = prop.get("rich_text") or prop.get("title") or []
    return "".join(item.get("plain_text", "") for item in values)


def page_properties(
    path: Path, metadata: dict[str, Any], digest: str
) -> dict[str, Any]:
    relative = path.relative_to(VAULT_DIR).as_posix()
    source_files = metadata.get("source_files", [])
    if not isinstance(source_files, list):
        source_files = [source_files]
    authors = metadata.get("authors", [])
    if not isinstance(authors, list):
        authors = [authors]

    def text(value: object) -> dict[str, Any]:
        return {"rich_text": plain_rich_text(str(value))[:1]}

    properties: dict[str, Any] = {
        "제목": {
            "title": plain_rich_text(str(metadata.get("title") or path.stem))[:1]
        },
        "문서 유형": {
            "select": {"name": TYPE_MAP.get(str(metadata.get("type")), "기타")}
        },
        "연구 영역": {
            "select": {
                "name": AREA_MAP.get(path.relative_to(VAULT_DIR).parts[0], "시작·인덱스")
            }
        },
        "상태": {
            "select": {"name": STATUS_MAP.get(str(metadata.get("status")), "초안")}
        },
        "GLOBE++ 관련성": {
            "select": {
                "name": RELEVANCE_MAP.get(
                    str(metadata.get("globe_relevance")), "중간"
                )
            }
        },
        "신뢰도": {
            "select": {
                "name": CONFIDENCE_MAP.get(str(metadata.get("confidence")), "중간")
            }
        },
        "저자": text(", ".join(str(author) for author in authors)),
        "출판 연도": text(metadata.get("year", "")),
        "원본 파일": text(", ".join(str(item) for item in source_files)),
        "Obsidian 경로": text(relative),
        "Source Hash": text(metadata.get("source_hash", "")),
        "콘텐츠 Hash": text(digest),
        "최종 동기화": {
            "date": {"start": datetime.now().astimezone().isoformat(timespec="seconds")}
        },
    }
    return properties


def load_state() -> dict[str, Any]:
    if not SYNC_STATE.exists():
        return {"pages": {}}
    return json.loads(SYNC_STATE.read_text(encoding="utf-8"))


def save_state(state: dict[str, Any]) -> None:
    SYNC_STATE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--refresh-properties",
        action="store_true",
        help="본문은 건드리지 않고 Notion 속성만 다시 계산해 갱신한다.",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    load_env()
    token = os.environ.get("NOTION_TOKEN", "")
    data_source_id = os.environ.get(
        "NOTION_DATA_SOURCE_ID", DEFAULT_DATA_SOURCE_ID
    ).removeprefix("collection://")
    paths = markdown_files()
    if args.limit:
        paths = paths[: args.limit]
    if args.dry_run:
        print(f"동기화 대상: {len(paths)}개 Markdown 문서")
        print(f"Notion data source: {data_source_id}")
        print("Obsidian이 원본이며 Notion은 단방향 복제본으로 갱신된다.")
        return 0
    if not token:
        print("NOTION_TOKEN이 없다. .env.example을 참고해 .env에 설정해야 한다.")
        return 2

    state = load_state()
    client = NotionClient(token)
    existing = client.existing_pages(data_source_id)
    for relative, saved in state.get("pages", {}).items():
        if saved.get("page_id"):
            existing[relative] = {
                "id": saved["page_id"],
                "hash": saved.get("content_hash", ""),
            }
    created = updated = skipped = 0
    for path in paths:
        metadata, body = read_markdown(path)
        relative = path.relative_to(VAULT_DIR).as_posix()
        digest = content_hash(path)
        remote = existing.get(relative)
        if (
            remote
            and remote.get("hash") == digest
            and not args.force
            and not args.refresh_properties
        ):
            skipped += 1
            continue
        properties = page_properties(path, metadata, digest)
        blocks = markdown_to_blocks(body)
        if remote:
            page_id = remote["id"]
            client.request("PATCH", f"/pages/{page_id}", {"properties": properties})
            if not args.refresh_properties:
                client.clear_page(page_id)
                client.append_blocks(page_id, blocks)
            updated += 1
        else:
            page = client.request(
                "POST",
                "/pages",
                {
                    "parent": {
                        "type": "data_source_id",
                        "data_source_id": data_source_id,
                    },
                    "properties": properties,
                },
            )
            page_id = page["id"]
            client.append_blocks(page_id, blocks)
            created += 1
        state.setdefault("pages", {})[relative] = {
            "page_id": page_id,
            "content_hash": digest,
        }
        print(f"동기화: {relative}")
    save_state(state)
    print(f"완료: 생성 {created}, 갱신 {updated}, 변경 없음 {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
