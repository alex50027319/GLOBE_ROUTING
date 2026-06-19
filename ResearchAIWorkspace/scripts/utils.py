"""Shared filesystem and frontmatter helpers for the local research wiki."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # Core wiki maintenance remains usable before installation.
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw"
VAULT_DIR = ROOT / "vault"
INDEX_FILE = ROOT / "processed_index.json"

WIKI_LINK_RE = re.compile(r"(?<!!)\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")


def now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")


def today() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_root(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def assert_raw_read_only(path: Path) -> None:
    resolved = path.resolve()
    if resolved != RAW_DIR.resolve() and RAW_DIR.resolve() not in resolved.parents:
        raise ValueError(f"Expected a path under raw/: {path}")


def load_registry() -> dict[str, Any]:
    if not INDEX_FILE.exists():
        return {"version": 1, "updated": "", "files": {}}
    with INDEX_FILE.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    data.setdefault("version", 1)
    data.setdefault("updated", "")
    data.setdefault("files", {})
    return data


def save_registry(data: dict[str, Any]) -> None:
    data["updated"] = now()
    with INDEX_FILE.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    marker = text.find("\n---\n", 4)
    if marker < 0:
        return {}, text
    raw = text[4:marker]
    parsed = _yaml_load(raw)
    return parsed if isinstance(parsed, dict) else {}, text[marker + 5 :]


def read_markdown(path: Path) -> tuple[dict[str, Any], str]:
    return split_frontmatter(path.read_text(encoding="utf-8"))


def markdown_files() -> list[Path]:
    return sorted(VAULT_DIR.rglob("*.md"))


def page_names() -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for path in markdown_files():
        meta, _ = read_markdown(path)
        names = {path.stem, str(meta.get("title", "")).strip()}
        for name in names - {""}:
            result.setdefault(name.casefold(), []).append(path)
    return result


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^\w\s.-]", "", value, flags=re.UNICODE)
    cleaned = re.sub(r"[\s.]+", "_", cleaned).strip("_")
    return cleaned or "untitled"


def yaml_document(metadata: dict[str, Any], body: str) -> str:
    frontmatter = _yaml_dump(metadata)
    return f"---\n{frontmatter}\n---\n\n{body.rstrip()}\n"


def _scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "Null", "NULL", "~"}:
        return "" if value == "" else None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        content = value[1:-1].strip()
        if not content:
            return []
        return [_scalar(item) for item in content.split(",")]
    try:
        return int(value)
    except ValueError:
        return value


def _yaml_load(raw: str) -> dict[str, Any]:
    if yaml is not None:
        parsed = yaml.safe_load(raw) or {}
        return parsed if isinstance(parsed, dict) else {}
    result: dict[str, Any] = {}
    active_list: str | None = None
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("  - ") and active_list:
            result[active_list].append(_scalar(line[4:]))
            continue
        match = re.match(r"^([A-Za-z0-9_+.-]+):(?:\s*(.*))?$", line)
        if not match:
            active_list = None
            continue
        key, value = match.group(1), match.group(2) or ""
        if value == "":
            result[key] = []
            active_list = key
        else:
            result[key] = _scalar(value)
            active_list = None
    return result


def _quote_yaml(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _yaml_dump(metadata: dict[str, Any]) -> str:
    if yaml is not None:
        return yaml.safe_dump(
            metadata, allow_unicode=True, sort_keys=False, default_flow_style=False
        ).strip()
    lines = []
    for key, value in metadata.items():
        if isinstance(value, list):
            if not value:
                lines.append(f"{key}: []")
            else:
                rendered = ", ".join(_quote_yaml(str(item)) for item in value)
                lines.append(f"{key}: [{rendered}]")
        elif isinstance(value, bool):
            lines.append(f"{key}: {'true' if value else 'false'}")
        elif value is None:
            lines.append(f"{key}: null")
        elif isinstance(value, (int, float)):
            lines.append(f"{key}: {value}")
        else:
            lines.append(f"{key}: {_quote_yaml(str(value))}")
    return "\n".join(lines)


def append_log(action: str, target: str, details: dict[str, str]) -> None:
    path = VAULT_DIR / "00_Index" / "log.md"
    if not path.exists():
        raise FileNotFoundError("Initialize vault/00_Index/log.md before logging")
    lines = [
        "",
        f"## [{now()}] {action} | {target}",
        "",
        f"- 생성: {details.get('created', '')}",
        f"- 갱신: {details.get('updated', '')}",
        f"- 연결: {details.get('linked', '')}",
        f"- 메모: {details.get('notes', '')}",
        f"- 경고: {details.get('warnings', '')}",
        "",
    ]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def extract_links(text: str) -> list[str]:
    return [match.group(1).strip() for match in WIKI_LINK_RE.finditer(text)]


def first_summary_line(body: str) -> str:
    for line in body.splitlines():
        value = line.strip()
        if value and not value.startswith(("#", "|", "-", "```", ">")):
            value = value.replace("[[", "").replace("]]", "")
            return value[:180]
    return ""
