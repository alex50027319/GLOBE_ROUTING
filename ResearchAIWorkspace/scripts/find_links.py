"""Report outgoing, incoming, and unresolved Obsidian links."""

from __future__ import annotations

from collections import Counter

from utils import VAULT_DIR, extract_links, markdown_files, page_names, read_markdown


def main() -> int:
    names = page_names()
    incoming: Counter[str] = Counter()
    broken: list[tuple[str, str]] = []
    total = 0
    for path in markdown_files():
        _, body = read_markdown(path)
        links = extract_links(body)
        total += len(links)
        for link in links:
            key = link.casefold()
            if key in names:
                incoming[key] += 1
            else:
                broken.append((path.relative_to(VAULT_DIR).as_posix(), link))
    print(f"Pages: {len(markdown_files())}")
    print(f"Links: {total}")
    print(f"Resolved targets: {len(incoming)}")
    print(f"Broken links: {len(broken)}")
    for source, target in broken:
        print(f"- {source}: [[{target}]]")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
