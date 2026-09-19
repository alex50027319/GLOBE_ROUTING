from pathlib import Path
import re


root = Path(__file__).resolve().parent
main = (root / "main.tex").read_text(encoding="utf-8")
supplement = (root / "supplementary.tex").read_text(encoding="utf-8")
bibliography = (root / "references.bib").read_text(encoding="utf-8")

citation_keys = {
    key.strip()
    for group in re.findall(r"\\cite\{([^}]+)\}", main)
    for key in group.split(",")
}
bibliography_keys = set(re.findall(r"^@\w+\{([^,]+),", bibliography, flags=re.MULTILINE))

figure_paths = re.findall(
    r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", main + "\n" + supplement
)
missing_figures = [path for path in figure_paths if not (root / path).exists()]

labels = re.findall(r"\\label\{([^}]+)\}", main + "\n" + supplement)
duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})

print(f"Citation keys: {len(citation_keys)}")
print(f"Bibliography entries: {len(bibliography_keys)}")
print(f"Missing citation keys: {sorted(citation_keys - bibliography_keys)}")
print(f"Referenced figures: {len(figure_paths)}")
print(f"Missing figures: {missing_figures}")
print(f"Duplicate labels: {duplicate_labels}")

if citation_keys - bibliography_keys or missing_figures or duplicate_labels:
    raise SystemExit(1)
