# Fused Exact SwitchGLOBE — IEEE Access draft package

Primary target: **IEEE Access** (Research Article). This package uses the official IEEE Access LaTeX class distributed on 2026-05-13. It is a complete technical draft, but author identities, affiliations, ORCID iDs, funding statements, and the public artifact DOI must be finalized before submission.

## Core files

- `main.tex`: double-column IEEE Access manuscript.
- `supplementary.tex`: 20-figure supplementary analysis and reproducibility material.
- `cover_letter.tex`: IEEE Access cover-letter draft.
- `references.bib`: BibTeX database, including the required AI-tool disclosure citation.
- `ieeeaccess.cls`, `IEEEtran.bst`, `spotcolor.sty`, logos, and bundled Type1 fonts: official template assets.
- `figures/`: five publication figures in PDF, SVG, and PNG.
- `supplementary_figures/`: 20 analysis figures in PNG and SVG.
- `figure_data/`: per-figure CSV data and paired-latency statistics.
- `tables/`: A100 timing, operator profiling, ablation, and statistical tables.
- evidence, novelty, literature, reproducibility, and review documents: internal audit trail.

## Scientific interpretation

The final proposal is **Fused Exact SwitchGLOBE**. It preserves the original SwitchGLOBE routing decision exactly while eliminating redundant branch evaluations: each original branch is evaluated once, and the same intermediate logits are reused for action selection and diagnostics. FastSwitchGLOBE remains an approximate speed–quality candidate because it does not satisfy the predefined reliability gate. Early Exit is retained as a negative result.

## Build

The official class depends on PDFLaTeX primitives used by `spotcolor.sty`; therefore build with PDFLaTeX/BibTeX rather than XeTeX. The packaged PDFs are generated with a Codex-managed TeX Live 2026 runtime and are visually checked after rendering every page to PNG.

## Submission status

The package is suitable for co-author review. It is not ready for direct journal upload until every unchecked item in `submission_checklist.md` is resolved.
