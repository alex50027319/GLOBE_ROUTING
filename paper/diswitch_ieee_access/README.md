# DiSwitch — IEEE Access draft package

Primary target: **IEEE Access** (Research Article). This package uses the official IEEE Access LaTeX class distributed on 2026-05-13. Kyung Eun Kim and Howon Lee are listed with the Department of Military Digital Convergence, Ajou University. ORCID iDs, any applicable funding statement, and the public artifact DOI should be finalized before submission.

## Core files

- `main.tex`: double-column IEEE Access manuscript.
- `supplementary.tex`: 20-figure supplementary analysis and reproducibility material.
- `cover_letter.tex`: IEEE Access cover-letter draft.
- `references.bib`: BibTeX database, including the required AI-tool disclosure citation.
- `ieeeaccess.cls`, `IEEEtran.bst`, `spotcolor.sty`, logos, and bundled Type1 fonts: official template assets.
- `figures/`: six publication figures in PDF, SVG, and PNG, including separate training and deployment architecture diagrams.
- `supplementary_figures/`: 20 analysis figures in PNG and SVG.
- `figure_data/`: per-figure CSV data and paired-latency statistics.
- `tables/`: A100 timing, operator profiling, ablation, and statistical tables.
- evidence, novelty, literature, reproducibility, and review documents: internal audit trail.

## Scientific interpretation

The final proposal is **DiSwitch (Distilled Policy Switch)**. It combines global-to-local policy distillation with a local risk switch and preserves the validated two-branch routing decision while eliminating redundant branch evaluations. Each branch is evaluated once, and the same intermediate logits are reused for action selection and diagnostics. Fast-DiSwitch remains an approximate speed--quality candidate because it does not satisfy the predefined reliability gate. Early Exit is retained as a negative result.

## Build

The official class depends on PDFLaTeX primitives used by `spotcolor.sty`; therefore build with PDFLaTeX/BibTeX rather than XeTeX. The packaged PDFs are generated with a Codex-managed TeX Live 2026 runtime and are visually checked after rendering every page to PNG.

## Submission status

The package is suitable for co-author review. It is not ready for direct journal upload until every unchecked item in `submission_checklist.md` is resolved.
