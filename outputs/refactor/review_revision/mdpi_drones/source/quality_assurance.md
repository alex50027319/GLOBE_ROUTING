# PDF and package quality assurance

## Build environment

- Engine: PDFLaTeX from Codex-managed TeX Live 2026.
- Bibliography: BibTeX with `IEEEtran.bst`.
- Official `ieeeaccess.cls` retained without modification.
- XeTeX/Tectonic is intentionally not used because the official spot-color implementation requires PDFLaTeX primitives.

## Completed checks

- [x] Manuscript, supplement, and cover letter compile without fatal errors.
- [x] References and cross-references contain no unresolved `??` markers.
- [x] Every PDF page was rendered to PNG and inspected using contact sheets.
- [x] High-risk pages containing wide tables, equations, algorithms, and multi-panel figures were inspected at full resolution.
- [x] No text, figure, table, caption, footer, or biography is clipped or obscured.
- [x] All detected fonts are embedded and pages have the expected size.
- [x] Text extraction contains no `TODO`, `FIXME`, absolute user path, unresolved-reference marker, or build diagnostic.
- [x] All 20 citation keys resolve; all 34 referenced figure instances exist; no duplicate labels were detected.
- [x] The former combined architecture was split into independent training and deployment figures; both were inspected at full resolution.
- [x] The execution-reuse and evaluation-protocol diagrams were regenerated with contained text and increased spacing.
- [x] Visible manuscript, supplementary, cover-letter, and figure text consistently uses the DiSwitch name.
- [x] Final PDFs open successfully and pass structural checks.
- [x] Each PDF is well below the IEEE Access 40 MB upload limit.

The official class intentionally constructs its running header and footer with zero-width boxes; PDFLaTeX therefore reports repeated `505.12177 pt` output-routine overfull messages. The front-matter decoration also reports two `9.2679 pt` messages. These are template-internal diagnostics rather than overflowing manuscript content, and every affected page was visually inspected.

## Final PDF record

| Artifact | Pages | Page size | Bytes | SHA-256 |
|---|---:|---|---:|---|
| Manuscript | 16 | 576 × 782.929 pt | 1,162,537 | `df203fca34a25ba766df338a31c12d23c4275e4ba5000ca8a87eaac917a3960c` |
| Supplementary material | 15 | 576 × 782.929 pt | 1,950,922 | `d0baee856c179cbdf44897a7e5113cb88801a2f03192310c02eaf8e8f2657979` |
| Cover letter | 1 | A4, 595.276 × 841.89 pt | 33,865 | `f7ee1e713afeb0adc91bb775ad7ae9e6e69d1f2b8715d01a86f5d6540eaa3f0c` |

The final package archive passed `unzip -t`; its external SHA-256 is recorded in
`outputs/ieee_access/DiSwitch_SHA256SUMS_2026-09-07.txt` so that recording the digest does not alter
the archive being measured.
