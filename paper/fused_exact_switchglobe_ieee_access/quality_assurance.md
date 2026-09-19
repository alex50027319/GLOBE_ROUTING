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
- [x] All 20 citation keys resolve; all 33 referenced figure instances exist; no duplicate labels were detected.
- [x] Final PDFs open successfully and pass structural checks.
- [x] Each PDF is well below the IEEE Access 40 MB upload limit.

The official class intentionally constructs its running header and footer with zero-width boxes; PDFLaTeX therefore reports repeated `505.12177 pt` output-routine overfull messages. The front-matter decoration also reports two `9.2679 pt` messages. These are template-internal diagnostics rather than overflowing manuscript content, and every affected page was visually inspected.

## Final PDF record

| Artifact | Pages | Page size | Bytes | SHA-256 |
|---|---:|---|---:|---|
| Manuscript | 16 | 576 × 782.929 pt | 1,121,674 | `e64b872f05c383f22a7ccd37fb59fd7dc0ca10d1ff7e481ce578bead441c9179` |
| Supplementary material | 14 | 576 × 782.929 pt | 1,888,974 | `bf085ea5244c0ac31e336fa55e30c14835dd76474bf798f5129a91dd48100f56` |
| Cover letter | 1 | A4, 595.276 × 841.89 pt | 34,431 | `52d9986fb91132438e2d322346512756ecdcd4edb569bed0e8c75d4d4a9ab6df` |

The final package archive passed `unzip -t`; its external SHA-256 is recorded in
`outputs/ieee_access/SHA256SUMS.txt` so that recording the digest does not alter
the archive being measured.
