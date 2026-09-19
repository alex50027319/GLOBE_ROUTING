---
name: update-switchglobe-manuscript
description: Update the SwitchGLOBE manuscript from verified full-run artifacts while preserving method scope, citations, and LaTeX integrity.
---

# Update SwitchGLOBE Manuscript

Before editing `submission/ad_hoc_networks_overleaf/`:

1. audit the source artifact and verify all five seeds;
2. map each number to campaign, scenario, method, metric, and file path;
3. keep Phase 13/P+ outside the final algorithm and results;
4. distinguish SwitchGLOBE ablation from external baseline comparison;
5. preserve proxy units and denominator definitions.

Update prose, tables, figures, captions, limitations, and supplementary references consistently.
Compile the LaTeX project and inspect unresolved references, overflow, and stale figures. Unsupported
claims remain unchanged or are marked `needs-verification`.
