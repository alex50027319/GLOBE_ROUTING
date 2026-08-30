---
name: audit-switchglobe-results
description: Audit SwitchGLOBE or external-comparison artifacts for five-seed completeness, metric validity, and claim readiness.
---

# Audit SwitchGLOBE Results

Ground the campaign and distinguish SwitchGLOBE ablation from external comparison.

Check:

- manifest mode, completion flag, config, checkpoint paths, and seeds;
- expected and observed episode/summary rows;
- duplicate and missing scenario-method-seed cells;
- overall versus connected-pair PDR denominators;
- deadline, p95 delay, energy proxy, drop reasons, decision latency, input bytes;
- CSV, generated tables, figures, and manuscript consistency.

Use `.claude/templates/result-audit.md`. Lead with findings and evidence paths. Do not repair raw
results or fill missing seeds with smoke data.
