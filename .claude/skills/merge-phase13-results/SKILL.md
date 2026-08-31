---
name: merge-phase13-results
description: Merge independently produced Phase 13 seed artifacts and validate completeness.
---

# Merge Phase 13 Results

1. List input ZIPs and extract their observed training seeds.
2. Require the three raw CSV files expected by `merge_phase13_artifacts.py`.
3. Record missing and duplicated seed/scenario/method combinations before merging.
4. Run the existing merge script with a new explicit output directory.
5. Verify manifest seed set, raw row counts, tables, and figures after merging.
6. Report gaps without fabricating rows or replacing them with smoke results.
