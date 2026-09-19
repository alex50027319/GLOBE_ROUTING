# Reproducibility guide

1. Use the exact commit and five checkpoint archives recorded in the latency and routing manifests.
2. Validate SHA-256 before execution.
3. Run the common-simulator evaluation for seeds 42, 77, 123, 314, 2718; all 14 scenarios; 200 episodes each.
4. Verify 14,000 episode rows per method and 112,000 combined rows.
5. Aggregate per seed/scenario before scenario-macro averaging.
6. Run batch-one latency with 50 warm-ups and 2,000 repetitions, synchronizing CUDA at the timing boundaries.
7. Keep operator profiling separate from latency measurement.
8. Regenerate figures with `MPLBACKEND=Agg MPLCONFIGDIR=<writable-dir> .venv/bin/python paper/fused_exact_switchglobe/generate_paper_figures.py`.
9. Compile `main.tex`, `supplementary.tex`, and `cover_letter.tex`; render all PDF pages and visually inspect clipping, overlap, missing glyphs, tables, captions, and references.

Recommended strict rerun: clean git tag, 10 seeds, raw randomized-block timing CSV, target UAV companion computer, measured joules, routing-table update time, and wireless control overhead.

