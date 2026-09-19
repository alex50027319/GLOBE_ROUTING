# Confirmatory A100 run failure record

- Requested source commit: `29a8b4f768fc573cd273c7ea5dba463de17b80eb`
- Intended additions: P90, maximum, standard deviation, coefficient of variation, CUDA peak device memory, and operator-level profiler.
- A100 allocation and source/checkpoint archive SHA validation succeeded.
- The remote benchmark started in session `switchglobe-globev2-confirmatory-final` and remained busy for an extended run.
- Colab backend subsequently returned `Session not found`; no confirmatory result ZIP or benchmark CSV was produced.
- The local CLI process was terminated after the backend session disappeared.
- The session was not retried automatically to avoid repeated compute-unit consumption.

This is a negative execution result, not an A100 result. The valid A100 evidence remains the primary run in `results/` (commit `92d17df`, 5 seeds, warm-up 50, repeats 2,000, A100-SXM4-40GB). The expanded metric code and profiler remain in the repository for a future shorter or split run.
