---
name: run-phase13-seeds
description: Plan and run explicitly approved Phase 13 evaluation jobs one training seed at a time.
---

# Run Phase 13 Seeds

1. Read `config/phase13.yaml` and confirm requested seeds are in the configured set.
2. Check whether each `phase13_seeds_<seed>_results.zip` already exists.
3. Run relevant Phase 13 pytest smoke checks.
4. Show the session names, GPU, timeout, output ZIPs, and logs before remote execution.
5. Start `scripts/run_phase13_seed_queue.py` only when the user explicitly requested execution.
6. Never infer liveness from an old log; check process/session state and modification time.
7. Summarize each seed independently and preserve partial success.
