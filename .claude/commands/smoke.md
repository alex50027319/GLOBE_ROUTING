---
description: Quick code-level smoke pytest pass for one Lite-GLOBE phase (or the whole suite if no phase given)
argument-hint: "[phase number]"
---

If `$ARGUMENTS` is a number N, run:
```bash
pytest tests/lite_globe/test_phase${N}_*.py -v
```

If `$ARGUMENTS` is empty, run the full suite instead:
```bash
pytest -q
```

Summarize pass/fail counts. If anything fails, show just the failing test
names and the final error line, not the full traceback, unless the user asks
for more detail.
