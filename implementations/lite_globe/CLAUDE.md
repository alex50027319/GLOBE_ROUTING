# SwitchGLOBE Implementation

- Public final entry point is `run_switchglobe.py`; historical Phase names remain for lineage compatibility.
- Keep environment observations local at deployment and privileged information training-only.
- Preserve action-mask and explicit DROP semantics across Teacher, Students, adapters, and evaluators.
- Changes to shared environment or observation fields require focused environment, policy, and reporting tests.
- Do not add Phase 13/P+ features to `SwitchGlobePolicy` unless the user explicitly changes the final method definition.
- Generated checkpoints and reports belong under `artifacts/`, not the package tree.
