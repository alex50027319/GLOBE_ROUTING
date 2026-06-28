# Reviewer Risk Register

| Risk | Why It Matters | Likely Reviewer Question | Mitigation |
| --- | --- | --- | --- |
| Novelty sounds like generic global context | CTDE, MARL, 2-hop routing, and prediction are already known | "How is this different from existing CTDE/MARL routing?" | Frame novelty as global-to-local routing policy distillation plus local execution and risk switching |
| Phase 13 full results missing | The draft presents P+ as final method | "Where is the full validation of the proposed method?" | Run Phase 13 full protocol and add ablation table |
| DRAMA is close baseline | DRAMA uses emergent communication and MARL | "Why not compare communication overhead explicitly?" | Add control bytes per delivered packet |
| Evo-QGeo beats Phase 12 in one predictive-break case | Weakens universal superiority claim | "Your method is not always best" | Present honest limitation and show Phase 13 link-loss gate improvement after validation |
| Energy proxy is weak | Real UAV energy depends on radio and flight dynamics | "Does this energy metric mean physical energy?" | Call it proxy unless calibrated; add sensitivity analysis |
| References incomplete | Local PDFs need exact metadata | "Citations are incomplete" | Complete BibTeX from PDFs before submission |
| AODV/OLSR absent | Classical routing reviewers may expect them | "Why no MANET baselines?" | Implement or state scope clearly |
| Input bytes are not control bytes | Input feature footprint differs from network signaling overhead | "Does lower input bytes mean lower communication overhead?" | Separate policy input footprint from control signaling |
| Teacher uses privileged state | Could be seen as unrealistic | "How is teacher information available?" | Emphasize teacher is offline only, not deployed |
| Simulator-only evidence | No testbed | "Does it generalize to real UAV networks?" | Add sensitivity/OOD stress tests and future testbed plan |
