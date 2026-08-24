# Evidence Map

This document maps the primary academic claims of the **GLOBE++ Lite-P+** paper to actual implemented source files and raw evaluation artifacts within the workspace.

| Academic Claim / Component | Status | Evidence File(s) | Description / Notes | Can Claim? |
| :--- | :---: | :--- | :--- | :---: |
| **FANET Simulator Environment** | `implemented` | [env/fanet_env.py](file:///Users/alex/Documents/GLOBE_ROUTING/ResearchAIWorkspace/implementations/lite_globe/env/fanet_env.py) | Dynamic topology generator, RWP mobility model, communication ranges. | **Yes** |
| **Dec-POMDP Formulation** | `implemented` | [assumptions.md](file:///Users/alex/Documents/GLOBE_ROUTING/ResearchAIWorkspace/implementations/lite_globe/assumptions.md) | Standard state, action, observation, and reward constraints. | **Yes** |
| **GNN Global Teacher Model** | `implemented` | [models/teacher_gnn.py](file:///Users/alex/Documents/GLOBE_ROUTING/ResearchAIWorkspace/implementations/lite_globe/models/teacher_gnn.py) | 2-layer Edge-conditioned Message Passing GNN with centralized critic. | **Yes** |
| **MLP Local Student Model** | `implemented` | [models/student_mlp.py](file:///Users/alex/Documents/GLOBE_ROUTING/ResearchAIWorkspace/implementations/lite_globe/models/student_mlp.py) | Permutation-invariant mean pooling architecture using localized 1-hop inputs. | **Yes** |
| **Policy Distillation Loss** | `implemented` | [run_phase4.py](file:///Users/alex/Documents/GLOBE_ROUTING/ResearchAIWorkspace/implementations/lite_globe/run_phase4.py) | Temperature-scaled KL divergence minimizing action distribution drift. | **Yes** |
| **Predictive Risk Switch (PRS)** | `implemented` | [models/risk_switch.py](file:///Users/alex/Documents/GLOBE_ROUTING/ResearchAIWorkspace/implementations/lite_globe/models/risk_switch.py) | Toggles nominal distilled branch with predictive branch via Danger score gates. | **Yes** |
| **P+ Features ($x_i^+$)** | `implemented` | [run_phase13.py](file:///Users/alex/Documents/GLOBE_ROUTING/ResearchAIWorkspace/implementations/lite_globe/run_phase13.py) | Top-$k$ onward stability, link survival probability, energy proxy. | **Yes** |
| **PDR Improvements** | `verified` | [artifacts/lite_globe/phase12/](file:///Users/alex/Documents/GLOBE_ROUTING/ResearchAIWorkspace/artifacts/lite_globe/phase12/) | Merged statistics showing PDR of 86.4\% (32.5\% absolute gain over GPSR). | **Yes** (Phase 12) |
| **Tail Latency Reduction** | `verified` | [artifacts/lite_globe/phase12/](file:///Users/alex/Documents/GLOBE_ROUTING/ResearchAIWorkspace/artifacts/lite_globe/phase12/) | Merged statistics showing p95 delay of 4.264 seconds. | **Yes** (Phase 12) |
| **Overhead Minimization** | `verified` | [artifacts/lite_globe/phase12/](file:///Users/alex/Documents/GLOBE_ROUTING/ResearchAIWorkspace/artifacts/lite_globe/phase12/) | Input-byte proxy showing 128 bytes footprint (lower than Evo-QGeo/DRAMA). | **Yes** (Phase 12) |
| **Phase 13 P+ Statistical Sweep** | `missing` | [run_phase13.py](file:///Users/alex/Documents/GLOBE_ROUTING/ResearchAIWorkspace/implementations/lite_globe/run_phase13.py) | Smoke-tests pass, but full multi-seed evaluation results are not yet complete. | **No** (Mark as TODO) |
| **Physical MAC/PHY validation** | `not found` | N/A | Local simulator abstracts away collisions and multipath fading. | **No** (Mark as limitation) |
