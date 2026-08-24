# Missing Experiments

This document identifies required experiments to elevate the preliminary manuscript to a high-impact journal submission.

| Missing Experiment / validation | Purpose | Priority | Required Resources |
| :--- | :--- | :---: | :--- |
| **Phase 13 P+ Statistical Sweep** | Complete the 5-seed evaluation across all 14 scenarios to verify the absolute PDR gains of the P+ features (link-loss gate, onward k-stability, energy tie, drop suppression) over Evo-QGeo. | **High** | execution of `run_phase13.py` on GPU environment (Google Colab). |
| **ns-3 Network Simulator Cross-Validation** | Validate the distilled local student policy in a realistic ns-3 environment to verify PDR and latency under physical-layer fading and IEEE 802.11g/n MAC collisions. | **High** | ns-3 simulator environment, python binding, and trajectory exporters. |
| **Traditional Ad Hoc Baselines (AODV/OLSR)** | Evaluate standard MANET/FANET routing protocols under the same mobility scenarios to compare path stretch and protocol-level control signaling overhead. | **Medium** | OLSR/AODV protocol stack in the target simulator. |
| **Node Scalability Sweeps ($N > 24$)** | Sweep the network density from 10 to 50 nodes to verify that the local student policy scales gracefully without exponential inference latency or memory allocation. | **Medium** | Scalability scenario configs in Lite-GLOBE. |
| **Real Radio Energy Model Calibration** | Replace the geometric distance energy proxy with a physical RF transmission energy model (e.g., free-space/two-ray ground path loss) to measure real Watt-hour savings. | **Low** | Mathematical RF energy model integration in env. |
