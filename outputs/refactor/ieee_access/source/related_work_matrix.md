# Related-work matrix

The full 45-record metadata export is `literature_comparison.csv`. The closest comparators are separated by what their delay quantity means.

| Work | Method | Routing scope | Deployment evidence | Delay/latency scope | Relation to DiSwitch |
|---|---|---|---|---|---|
| Predictive-Q (Dong et al., 2026) | Q-learning, predictive dual path | Per-packet UAV routing | Standard laptop/MATLAB | Per-packet and per-slot compute reported | Motivates lightweight backup preparation; table-based state differs from local neural policy |
| RDQN-HERP (Shou et al., 2026) | Recurrent dueling Double DQN | Distributed next-hop routing | Simulation/runtime disclosure | Decision time reported for a 50-node case | Closest compute-latency comparator, but hardware and timed scope are not identical |
| RLFR/DRLFR (Li et al., 2024) | Q-learning + Double DQN | Energy-efficient local forwarding | Raspberry Pi 4B and small UAV testbed | Network/slot forwarding delay, not pure policy call | Strong small-device precedent |
| RoutePPO (Cürmen et al., 2026) | PPO + eBPF/P4 | Adaptive path installation | Linux forwarding plane | Sub-ms update path; not neural inference only | Motivates splitting inference and route-update latency |
| KG-DDRL (Zhang et al., 2026) | KAN-GNN + DQN | Routing and power | Simulation | No directly comparable decision timer | Motivates sparse/local encoding |
| DFS-PPO (Chen et al., 2026) | Pruning + PPO | Routing/scheduling/load balance | Simulation on desktop CPU | Online-flow solve time | Motivates structural action pruning; different action problem |
| Drones harsh-environment routing (Sun et al., 2026) | GCN + DQN | Cluster maintenance and routing | Simulation | Network delay | Direct venue precedent; no batch-one compute benchmark |
| GAT Double DQN (Zhang et al., 2025) | GAT + Double DQN | Resilient next-hop routing | Simulation | Network delay | Graph-RL baseline family |
| TARRAQ (Cui et al., 2022) | Adaptive Q-learning | Topology-aware FANET routing | Simulation | Network delay | Topology-aware value-based prior art |

Conclusion: E2E delay is common, but explicitly scoped policy decision latency is uncommon. Cross-paper numbers remain contextual unless hardware, batch size, synchronization, and timed boundaries match.

