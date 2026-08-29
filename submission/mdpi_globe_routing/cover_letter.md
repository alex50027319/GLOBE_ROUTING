# Cover Letter Draft — Drones

Dear Editors,

We submit the manuscript “Risk-Switch Lite-GLOBE-P: Global-to-Local Policy Distillation with Predictive Recovery for Decentralized FANET Routing” for consideration in *Drones*, preferably in the Drone Communications section.

The manuscript addresses a deployment gap in learning-based FANET routing. A graph-aware PPO teacher is used only during offline training, while the deployed actor is restricted to a masked one-hop observation. The actor combines a geographic residual policy with a calibrated predictive risk switch that is activated only for unsafe or imminently failing links. This separates the benefit of global supervision from the communication cost of global or message-passing execution.

The Phase 12 full evaluation uses five training seeds, 14 held-out and stress scenarios, 200 episodes per scenario, and 84,000 episode records. Risk-Switch Lite-GLOBE-P achieves the strongest aggregate connected-pair PDR (0.905) and deadline-delivery ratio (0.838) among the compared policies. The paper also reports its limitations: a modest energy-proxy penalty relative to DRAMA, slightly higher aggregate delay than the Phase 8 predecessor, and weaker performance than Evo-QGeo in one highly lossy predictive-break condition.

This work is original, is not under consideration elsewhere, and has been approved by all authors. The simulator-level limitations, input-byte definition, and absence of a packet-level AODV/OLSR comparison are stated explicitly, and we plan to release the exact configuration and Phase 12 archive with the final repository DOI.

Sincerely,\\
First Author\\
Corresponding author: third.author@example.com
