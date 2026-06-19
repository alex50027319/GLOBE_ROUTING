"""Finalize the reviewed paper cards and source summaries for the initial batch."""

from __future__ import annotations

from pathlib import Path

from utils import ROOT, VAULT_DIR, load_registry, save_registry, today, yaml_document

PAPER_DIR = VAULT_DIR / "03_Papers" / "Paper_Cards"
SOURCE_DIR = VAULT_DIR / "01_Sources" / "Papers"

PAPERS = {
    "evo_qgeo": {
        "title": "Reinforcement-Learning-Based Geographic Routing Considering Future Evolution of Link States for UAV Networks",
        "authors": ["Ming Xu", "Yu Xia", "Wei Liu", "Daqing Huang"],
        "year": 2026,
        "venue": "Drones 10(2), 150",
        "doi": "10.3390/drones10020150",
        "status": "reviewed",
        "methods": ["Evo-QGeo", "Q-learning", "future link-state estimation", "two-hop routing-hole bypass"],
        "baselines": ["GPSR", "QGeo"],
        "metrics": ["PDR", "energy consumption", "end-to-end delay", "end-to-end ETX"],
        "concepts": ["FANET", "UAV Routing", "Partial Observability", "Ablation Study"],
        "summary": "Evo-QGeo combines Q-learning with a hand-designed future link-state score and two-hop routing-hole bypass logic for highly mobile FANETs.",
        "problem": "Reliable geographic forwarding under link fluctuations and local routing holes in highly mobile UAV networks.",
        "method": "A tabular Q-learning relay policy initializes/updates values from a fused link-state estimate. The link score combines predicted distance, received power, and link lifetime. A beacon-derived two-hop topology and regional bypass rule handles routing holes.",
        "training": "Online Q-learning with an epsilon policy; reward is +10 at destination, -10 at a routing hole, otherwise the estimated link state.",
        "execution": "Distributed geographic forwarding using local and two-hop beacon information; no global topology is required, but handcrafted link metrics and bypass regions directly shape decisions.",
        "experiment": "Numerical simulation varies 20-60 UAVs, 5-60 m/s, 250-450 m range, and cubic deployment areas. RWP mobility, log-normal/Rayleigh loss, IEEE 802.11b, and 10 repetitions per scenario are reported. A hardware-in-the-loop simulator uses real UAV trajectories.",
        "results": "The paper reports PDR gains of 3.15-44.61% over GPSR and 0.81-31.35% over QGeo; energy reductions of 38.79-74.47% and 29.22-48.40%; delay reductions of 7.43-38.68% and 4.55-21.63%. HIL trends agree with numerical results.",
        "theory": "No Dec-POMDP or policy-performance theory. The method is justified by link-state prediction and routing-hole logic.",
        "assumptions": "GPS/velocity information, periodic beacons, two-hop neighbor summaries, RWP mobility, fixed radio parameters, and a shaped link-state reward.",
        "strengths": "Direct FANET routing relevance; broad mobility/density/range sweeps; 10 repetitions; HIL trajectory validation; reports PDR, delay, energy, and ETX.",
        "weaknesses": "Only GPSR and QGeo baselines; no neural MARL comparison; no component ablation separating future-link fusion from two-hop bypass; hand-designed score and bypass logic dominate the policy; statistical intervals are not shown.",
        "fragile": "Performance depends on timely two-hop beacons and accurate motion/link prediction. HIL still simulates most network behavior rather than performing a full over-the-air swarm experiment.",
        "repro": "medium",
        "globe": "high",
        "connection": "A strong heuristic-rich baseline for GLOBE++. It supports evaluating future-link information and routing holes, but also motivates a mask-only policy: GLOBE++ must show that learned ego-graph decisions can match or exceed these engineered rules without hidden two-hop scoring.",
        "supports": "Related work on RL/geographic FANET routing; baseline selection; metric and mobility sweep design; discussion of heuristic dependence.",
        "cannot": "It cannot support policy distillation, Dec-POMDP theory, CTDE, GNN execution, or an irreducible partial-observability gap.",
        "evidence": [
            ("Protocol definition and reported headline gains", "pp. 1-2", "direct-claim"),
            ("Two-hop routing-hole bypass and shaped reward", "p. 10", "direct-claim"),
            ("10 repetitions and simulation/HIL settings", "pp. 10-11", "experimental-result"),
            ("PDR, energy, delay, ETX and HIL results", "pp. 12-15", "experimental-result"),
        ],
    },
    "glomappo": {
        "title": "GLo-MAPPO: Multi-Agent Deep Reinforcement Learning for Energy-Efficient UAV-Assisted LoRa Networks",
        "authors": ["Abdullahi Isa Ahmed", "Jamal Bentahar", "El Mehdi Amhoud"],
        "year": 2026,
        "venue": "arXiv:2509.17676v2; submitted to IEEE",
        "doi": "",
        "status": "reviewed",
        "methods": ["MAPPO", "CTDE", "GRU", "gain-based association", "joint trajectory/resource allocation"],
        "baselines": ["MAA2C", "COMA", "VDN", "QMIX", "IPPO"],
        "metrics": ["weighted energy efficiency", "power consumption", "cumulative reward", "robustness", "scalability"],
        "concepts": ["MAPPO", "CTDE", "Partial Observability", "OOD Generalization", "Ablation Study"],
        "summary": "GLo-MAPPO formulates multi-UAV LoRa gateway trajectory and radio-resource optimization as a partially observable stochastic game solved with CTDE MAPPO.",
        "problem": "Jointly optimize spreading factor, transmit power, UAV trajectory, and end-device association while satisfying QoS, propulsion, collision, and mission constraints.",
        "method": "Decentralized GRU actors use local observations; a centralized critic stabilizes training. Device association is deliberately decoupled from MARL and handled by a gain-based algorithm to reduce action-space complexity.",
        "training": "MAPPO with GAE, clipped PPO, 2 million environment steps, five fixed seeds, and hyperparameter sensitivity analysis.",
        "execution": "Each UAV actor executes from local observations, while association still relies on the gain-based scheme. This is CTDE but not routing-policy distillation.",
        "experiment": "2-5 UAV gateways, LoRa end devices, 1 km square, 400 time steps per mission, five seeds. Benchmarks cover policy-gradient and value-decomposition MARL. Ablations fix spreading factor, power, position, and association; noisy CSI and 100-400 device scaling are tested.",
        "results": "The paper reports the best cumulative reward and energy efficiency across tested densities, with energy-efficiency gains ranging from 18.56% to 152.72% over the best baseline depending on device count. Dynamic association and adaptive transmit power are major contributors.",
        "theory": "POSG formulation and computational complexity are provided, but no bound connects partial observability or policy error to performance.",
        "assumptions": "LoRa channel model, fixed altitude, discretized action components, decoupled association, perfect/noisy CSI model, and a simulation-only evaluation.",
        "strengths": "Explicit partial observability and CTDE; five seeds; broad MARL baselines; ablation, robustness, hyperparameter sensitivity, and scalability studies.",
        "weaknesses": "The task is gateway mobility/resource allocation rather than packet next-hop routing. The gain-based association remains hand-designed. No routing overhead, PDR, or end-to-end packet simulation.",
        "fragile": "The claimed joint optimization is partly decomposed outside MARL. Energy-efficiency conclusions depend on modeled LoRa and propulsion costs.",
        "repro": "medium-high",
        "globe": "medium-high",
        "connection": "Useful evidence for MAPPO/CTDE design, seed reporting, recurrent local policies, and partial-observability robustness. It is not a direct FANET routing baseline.",
        "supports": "Method design for CTDE MAPPO, multi-seed reporting, robustness tests, and ablation structure.",
        "cannot": "It cannot support next-hop routing claims, global-to-local distillation, routing overhead, or heuristic-free forwarding.",
        "evidence": [
            ("POSG, CTDE, objectives and contributions", "pp. 1-2", "direct-claim"),
            ("Training algorithm and complexity", "p. 10", "theoretical-claim"),
            ("Parameters, five seeds and baselines", "pp. 10-11", "experimental-result"),
            ("Ablation, robustness and scalability", "pp. 12-14", "experimental-result"),
        ],
    },
    "wpt_routing": {
        "title": "Deep Reinforcement Learning for Online Routing of Unmanned Aerial Vehicles with Wireless Power Transfer",
        "authors": ["Kaiwen Li", "Tao Zhang", "Rui Wang", "Ling Wang"],
        "year": 2022,
        "venue": "arXiv:2204.11477v1",
        "doi": "",
        "status": "reviewed",
        "methods": ["attention encoder-decoder", "REINFORCE", "greedy rollout baseline", "neural combinatorial optimization"],
        "baselines": ["Google OR-Tools", "Clarke-Wright", "PSO", "ACO", "ALNS"],
        "metrics": ["route cost", "runtime", "optimality gap", "solved instances"],
        "concepts": ["UAV Routing", "Graph Neural Network", "OOD Generalization"],
        "summary": "An attention-based sequence model is trained offline with REINFORCE to solve a UAV tour-planning problem with battery depletion and wireless charging.",
        "problem": "Minimize total mission time for a single UAV visiting task locations and returning to a wireless charging base when needed.",
        "method": "A Transformer-like encoder-decoder autoregressively constructs routes; invalid selections are masked. Training uses REINFORCE with the best greedy rollout policy as baseline.",
        "training": "100 epochs, 320,000 random instances per epoch, batch 256, Adam, 128-dimensional hidden layers. Greedy, sampling, and beam-search decoding are evaluated.",
        "execution": "Centralized offline-trained combinatorial solver; it is not packet routing and not decentralized multi-agent execution.",
        "experiment": "100 random test instances at 20-200 nodes, same Python machine for all solvers, and a 1000-second limit for large cases.",
        "results": "The learned method is typically much faster than search solvers and close in cost to OR-Tools. It is 0.7% worse on 100-node and 1.1% worse on 150-node cases but 5.6x and 592x faster, respectively; OR-Tools fails all 200-node cases within the time limit.",
        "theory": "No guarantee of optimality or generalization; evaluation is empirical.",
        "assumptions": "Single UAV, fixed speed, generated Euclidean instances, known task locations, one charging base, modeled WPT, and no wireless packet-network dynamics.",
        "strengths": "Clear optimization objective; strong non-learning solvers; runtime/quality tradeoff; large-instance scaling.",
        "weaknesses": "The word routing refers to vehicle/tour routing, not FANET packet forwarding. Only 100 test instances and no confidence intervals; generated topology distribution may be narrow.",
        "fragile": "Claims of arbitrary-topology scaling are within the same synthetic problem generator. Fixed speed and simplified charging omit flight dynamics.",
        "repro": "medium",
        "globe": "low-medium",
        "connection": "Relevant mainly for graph/attention policy design, action masking, and learned heuristic scaling. It should not be cited as a direct FANET routing method.",
        "supports": "Attention-based graph decision architecture and runtime-quality evaluation.",
        "cannot": "It cannot support MARL, CTDE, packet-routing metrics, partial observability, or decentralized FANET execution.",
        "evidence": [
            ("Task definition and contributions", "pp. 1-2", "direct-claim"),
            ("REINFORCE and greedy rollout baseline", "p. 7", "direct-claim"),
            ("Settings and solver baselines", "pp. 7-8", "experimental-result"),
            ("Small/large-instance results and limitations", "pp. 8-10", "experimental-result"),
        ],
    },
    "de_maddpg": {
        "title": "Building a Connected Communication Network for UAV Clusters Using DE-MADDPG",
        "authors": ["Zixiong Zhu", "Nianhao Xie", "Kang Zong", "Lei Chen"],
        "year": 2021,
        "venue": "Symmetry 13, 1537",
        "doi": "10.3390/sym13081537",
        "status": "reviewed",
        "methods": ["DE-MADDPG", "reward decomposition", "reward reshaping", "virtual leader-follower"],
        "baselines": ["MADDPG"],
        "metrics": ["success rate", "steps to connectivity", "reward", "state-space violation"],
        "concepts": ["CTDE", "Partial Observability", "FANET"],
        "summary": "DE-MADDPG controls UAV motion to rebuild a connected cluster network using reward decomposition, shaping, and a virtual leader-follower coordinate model.",
        "problem": "Restore single connectivity after UAV failures or communication interruptions when neighbor motion information may be unavailable.",
        "method": "MADDPG is decomposed into global and local critics with shaped intermediate rewards. A virtual navigator stabilizes the moving coordinate/state space.",
        "training": "20,000 episodes, replay buffer 200,000, batch 1024, actor/critic learning rates 0.001, centralized training and local actor execution.",
        "execution": "Each UAV outputs a discretized 3D movement action from local observation; this controls topology rather than selecting packet next hops.",
        "experiment": "Python/PaddlePaddle simulation in a moving 100 m cube. Training curves, 100-episode tests, 10 training runs, and varying UAV counts compare DE-MADDPG with MADDPG.",
        "results": "DE-MADDPG learns faster, reduces state-space violations, and achieves higher connectivity success than MADDPG. More UAVs make convergence harder.",
        "theory": "A POMDP is stated, but no formal convergence or partial-observability bound is established.",
        "assumptions": "Homogeneous UAVs, simplified discrete headings/elevations, a virtual navigator, idealized communication radius, and single-connectivity objective.",
        "strengths": "Clear CTDE separation; repeated training curves; explicit hyperparameters; exposes scaling difficulty with UAV count.",
        "weaknesses": "Only MADDPG baseline; no packet-routing metrics; simplified movement/action model; no energy/radio stack; single connectivity is fragile.",
        "fragile": "Reward shaping and virtual-coordinate construction may encode much of the solution. Success in a bounded cube may not transfer to dynamic routing.",
        "repro": "medium",
        "globe": "medium",
        "connection": "Useful for CTDE, reward shaping critique, and topology-control/routing separation. GLOBE++ should avoid conflating connectivity control with packet routing.",
        "supports": "CTDE motivation, local execution, partial observation, and scaling discussion.",
        "cannot": "It cannot support routing PDR/delay claims, policy distillation, graph routing, or heuristic-free next-hop selection.",
        "evidence": [
            ("POMDP, DE-MADDPG contributions and CTDE", "pp. 1-2", "direct-claim"),
            ("Training settings and network architecture", "pp. 11-12", "experimental-result"),
            ("Repeated tests and comparison with MADDPG", "pp. 12-15", "experimental-result"),
            ("Single-connectivity limitation", "p. 15", "limitation"),
        ],
    },
    "drama": {
        "title": "DRAMA: A Dynamic Packet Routing Algorithm using Multi-Agent Reinforcement Learning with Emergent Communication",
        "authors": ["Wang Zhang", "Chenguang Liu", "Yue Pi", "Yong Zhang", "Hairong Huang", "Baoquan Rao", "Yulong Ding", "Shuanghua Yang", "Jie Jiang"],
        "year": 2025,
        "venue": "arXiv:2504.04438v1",
        "doi": "",
        "status": "reviewed",
        "methods": ["MARL", "emergent communication", "attention aggregation", "graph-based Q-network"],
        "baselines": ["SPF", "Backpressure", "PPO", "Q-routing", "MADDPG", "DQRC"],
        "metrics": ["packet delivery rate", "latency", "communication overhead", "topology adaptation"],
        "concepts": ["Graph Neural Network", "Partial Observability", "Routing Overhead", "OOD Generalization"],
        "summary": "DRAMA uses learned inter-router messages and a topology-flexible Q-network to adapt packet routing to load, link/node failures, and added routers.",
        "problem": "Distributed packet routing under congestion and topology changes, including router addition that fixed-size neural policies cannot handle.",
        "method": "Routers encode local observations into learned messages, aggregate C-hop messages with attention, and predict neighbor-specific Q-values. TD loss is supplemented with a weighted-shortest-path estimated-cost loss.",
        "training": "Off-policy Q-learning-style MARL with replay, target network, emergent communication, and auxiliary cost supervision.",
        "execution": "Distributed but requires runtime learned communication among routers. Message rounds and quantization directly trade overhead for performance.",
        "experiment": "Python packet simulator with Poisson traffic and fixed link bandwidth; 10 simulations of 512 steps for standard tests, 50 random failure/extension tests, synthetic topology and the real ATT topology.",
        "results": "DRAMA reaches 100% delivery at the heaviest toy load with 18.15 ms latency, adapts to failures and router addition without retraining, and dominates baselines on ATT. Quantized, 10-step messages use 0.3% of original communication volume with modest degradation.",
        "theory": "No Dec-POMDP/performance bound. Scalability is argued through topology-flexible network structure and tested on ATT.",
        "assumptions": "Simplified queues/constant links, learned message exchange at deployment, auxiliary shortest-path targets, and latency computed only for delivered packets.",
        "strengths": "Strong and diverse baselines; dynamic node/link tests; real topology; explicit communication-overhead ablation; direct packet-routing relevance.",
        "weaknesses": "Not a UAV PHY/MAC simulation; runtime communication conflicts with strictly local execution; auxiliary shortest-path supervision is a handcrafted bias; delivered-only latency risks survivorship bias.",
        "fragile": "Results may depend on message synchronization and reliable neighbor communication. Larger/richer topology scaling is explicitly left as future work.",
        "repro": "medium-high",
        "globe": "high",
        "connection": "A key comparison for graph-structured local routing and dynamic-topology generalization. GLOBE++ must clearly contrast zero/limited execution communication with DRAMA's learned message exchange and account for ego-graph acquisition overhead.",
        "supports": "Dynamic-topology tests, communication-overhead analysis, graph policy design, and routing baselines.",
        "cannot": "It cannot support global teacher distillation, FANET radio performance, or an irreducible observation-gap theory.",
        "evidence": [
            ("Emergent communication and topology-flexible routing", "pp. 1-2", "direct-claim"),
            ("Simulator, metrics and six baselines", "p. 6", "experimental-result"),
            ("Load, ablation and overhead results", "pp. 7-8", "experimental-result"),
            ("Failure, node addition, ATT topology and limitations", "pp. 8-10", "experimental-result"),
        ],
    },
    "edp": {
        "title": "EDP Protocol: Advancing Mobility-Aware Drone Network Connectivity with Adaptive Routing",
        "authors": ["Jingjing Wang", "Houze Feng", "Jianrui Chen", "Lin Zhou", "Mengyuan Zhang", "Chunxiao Jiang"],
        "year": "needs-verification",
        "venue": "IEEE/ACM Transactions on Networking manuscript",
        "doi": "",
        "status": "reviewed",
        "methods": ["adaptive Kalman mobility prediction", "probabilistic flooding", "multi-metric route maintenance"],
        "baselines": ["AODV", "LEPR Variant", "MPR Variant", "NC Variant"],
        "metrics": ["end-to-end delay", "PDR", "routing overhead ratio", "route lifetime", "route discovery frequency"],
        "concepts": ["FANET", "UAV Routing", "Routing Overhead", "End-to-End Delay"],
        "summary": "EDP is a non-RL FANET baseline combining mobility prediction, coverage-aware probabilistic flooding, and link/load/lifetime-aware route maintenance.",
        "problem": "Reduce flooding overhead, route breakage, and delay in sparse, rapidly moving drone networks.",
        "method": "An adaptive Kalman predictor estimates mobility; conical neighbor-coverage forwarding prunes route-request floods; route selection and recovery use distance, traffic load, expected delay, and route lifetime.",
        "training": "No learned policy. Parameters and formulas are designed analytically.",
        "execution": "Distributed reactive routing with periodic mobility/link information and hand-engineered route discovery and maintenance.",
        "experiment": "NS-3, IEEE 802.11b, YansWifiPhy, Friis loss, custom 3D Gauss-Markov mobility, 10-80 drones, 4-20 m/s, 10-50 KB/s generation. The paper says scenarios are simulated extensively but does not state seed count.",
        "results": "Reported gains include 17.6% lower delay on long paths, 15.8% higher PDR at 40 drones, PDR above 0.65 at 20 m/s, overhead below 0.18 at high speed, and 29.4% fewer rediscoveries at 12 m/s versus AODV.",
        "theory": "Provides overhead and M/G/1 delay derivations under simplifying assumptions; these are protocol analyses rather than RL performance theory.",
        "assumptions": "Accurate enough mobility estimates, multi-metric normalization, customized Gauss-Markov motion, simplified path loss, and adapted component baselines.",
        "strengths": "Direct FANET/NS-3 relevance; metrics include overhead and stability; component-aligned variants clarify mechanism contributions; explicit extreme-mobility limitation.",
        "weaknesses": "Publication year/status and seed count are not stated; variants are adapted by the authors rather than independent implementations; no learned routing comparison; no confidence intervals.",
        "fragile": "Route lifetime still collapses at extreme speed. Prediction computation and asynchronous update cost are acknowledged but not fully evaluated.",
        "repro": "medium-low",
        "globe": "high",
        "connection": "An important non-learning baseline and source of mobility/overhead tests. It is also a counterexample to heuristic-free purity: strong performance comes from engineered prediction, flooding, and route scores.",
        "supports": "NS-3 scenario design, overhead/stability metrics, and predictive routing baselines.",
        "cannot": "It cannot support MARL, policy distillation, GNN execution, or partial-observability theory.",
        "evidence": [
            ("Protocol components and motivation", "pp. 1-2", "direct-claim"),
            ("Analytical overhead and delay model", "pp. 8-9", "theoretical-claim"),
            ("NS-3 setup and baseline variants", "pp. 9-10", "experimental-result"),
            ("Delay, PDR, overhead, stability and limitations", "pp. 10-12", "experimental-result"),
        ],
    },
    "iqmr": {
        "title": "Improved Q-learning based Multi-hop Routing for UAV-Assisted Communication",
        "authors": ["N P Sharvari", "Dibakar Das", "Jyotsna Bapat", "Debabrata Das"],
        "year": 2024,
        "venue": "arXiv:2408.09109v1",
        "doi": "",
        "status": "reviewed",
        "methods": ["Q(lambda)-learning", "multi-objective reward", "operational modes", "Gauss-Markov mobility"],
        "baselines": ["QMR", "Q-FANET"],
        "metrics": ["residual energy", "data throughput", "cumulative reward", "convergence"],
        "concepts": ["FANET", "UAV Routing", "Partial Observability"],
        "summary": "IQMR uses Q(lambda)-learning and a multi-objective state/reward design for energy-aware, collision-constrained multi-hop forwarding from UAVs to a terrestrial base station.",
        "problem": "Route surveillance data despite intermittent connectivity, energy depletion, collision risk, and network fragmentation without predefined UAV paths.",
        "method": "Tabular Q(lambda) updates next-hop values using residual energy, ACK status, coverage probability, collision probability, and source-destination alignment. UAVs switch among discovery, receive, transmit, and charge modes.",
        "training": "Online epsilon-greedy Q(lambda) learning; the paper studies learning/exploration rates and reports convergence in fewer than 500 episodes.",
        "execution": "Distributed neighbor-table routing, but with a heavily engineered objective, constraints, and tie-breaking geometry.",
        "experiment": "MATLAB simulation, 50 UAVs, 3D Gauss-Markov mobility, Nakagami fading, 802.11n, 2 Mbps CBR, 1000 m radius/300 m height. Fragmentation and rejoining behavior are tested.",
        "results": "The paper reports 32.27-36.35% better energy-consumption efficiency and 25.19-32.05% higher throughput than QMR and Q-FANET.",
        "theory": "No formal MARL/Dec-POMDP analysis or performance guarantee.",
        "assumptions": "GPS, ACKs, energy and collision estimates, one TBS, 50 UAVs, MATLAB network model, and manually selected objective priorities.",
        "strengths": "Direct UAV multi-hop routing; models energy, fragmentation, rejoining and collision constraints; compares against Q-learning FANET methods.",
        "weaknesses": "Only two baselines; no seed count or confidence intervals; throughput/energy dominate while PDR, delay, and routing overhead are absent; simulator fidelity is unclear.",
        "fragile": "Performance depends on normalized multi-objective weights and access to several estimated quantities. Tabular Q-values may scale poorly.",
        "repro": "low-medium",
        "globe": "high",
        "connection": "A direct Q-learning baseline for GLOBE++. Its hand-built state, reward, constraints, and tie-breaking illustrate exactly what a policy-learning method must ablate against.",
        "supports": "Energy/fragmentation scenarios and Q-learning routing baseline design.",
        "cannot": "It cannot support CTDE, GNN, policy distillation, rigorous statistics, or partial-observability bounds.",
        "evidence": [
            ("IQMR motivation and contributions", "pp. 1-2", "direct-claim"),
            ("Algorithm and engineered decision variables", "p. 9", "direct-claim"),
            ("MATLAB simulation parameters", "p. 10", "experimental-result"),
            ("Energy/throughput comparison and limitations", "pp. 11-12", "experimental-result"),
        ],
    },
    "globe_original": {
        "title": "GLOBE-Routing: Global-to-Local Knowledge-Distilled Graph-MAPPO for Decentralized Routing in UAV Swarm FANETs",
        "authors": ["Author Name", "Coauthor Name", "Advisor Name"],
        "year": 2026,
        "venue": "Unpublished research draft",
        "doi": "",
        "status": "draft",
        "methods": ["Graph-MAPPO", "CTDE", "latent context distillation", "two-hop forwardability"],
        "baselines": ["planned only"],
        "metrics": ["planned PDR", "delay", "throughput", "overhead", "energy", "routing holes"],
        "concepts": ["MAPPO", "CTDE", "Graph Neural Network", "Latent Distillation", "Ego-Graph"],
        "summary": "The current GLOBE-Routing draft proposes global graph-context latent distillation into local predictors plus a two-hop forwardability representation, but contains no validated experimental results.",
        "problem": "Use training-time LEO/base-station global visibility to improve decentralized FANET next-hop decisions without online global commands.",
        "method": "A global GNN encoder produces latent context z; local history predicts that latent using squared error. A lightweight MLP actor consumes local/2-hop features and predicted context. Reward includes many engineered routing terms.",
        "training": "Planned MAPPO with centralized critic and latent L2 distillation. The manuscript describes an algorithm but no completed training campaign.",
        "execution": "Local/2-hop observations, predicted global context, action masks, and two-hop forwardability. The actor shown is an MLP, not a local GNN.",
        "experiment": "Only a proposed protocol: Python training followed by NS-3/ns3-gym validation, 20-100 UAVs, 5-10 seeds, traditional and RL baselines, and component ablations.",
        "results": "No numerical results are reported. The manuscript explicitly says figures/tables must be generated from validated logs.",
        "theory": "Complexity expressions are provided, but no Dec-POMDP formulation, information-gap decomposition, or KL/performance bound.",
        "assumptions": "Training-time global observer, reliable local/2-hop summaries, a latent predictor that can recover global context, many shaped reward terms, and LDT/SINR/energy thresholds in action masking.",
        "strengths": "Clear training/execution split; explicit no-fabrication statement; broad planned baseline/metric suite; identifies global-to-local information transfer.",
        "weaknesses": "Latent L2 is not decision-aligned; local actor is MLP; two-hop forwardability and rich reward are strong heuristics; mask includes quality thresholds beyond invalidity; no results, authors, or validated citations.",
        "fragile": "The central claim depends on predicting global latent context from aliased local histories. The draft does not separate irreducible information loss from predictor error.",
        "repro": "not-yet-evaluable",
        "globe": "critical",
        "connection": "This is the direct predecessor/design draft. It establishes the exact delta for GLOBE++: replace latent L2 with policy KL, use a local ego-graph GNN, keep only invalid-action masks, remove two-hop score heuristics, and formalize partial observability.",
        "supports": "Internal design history and experiment planning, not external evidence.",
        "cannot": "It cannot support any empirical superiority or peer-reviewed novelty claim.",
        "evidence": [
            ("Draft title, architecture and claimed contributions", "pp. 1-2", "direct-claim"),
            ("Latent L2 distillation and MLP actor", "p. 4", "direct-claim"),
            ("Planned baselines, metrics and seeds", "pp. 5-6", "assumption"),
            ("No validated results and explicit limitations", "pp. 5-7", "limitation"),
        ],
    },
    "alam_thesis": {
        "title": "Routing Algorithms Based on Reinforcement Learning for Unmanned Aerial Vehicle Swarm Networks",
        "authors": ["Muhammad Morshed Alam"],
        "year": 2023,
        "venue": "Doctoral dissertation, Chosun University",
        "doi": "",
        "status": "reviewed",
        "methods": ["JTCR", "TAQR", "QRIFC", "DMA-DDPG", "JTFR", "two-hop state"],
        "baselines": ["multiple protocol-specific baselines"],
        "metrics": ["PDR", "end-to-end delay", "retransmissions", "energy", "connectivity", "coverage", "control overhead"],
        "concepts": ["FANET", "UAV Routing", "CTDE", "Partial Observability", "Routing Overhead"],
        "summary": "This dissertation develops three coupled mobility/routing systems: JTCR with topology-aware Q-routing, QRIFC with two-hop predictive Q-learning, and JTFR with distributed multi-agent DDPG.",
        "problem": "Jointly manage swarm mobility, topology, radio resources, and packet routing under dynamic links, delay, interference, and energy constraints.",
        "method": "JTCR combines virtual-force mobility, fuzzy clustering, and Q-routing; QRIFC combines adaptive flocking with two-hop Q-learning; JTFR jointly selects trajectory, frequency, and relay using LSTM actors and multi-head attentional critics.",
        "training": "Protocol-specific online Q-learning or DMA-DDPG. Rich multi-objective rewards include link duration, packet transmission success, SINR, queue delay, and residual energy.",
        "execution": "Distributed/local decisions use up to two-hop neighbor information and substantial mobility/topology-control coordination.",
        "experiment": "Extensive custom computer simulations across node count and velocity. The dissertation reports mission and communication metrics and compares each contribution with its own baselines.",
        "results": "Abstracted results report JTCR: 7-21% PDR gain, 9-37% delay reduction, 15-23% energy reduction; QRIFC: 9-23% PDR gain and 21-40% delay reduction; JTFR: 15-32% PDR gain, 30-60% delay reduction, and 20-46% energy reduction.",
        "theory": "Optimization formulations and complexity discussions are present, but no Dec-POMDP information-gap or policy-distillation theory.",
        "assumptions": "Behavior-based swarm mobility, two-hop control exchange, predictive link duration, multi-objective reward weights, and custom simulation models.",
        "strengths": "Broad, directly relevant FANET routing treatment; multiple algorithms; coverage/connectivity/overhead/energy metrics; explicit coupling of mobility and routing.",
        "weaknesses": "Three large systems make causal attribution difficult; many hand-designed modules and reward terms; no common modern MARL baseline across all contributions; statistical seed reporting is not prominent.",
        "fragile": "Performance may rely on topology control and two-hop information rather than the routing learner itself. Joint mobility changes the task distribution relative to fixed-mobility baselines.",
        "repro": "medium-low",
        "globe": "high",
        "connection": "A major design and baseline source for GLOBE++. It highlights two-hop predictive features, multi-objective rewards, and mobility-routing coupling, but also motivates clean ablations and strict accounting of information/overhead.",
        "supports": "FANET design issues, metric selection, Q-learning/MARL baselines, topology-control confounds, and two-hop overhead questions.",
        "cannot": "It cannot support global-to-local policy distillation, local GNN novelty, or irreducible observation-gap theory.",
        "evidence": [
            ("Dissertation identity and scope", "pp. 2-4", "direct-claim"),
            ("Three proposed routing/control systems", "pp. 13-15", "direct-claim"),
            ("Reported aggregate improvements", "p. 15", "experimental-result"),
            ("Conclusions, limitations and future directions", "pp. 163-165", "limitation"),
        ],
    },
}

SOURCES = [
    ("evo_qgeo", "a7f805f88f8c4ddfd63e073752b6b640c060895ca0853a5c5475e16c196c50b6", "1. Reinforcement-Learning-Based Geographic Routing Considering.pdf"),
    ("glomappo", "8a2e4c995fb736af9fbcc21704a4194a40f193f4d52c9eb3de4fc2cce5a85b87", "1GLo-MAPPO_ Multi-Agent Deep Reinforcement Learning for Energy-Efficient UAV-Assisted LoRa Networks.pdf"),
    ("wpt_routing", "2f45c3e066ffc75d21af60551d1d7cfca47cdfe83d8f0e932bacefe520085afc", "2204.11477v1.pdf"),
    ("de_maddpg", "c0dcfc5472f95ce602ae316d25e5cac61e3dd33f2c75b777f4a86a1343e6de9c", "Building_a_Connected_Communication_Network_for_UAV.pdf"),
    ("drama", "5359264f780f205d643fe534a561380760d02e584e1b9c1e872dedd812d5c395", "DRAMA_ A Dynamic Packet Routing Algorithm using Multi-Agent Reinforcement Learning with Emergent Communication.pdf"),
    ("edp", "d02af4c8366e1af9732cb2a9831ed111a5c51358cd40d484bb646121905b024e", "EDP Protocol Advancing Mobility-Aware Drone.pdf"),
    ("iqmr", "e2f7cbd36779dd9fac34f0bba5fe3e21ecc2d9090ac8770e3cc13f0920d90304", "Improved Q-learning based Multi-hop Routing for UAV-Assisted Communication.pdf"),
    ("evo_qgeo", "a7187e06116f47121834372c45cca137bdce84b073879038e06ed716e9d41670", "Reinforcement-Learning-Based Geographic Routing Considering Future Evolution of Link States for UAV Networks (1).pdf"),
    ("evo_qgeo", "61202dcc5cc73c9a6e2aeef4323502a30c363fe618d603eeece644e358694e9f", "Reinforcement-Learning-Based Geographic Routing Considering Future Evolution of Link States for UAV Networks.pdf"),
    ("globe_original", "7f702b9995e30b71465dffa158c1a973be05e2a3a39079b8ca1c0a8e6eaf1b02", "globe_routing_manuscript.pdf"),
    ("alam_thesis", "55f86deceb9f6232cd7e500f6bb87ba9ce55588221eb9d444a86f355ab67a121", "무인 비행체 군집 네트워크를 위한 강화학습 기반 라우팅 알고리즘.pdf"),
]


def paper_card(key: str, paper: dict, source_rows: list[tuple[str, str]]) -> str:
    title = paper["title"]
    source_files = [name for _, name in source_rows]
    source_notes = "\n".join(
        f"- Source file: `{name}` | Source path: `raw/papers/{name}` | Source hash: `{source_hash}`"
        for source_hash, name in source_rows
    )
    evidence = "\n".join(
        f"| {claim} | {pages} | {kind} | high |" for claim, pages, kind in paper["evidence"]
    )
    metadata = {
        "title": title,
        "type": "paper-card",
        "created": today(),
        "updated": today(),
        "status": paper["status"],
        "tags": ["paper", "literature", "reviewed"],
        "source_files": source_files,
        "paper_title": title,
        "authors": paper["authors"],
        "year": paper["year"],
        "venue": paper["venue"],
        "doi": paper["doi"],
        "url": "",
        "research_area": ["UAV", "routing", "reinforcement learning"],
        "methods": paper["methods"],
        "baselines": paper["baselines"],
        "metrics": paper["metrics"],
        "datasets_or_simulators": [],
        "related_concepts": paper["concepts"],
        "related_papers": [],
        "globe_relevance": paper["globe"],
        "confidence": "high",
        "uses_gnn": "Graph" in " ".join(paper["methods"]) or "attention" in " ".join(paper["methods"]).lower(),
        "uses_marl": any(token in " ".join(paper["methods"]) for token in ["MAPPO", "MADDPG", "MARL", "multi-agent"]),
        "uses_kd": "distillation" in " ".join(paper["methods"]).lower(),
        "handles_partial_observability": "Partial Observability" in paper["concepts"],
        "simulator": paper["experiment"].split(".")[0],
        "seed_count": "See experimental setup",
        "statistical_reporting": "See reviewer critique",
        "reproducibility": paper["repro"],
    }
    concept_links = ", ".join(f"[[{item}]]" for item in paper["concepts"])
    body = f"""# {title}

## 1. One-line Summary

{paper["summary"]}

## 2. Problem Setting

{paper["problem"]}

## 3. Core Contribution

### Paper Claim

{paper["method"]}

### Agent Assessment

{paper["connection"]}

## 4. Method

### 4.1 Model / Algorithm

{paper["method"]}

### 4.2 Training Setup

{paper["training"]}

### 4.3 Execution Assumption

{paper["execution"]}

## 5. Experimental Setup

{paper["experiment"]}

| Item | Description |
| --- | --- |
| Baselines | {", ".join(paper["baselines"])} |
| Metrics | {", ".join(paper["metrics"])} |
| Reproducibility | {paper["repro"]} |

## 6. Main Results

**Paper Claim:** {paper["results"]}

## 7. Theoretical Claims

{paper["theory"]}

## 8. Assumptions

{paper["assumptions"]}

## 9. Limitations

{paper["weaknesses"]}

## 10. Reviewer-level Critique

### Strengths

{paper["strengths"]}

### Weaknesses

{paper["weaknesses"]}

### Hidden Fragile Assumptions

{paper["fragile"]}

### Reproducibility Risk

**{paper["repro"]}**. Bibliographic fields and claims were read from the supplied PDF; missing seed or publication metadata remains marked where applicable.

## 11. Connection to GLOBE++

{paper["connection"]}

Related concepts: {concept_links}.

## 12. What This Paper Can Support in My Manuscript

{paper["supports"]}

## 13. What This Paper Cannot Support

{paper["cannot"]}

## 14. Follow-up Questions

- Which information is genuinely available to a decentralized UAV at execution?
- Which gain survives equal-budget multi-seed comparison and component ablation?
- Is performance caused by learning, privileged information, or engineered heuristics?

## 15. Source Evidence

| Evidence | Page/Section | Evidence Type | Confidence |
| --- | --- | --- | --- |
{evidence}

### Source Files

{source_notes}
"""
    return yaml_document(metadata, body)


def source_summary(key: str, paper: dict, source_hash: str, filename: str, duplicates: int) -> str:
    title = f"Source - {paper['title']} - {source_hash[:8]}"
    duplicate_note = (
        f"This is one of {duplicates} byte-distinct PDFs with identical extracted text for this paper."
        if duplicates > 1
        else "No same-title duplicate was identified in this batch."
    )
    evidence = "\n".join(
        f"| {claim} | {pages} | {kind} | high |" for claim, pages, kind in paper["evidence"]
    )
    metadata = {
        "title": title,
        "type": "source-summary",
        "created": today(),
        "updated": today(),
        "status": "reviewed",
        "tags": ["raw-summary", "paper"],
        "source_files": [filename],
        "source_file": filename,
        "source_path": f"raw/papers/{filename}",
        "source_hash": source_hash,
        "source_type": "pdf",
        "processed_by": "Codex",
        "related_concepts": paper["concepts"],
        "related_papers": [paper["title"]],
        "globe_relevance": paper["globe"],
        "confidence": "high",
    }
    concepts = ", ".join(f"[[{item}]]" for item in paper["concepts"])
    body = f"""# {title}

## 1. Source Metadata

| Field | Value |
| --- | --- |
| Actual paper title | [[{paper["title"]}]] |
| Authors | {", ".join(paper["authors"])} |
| Year / venue | {paper["year"]} / {paper["venue"]} |
| Source file | `{filename}` |
| Source path | `raw/papers/{filename}` |
| Source hash | `{source_hash}` |
| Processed date | {today()} |

## 2. Core Summary

{paper["summary"]}

## 3. Key Concepts

{concepts}

## 4. Important Details

- Method: {paper["method"]}
- Experiment: {paper["experiment"]}
- Duplicate handling: {duplicate_note}

## 5. Extracted Claims

| Claim | Page/Section | Evidence Type | Confidence |
| --- | --- | --- | --- |
{evidence}

## 6. Relevance to My Research

### Directly useful for GLOBE++

{paper["connection"]}

### Indirectly useful

{paper["supports"]}

### Not useful / weak relevance

{paper["cannot"]}

## 7. Concept Pages to Create or Update

{concepts}

## 8. Paper Cards to Create or Update

[[{paper["title"]}]]

## 9. Contradictions or Tensions with Existing Wiki

{paper["weaknesses"]}

## 10. Follow-up Questions

- Which reported findings require reproduction?
- Which assumptions conflict with mask-only, local ego-graph execution?

## 11. Citation Notes

- Source file: `{filename}`
- Source path: `raw/papers/{filename}`
- Source hash: `{source_hash}`
- Evidence table above records page references and evidence type.
"""
    return yaml_document(metadata, body)


def main() -> int:
    grouped: dict[str, list[tuple[str, str]]] = {}
    for key, source_hash, filename in SOURCES:
        grouped.setdefault(key, []).append((source_hash, filename))

    for placeholder in PAPER_DIR.glob("*_paper.md"):
        placeholder.unlink()

    registry = load_registry()
    card_paths: dict[str, Path] = {}
    for key, paper in PAPERS.items():
        card_path = PAPER_DIR / f"{key}.md"
        card_path.write_text(paper_card(key, paper, grouped[key]), encoding="utf-8")
        card_paths[key] = card_path

    for key, source_hash, filename in SOURCES:
        entry = registry["files"][source_hash]
        summary_path = ROOT / entry["created_pages"][0]
        summary_path.write_text(
            source_summary(key, PAPERS[key], source_hash, filename, len(grouped[key])),
            encoding="utf-8",
        )
        card_rel = card_paths[key].relative_to(ROOT).as_posix()
        entry["created_pages"] = [summary_path.relative_to(ROOT).as_posix(), card_rel]
        entry["status"] = "reviewed"
        entry["notes"] = "PDF title verified from content; reviewer-level analysis completed."
    save_registry(registry)
    from localize_korean import main as localize_korean

    localize_korean()
    print(f"Finalized {len(PAPERS)} paper cards and {len(SOURCES)} source summaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
