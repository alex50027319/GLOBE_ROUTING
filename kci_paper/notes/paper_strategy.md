# Paper Strategy Formulation

This note documents the strategic selection of the KCI paper structure based on the project audit.

## Selected Strategy: **실험 중심 KCI 논문 (Evaluation-focused KCI Paper)**

### Rationale:
1. **Sufficient Empirical Evidence**: The workspace contains a completed full-scale Phase 12 validation dataset representing 84,000 distinct episode runs over 14 scenarios and 5 random seeds.
2. **Established Comparative Baselines**: Results are compared directly against classical GPSR and three state-of-the-art baselines:
   - *Predictive Geographic Routing* (Heuristic)
   - *Evo-QGeo* (Predictive RL)
   - *DRAMA* (MARL with emergent communication)
3. **Verified Performance Gains**: We can make concrete claims on Packet Delivery Ratio (PDR) improvements (+32.5% over GPSR, +1.5% over Predictive Geo) and control footprint efficiency (128 bytes vs. 320 bytes for DRAMA).
4. **Active Implementation Pipeline**: The Phase 13 P+ code is fully implemented and tested, allowing us to present it as the final proposed model while stating its full multi-seed evaluation as immediate next steps.

### Outline of the Manuscript:
- **I. 서론**: Introduce FANET, Dec-POMDP, and the deployment gap of MARL.
- **II. 관련 연구**: Discuss MANET/FANET geographic, predictive, and learning-based routing protocols.
- **III. 시스템 모델 및 문제 정의**: Formulate routing as a Dec-POMDP with local observations.
- **IV. 제안 기법**: Detail GLOBE++ global-local distillation and the Predictive Risk Switch (PRS).
- **V. 시뮬레이션 환경**: Outline the scenarios, parameters, and compared baselines.
- **VI. 성능 평가 및 분석**: Present the Phase 12 results (PDR, latency, footprint) and scenario-level analysis.
- **VII. 논의 및 한계점**: Address simulator abstractions and Phase 13 validation plans.
- **VIII. 결론**: Summarize findings and future work.
