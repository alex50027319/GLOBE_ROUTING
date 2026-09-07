# Metric definitions

- **Connected-pair PDR:** delivered packets divided by episodes whose source–destination pair is initially connected.
- **Overall PDR:** delivered packets divided by all episodes.
- **Deadline delivery ratio:** packets delivered no later than the simulator deadline divided by all episodes.
- **p95 successful delay:** 95th percentile of delay steps among delivered packets only. A no-delivery cell is N/A, never zero.
- **Decision latency:** wall-clock observation-to-action software time. CUDA timing is synchronized. This excludes radio transmission and multi-hop queueing.
- **Network E2E delay:** route-level delay in simulator steps. It is not the neural microbenchmark.
- **Energy per delivered packet:** simulator proxy under the specified aggregation. It is not joules.
- **Policy input bytes:** in-memory bytes presented to the policy. It is not wireless control overhead.
- **Directional paired difference:** candidate minus reference for higher-is-better metrics and reference minus candidate for lower-is-better metrics; positive always favors the candidate.

