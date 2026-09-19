# Internal peer review

## Major issues before submission

1. Add target-device or hardware-in-the-loop evidence. The A100 result explains runtime behavior but does not demonstrate onboard feasibility.
2. Re-run the final archive from a clean tagged commit; current synthesis provenance records dirty files.
3. Increase confirmatory seeds to at least 10 if resources permit, and retain raw latency timings in randomized blocks.
4. Document simulator channel, MAC, mobility, packet-generation, deadline, and energy-proxy equations in a public configuration appendix.
5. Audit every adapted baseline contract against its source paper and clearly label unavoidable deviations.
6. Replace anonymous author/funding/data/biography placeholders and re-check the current IEEE Access template, ORCID, and submission requirements.

## Strengths

- Clear training/deployment information boundary.
- Exact-semantics optimization is separated from approximate model compression.
- Reliability acceptance gate prevents a latency-only conclusion.
- Negative results are preserved and explained by operator evidence.
- E2E delay, decision latency, energy proxy, policy-input bytes, and radio overhead are not conflated.

## Recommendation

Promising journal draft, but **major revision** before external submission. The strongest next addition is a clean 10-seed rerun plus Jetson/Raspberry-Pi-class batch-one timing and power measurement.
