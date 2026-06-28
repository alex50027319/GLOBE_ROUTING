# Missing Experiments

## Required Before Submission

1. Full Phase 13 P+ run with the same protocol as Phase 12:
   - 5 seeds: 42, 77, 123, 314, 2718
   - 14 scenarios
   - 200 episodes per scenario
   - same metrics and paired effects

2. Phase 13 ablations:
   - no link-loss gate
   - no energy tie
   - no drop suppression
   - top-1 onward only

3. Control overhead:
   - control bytes per generated packet
   - control bytes per delivered packet
   - online messages per hop
   - compare especially against DRAMA

4. Energy sensitivity:
   - current proxy energy
   - distance-squared transmit model sensitivity
   - optional radio model if available

5. Classical routing baselines if claimed:
   - AODV
   - OLSR
   - only include if implemented in the same simulator conditions

## Recommended Before Submission

1. Larger topology scalability:
   - more node counts
   - denser and sparser networks

2. Mobility stress:
   - speed sweep
   - direction-change frequency
   - stale neighbor state

3. Link-loss stress:
   - stochastic link loss probability sweep
   - delayed neighbor-table update

4. Statistical reporting:
   - paired seed differences
   - confidence intervals
   - effect sizes

5. Failure analysis:
   - routing hole failures
   - link break failures
   - agent drop failures
   - TTL/deadline failures
