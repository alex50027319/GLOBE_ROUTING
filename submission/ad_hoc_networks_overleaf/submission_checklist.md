# Ad Hoc Networks Submission Checklist

Official guide: <https://www.sciencedirect.com/journal/ad-hoc-networks/publish/guide-for-authors>

## Scientific readiness

- [ ] SwitchGLOBE full run is complete for seeds 42, 77, 123, 314, and 2718.
- [ ] Merged manifest says `mode=full` and contains the complete seed set.
- [ ] Raw rows have no missing or duplicate scenario-method-seed combinations.
- [ ] Calibration and final evaluation are separated.
- [ ] Primary claims use paired effects and uncertainty.
- [ ] PDR, deadline, p95 delay, energy, input bytes, switch behavior, and drop are reported together.
- [ ] Unfavorable results and failed scenarios are retained.
- [ ] External baseline results and SwitchGLOBE internal ablations are not presented as one campaign unless their simulator contracts are identical.

## Reviewer-risk closure

- [ ] AODV/OLSR or equivalent protocol baselines are included or their absence is justified.
- [ ] Input bytes are not called actual control overhead.
- [ ] The teacher is called a privileged reference policy, not an optimal oracle.
- [ ] The paper distinguishes structural holes from predictive link breaks.
- [ ] Simulator units and energy/link proxies are disclosed.
- [ ] Custom-simulator limitations are stated prominently.
- [ ] Direct competitors from TMC, TVT, TCOM, Drones, and RoutePPO are discussed fairly.
- [ ] Novelty is framed as global-to-local deployment plus selective recovery, not “PPO routing.”

## Manuscript files

- [ ] Author names, affiliations, email, ORCID, and corresponding author are final.
- [ ] Abstract contains audited numbers only and no citations.
- [ ] Keywords are final.
- [ ] Every figure and table is cited and has a self-contained caption.
- [ ] Figures are vector PDF/EPS or meet the current raster-resolution guidance.
- [ ] Highlights contain 3–5 bullets and meet the current character limit.
- [ ] Graphical abstract is prepared if used.
- [ ] References and DOI metadata are verified.
- [ ] Cover letter is updated with final contributions.

## Declarations

- [ ] CRediT author contributions are agreed by all authors.
- [ ] Funding text and grant numbers are exact.
- [ ] Competing-interest declaration is confirmed.
- [ ] Data and code availability matches the actual public release.
- [ ] Generative-AI declaration reflects actual tool use.
- [ ] All authors approve the final manuscript and submission.
- [ ] The manuscript is not under consideration elsewhere.

## Final technical check

- [ ] The Overleaf project compiles from a clean cache.
- [ ] There are no undefined references or citations.
- [ ] The PDF has no clipped tables, figures, or equations.
- [ ] The ZIP contains `main.tex`, `references.bib`, and every figure/source file.
- [ ] The official Guide for Authors was rechecked on the submission date.
