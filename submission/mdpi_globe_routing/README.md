# MDPI Drones manuscript draft

This folder is an Overleaf-ready draft for the MDPI journal *Drones*, targeting the Drone Communications section. It uses the MDPI `Definitions/mdpi` class bundle and includes the Phase 12 figures in the manuscript.

## Official guidance

- [MDPI LaTeX preparation](https://www.mdpi.com/authors/latex)
- [Drones aims and scope](https://www.mdpi.com/journal/Drones/about)
- [Drones instructions for authors](https://www.mdpi.com/journal/Drones/instructions)
- [Drone Communications section](https://www.mdpi.com/journal/drones/sections/drone_communications)

MDPI recommends a LaTeX ZIP containing all source files and images. The official template is updated periodically, so compare this bundled class with the current MDPI template immediately before submission.

## Overleaf

1. Upload `mdpi_globe_routing_overleaf.zip` using **New Project → Upload Project**.
2. Set `main.tex` as the main document if Overleaf does not select it automatically.
3. Compile with pdfLaTeX (the source also compiles locally with Tectonic/XeTeX after removing the `pdftex` option, as currently configured).
4. Replace author metadata, repository DOI, funding, and acknowledgments before submission.

## Scientific gates

All quantitative claims in this draft come from the Phase 12 full archive: five training seeds, 14 scenarios, 200 episodes per scenario, and 84,000 episode rows. The draft intentionally does not claim Phase 13/P+ variants. Red TODO markers are administrative fields that must be confirmed by the authors.

The aggregate metrics are connected-pair PDR 0.905, deadline delivery 0.838, p95 delay 4.264, energy proxy 1.779, and policy input bytes 4821. The draft reports the energy and highly lossy predictive-break limitations rather than claiming universal superiority.
