# Data and code availability — author completion draft

The manuscript is supported by the following internal artifact classes:

- common-simulator episode results for eight routing methods;
- five-seed, 14-scenario aggregated routing tables;
- paired statistical summaries and ablation tables;
- A100 Exact/Fused/Fast benchmark outputs and operator profiles;
- five main-figure source tables and 20 supplementary-figure source tables;
- implementation and checkpoint archives with recorded SHA-256 hashes;
- environment and package-freeze records.

Before submission, the authors should create a clean public release that excludes credentials, private Colab metadata, personally identifying paths, oversized redundant checkpoints, and unlicensed third-party materials. The release should preserve the scripts and immutable result files needed to reproduce each table and figure.

Suggested final statement:

> The code, configuration files, trained checkpoints, aggregate and episode-level evaluation data, profiling traces, and scripts used to reproduce all tables and figures will be made publicly available in an archival repository at [DOI/URL to be inserted after acceptance or upon submission, according to journal policy]. The anonymized review artifact is available at [anonymous URL, if used]. All primary numerical values reported in the article are also provided in machine-readable form in the supplementary package.

Author checks before release:

- [ ] assign a persistent DOI (for example, Zenodo);
- [ ] add a software license and data license;
- [ ] pin the exact commit and environment lock/freeze;
- [ ] document checkpoint provenance and SHA-256 hashes;
- [ ] supply a one-command or notebook-based reproduction path;
- [ ] test the release from a clean environment;
- [ ] confirm that the public artifact matches the values frozen in the manuscript.
