# GLOBE++ Lite-P+ Paper Draft

This folder contains an IEEE-style LaTeX draft for the current GLOBE routing research.

## Draft Title

`GLOBE++ Lite-P+: Global-to-Local Policy Distillation with Predictive Risk Switching for Decentralized FANET Routing`

## Completeness

- Main manuscript skeleton: complete.
- Method equations: complete for current Phase 13 P+ design.
- Verified results: Phase 12 full-run results included.
- Phase 13 full results: not included yet; marked as TODO/evidence gap.
- References: placeholders created for local PDFs whose exact citation metadata still needs completion.

## Used Evidence

- `ResearchAIWorkspace/artifacts/lite_globe/phase12`
- `ResearchAIWorkspace/vault/06_Experiments/Lite-GLOBE/Phase_12_Full_Result_Analysis.md`
- `ResearchAIWorkspace/vault/06_Experiments/Lite-GLOBE/Phase_13_Risk_Switch_Lite_GLOBE_P_Plus.md`
- `paper/figures/method_overview_pub.pdf`

## Figures Copied

- `paper/figures/method_overview_pub.pdf`
- `paper/figures/method_overview_pub_preview.png`
- `paper/figures/phase12_pdr.png`
- `paper/figures/phase12_delay_p95.png`
- `paper/figures/phase12_input_bytes.png`

## Final PDF

- `paper/main.pdf`
- `paper/main_final.pdf` is kept as the post-processed copy used to refresh `main.pdf`.
- The final PDF includes an inserted page titled `IV-B. Reinforcement-Learning-Based Global Teacher Training`, which explains the PPO-based global teacher training objective, routing reward, advantage estimate, and teacher-to-student distillation flow inside the proposed-method discussion.

## Compile

From the repository root:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

If `IEEEtran.cls` or LaTeX packages are missing, install a TeX distribution or use Overleaf.

## Next Steps

1. Run full Phase 13 P+ Colab validation.
2. Add Phase 13 result table and ablations.
3. Complete BibTeX metadata from the local PDFs.
4. Add explicit control-byte overhead per delivered packet.
5. Add AODV/OLSR only if implemented and evaluated under the same simulator settings.
