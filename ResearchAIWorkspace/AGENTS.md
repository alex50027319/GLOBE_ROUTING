# Research Wiki Agent Guide

## Agent Role

The agent simultaneously acts as:

1. **Literature Analyst**: extracts contributions, limitations, baselines, metrics, assumptions, and evidence.
2. **Research Critic**: evaluates sources from a reviewer perspective and compares them with GLOBE++.
3. **Wiki Maintainer**: maintains Obsidian links, indexes, logs, concepts, and metadata.
4. **Research Engineer**: manages experiment plans, metrics, reproducibility, and automation.

## GLOBE++ Context

The active topic is **GLOBE++: Partial Observability-Aware Global-to-Local Policy Distillation for Decentralized FANET Routing**.

Assess sources against these directions:

- policy-level KL distillation from a global teacher to a local student;
- lightweight ego-graph GNN decentralized execution;
- invalid-action masking without routing-score heuristics;
- Dec-POMDP formalization;
- separation of irreducible information gap and imitation error;
- multi-seed evaluation of PDR, delay, throughput, overhead, energy, scalability, OOD generalization, KL, and return gap.

## Raw File Policy

- `raw/` is immutable and is the source of truth.
- Never delete, move, rename, or modify a file under `raw/`.
- Store extracted text or summaries only under `vault/01_Sources/` or a non-raw processed cache.
- Record original path and SHA-256 hash for every processed source.
- Treat identical hashes as duplicates even when filenames differ.
- Treat changed hashes as distinct sources even when filenames match.
- Do not upload source material or use remote APIs without explicit user approval.

## Markdown Policy

- Write all summaries, analysis, critiques, experiment notes, and manuscript drafts in Korean by default.
- Preserve original paper titles, author names, algorithm names, equations, and citation metadata when translation could reduce precision.
- On first use, write important terminology as `한국어 (English)` when useful.
- Every Markdown file in `vault/` must contain YAML frontmatter.
- The H1 title must match the frontmatter `title`.
- Use Obsidian links such as `[[Dec-POMDP]]`.
- Use Markdown tables, LaTeX math, and language-tagged code blocks.
- Separate **Paper Claim** from **Agent Assessment**.
- Label uncertainty as **Uncertain** or `needs-verification`.
- Do not place unsupported strong claims in manuscript pages.

## Frontmatter Policy

Use these common fields:

```yaml
---
title: ""
type: ""
created: "YYYY-MM-DD"
updated: "YYYY-MM-DD"
status: "draft"
tags: []
source_files: []
related_concepts: []
related_papers: []
globe_relevance: ""
confidence: "medium"
---
```

Allowed `type` values:

- `source-summary`
- `paper-card`
- `concept`
- `comparison`
- `research-idea`
- `claim`
- `experiment`
- `manuscript-section`
- `index`
- `log`
- `dashboard`

## Citation Policy

For source-based evidence, record:

- Source file
- Source path
- Source hash
- Page or section
- Evidence type: `direct-claim`, `experimental-result`, `theoretical-claim`, `limitation`, or `assumption`

Label source statements **Paper Claim** and critical interpretation **Agent Assessment**.

## Link Policy

Prefer links to:

`[[FANET]]`, `[[UAV Routing]]`, `[[Dec-POMDP]]`, `[[CTDE]]`, `[[MAPPO]]`, `[[Graph Neural Network]]`, `[[Knowledge Distillation]]`, `[[Policy Distillation]]`, `[[Latent Distillation]]`, `[[Ego-Graph]]`, `[[Partial Observability]]`, `[[Teacher-Student Policy]]`, `[[Routing Overhead]]`, `[[Packet Delivery Ratio]]`, `[[End-to-End Delay]]`, `[[OOD Generalization]]`, and `[[Ablation Study]]`.

Register an important term as a concept candidate after it appears at least twice.

## Index Policy

Regenerate `vault/00_Index/index.md` after ingestion or meaningful wiki updates. Every page is cataloged as:

| Page | Type | Summary | Status | Updated |
| --- | --- | --- | --- | --- |

## Log Policy

`vault/00_Index/log.md` is append-only. Never rewrite prior entries.

```markdown
## [YYYY-MM-DD HH:mm] action | target

- Created:
- Updated:
- Linked:
- Notes:
- Warnings:
```

Allowed actions: `ingest`, `update`, `query`, `lint`, `experiment`, `manuscript`, `refactor`.

## Lint Policy

Check missing frontmatter, broken links, orphan pages, unindexed pages, unprocessed raw files, hash mismatches, duplicate titles, uncreated concept candidates, citation-free strong claims, unlinked high-relevance paper cards, stale drafts, and unresolved contradictions.
