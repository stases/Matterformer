# Matterformer Codex Instructions

This repository is part of the server-native materials research workflow.

## Research Vault

The research vault is:

`~/research-vault`

Write new research-memory transactions to:

`~/research-vault/_inbox/transactions/pending/`

Do not directly rewrite curated wiki pages unless explicitly asked.

## Local Commands

- Inspect setup: `kb-status`
- Submit SLURM with provenance: `research-sbatch <sbatch-options> <script.sbatch>`
- Mark job start inside SLURM scripts: `research-run-start`
- Mark job finish inside SLURM scripts: `trap 'research-run-finish $?' EXIT`
- Harvest completed runs: `research-harvest`
- Ingest pending notes when Hermes is installed: `kb-ingest`
- Commit/push vault changes: `kb-push`

## Experiment Closeout

When closing out substantial work, preserve:

- goal and hypothesis
- changed files
- commands and configs
- datasets
- model and baseline details
- metrics and interpretation
- failures or surprises
- decisions
- research ideas
- open questions
- next actions
- artifact paths
- SLURM job ids when relevant

