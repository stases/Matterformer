# Matterformer

Minimal active repo for shared periodic and non-periodic atomistic transformers.

## Layout

- `src/matterformer`: active package
- `scripts`: thin training and sampling entrypoints
- `tests`: unit and smoke tests for adapters, trunk, QM9 data, and models
- `legacy`: frozen imported research code kept for reference only

## Active Scope

- Shared dense-token transformer trunk with standard multi-head or 2-simplicial attention
- Geometry adapters for periodic and non-periodic inputs
- Shared QM9 data pipeline
- QM9 regression
- QM9 unconditional EDM generation
