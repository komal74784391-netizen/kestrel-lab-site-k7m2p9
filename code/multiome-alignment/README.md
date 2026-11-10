# Anchored Factor Graph Multiome Alignment

Code release for the paper **Joint Multiome Alignment with Anchored Factor
Graphs** (GIC 2022).

The implementation covers anchor selection from paired multiome assays,
factor-graph construction and block-coordinate optimisation of the joint
alignment objective.

## Layout

- `src/afa/factor_graph.py` — graph definition and message schedule
- `src/afa/align.py` — block-coordinate alignment solver
- `scripts/run_alignment.py` — run alignment on a paired dataset
- `scripts/evaluate_alignment.py` — label transfer and alignment metrics
- `configs/align.yaml` — hyperparameters

## Usage

```bash
pip install -r requirements.txt
python scripts/run_alignment.py --rna data/rna.h5ad --atac data/atac.h5ad
```
