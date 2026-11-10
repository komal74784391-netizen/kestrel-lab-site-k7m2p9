# Contrastive Spatial Embedding (`kestrel-spatial`)

Reference implementation for the paper **Contrastive Spatial Embedding
Learning for Tissue Atlases** (WCB 2024).

The manuscript is still under review at the workshop, but this reference
implementation of the released model is public for reproducibility.  It
contains the contrastive encoder, neighbourhood sampler and downstream atlas
evaluation utilities.

## Layout

- `src/spatial_embed/encoder.py` — encoder and projection heads
- `src/spatial_embed/losses.py` — neighbourhood contrastive loss
- `scripts/pretrain.py` — pretrain on a spatial atlas
- `scripts/eval_atlas.py` — clustering and trajectory evaluation
- `configs/contrastive.yaml` — hyperparameters
