# LAM experiments

Scratch experiments that found and fixed the latent action model's failure modes (Sep 2026).
All paths are relative to the repo root; outputs go to `data/experiments/` and `data/checkpoints/`.

- `vq_variants.py`  VQ with data-init codes, usage-threshold restarts, optional entropy penalty; stride-N clip dataset that also returns the real DOOM actions.
- `arch_variants.py`  encoder/decoder probes: `diffenc` (code the change in pooled features), `diffspatial` (same, but keeps where the change happened), `resdec` (residual decoder).
- `train_scored.py NAME EPOCHS ARCH`  training with prediction-based scoring every epoch (perplexity, dPSNR, consistency across contexts, grouped latent->real mapping). Env knobs: WIDTH BLOCKS HEADS BATCH NUM_WORKERS.
- `eval_lam.py CKPT OUT_DIR [N]`  val metrics + PNG grids.   `diagnose_codes.py CKPT`  what the codes correlate with.   `pan_sanity.py`  does the decoder render pans.
- `train_variant.py NAME EPOCHS STRIDE NORMALIZE [ENTROPY] [ARCH]`  the earlier ablation trainer.

TensorBoard: `python3 -m tensorboard.main --logdir data/experiments/tb --port 6006`
