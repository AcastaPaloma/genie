# Dynamics training loop: spec (v1)

Spec for a hand-written `train_dynamics` loop. Companion to `HYPERPARAMETERS.md`.
Paper reference: Genie 1, Section 2.3 and Appendix (MaskGIT-style masked token
prediction, Tables 9 and 12). Everything below is grounded in this repo's code;
anything marked *local choice* is ours, not the paper's.

## 1. Scope of v1

- Single device (cuda / mps / cpu). No DDP, no gradient accumulation.
- Same shape as `train_tokenizer` (vae.py) and `train_lam` (lam.py): epoch loop
  over local ArrayRecord shards, argparse `__main__`, atomic checkpoint per epoch.
- Frozen pretrained tokenizer (`VAE`) and frozen pretrained LAM. Only the
  `DynamicsModel` parameters get gradients. (Paper co-trains LAM + dynamics; v2.)
- Video representation: **integer token IDs** `(B,T,N)`, not the vector path.
  Masking is then an ID swap, and MaskGIT sampling produces IDs that feed
  straight back in. Pick one path and never mix them (their input layers differ).
- Online encoding: every step runs the frozen encoders under `no_grad` on the raw
  clip. A precomputed token cache is v2.
- Loss: cross-entropy on masked positions only. No VQ loss, no reconstruction
  loss; those belong to the tokenizer and LAM.

## 2. Data contract

Source: `experiments/vq_variants.StrideClipDataset(data_dir, T=16, stride=4, start_step=...)`.
Each item is `(video, real_actions)`:

| Field | Shape | dtype / range |
|---|---|---|
| `video` | `(16, 96, 160, 3)` | float32 in [0, 1], already resized + padded |
| `real_actions` | `(15,)` | int64 DOOM ids 0..17, diagnostics only, never fed to the model in v1 |

- Stride 4 = one frame per real decision. The LAM checkpoints were trained at
  stride 4; the tokenizer (vae.py) at stride 1. The tokenizer encoder is mostly
  per-frame (its temporal attention is causal), so stride-4 clips should
  tokenize fine, but verify at Gate 0: reconstruction PSNR on stride-4 val clips
  within about 1 dB of stride-1 val clips. If not, retrain the tokenizer on
  stride 4 before anything else matters.
- DataLoader: `shuffle=True`, `drop_last=False`, seeded `torch.Generator`,
  `multiprocessing_context="spawn"` + `persistent_workers=True` when
  `num_workers > 0` (forked ArrayRecord readers segfault).
- Local data: `data/train` has shards 0021 and 0022, `data/val` has 0000.

## 3. Per-step pipeline (the shapes are the acceptance criteria)

| # | Op | Output | dtype / range |
|---|---|---|---|
| 1 | `video.to(device)` | `(B,16,96,160,3)` | float32 [0,1] |
| 2 | `tokenizer.encode(video)` under `torch.no_grad()`, keep `indices` | `(B,16,960)` | int64 in [0,1023] |
| 3 | `lam.encode(video)` under `torch.no_grad()`, keep `z_q` | `(B,15,32)` | float32 |
| 4 | sample mask (Section 4) | `(B,16,960)` | bool, frame 0 all False, per-sequence mean over frames 1..15 in [0.5,1.0] |
| 5 | `logits = dynamics(indices, z_q, mask=mask)` | `(B,16,960,1024)` | float32 (bf16 under autocast) |
| 6 | `loss = F.cross_entropy(logits[mask].float(), indices[mask])` | scalar | about 7.1 at init (ln 1024 + head init) |
| 7 | backward, clip, `optimizer.step()`, `scheduler.step()` | | |

Notes:
- Call `tokenizer.eval()` and `lam.eval()` once, then `requires_grad_(False)` on
  both. Keep them as plain local objects, NOT submodules of the dynamics model,
  so `model.train()` can never flip them back (see Trap 1).
- Step 5 masks internally: the model replaces `indices[mask]` with
  `mask_id = 1024` before embedding. Pass ORIGINAL ids plus the bool mask; never
  pre-fill ids with 1024 yourself, or the targets become 1024 too.
- Step 6 excludes unmasked positions. Including them teaches an identity map and
  inflates apparent accuracy.
- The model detaches `action_tokens` and never sees pixels, so the tokenizer and
  LAM get no gradient anyway; freezing them is for memory and clarity.

## 4. Mask sampling (paper: Bernoulli rate ~ U[0.5, 1])

```python
lo, hi = MASK_RATE_RANGE                                     # (0.5, 1.0) in config.py
rate = lo + (hi - lo) * torch.rand(B, 1, 1, device=device)   # one rate per sequence
mask = torch.rand(B, T, N, device=device) < rate             # one coin per token
mask[:, 0] = False                                           # frame 0 is never hidden
```

- Per-sequence rate, per-token coin, over frames 1..T-1 only. Frame 0 is never
  masked and never scored. The paper masks input tokens z_{2:T-1} and scores
  predictions z_{2:T}; frame 1 (our index 0) has no incoming action, and at
  inference it is always the visible prompt. (Open reimplementations jafar and
  1xgpt do the same.)
- Guard: if `mask.sum() == 0` (practically impossible at rate >= 0.5 over 15,360
  tokens) skip the step rather than divide by zero.
- Do NOT mask actions. Do NOT mask whole frames in training v1. Whole-frame
  masking of the last frame is the *eval* regime (Section 7).
- Use a separate seeded `torch.Generator` for the val mask so eval is
  deterministic across epochs.

## 5. Model config

| Config | Params | Use |
|---|---:|---|
| `DynamicsModel(d_width=256, st_blocks=4, st_heads=4, head_dim=64)` | 5.0M | local iteration on MPS, batch 4..8 |
| `DynamicsModel(d_width=512, st_blocks=8, st_heads=8, head_dim=64)` | 35M | VM |
| `DynamicsModel()` (Table 12) | 19.1B | not runnable locally or on one A100 40GB |

- `T=16`, `N=960`, `K=1024` come from config and must match the tokenizer
  checkpoint's `num_codes`, `img`, `patch`.
- Expose `--width --blocks --heads` on the CLI like `experiments/train_scored.py`
  so one loop serves both scales.
- Add `self.model_config = dict(...)` to `DynamicsModel.__init__` (VAE and LAM
  have it; the dynamics class does not yet) so a checkpoint can rebuild the model
  with `DynamicsModel(**ckpt["model_config"])`.

## 6. Optimizer, schedule, precision

- `build_optimizer(model, replace(DYNAMICS_TRAINING, max_lr=lr, min_lr=lr_min,
  warmup_steps=w, steps=epochs * len(loader), global_batch_size=batch))`.
  Scheduler stepped once per optimizer step, after it.
- Paper: 3e-5 to 3e-6 cosine, 5k warmup, 125k steps, global batch 512. That is
  for a 10B model at batch 512. Locally at 5M params and batch 8 start at 3e-4
  constant (what worked for the LAM), warmup 100..500.
- Precision: bf16 autocast on CUDA only; float32 on MPS/CPU (repo convention).
  Do the cross-entropy in float32 (`logits.float()`); the `(B,T,N,K)` logits in
  bf16 for the forward are fine.
- Gradient clipping at global norm 1.0: not in vae.py/lam.py; recommended here
  (deep causal transformer on masked CE, cheap insurance). *Local choice*; note
  it in HYPERPARAMETERS.md if you keep it. With a GradScaler, unscale first.
- Nonfinite loss: raise, same as the other trainers.
- Memory: 16 frames x 960 tokens keep about 4 GB of activations per clip at
  width 512 / 8 blocks (measured: batch 4 needs 17 GB on the A100). Use
  `--gradient-checkpointing` (recompute each ST block in backward, verified
  bit-identical gradients) for anything above batch 2 at that size.

## 7. Evaluation (every epoch, fixed val subset of ~128 clips, fixed mask seed)

1. `val/masked_ce`: the training loss with the paper's random mask.
   Reference: ln(1024) = 6.931 at init.
2. `val/next_frame_ce`, `val/next_frame_acc`: frames 0..t visible (mask False),
   frame t+1 fully masked; loss and top-1 accuracy on frame t+1 only. Average
   over a few t (for example t = 3, 7, 14). This is the regime MaskGIT sampling
   starts from.
3. `val/action_delta_ce` = next_frame_ce(random LAM action at t->t+1) minus
   next_frame_ce(true LAM action). Positive means the model uses the action.
   This is the dynamics analogue of the LAM's dPSNR: if it stays ~0 the world
   model is not controllable no matter how low metric 1 gets.
4. v2: 25-step MaskGIT decode of frame t+1, `tokenizer.decode_tokens`, PSNR vs
   the true frame, and pan correlation with the LAM code (reuse `pan` from
   `experiments/train_scored.py`).

Log to stdout like the other trainers; TensorBoard optional under
`data/experiments/tb/<name>`.

## 8. Checkpoint (`data/checkpoints/dynamics_latest.pt`, write `.tmp` then replace)

Same keys as vae.py/lam.py: `model_config`, `model_state_dict`,
`optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, `epoch`,
`global_step`, `history`, `training_config`, `source_files`, `seed`.
Plus provenance, because a dynamics checkpoint is useless without knowing which
encoders produced its tokens: `tokenizer_checkpoint` (path), `lam_checkpoint`
(path), `stride`, `mask_rate_range`, `video_representation="ids"`, `evals`.

## 9. CLI and loading

```sh
python dynamics_model.py --tokenizer data/checkpoints/vae_latest.pt \
  --lam data/checkpoints/lam_scored_v6_spatial.pt \
  --epochs 1 --batch-size 8 --learning-rate 3e-4 --warmup-steps 100 \
  --width 256 --blocks 4 --heads 4 --stride 4 --num-workers 0 --seed 42 \
  --log-every 50 --eval-clips 128
```

- Tokenizer: `ckpt = torch.load(path, map_location="cpu", weights_only=False)`;
  `VAE(**ckpt["model_config"]).load_state_dict(ckpt["model_state_dict"])`.
- LAM: `experiments.arch_variants.load_checkpoint(path)` rebuilds the arch
  variant and the experiment VQ, returns `(model.eval(), ckpt)`. The plain
  `LAM(**model_config)` will not load `lam_scored_v6_spatial.pt`.

## 10. Traps specific to this repo

1. **The experiment VQ mutates its codebook in train mode.** `vq_variants.VQ.forward`
   does data-init on its first call and dead-code restarts every 50 steps when
   `self.training` is True, and its `initialized` buffer is non-persistent, so it
   is False after every load. A LAM left in train mode overwrites its trained
   codebook with random encoder outputs on the first batch. Always `.eval()`;
   never make it a submodule of anything you call `.train()` on.
2. **Tokenizer encoder is temporally causal** (`temporal_causal=True` default),
   so frame t's tokens do not leak frame t+1. Keep it that way; a non-causal
   tokenizer would let the dynamics model cheat.
3. **Stride mismatch** between LAM (4) and tokenizer (1): Section 2, Gate 0.
4. **Workers**: spawn context or segfault.
5. **No tokenizer checkpoint exists locally**; `data/checkpoints` holds only LAM
   checkpoints. Copy one from the VM, or train a small one for an epoch:
   `VAE(enc_width=256, enc_layers=4, st_enc_heads=4, dec_width=256, dec_layers=4, st_dec_heads=4)`
   is 9.0M params. Token quality affects final results, not whether the loop works.
6. `DynamicsModel.forward` rejects `T > 16`, non-bool masks, int32/int64
   mismatches, and `action_tokens` that are not `(B,T-1,32)` float. Its error
   messages name the contract you broke.
7. Logits are `(B,16,960,1024)`: 63 MB per sample in float32, but the block
   activations dominate (Section 6). Never `.cpu()` logits in the loop; reduce
   to the loss first.
8. Autocast on MPS: keep it off (repo convention).

## 11. Build order and gates

Show me the printed output at each gate before moving on.

- **Gate 0, encoders load.** Print tokenizer and LAM param counts, and the val
  reconstruction PSNR on a stride-4 batch vs a stride-1 batch.
- **Gate 1, one batch through steps 1..4.** Print shapes, dtypes,
  `indices.min()/max()`, `mask[:, 0].any()` (must be False), and
  `mask[:, 1:].float().mean(dim=(1, 2))` (each in [0.5, 1]).
- **Gate 2, init loss.** Steps 5..6 on that batch with an untrained dynamics
  model: loss in [7.0, 7.2]. Measured 7.11 at width 256 and 7.10 at width 512:
  ln(1024) = 6.93 plus about 0.17 from the default init of the output head.
  Below 6.9 means a leak, most likely pre-masked ids passed together with the
  mask, or unmasked positions in the loss.
- **Gate 3, overfit.** 200 steps on one fixed batch with a fixed mask at lr
  3e-4: loss below 0.5. Then with a fresh mask each step: loss still clearly
  below 6.9. Verifies backward, optimizer, and scheduler wiring.
- **Gate 4, leak and causality probes** on the Gate 3 model: (a) replace all
  visible tokens with random ids, loss rises; (b) permute frame t+2's tokens,
  frame t+1's logits are unchanged under `torch.allclose`.
- **Gate 5, one epoch** on the two local shards with the Section 7 eval:
  `val/masked_ce` well below 6.93, `val/next_frame_ce` below init,
  `val/action_delta_ce` above 0. Checkpoint written and reloadable via
  `DynamicsModel(**ckpt["model_config"])`.
- **Gate 6, resume** from the checkpoint continues `global_step` and the LR.

## 12. v2 backlog (do not build yet)

Precomputed token cache; gradient accumulation toward global batch 512; DDP;
LAM co-training (Table 9 schedule, unfreeze the LAM, add its losses); MaskGIT
sampler (25 steps, temperature 2.0) and rollouts; a real-action baseline
(18-way embedding to 32-d through the same loop) to separate LAM failures from
dynamics failures.
