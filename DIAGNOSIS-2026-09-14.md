# Tokenizer collapse diagnosis and repair

## Confirmed failure mechanism

The archived full-size checkpoint at step 78,000 had decoder residual RMS rising
from 13.19 after block 1 to 204.55 after block 20. There was no normalization
between the final residual stream and pixel projection. Pixel logits ranged
from -2,464 to +370 (RMS 610.54). The following sigmoid produced exactly zero for
91.67% of pixels and exactly one for 8.33%.

Backpropagating reconstruction MSE through that checkpoint gave **exactly zero**
gradients at pixel logits, pixel-head weights, and the encoder's continuous code
vectors. Applying sigmoid to the saved logits in float32 also gave zero gradient.
This is not solved merely by disabling BF16. Once reconstruction gradients died,
the remaining VQ/commitment optimization could minimize its objective without
preserving video information. Validation used a single code (74) for all 61,440
tokens across four different clips, with bit-identical reconstructions.

The repeating colored grid is a learned saturated pixel patch repeated across
spatial positions, not a visualization/layout error: patchify/unpatchify round
trip error on real video was exactly zero. Reconstruction loss was unchanged at
approximately 0.11370548 from epochs 2 through 8.

The actual launch used LR 3e-4 immediately, zero warmup, and no gradient clipping.
The code had overridden the configured warmup of 10,000 steps. There was
also no code-usage or reconstruction validation to detect failure during the
overnight job. Those were mistakes in the training setup and monitoring.

A second confirmed numerical bug affected the quantizer's straight-through
expression `features + (code - features).detach()`. It is mathematically equal to
the code in the forward pass, but subtractive cancellation destroys that identity
at finite precision when codes are much smaller than features. On the original
initialization under BF16, feature RMS was 0.87325 versus code RMS 0.000596:
81.17% of returned components became zero, although no selected code components
were zero. Only 0.388% matched the selected code exactly. The equivalent expression
`code.detach() + (features - features.detach())` matched it exactly for 100% of
components and preserves the identity gradient to the encoder. This is a real
code defect in addition to the independently demonstrated decoder saturation.

## Reproduction and tests

A full-size single-H100 reproduction cycling eight fixed training clips began
with max absolute pixel logit 2.72 and 24 active codes. By step 2 the max logit was
15.19; by step 20 it was 456; by step 200 it was 2,400, all pixels were saturated,
and four checked clips reconstructed identically using one code. This reproduces
the failure without DDP and rules out multi-GPU synchronization as a necessary
cause. It does not isolate the relative contribution of every hyperparameter.

A staged stabilization test using output normalization, a small initialized
pixel head, lower LR with warmup, and clipping avoided saturation through 400
steps and retained input dependence. Adding initialization from actual encoder
features was then tested for 800 steps on the same eight-clip subset. It used 26
codes across four checked clips; their reconstruction MSEs were 0.01963, 0.00323,
0.00283 and 0.00283. Saturation was zero and outputs differed across inputs.
These subset results demonstrate restored learning, not final generalization.

## Implemented changes

- Parameter-free LayerNorm before the encoder code projection and decoder pixel
  head controls the otherwise unnormalized residual stream.
- Small pixel-head initialization (normal std 0.01, zero bias) and a float32
  sigmoid keep initial pixel predictions away from saturation.
- Fresh codebook entries are sampled from encoder features across training clips
  before DDP synchronizes parameters, instead of starting optimization with all
  embeddings in the tiny uniform initialization range.
- The straight-through quantizer uses a cancellation-safe expression that
  preserves the selected code exactly while keeping encoder gradients unchanged.
- The distributed trainer defaults to LR 1e-4, 1,000 warmup steps, and gradient
  norm clipping at 1.0. Model widths, depths, vocabulary and video layout remain
  the same. These are implementation repairs, not claimed paper settings.
- Fixed held-out validation every 250 steps writes metrics and reconstruction
  previews, retains the best MSE checkpoint, and stops with a saved checkpoint
  on the observed collapse conditions after an initial 500-step grace period.

The original and training/repair checks cover
including global-batch/DDP gradient equivalence, exact sample coverage and
checkpoint/resume, nonzero reconstruction gradients under large residuals,
an integration test that saves and stops on identical validation outputs, and
exact forward-code preservation/identity encoder gradients in float32 and BF16.

The failed checkpoint, source and logs are preserved on the server under
`data/diagnostics/collapse-20260914/`. Full-dataset restart artifacts are isolated
under `data/checkpoints/tokenizer-repair-20260914/`, with logs in
`logs/vae-repair.log` and tmux session `vae`.

## Full-dataset restart verification

The repaired full-size model is training on both H100s with two clips per GPU,
global batch four, targeting ten epochs. Six training/repair checks and the five
original tests passed. The run was checkpointed and resumed at step 442 to load
the final straight-through arithmetic fix. The resumed job passed its 1,000-step
warmup and was verified still training at step 1,150 at LR 1e-4. The step-1,000
checkpoint records `implementation_version=stabilized-v2`.

The archived original implementation and repaired checkpoint were evaluated on
the **same** four validation windows, indices `[0, 2099, 4199, 6299]`:

| Metric | Failed checkpoint, step 78,000 | Repaired, step 1,000 |
|---|---:|---:|
| Mean reconstruction MSE | 0.166258 | 0.005597 |
| Active codes / 1,024 | 1 | 61 |
| Code perplexity | 1.00 | 12.62 |
| Saturated pixel fraction | 100% | 0% |
| Max output difference across clips | 0 | 0.814287 |

The same clips' per-clip mean-color baseline MSE is 0.034088. The new images show
coarse scene-dependent structure; they are still early-training outputs. Four
fixed validation clips and short-run stability do not establish final tokenizer
quality across the whole dataset. Validation previews continue to be saved every
250 steps, and the watchdog stops on the demonstrated collapse conditions.
