# Genie 1 settings for the DOOM implementation

`config.py` is the shared source for model and training defaults. The requested
full model is used; there is no automatic smaller model based on local hardware.
These are Genie **1** settings, not disclosed Genie 3 internals.

## Published settings

| Component | Width | ST blocks | Heads | Q/K head width | Patch |
|---|---:|---:|---:|---:|---:|
| Video encoder | 512 | 12 | 8 | 64 | 4 |
| Video decoder | 1024 | 20 | 16 | 64 | 4 |
| LAM encoder | 1024 | 20 | 16 | 64* | 16 |
| LAM decoder | 1024 | 20 | 16 | 64* | 16 |
| Dynamics | 5120 | 48 | 36 | 128 | token input |

Source: [Genie, Tables 5, 7, 12 and Figure 3](https://arxiv.org/html/2402.15391v1).
`num_layers` is instantiated as the number of complete ST blocks. Each contains
spatial attention, causal temporal attention, and one FFN. *LAM head width is
inferred as width/heads; Table 5 does not specify it.

Video/action codebooks are respectively `[1024,32]` and `[8,32]`. Context is 16
frames at 10 FPS. Dynamics uses QK normalization and bfloat16. Sampling settings
are 25 MaskGIT steps, temperature 2.0. [Section 3](https://arxiv.org/html/2402.15391v1#S3)

| Training setting | Tokenizer | Dynamics |
|---|---:|---:|
| AdamW max/min LR | 3e-4 / 3e-4 | 3e-5 / 3e-6 |
| Betas | 0.9, 0.9 | 0.9, 0.9 |
| Weight decay | 1e-4 | 1e-4 |
| Warmup steps | 10,000 | 5,000 |
| Training steps | 300,000 | 125,000 |
| Global batch | 384 | 512 |

Sources: [Tables 6, 8, 9 and Appendix D](https://arxiv.org/html/2402.15391v1#SApp3).
Tokenizer batch 384 selects the larger reported experiment. Equal tokenizer
max/min LR is intentional. `build_optimizer` implements warmup then cosine.

## Unspecified details and parameter-count discrepancy

This is an implementation of the published settings, not an exact recovered
DeepMind implementation. The following are explicit local choices:

- Separate spatial/temporal attention weights, pre-LayerNorm, GELU FFNs with 4x
  expansion, biases enabled, dropout zero. The paper's tables do not specify all
  of these. Learned absolute space/time positions use normal initialization,
  standard deviation 0.02; LayerNorm uses PyTorch's default epsilon 1e-5.
- QK normalization uses per-head LayerNorm, followed by scaled dot products.
  Values use the same per-head width as queries/keys. The exact normalization
  parameterization and value width are not fully specified by the tables.
- The dynamics network supports either integer video IDs via an independent
  embedding table or quantized video vectors via `Linear(32,5120)`. Pick one
  representation consistently for training and generation. These input paths
  have different parameters and are not interchangeable without training.
- LAM codebook vectors enter dynamics via `Linear(32,5120)`. Stop-gradient is
  enforced on action vectors and on quantized video inputs. The exact linear
  projection modules are implementation choices, not verbatim paper details.
- Actions condition their destination frame. `forward(video, actions)` expects
  T video positions and T-1 transition vectors. The first frame gets zero action
  embedding. To generate a new frame, append masked positions and its action.
- VQ commitment beta 0.25, uniform codebook initialization, MSE reconstruction,
  LAM spatial mean pooling, tokenizer sigmoid, a dynamics final LayerNorm, and
  an untied logits head are local choices. Codebook distance search is chunked
  in float32; chunk size 4096 changes memory usage, not codebook capacity.
- LAM optimizer settings follow dynamics here. Tokenizer/LAM bfloat16 is also a
  local choice. The loop must actually enable autocast; config alone does not.

**Actual parameter counts with these defaults:**

| Component | Parameters in this implementation |
|---|---:|
| Tokenizer | 388,038,224 |
| LAM | 674,451,488 |
| Dynamics | 19,146,434,560 |

The paper reports approximately 200M/300M/10.1B respectively. The literal
width/depth settings combined with this ST-block implementation do not reproduce
those totals. For example, each dynamics block has two independent attentions
with internal width `36 * 128 = 4608`, plus a `5120 -> 20480 -> 5120` FFN.
The two attentions alone contribute about 9.06B parameters over 48 blocks;
the FFNs add about 10.07B. Extra input embeddings cannot explain this discrepancy.

Do not label this model "10.1B" or size a VM using that number. Resolving the gap
requires additional authoritative architectural detail (e.g. layer counting,
weight sharing, FFN dimensions); we have not guessed or halved the requested
block counts. `python main.py` measures the full architecture on the meta device
without allocating its weight storage.

## DOOM data contract

All models now share normalized channels-last RGB: `(B,T,H,W,3)`. The loader
resizes content to 160x90 and replicate-pads its bottom to 160x96, making both
patch sizes divide the input. Padding is a local choice, not a published Genie
preprocessing claim. This yields **960 video tokens** and **60 LAM patches**
per frame. To recover unpadded content, crop reconstruction `[:, :, :90, :, :]`.

Supply:

- `data/doom/frames.npy`: uint8 `(frames,height,width,3)`, concatenated episodes.
- `data/doom/episodes.npy`: integer `(start,length)` rows, sorted/nonoverlapping.
- `source_fps` to `build_loader` if the recordings are not already 10 FPS. The
  loader samples timestamp-spaced frame indices and stays within each episode.
  Separate episodes at resets/deaths/cuts; it cannot infer these from pixels.

Grayscale or channels-first recordings require the explicit documented layout;
grayscale is repeated into RGB. Data is memory-mapped, not all loaded into RAM.
No Atari download, unsafe pickle loading, or dataset preprocessing runs implicitly.
Existing 64x64 channels-first data requires `layout="TCHW"` and is resized.

Global batch means `local microbatch * world_size * gradient_accumulation_steps`.
`build_loader(batch_size=...)` controls only its local batch. Choose that and
sharding for the VM; they do not alter model widths or block counts.

## What is implemented and checked

The tokenizer, LAM, and dynamics forward paths, data loader, and optimizer
factory are implemented. `main.py` is an architecture audit, **not a trainer**.
Masking rates/sampling settings are configuration for the future training and
MaskGIT loops; no distributed training loop, checkpoint pipeline, iterative
sampler, DOOM data collector, or playable application exists yet. Changing
hyperparameters does not establish rollout quality or playability.

The dynamics output is `(B,T,960,1024)` logits. Supervise masked target positions
against original tokenizer IDs; never feed a target its own visible answer.
Use the trained tokenizer decoder only for reconstruction/display.

Run checks:

```sh
python main.py
python -m unittest discover -s tests -v
```

Tests inspect all full-size modules on `meta`, and use explicitly tiny models
only for functional forward/backward tests. They cover fused-attention parity,
temporal causality, masked targets, both dynamics input representations, action
conditioning/stop-gradient, codebook search, patch reconstruction, FPS/episode
boundaries, optimizer schedules, and LAM train/eval behavior. These CPU tests do
not establish that full-size CUDA training or long rollouts work.

## Tokenizer training entrypoint

`vae.py` now has a single-device, tokenizer-only epoch loop:

```sh
python vae.py --epochs 1 --batch-size 1 --device cuda
```

`--batch-size` is the number of sequences, each containing exactly 16 consecutive
stored frames from one ArrayRecord video. Defaults read `data/train` relative to
this project, not the working directory. There is no fixed batch-count limit.
The loader enumerates every chunk and shuffles sequence order each epoch; it
never shuffles frames or samples across records. The last optimizer batch is
kept even when it is smaller than the requested batch size.

A 160-frame record yields ten sequences. A 40-frame record yields windows
`[0:16]`, `[16:32]`, `[24:40]`: the final window overlaps to cover the tail without
padding or dropping frames. The downloaded train split therefore yields 37,800
sequences per epoch from 4,200 records / 600,000 original frames. No FPS-based
subsampling is applied on this ArrayRecord path. Records shorter than 16 frames
are rejected explicitly.

The entrypoint uses full-size VAE defaults, AdamW at 3e-4 and the shared betas /
weight decay. For this epoch-based entrypoint, LR is constant by default;
`--warmup-steps 10000` opts into the paper's tokenizer warmup. CUDA uses bf16
where supported, otherwise scaled fp16; CPU/MPS use float32. `--num-workers`
controls spawned data-loader workers. This does not implement distributed
training or gradient accumulation.

After each epoch, `data/checkpoints/vae_latest.pt` is atomically replaced with
model weights, constructor settings, optimizer/scheduler/scaler states, metrics,
epoch and step counts. Override the path with `--checkpoint`. The model can be
reconstructed from `model_config` and `model_state_dict`; CLI resume is not yet
implemented. `model.train()` and `model.eval()` retain normal PyTorch behavior.

Run focused checks with `python -m unittest discover -s tests -v`. Tests verify
consecutive frames, complete frame coverage, record boundaries, spawned readers,
the final partial batch, all sequences across multiple epochs, optimizer updates,
and checkpoint restoration. Full production training is not run by these tests.

## Download the same subset on a VM

After cloning the repository, run `python3 scripts/download_doom.py`. This uses
only the Python 3.11+ standard library and downloads the exact pinned 42/7/7
train/validation/test shards into `data/`. Paths, sizes and SHA-256 hashes are
committed in `scripts/doom_split.json`. Completed matching files are reused;
rerun the command after an interrupted download. Use `--output /path/to/data`
for another destination, then pass its `train` directory to `vae.py --data-dir`.
