"""Genie 1 dynamics: full Table 12 width, depth and attention configuration.

The paper leaves the exact input embedding modules unspecified. Support the
user's codebook-vector path as well as discrete video token IDs. Both produce
model-width embeddings before action addition. train_dynamics below is the
single-device training entrypoint; MaskGIT sampling is not implemented yet.
"""

import math

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

from config import (DYNAMICS as ARCH, VIDEO_CODES, CODE_WIDTH, IMAGE_SIZE,
                    VIDEO_PATCH, SEQUENCE_LENGTH, MASK_RATE_RANGE, MASKGIT_STEPS,
                    SAMPLING_TEMPERATURE)
from patches import VideoPatches, position_parameter
from transformer import SpatioTemporalTransformerBlock


class DynamicsModel(nn.Module):
    def __init__(self, d_width=ARCH.width, st_blocks=ARCH.layers,
                 st_heads=ARCH.heads, head_dim=ARCH.head_dim, *,
                 img=IMAGE_SIZE, patch=VIDEO_PATCH, T=SEQUENCE_LENGTH,
                 num_codes=VIDEO_CODES, action_width=CODE_WIDTH,
                 video_width=CODE_WIDTH, qk_norm=ARCH.qk_norm,
                 ffn_expansion=ARCH.ffn_expansion):
        super().__init__()
        if min(d_width, st_blocks, T, num_codes, action_width, video_width) < 1:
            raise ValueError("Widths, blocks, context and vocabulary must be positive")
        self.model_config = dict(d_width=d_width, st_blocks=st_blocks, st_heads=st_heads,
                                 head_dim=head_dim, img=img, patch=patch, T=T, num_codes=num_codes,
                                 action_width=action_width, video_width=video_width,
                                 qk_norm=qk_norm, ffn_expansion=ffn_expansion)
        self.n_patches = VideoPatches(img, patch, 3).n_patches
        self.d_width, self.num_codes, self.mask_id = d_width, num_codes, num_codes
        self.action_width, self.video_width = action_width, video_width
        self.video_embedding = nn.Embedding(num_codes + 1, d_width)
        self.video_proj = nn.Linear(video_width, d_width)
        self.action_proj = nn.Linear(action_width, d_width)
        self.spatial_pos = position_parameter(self.n_patches, d_width)
        self.temporal_pos = position_parameter(T, d_width)
        self.trans_blocks = nn.ModuleList([
            SpatioTemporalTransformerBlock(
                d_width, st_heads, head_dim=head_dim, qk_norm=qk_norm,
                ffn_expansion_factor=ffn_expansion, temporal_causal=True,
            ) for _ in range(st_blocks)
        ])
        self.output_norm = nn.LayerNorm(d_width)
        self.to_logits = nn.Linear(d_width, num_codes)
        # Recompute block activations in backward: 16 frames x 960 tokens keep
        # about 4 GB per clip per 8 blocks at width 512 otherwise. Training-only.
        self.gradient_checkpointing = False

    @property
    def blocks(self):
        return self.trans_blocks

    def forward(self, video_tokens, action_tokens, *, mask=None):
        """Return token logits (B,T,N,K), not a shape or decoded video.

        video_tokens: integer IDs (B,T,N), or quantized vectors (B,T,N,32).
        action_tokens: LAM codebook vectors (B,T-1,32), for transitions t -> t+1.
        mask: boolean (B,T,N), True at positions to replace with learned MASK.

        The selected action is added to every patch of its destination frame;
        the first frame has no incoming action. Use one video representation
        consistently during training and generation; its input layers are distinct.
        The training loop masks targets and computes token cross-entropy.
        """
        is_ids = video_tokens.ndim == 3 and video_tokens.dtype in (torch.int32, torch.int64)
        is_vectors = (video_tokens.ndim == 4 and video_tokens.is_floating_point()
                      and video_tokens.shape[-1] == self.video_width)
        if not (is_ids or is_vectors) or video_tokens.shape[2] != self.n_patches:
            raise ValueError("Expected integer IDs (B,T,N) or quantized vectors (B,T,N,video_width)")
        b, t, n = video_tokens.shape[:3]
        if not 1 <= t <= self.temporal_pos.shape[0]:
            raise ValueError("Frame count exceeds configured dynamics context")
        if tuple(action_tokens.shape) != (b, t - 1, self.action_width):
            raise ValueError("Expected T-1 transition action vectors (B,T-1,action_width)")
        if not action_tokens.is_floating_point():
            raise TypeError("Actions must be floating-point LAM codebook vectors")
        if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != (b, t, n)):
            raise ValueError("mask must be boolean with shape (B,T,N)")
        if is_ids:
            if mask is not None:
                video_tokens = video_tokens.masked_fill(mask, self.mask_id)
            x = self.video_embedding(video_tokens)
        else:
            # Tokenizer is pretrained/frozen during dynamics training.
            x = self.video_proj(video_tokens.detach())
            if mask is not None:
                x = torch.where(mask.unsqueeze(-1), self.video_embedding.weight[self.mask_id], x)
        actions = self.action_proj(action_tokens.detach())
        # Pad AFTER projection: first frame gets zero even when projection has bias.
        actions = F.pad(actions, (0, 0, 1, 0)).unsqueeze(2)
        x = x + actions + self.spatial_pos[None, None] + self.temporal_pos[None, :t, None]
        for block in self.trans_blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.to_logits(self.output_norm(x))



def _resolve_device(device):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(device)


def load_tokenizer(path, device):
    """Frozen pretrained video tokenizer from a vae.py checkpoint."""
    from vae import VAE

    import inspect

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    unknown = set(config) - set(inspect.signature(VAE.__init__).parameters)
    if unknown:
        # Loading weights into a class with different forward math gives valid
        # shapes and garbage tokens, so refuse instead of guessing.
        raise ValueError(f"{path} was trained by a different vae.py "
                         f"(version {checkpoint.get('implementation_version', '?')}); "
                         f"local VAE does not accept {sorted(unknown)}")
    tokenizer = VAE(**config)
    tokenizer.load_state_dict(checkpoint["model_state_dict"])
    return tokenizer.to(device).eval().requires_grad_(False)


def load_lam(path, device):
    """Frozen pretrained latent action model, including the experiment variants."""
    try:
        from experiments.arch_variants import load_checkpoint
    except ImportError as error:
        raise ImportError("experiments/arch_variants.py is required to load LAM checkpoints") from error
    lam, _ = load_checkpoint(path, device=device)
    # eval() is mandatory: the experiment VQ rewrites its codebook in train mode.
    return lam.eval().requires_grad_(False)


def action_vectors(lam, indices):
    """Codebook vectors for integer latent action IDs, for either VQ implementation."""
    cookbook = lam.cookbook
    return cookbook.lookup(indices) if hasattr(cookbook, "lookup") else cookbook.cookbook(indices)


@torch.no_grad()
def encode_clips(tokenizer, lam, video):
    """Pixels (B,T,H,W,3) -> tokenizer IDs (B,T,N) and LAM action vectors (B,T-1,32)."""
    _, ids, _ = tokenizer.encode(video)
    actions, _, _ = lam.encode(video)
    return ids, actions.float()


def sample_mask(batch, frames, patches, *, rate_range=MASK_RATE_RANGE, generator=None):
    """Paper masking: rate ~ U[0.5,1] per sequence, one Bernoulli coin per token.

    Frame 0 is never hidden or scored: the paper masks z_{2:T-1} and scores
    z_{2:T}; frame 0 has no incoming action and is the visible prompt at play time.
    """
    low, high = rate_range
    rate = low + (high - low) * torch.rand(batch, 1, 1, generator=generator)
    mask = torch.rand(batch, frames, patches, generator=generator) < rate
    mask[:, 0] = False
    return mask


def masked_cross_entropy(logits, ids, mask):
    """Token cross-entropy at hidden positions only; visible tokens are never scored."""
    if not mask.any():
        raise ValueError("mask hides no tokens")
    return F.cross_entropy(logits[mask].float(), ids[mask])


@torch.no_grad()
def evaluate(model, lam, val_ids, val_actions, *, batch_size, device, autocast,
             seed=123, context_frames=(3, 7, 14)):
    """Fixed-mask metrics on pre-encoded validation clips.

    masked_ce: the training objective. next_frame_ce/acc: frames 0..t visible,
    frame t+1 fully hidden (the regime MaskGIT sampling starts from).
    action_delta_ce: next_frame_ce with a random latent action minus with the
    true one; positive means the model uses the action.
    """
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    num_codes = model.num_codes
    sums = dict(masked_ce=0.0, next_frame_ce=0.0, next_frame_acc=0.0, action_delta_ce=0.0)
    masked_count = next_count = 0
    for start in range(0, len(val_ids), batch_size):
        ids = val_ids[start:start + batch_size].to(device)
        actions = val_actions[start:start + batch_size].to(device)
        b, t, n = ids.shape
        mask = sample_mask(b, t, n, generator=generator).to(device)
        with autocast():
            logits = model(ids, actions, mask=mask)
        sums["masked_ce"] += masked_cross_entropy(logits, ids, mask).item() * b
        masked_count += b
        for context in context_frames:
            if context + 2 > t:
                continue
            context_ids, context_actions = ids[:, :context + 2], actions[:, :context + 1]
            frame_mask = torch.zeros_like(context_ids, dtype=torch.bool)
            frame_mask[:, -1] = True
            target = context_ids[:, -1].reshape(-1)
            wrong_actions = context_actions.clone()
            random_ids = torch.randint(0, lam.K, (b,), generator=generator).to(device)
            wrong_actions[:, -1] = action_vectors(lam, random_ids)
            with autocast():
                true_logits = model(context_ids, context_actions, mask=frame_mask)[:, -1].float()
                wrong_logits = model(context_ids, wrong_actions, mask=frame_mask)[:, -1].float()
            ce_true = F.cross_entropy(true_logits.reshape(-1, num_codes), target)
            ce_wrong = F.cross_entropy(wrong_logits.reshape(-1, num_codes), target)
            sums["next_frame_ce"] += ce_true.item() * b
            sums["next_frame_acc"] += (true_logits.argmax(-1).reshape(-1) == target).float().mean().item() * b
            sums["action_delta_ce"] += (ce_wrong - ce_true).item() * b
            next_count += b
    model.train(was_training)
    return {key: value / (masked_count if key == "masked_ce" else max(1, next_count))
            for key, value in sums.items()}


@torch.no_grad()
def sample_next_frame(model, ids, actions, *, steps=MASKGIT_STEPS, temperature=SAMPLING_TEMPERATURE):
    """MaskGIT-decode the frame after `ids`. Returns its token IDs (B,N).

    ids: visible context (B,t,N). actions: (B,t,action_width), transitions into
    context frames 2..t followed by the one into the new frame. Every pass
    samples all still-hidden tokens from softmax(logits / temperature), keeps
    the most confident, and re-hides the rest on a cosine schedule; the last
    pass leaves nothing hidden. Uses torch's global RNG; seed for repeatability.
    """
    b, t, n = ids.shape
    if tuple(actions.shape[:2]) != (b, t):
        raise ValueError("Need one action per context frame, the last one entering the new frame")
    if t + 1 > model.temporal_pos.shape[0]:
        raise ValueError("Context already fills the model's temporal window; drop the oldest frame")
    if steps < 1:
        raise ValueError("steps must be positive")
    was_training = model.training
    model.eval()
    frame = torch.zeros(b, n, dtype=ids.dtype, device=ids.device)
    hidden = torch.ones(b, n, dtype=torch.bool, device=ids.device)
    for step in range(1, steps + 1):
        mask = torch.zeros(b, t + 1, n, dtype=torch.bool, device=ids.device)
        mask[:, -1] = hidden
        logits = model(torch.cat([ids, frame[:, None]], dim=1), actions, mask=mask)[:, -1].float()
        probs = torch.softmax(logits / temperature, dim=-1)
        sampled = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(b, n)
        confidence = probs.gather(-1, sampled[..., None]).squeeze(-1)
        frame = torch.where(hidden, sampled, frame)
        keep_hidden = int(n * math.cos(math.pi / 2 * step / steps))
        if keep_hidden == 0:
            break
        confidence = confidence.masked_fill(~hidden, float("inf"))  # fixed tokens stay fixed
        lowest = confidence.topk(keep_hidden, dim=-1, largest=False).indices
        hidden = torch.zeros_like(hidden).scatter_(1, lowest, True)
    model.train(was_training)
    return frame


@torch.no_grad()
def rollout(model, ids, actions, num_steps, **sampling):
    """Generate num_steps frames autoregressively from prompt frames ids (B,t,N).

    actions: (B, t-1+num_steps, action_width): transitions inside the prompt,
    then one per generated frame. The context slides once it fills the model's
    window. Returns generated token IDs (B,num_steps,N).
    """
    b, t, n = ids.shape
    if actions.shape[1] != t - 1 + num_steps:
        raise ValueError("Need t-1 prompt transitions plus one action per generated frame")
    limit = model.temporal_pos.shape[0]
    sequence, generated = ids, []
    for k in range(num_steps):
        context = min(sequence.shape[1], limit - 1)
        end = t + k  # exclusive index of the transition into the new frame, plus one
        new = sample_next_frame(model, sequence[:, -context:], actions[:, end - context:end], **sampling)
        sequence = torch.cat([sequence, new[:, None]], dim=1)
        generated.append(new)
    return torch.stack(generated, dim=1)


def encode_dataset(tokenizer, lam, dataset, *, batch_size, device, autocast):
    """Run the frozen encoders once over a dataset; returns CPU IDs and action vectors."""
    ids, actions = [], []
    for start in range(0, len(dataset), batch_size):
        video = torch.stack([dataset[i][0] for i in range(start, min(start + batch_size, len(dataset)))])
        with autocast():
            batch_ids, batch_actions = encode_clips(tokenizer, lam, video.to(device))
        ids.append(batch_ids.cpu())
        actions.append(batch_actions.cpu())
    return torch.cat(ids), torch.cat(actions)


def save_checkpoint(path, state):
    """Atomic replacement avoids a half-written checkpoint if saving is interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train_dynamics(*, tokenizer, lam, data_dir=None, val_dir=None, epochs=1, batch_size=1,
                   device=None, learning_rate=None, min_learning_rate=None, warmup_steps=0,
                   grad_clip=1.0, num_workers=0, seed=42, checkpoint=None, log_every=100,
                   width=None, blocks=None, heads=None, head_dim=None, stride=4,
                   eval_clips=128, max_windows=None, resume=False, model=None,
                   gradient_checkpointing=False):
    """Train the dynamics model on frozen tokenizer IDs and frozen LAM actions.

    tokenizer/lam are checkpoint paths. batch_size counts 16-frame clips taken
    every `stride` stored frames (stride 4 = one frame per DOOM decision, which
    is what the LAM checkpoints were trained on). Single device, no gradient
    accumulation. Model size stays at full Table 12 defaults unless width/blocks/
    heads/head_dim or a model are given. max_windows caps the training windows
    for smoke tests only.
    """
    from dataclasses import replace
    from pathlib import Path
    import random
    import numpy as np
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from config import DYNAMICS_TRAINING, build_optimizer
    from experiments.vq_variants import StrideClipDataset

    if min(epochs, batch_size, log_every, stride, eval_clips) < 1 or num_workers < 0 or warmup_steps < 0:
        raise ValueError("epochs, batch_size, log_every, stride, eval_clips must be positive; workers/warmup nonnegative")
    if grad_clip < 0 or (max_windows is not None and max_windows < 1):
        raise ValueError("grad_clip must be nonnegative and max_windows positive")
    root = Path(__file__).resolve().parent
    data_dir = Path(data_dir) if data_dir is not None else root / "data/train"
    val_dir = Path(val_dir) if val_dir is not None else root / "data/val"
    checkpoint = Path(checkpoint) if checkpoint is not None else root / "data/checkpoints/dynamics_latest.pt"
    learning_rate = DYNAMICS_TRAINING.max_lr if learning_rate is None else learning_rate
    min_learning_rate = learning_rate if min_learning_rate is None else min_learning_rate
    if not 0 < min_learning_rate <= learning_rate:
        raise ValueError("Need 0 < min_learning_rate <= learning_rate")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _resolve_device(device)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    def autocast():
        return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp)

    resumed = None
    if resume:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"--resume needs an existing checkpoint at {checkpoint}")
        resumed = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = DynamicsModel(**resumed["model_config"])
    elif model is None:
        model = DynamicsModel(d_width=width or ARCH.width, st_blocks=blocks or ARCH.layers,
                              st_heads=heads or ARCH.heads, head_dim=head_dim or ARCH.head_dim)
    model = model.to(device)
    model.gradient_checkpointing = gradient_checkpointing

    print(f"Loading frozen tokenizer {tokenizer} and LAM {lam} ...", flush=True)
    tokenizer_path, lam_path = str(tokenizer), str(lam)
    tokenizer, lam = load_tokenizer(tokenizer_path, device), load_lam(lam_path, device)
    if tokenizer.n_patches != model.n_patches or tokenizer.cb.K != model.num_codes:
        raise ValueError("Tokenizer patch grid / codebook size do not match the dynamics model")
    if lam.cookbook.code_width != model.action_width:
        raise ValueError("LAM code width does not match the dynamics action width")
    if min(tokenizer.temporal_pos.shape[0], lam.temporal_pos.shape[0], model.temporal_pos.shape[0]) < 16:
        raise ValueError("All three models must accept 16-frame sequences")

    print(f"Indexing stride-{stride} training windows in {data_dir} ...", flush=True)
    dataset = StrideClipDataset(data_dir, T=16, stride=stride, start_step=8, max_windows=max_windows)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False,
        num_workers=num_workers, pin_memory=use_amp,
        generator=torch.Generator().manual_seed(seed),
        # Readers are opened in each spawned worker, never inherited from a fork.
        **({"multiprocessing_context": "spawn", "persistent_workers": True} if num_workers else {}),
    )
    val_dataset = StrideClipDataset(val_dir, T=16, stride=stride, start_step=16, max_windows=eval_clips)
    print(f"Encoding {len(val_dataset)} validation clips once ...", flush=True)
    val_ids, val_actions = encode_dataset(tokenizer, lam, val_dataset, batch_size=batch_size, device=device, autocast=autocast)
    print(f"{len(dataset):,} training windows -> {len(loader):,} batches/epoch; "
          f"dynamics {sum(p.numel() for p in model.parameters()):,} params; device={device}", flush=True)

    training_config = replace(DYNAMICS_TRAINING, max_lr=learning_rate, min_lr=min_learning_rate,
                              warmup_steps=warmup_steps, steps=epochs * len(loader),
                              global_batch_size=batch_size)
    optimizer, scheduler = build_optimizer(model, training_config)
    history, evals, global_step, start_epoch = [], [], 0, 1
    if resumed is not None:
        model.load_state_dict(resumed["model_state_dict"])
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        scheduler.load_state_dict(resumed["scheduler_state_dict"])
        scaler.load_state_dict(resumed["scaler_state_dict"])
        history, evals = resumed["history"], resumed["evals"]
        global_step, start_epoch = resumed["global_step"], resumed["epoch"] + 1
        print(f"Resumed from {checkpoint}: epoch {resumed['epoch']}, step {global_step}", flush=True)
        del resumed

    def run_eval():
        metrics = evaluate(model, lam, val_ids, val_actions, batch_size=batch_size, device=device, autocast=autocast)
        metrics["step"] = global_step
        evals.append(metrics)
        print(f"[eval @ step {global_step}] masked_ce={metrics['masked_ce']:.4f} "
              f"next_frame_ce={metrics['next_frame_ce']:.4f} next_frame_acc={metrics['next_frame_acc']:.4f} "
              f"action_delta_ce={metrics['action_delta_ce']:+.4f}", flush=True)

    if start_epoch == 1:
        run_eval()  # untrained reference: about ln(num_codes) + 0.17
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss, sequences_seen = 0.0, 0
        for batch_number, (video, _real_actions) in enumerate(loader, 1):
            video = video.to(device, non_blocking=use_amp)
            with autocast():
                ids, actions = encode_clips(tokenizer, lam, video)
            b, t, n = ids.shape
            mask = sample_mask(b, t, n).to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(ids, actions, mask=mask)
            loss = masked_cross_entropy(logits, ids, mask)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, batch {batch_number}")
            previous_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            grad_norm = float("nan")
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(model.parameters(), grad_clip).item()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            global_step += 1
            sequences_seen += b
            total_loss += loss.item() * b
            if batch_number == 1 or batch_number % log_every == 0 or batch_number == len(loader):
                print(f"epoch {epoch}/{epochs} batch {batch_number}/{len(loader)} loss={loss.item():.4f} "
                      f"masked={mask[:, 1:].float().mean().item():.2f} grad_norm={grad_norm:.2f} "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)
        history.append(dict(epoch=epoch, step=global_step, loss=total_loss / sequences_seen,
                            sequences=sequences_seen, batches=len(loader)))
        run_eval()
        save_checkpoint(checkpoint, {
            "model_config": model.model_config, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(), "epoch": epoch, "global_step": global_step,
            "history": history, "evals": evals, "training_config": vars(training_config),
            "source_files": dataset.paths, "seed": seed, "sequence_length": 16, "stride": stride,
            "mask_rate_range": MASK_RATE_RANGE, "video_representation": "ids",
            "tokenizer_checkpoint": tokenizer_path, "lam_checkpoint": lam_path,
        })
        print(f"Finished epoch {epoch}: {sequences_seen:,} sequences; saved {checkpoint}", flush=True)

    model.eval()
    return model, history, evals


if __name__ == "__main__":
    # Example (local smoke): python dynamics_model.py --tokenizer vae_latest_best.pt \
    #   --lam data/checkpoints/lam_scored_v6_spatial.pt --width 256 --blocks 4 --heads 4 \
    #   --batch-size 4 --learning-rate 3e-4 --max-windows 64 --eval-clips 16
    import argparse

    parser = argparse.ArgumentParser(description="Train the dynamics model on frozen tokenizer IDs and LAM actions.")
    parser.add_argument("--tokenizer", required=True, help="vae.py checkpoint path")
    parser.add_argument("--lam", required=True, help="LAM checkpoint path (lam.py or experiments/train_scored.py)")
    parser.add_argument("--data-dir", default=None, help="ArrayRecord split directory (default: data/train)")
    parser.add_argument("--val-dir", default=None, help="Validation split directory (default: data/val)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1, help="16-frame clips per optimizer batch")
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:1, mps, cpu")
    parser.add_argument("--learning-rate", type=float, default=None, help="Default: paper 3e-5; use ~3e-4 for small models")
    parser.add_argument("--min-learning-rate", type=float, default=None,
                        help="Default: constant LR; 3e-6 opts into the paper's cosine decay")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Paper: 5000")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Global grad-norm clip; 0 disables (local choice)")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=None, help="Default: data/checkpoints/dynamics_latest.pt")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--width", type=int, default=None, help="Default: Table 12 width 5120")
    parser.add_argument("--blocks", type=int, default=None, help="Default: Table 12 depth 48")
    parser.add_argument("--heads", type=int, default=None, help="Default: Table 12 heads 36")
    parser.add_argument("--head-dim", type=int, default=None, help="Default: Table 12 head width 128")
    parser.add_argument("--stride", type=int, default=4, help="Stored frames between clip frames (4 = one DOOM decision)")
    parser.add_argument("--eval-clips", type=int, default=128)
    parser.add_argument("--max-windows", type=int, default=None, help="Smoke tests only: cap training windows")
    parser.add_argument("--resume", action="store_true", help="Continue from --checkpoint")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Recompute ST-block activations in backward (~4x less activation memory, ~30%% slower)")
    train_dynamics(**vars(parser.parse_args()))
