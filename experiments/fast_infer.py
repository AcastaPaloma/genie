"""Inference-only fast paths for the trained Genie modules. Nothing here changes a model class or a weight.

Everything is the same computation as the modules' own forward passes, only arranged for speed:
- every linear runs on a canonically strided 2-D view (on MPS a linear over a transposed view is ~8x slower);
- q/k/v are one matmul per attention module (the three weights are concatenated once and cached);
- a lone frame's temporal attention is out_proj(v_proj(x)) exactly (softmax over one key is 1), folded into
  one matmul; that is the decoder's whole temporal path when it decodes a single frame;
- on MPS attention is plain matmul + softmax + matmul, 1.5-3x faster there than the fused SDPA kernel;
- the dynamics model keeps the temporal keys/values of frames it has already seen in preallocated buffers
  (`Cache`), so each MaskGIT pass and each committed frame costs one frame of compute instead of the window;
- MaskGIT tokens are sampled with the Gumbel-max trick (same distribution as torch.multinomial).
`check(model, tok, ids, actions)` verifies all of it against the modules' own forward passes.
"""
import math, weakref
import torch
from torch.nn import functional as F

_FUSED = weakref.WeakKeyDictionary()    # SelfAttention -> (weight, bias) of the fused q/k/v projection
_FOLDED = weakref.WeakKeyDictionary()   # SelfAttention -> (weight, bias) of out_proj o v_proj, for a lone position


def _stale(entry, weight):
    return entry is None or entry[0].device != weight.device or entry[0].dtype != weight.dtype


def _fused_qkv(attn, x2d):
    entry, w = _FUSED.get(attn), attn.q_proj.weight
    if _stale(entry, w):
        weight = torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], 0)
        bias = None if attn.q_proj.bias is None else torch.cat([attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias], 0)
        entry = _FUSED[attn] = (weight.detach(), None if bias is None else bias.detach())
    return F.linear(x2d, entry[0], entry[1]).split(attn.attention_dim, dim=-1)


def _folded(attn):
    entry, w = _FOLDED.get(attn), attn.v_proj.weight
    if _stale(entry, w):
        wo, wv = attn.out_proj.weight.float(), attn.v_proj.weight.float()
        bias = torch.zeros(wo.shape[0], device=w.device)
        if attn.v_proj.bias is not None: bias = bias + wo @ attn.v_proj.bias.float()
        if attn.out_proj.bias is not None: bias = bias + attn.out_proj.bias.float()
        entry = _FOLDED[attn] = ((wo @ wv).to(w.dtype).detach(), bias.to(w.dtype).detach())
    return entry


def _attend(q, k, v, *, causal, past):
    """softmax(q k^T / sqrt(d)) v for (rows, heads, Lq, d) queries over (rows, heads, past+Lq, d) keys."""
    lq, lk = q.shape[-2], k.shape[-2]
    if q.device.type == "mps":
        if q.dtype == torch.float16: q, k = q.float(), k.float()        # fp16 scores overflow without QK norm
        scores = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
        if causal:
            scores = scores.masked_fill(torch.ones(lq, lk, dtype=torch.bool, device=q.device).triu(diagonal=past + 1), float("-inf"))
        return torch.softmax(scores, -1).to(v.dtype) @ v
    if causal and past == 0: return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    if causal:                 # SDPA's is_causal aligns query i with key i; these queries start at key `past`
        return F.scaled_dot_product_attention(q, k, v, attn_mask=torch.ones(lq, lk, dtype=torch.bool, device=q.device).tril(diagonal=past))
    return F.scaled_dot_product_attention(q, k, v)


def attention(attn, x2d, rows, length, *, cache=None, causal=False):
    """transformer.SelfAttention over `length` positions of `rows` independent sequences.

    x2d: (rows*length, D), row-major as (rows, length, D). cache: None, or (k_buffer, v_buffer, past) with
    buffers (rows, heads, capacity, head_dim) holding `past` earlier positions (causal only): the new keys and
    values are written after them and attention runs over past+length positions. Returns (rows*length, D)."""
    h, d = attn.num_heads, attn.head_dim
    if cache is None and length == 1:            # a lone position attends only to itself: out_proj(v_proj(x))
        weight, bias = _folded(attn)
        return F.linear(x2d, weight, bias)
    q, k, v = _fused_qkv(attn, x2d)
    q = attn.q_norm(q.reshape(rows, length, h, d).transpose(1, 2))
    k = attn.k_norm(k.reshape(rows, length, h, d).transpose(1, 2))
    v = v.reshape(rows, length, h, d).transpose(1, 2)
    past = 0
    if cache is not None:
        if not causal: raise ValueError("a key/value cache only makes sense for causal attention")
        k_buffer, v_buffer, past = cache
        k_buffer[:, :, past:past + length] = k; v_buffer[:, :, past:past + length] = v
        k, v = k_buffer[:, :, :past + length], v_buffer[:, :, :past + length]
    out = _attend(q, k, v, causal=causal and length > 1, past=past)   # a single new query is last: sees everything
    return attn.out_proj(out.transpose(1, 2).reshape(rows * length, h * d))


def block(blk, x, cache=None):
    """transformer.SpatioTemporalTransformerBlock.forward on (B,T,N,D); `cache` feeds the temporal attention."""
    b, t, n, d = x.shape
    x = x + attention(blk.spatial_attention, blk.spatial_norm(x).reshape(b * t * n, d), b * t, n).reshape(b, t, n, d)
    u = attention(blk.temporal_attention, blk.temporal_norm(x).transpose(1, 2).reshape(b * n * t, d), b * n, t,
                  cache=cache, causal=blk.temporal_causal)
    x = x + u.reshape(b, n, t, d).transpose(1, 2)
    return x + blk.ffn(blk.ffn_norm(x).reshape(b * t * n, d)).reshape(b, t, n, d)


class Cache:
    """Per-block temporal key/value buffers for the dynamics model's committed (fully known) context frames.

    Buffers are allocated on first use, one window (model.temporal_pos) deep. Positions are absolute, so a
    cache is only valid for the window it was built in: start a new one from the last few frames before it
    fills (dynamics() raises otherwise)."""
    def __init__(self): self.k = self.v = None; self.frames = 0

    def buffers(self, model, rows, dtype, device):
        if self.k is None:
            attn = model.trans_blocks[0].temporal_attention
            shape = (rows, attn.num_heads, model.temporal_pos.shape[0], attn.head_dim)
            self.k = [torch.empty(shape, dtype=dtype, device=device) for _ in model.trans_blocks]
            self.v = [torch.empty(shape, dtype=dtype, device=device) for _ in model.trans_blocks]
        return self.k, self.v


@torch.no_grad()
def dynamics(model, ids, actions, mask=None, cache=None, *, commit=False):
    """dynamics_model.DynamicsModel.forward for frames at temporal positions cache.frames, cache.frames+1, ...

    ids (B,t,N) token IDs; mask (B,t,N) True where the learned MASK replaces the token; actions (B,t,32), the
    LAM vector entering each frame, or (B,t-1,32) when the frames start at position 0 (frame 0 has none).
    commit=True records the frames in `cache` (an int records only that many leading frames); only fully
    known frames may be recorded. Returns logits (B,t,N,K)."""
    b, t, n = ids.shape
    offset = cache.frames if cache is not None else 0
    if offset + t > model.temporal_pos.shape[0]:
        raise ValueError("Context window is full; start a new cache from fewer frames")
    if mask is not None: ids = ids.masked_fill(mask, model.mask_id)
    a = model.action_proj(actions)
    if offset == 0 and a.shape[1] == t - 1: a = F.pad(a, (0, 0, 1, 0))
    elif a.shape[1] != t or offset == 0: raise ValueError("Expected one action per frame (t-1 when starting at position 0)")
    x = (model.video_embedding(ids) + a[:, :, None] + model.spatial_pos[None, None]
         + model.temporal_pos[None, offset:offset + t, None])
    if cache is not None: k_buffers, v_buffers = cache.buffers(model, b * n, x.dtype, x.device)
    for i, blk in enumerate(model.trans_blocks):
        x = block(blk, x, None if cache is None else (k_buffers[i], v_buffers[i], offset))
    if commit: cache.frames = offset + (t if commit is True else int(commit))
    return model.to_logits(model.output_norm(x))


@torch.no_grad()
def sample(model, cache, action, *, steps, temperature=1.0, confidence_threshold=None, pending=None):
    """MaskGIT-decode the frame after the cached context (dynamics_model.sample_next_frame, cached).

    action (B,32) enters the new frame. Every pass samples all still-hidden tokens from softmax(logits/T)
    (Gumbel-max: argmax(logits + Gumbel noise)), keeps the most confident and re-hides the rest on a cosine
    schedule. confidence_threshold (0..1) fixes tokens sampled with at least that probability at once and
    stops when nothing is hidden. pending = (frame (B,N), action (B,32)): the previous frame, fully known but
    not yet in the cache; it is recorded during the first pass (one forward over two frames is cheaper than a
    separate commit pass). Returns the new frame's token IDs (B,N); it is NOT committed to the cache."""
    if steps < 1 or temperature <= 0: raise ValueError("steps must be positive and temperature > 0")
    b, n = action.shape[0], model.n_patches
    frame = torch.zeros(b, n, dtype=torch.long, device=action.device)
    hidden = torch.ones(b, n, dtype=torch.bool, device=action.device)
    for step in range(1, steps + 1):
        if pending is not None and step == 1:
            logits = dynamics(model, torch.stack([pending[0], frame], 1), torch.stack([pending[1], action], 1),
                              torch.stack([torch.zeros_like(hidden), hidden], 1), cache, commit=1)[:, 1]
        else:
            logits = dynamics(model, frame[:, None], action[:, None], hidden[:, None], cache)[:, 0]
        logits = logits.float() / temperature
        gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp_(1e-20, 1.0)))
        sampled = (logits + gumbel).argmax(-1)
        frame = torch.where(hidden, sampled, frame)
        keep_hidden = int(n * math.cos(math.pi / 2 * step / steps))
        if keep_hidden == 0: break
        confidence = torch.log_softmax(logits, -1).gather(-1, sampled[..., None]).squeeze(-1)
        confidence = confidence.masked_fill(~hidden, float("inf"))          # fixed tokens stay fixed
        if confidence_threshold is not None:
            confidence = confidence.masked_fill(confidence >= math.log(confidence_threshold), float("inf"))
            keep_hidden = min(keep_hidden, int(torch.isfinite(confidence).sum(-1).max()))
            if keep_hidden == 0: break
        hidden = torch.zeros_like(hidden).scatter_(1, confidence.topk(keep_hidden, dim=-1, largest=False).indices, True)
    return frame


@torch.no_grad()
def commit(model, cache, frame, action):
    """Append one fully known frame (B,N) and the action (B,32) that entered it to the cache."""
    dynamics(model, frame[:, None], action[:, None], None, cache, commit=True)


@torch.no_grad()
def decode(tok, ids):
    """vae.VAE.decode_tokens for (B,t,N) IDs with the fast blocks. The decoder is temporally causal and frame 0
    never has context, so decoding one frame alone is the decoder's own training regime for frame 0."""
    b, t, n = ids.shape
    x = (tok.proj_codebook_to_dec_width(tok.cb.cookbook(ids)) + tok.dec_spatial_pos[None, None]
         + tok.dec_temporal_pos[None, :t, None])
    for blk in tok.dec_blocks: x = block(blk, x)
    if hasattr(tok, "decoder_output_norm"): x = tok.decoder_output_norm(x)          # vae.py since 74ea0e3
    elif tok.stabilize: x = F.layer_norm(x, (x.shape[-1],))
    logits = tok.proj_dec_to_pp_width(x)
    return tok.unpatchify(logits.float().sigmoid() if tok.stabilize else logits.sigmoid())


@torch.no_grad()
def check(model, tok, ids, actions):
    """Compare every fast path with the modules' own forward passes on real tokens; returns the errors.

    ids (1,t,N) with t >= 4, actions (1,t-1,32), on the models' device/dtype. Raises beyond float noise."""
    b, t, n = ids.shape
    mask = torch.zeros_like(ids, dtype=torch.bool); mask[:, -1, ::3] = True          # a partly hidden last frame
    ref = model(ids, actions, mask=mask).float(); scale = ref.abs().max().item()
    errors = {"dynamics": (dynamics(model, ids, actions, mask).float() - ref).abs().max().item() / scale}
    cache = Cache(); dynamics(model, ids[:, :2], actions[:, :1], None, cache, commit=True)      # two-stage prefill
    dynamics(model, ids[:, 2:t - 1], actions[:, 1:t - 2], None, cache, commit=True)
    dynamics(model, ids[:, -1:], actions[:, -1:], torch.ones_like(mask[:, -1:]), cache)         # a scratch pass ...
    errors["cached"] = (dynamics(model, ids[:, -1:], actions[:, -1:], mask[:, -1:], cache)[:, 0].float() - ref[:, -1]).abs().max().item() / scale
    cache = Cache(); dynamics(model, ids[:, :t - 2], actions[:, :t - 3], None, cache, commit=True)      # pending frame merged into a pass
    merged = dynamics(model, ids[:, -2:], actions[:, -2:], mask[:, -2:], cache, commit=1)[:, 1]
    errors["merged"] = (merged.float() - ref[:, -1]).abs().max().item() / scale
    if cache.frames != t - 1: raise RuntimeError("commit count is wrong")
    errors["decoder_1_frame"] = (decode(tok, ids[:, -1:]).float() - tok.decode_tokens(ids[:, -1:]).float()).abs().max().item()
    errors["decoder_2_frames"] = (decode(tok, ids[:, -2:]).float() - tok.decode_tokens(ids[:, -2:]).float()).abs().max().item()
    def tolerance(param): return 1e-3 if param.dtype == torch.float32 else 3e-2      # half precision: rounding differs per op order
    bad = {k: v for k, v in errors.items() if v > tolerance(tok.dec_temporal_pos if k.startswith("decoder") else model.temporal_pos)}
    if bad: raise RuntimeError(f"fast_infer disagrees with the model forward: {bad}")
    return errors
