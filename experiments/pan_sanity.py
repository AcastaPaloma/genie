"""Does the decoder render camera pans at all? Compare pixel-estimated pan of (frame t -> model's own prediction)
against the true pan (frame t -> true frame t+1). Also check the estimator survives blur."""
# --- portable paths / device (added when moved from the scratchpad into the repo) ---
import os as _os, sys as _sys
from pathlib import Path as _Path
import torch as _torch
_ROOT = _Path(__file__).resolve().parents[1]                     # repo root
_EXP = _ROOT / "data" / "experiments"; _EXP.mkdir(parents=True, exist_ok=True)
_sys.path.insert(0, str(_ROOT)); _sys.path.insert(0, str(_Path(__file__).parent))
_DEV = "cuda" if _torch.cuda.is_available() else ("mps" if _torch.backends.mps.is_available() else "cpu")
_WORKERS = int(_os.environ.get("NUM_WORKERS", "8" if _DEV == "cuda" else "0"))
# -----------------------------------------------------------------------------------

import sys; pass; sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
import torch, torch.nn.functional as F
from vq_variants import StrideClipDataset
from arch_variants import load_checkpoint
model, ckpt = load_checkpoint(sys.argv[1] if len(sys.argv) > 1 else str(_ROOT / "data/checkpoints/lam_scored_v4.pt"), _DEV)
print("checkpoint epoch", ckpt["epoch"])
val = StrideClipDataset(str(_ROOT / "data/val"), stride=4, start_step=16, max_windows=128)
V = torch.stack([val[i][0] for i in range(len(val))])
def band(x): return x[..., :90, :, :].mean(-1)[..., 8:72, :]
def pan(a, b, max_shift=14):
    best = best_err = None
    for s in range(-max_shift, max_shift + 1):
        err = (((a[..., :, s:] - b[..., :, :160 - s]) if s >= 0 else (a[..., :, :160 + s] - b[..., :, -s:])) ** 2).mean(dim=(-2, -1))
        if best is None: best, best_err = torch.full_like(err, float(s)), err; continue
        better = err < best_err; best = torch.where(better, torch.full_like(err, float(s)), best); best_err = torch.minimum(err, best_err)
    return best
with torch.no_grad():
    preds = torch.cat([model.decoder(V[i:i+16, :-1].to(_DEV), model.encode(V[i:i+16].to(_DEV))[0]).cpu() for i in range(0, 128, 16)])
ft, ft1 = V[:, :-1], V[:, 1:]
true_pan = pan(band(ft), band(ft1)).flatten()
pred_pan = pan(band(ft), band(preds)).flatten()
blur = lambda x: F.avg_pool2d(x.flatten(0, 1).permute(0, 3, 1, 2), 16).repeat_interleave(16, -1).repeat_interleave(16, -2).permute(0, 2, 3, 1).reshape(x.shape)
blur_pan = pan(band(ft), band(blur(ft1))).flatten()
def corr(a, b): return torch.corrcoef(torch.stack([a, b]))[0, 1].item()
moving = true_pan.abs() >= 2
print(f"true pans: {moving.float().mean():.0%} of transitions move >=2px, mean |pan| {true_pan.abs().mean():.2f}px")
print(f"estimator on 16px-blurred TRUE next frame: corr with true pan = {corr(true_pan, blur_pan):.2f}  (does blur break the estimator?)")
print(f"model's own prediction:  corr with true pan = {corr(true_pan, pred_pan):.2f},  {(pred_pan.abs() >= 2).float().mean():.0%} of predictions move >=2px, mean |pan| {pred_pan.abs().mean():.2f}px")
print(f"on transitions that truly move: mean |pred pan| = {pred_pan[moving].abs().mean():.2f}px vs true {true_pan[moving].abs().mean():.2f}px; sign agreement {((pred_pan[moving].sign() == true_pan[moving].sign()).float().mean()):.0%}")
