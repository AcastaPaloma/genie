"""Relative code effect: for each context, apply every code and measure the pan of each prediction
MINUS the mean pan across codes in that same context. That removes the scene's default motion and
isolates what each code adds. A real 'turn left' button has a consistently negative relative pan."""
import os as _os, sys as _sys
from pathlib import Path as _Path
import torch as _torch
_ROOT = _Path(__file__).resolve().parents[1]; _sys.path.insert(0, str(_ROOT)); _sys.path.insert(0, str(_Path(__file__).parent))
_DEV = "cuda" if _torch.cuda.is_available() else ("mps" if _torch.backends.mps.is_available() else "cpu")
import sys, torch
from vq_variants import StrideClipDataset
from arch_variants import load_checkpoint

model, ckpt = load_checkpoint(sys.argv[1], _DEV); K = model.K
n_ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 128
val = StrideClipDataset(_ROOT / "data/val", stride=ckpt.get("stride", 4), start_step=16, max_windows=n_ctx)
V = torch.stack([val[i][0] for i in range(len(val))]); T = int(sys.argv[3]) if len(sys.argv) > 3 else 3   # index of the context frame the code is applied at; 0 = single-frame prompt
def band(x): return x[..., :90, :, :].mean(-1)[..., 8:72, :]
def pan(a, b, max_shift=14):
    best = best_err = None
    for s in range(-max_shift, max_shift + 1):
        err = (((a[..., :, s:] - b[..., :, :160 - s]) if s >= 0 else (a[..., :, :160 + s] - b[..., :, -s:])) ** 2).mean(dim=(-2, -1))
        if best is None: best, best_err = torch.full_like(err, float(s)), err; continue
        better = err < best_err; best = torch.where(better, torch.full_like(err, float(s)), best); best_err = torch.minimum(err, best_err)
    return best
pans = torch.zeros(K, n_ctx); lookup = model.cookbook.lookup
with torch.no_grad():
    for i in range(0, n_ctx, 16):
        c = V[i:i + 16, :T + 1].to(_DEV)
        if T == 0: base = torch.zeros(c.shape[0], 1, dtype=torch.long, device=_DEV)   # single prompt frame: no history to infer
        else: _, cidx, _ = model.encode(c); base = torch.cat([cidx, torch.zeros_like(cidx[:, :1])], 1)
        for k in range(K):
            ids = base.clone(); ids[:, T] = k
            pred = model.decoder(c, lookup(ids))[:, T].cpu()
            pans[k, i:i + 16] = pan(band(V[i:i + 16, T]), band(pred))
rel = pans - pans.mean(0, keepdim=True)                       # code effect relative to the other codes, per context
print(f"{sys.argv[1].split('/')[-1]}  epoch {ckpt['epoch']}  contexts={n_ctx}")
print(f"scene default motion: mean |pan across codes| = {pans.mean(0).abs().mean():.2f}px  (how much the context alone moves)")
print("\ncode  abs_pan  rel_pan  rel_std  sign_agreement   (rel = this code minus the mean of all codes, same scene)")
for k in range(K):
    r = rel[k]; sign = r.sign(); maj = 1.0 if (sign > 0).float().mean() >= (sign < 0).float().mean() else -1.0
    agree = ((sign == maj) | (r.abs() < 0.5)).float().mean().item()
    print(f"  {k}   {pans[k].mean():+6.2f}  {r.mean():+6.2f}   {r.std():5.2f}      {agree:.2f}   {'<-- consistent' if agree >= 0.8 and r.abs().mean() >= 1 else ''}")
# how much of the variance in pan is explained by the code vs by the context
tot = pans.var(); by_code = pans.mean(1).var(); by_ctx = pans.mean(0).var()
print(f"\nvariance of predicted pan explained by: code {by_code / tot:.0%}, context {by_ctx / tot:.0%}, interaction/noise {1 - (by_code + by_ctx) / tot:.0%}")
