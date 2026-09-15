"""Small LAM architecture probes that remove the 'encode the target frame's appearance' shortcut.
diffenc: the code is computed from the CHANGE in pooled features between frame t and t+1,
         so static appearance (brightness, layout) cancels and only motion survives.
resdec:  the decoder predicts frame t+1 as frame t plus a residual, so global appearance
         is free and a brightness code is worth nothing to it.
Both keep the paper's ST-transformer, pixel input, K=8, and additive action conditioning."""
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

import sys
pass
import torch
from torch import nn
from lam import LAM, LAMEncoder, LAMDecoder


class DiffPoolEncoder(LAMEncoder):
    def forward(self, video):
        x = self.patchify(video); t = video.shape[1]
        x = self.patch_proj(x) + self.spatial_pos[None, None] + self.temporal_pos[None, :t, None]
        for block in self.blocks: x = block(x)
        pooled = x.mean(dim=2)                                   # (B,T,D)
        change = pooled[:, 1:] - pooled[:, :-1]                  # (B,T-1,D): what moved, not what is there
        codes = self.to_code(change)
        return torch.cat([torch.zeros_like(codes[:, :1]), codes], dim=1)  # keep (B,T,32); position 0 is dropped by LAM.encode


class DiffSpatialEncoder(LAMEncoder):
    """Like DiffPoolEncoder, but keeps WHERE the change happened: per-patch feature change is
    squeezed to a few channels per patch, then all patches are flattened into the code.
    Mean pooling is permutation-invariant over patches, so it cannot tell left from right."""
    def __init__(self, *args, per_patch=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.squeeze = nn.Linear(self.d_width, per_patch)
        self.to_code = nn.Linear(self.n_patches * per_patch, self.to_code.out_features)
    def forward(self, video):
        x = self.patchify(video); t = video.shape[1]
        x = self.patch_proj(x) + self.spatial_pos[None, None] + self.temporal_pos[None, :t, None]
        for block in self.blocks: x = block(x)
        change = x[:, 1:] - x[:, :-1]                            # (B,T-1,N,D) per-patch change
        codes = self.to_code(self.squeeze(change).flatten(2))    # (B,T-1,N*per_patch) -> (B,T-1,32)
        return torch.cat([torch.zeros_like(codes[:, :1]), codes], dim=1)


class ResidualDecoder(LAMDecoder):
    def forward(self, video, z_q):
        x = self.patchify(video); b, t = video.shape[:2]
        x = self.patch_proj(x) + self.spatial_pos[None, None] + self.temporal_pos[None, :t, None]
        x = x + self.action_proj(z_q).unsqueeze(2)
        for block in self.blocks: x = block(x)
        return video + self.unpatchify(self.to_pixels(x))        # frame t plus predicted change


def apply_arch(model: LAM, arch: str):
    cfg = {k: v for k, v in model.model_config.items() if k != "K"}
    if arch in ("diffenc", "both"): model.encoder = DiffPoolEncoder(**cfg)
    if arch == "diffspatial": model.encoder = DiffSpatialEncoder(**cfg)
    if arch in ("resdec", "both"): model.decoder = ResidualDecoder(**cfg)
    return model


def load_checkpoint(path, device=_DEV):
    from vq_variants import VQ
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = LAM(**ckpt["model_config"])
    if ckpt.get("arch", "base") != "base": model = apply_arch(model, ckpt["arch"])
    if "vq_options" in ckpt: model.cookbook = VQ(K=model.K, code_width=32, **ckpt["vq_options"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt
