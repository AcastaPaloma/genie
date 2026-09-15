"""Anti-collapse VQ variants for the LAM experiments. cookbook.py is untouched.
- data init: codebook rows start as real encoder outputs (first training batch)
- threshold restarts: every `restart_every` steps, codes used for < min_share of
  assignments are replaced by live encoder outputs
- optional l2-normalized (cosine) lookup: fixes the norm so outputs cannot shrink
  to a single point (ViT-VQGAN style)
"""
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

import io, math, sys
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.nn import functional as F
pass
from cookbook import Cookbook
from dataloader import _VideoRecordUnpickler, prepare_video
from array_record.python.array_record_data_source import ArrayRecordDataSource


class VQ(Cookbook):
    def __init__(self, K=8, code_width=32, beta=0.25, *, normalize=False,
                 restart_every=50, min_share=0.02, entropy_weight=0.0):
        super().__init__(K=K, code_width=code_width, beta=beta)
        self.normalize, self.restart_every, self.min_share = normalize, restart_every, min_share
        self.entropy_weight = entropy_weight
        self.register_buffer("initialized", torch.tensor(False), persistent=False)
        self.register_buffer("since_restart", torch.zeros(K), persistent=False)
        self.step = 0

    def codes(self):
        w = self.cookbook.weight
        return F.normalize(w, dim=-1) if self.normalize else w

    def lookup(self, idx):  # what the decoder should receive for integer actions
        return self.codes()[idx]

    def forward(self, z):
        leading = z.shape[:-1]
        flat = z.reshape(-1, self.code_width).float()
        if self.normalize:
            flat = F.normalize(flat, dim=-1)
        if self.training and not bool(self.initialized):
            pick = torch.randperm(flat.shape[0])[: self.K]
            self.cookbook.weight.data.copy_(flat[pick].detach() + 0.01 * torch.randn_like(flat[pick]))
            self.initialized.fill_(True)
        codes = self.codes()
        with torch.no_grad():
            d = flat.detach().square().sum(-1, keepdim=True) + codes.square().sum(-1) - 2 * flat.detach() @ codes.T
            idx = d.argmin(-1)
        quantized = codes[idx]
        loss = F.mse_loss(quantized, flat.detach()) + self.beta * F.mse_loss(flat, quantized.detach())
        z_q = flat + (quantized - flat).detach()
        if self.entropy_weight > 0:
            # MaskGIT/MAGVIT-style usage entropy: soft-assign each output to codes, then
            # penalize low entropy of the batch-average assignment. Pushes the encoder
            # to spread outputs across codes instead of contracting to one point.
            d_soft = flat.square().sum(-1, keepdim=True) + codes.square().sum(-1) - 2 * flat @ codes.T
            tau = d_soft.mean().detach().clamp(min=1e-6)
            q_bar = torch.softmax(-d_soft / tau, dim=1).mean(0)
            usage_entropy = -(q_bar * (q_bar + 1e-10).log()).sum()
            loss = loss + self.entropy_weight * (math.log(self.K) - usage_entropy)
        if self.training:
            self.since_restart += torch.bincount(idx, minlength=self.K).float()
            self.step += 1
            if self.step % self.restart_every == 0:
                share = self.since_restart / self.since_restart.sum().clamp(min=1)
                dead = (share < self.min_share).nonzero().flatten()
                if len(dead):
                    pick = torch.randperm(flat.shape[0])[: len(dead)]
                    self.cookbook.weight.data[dead] = flat[pick].detach() + 0.01 * torch.randn_like(flat[pick])
                    print(f"  [restart] step {self.step}: revived {dead.tolist()} (shares {[round(s, 3) for s in share.tolist()]})", flush=True)
                self.since_restart.zero_()
        return z_q.reshape(*leading, self.code_width), idx.reshape(*leading), loss


class StrideClipDataset(torch.utils.data.Dataset):
    """T frames taken every `stride` stored frames, plus the real action driving each transition.
    Starts are multiples of 4 so windows align with the dataset's 4-frame action holds."""
    def __init__(self, data_dir, T=16, stride=1, start_step=16, max_windows=None):
        self.paths = sorted(str(p) for p in Path(data_dir).glob("*.array_record"))
        reader = ArrayRecordDataSource(self.paths)
        span = (T - 1) * stride + 1
        self.index = []
        for r in range(len(reader)):
            n = _VideoRecordUnpickler(io.BytesIO(reader[r])).load()["sequence_length"]
            self.index += [(r, s) for s in range(0, n - span + 1, start_step)]
            if max_windows and len(self.index) >= max_windows: break
        self.index = self.index[:max_windows] if max_windows else self.index
        self.T, self.stride, self._src = T, stride, None
    def __getstate__(self):
        state = self.__dict__.copy(); state["_src"] = None; return state   # workers open their own reader
    def __len__(self): return len(self.index)
    def __getitem__(self, i):
        r, s = self.index[i]
        if self._src is None: self._src = ArrayRecordDataSource(self.paths)
        rec = _VideoRecordUnpickler(io.BytesIO(self._src[r])).load()
        n = rec["sequence_length"]
        frames = np.frombuffer(rec["raw_video"], dtype=np.uint8).reshape(n, 60, 80, 3)
        sel = s + self.stride * np.arange(self.T)
        video = prepare_video(frames[sel])
        actions = torch.from_numpy(np.asarray(rec["actions"])[sel[:-1]].astype(np.int64))
        return video, actions
