"""
LAM data pipeline for p-doom/atari-breakout-dataset.

One-time preprocessing (ArrayRecord -> .npy), then a Dataset that yields
contiguous 16-frame clips that never cross an episode boundary.

Download first:
    huggingface-cli download --repo-type dataset \
        p-doom/atari-breakout-dataset --local-dir ./breakout
"""

import glob
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── config ────────────────────────────────────────────────────────────────
RAW_DIR    = "./breakout/train"
FRAMES_NPY = "frames.npy"
EPS_NPY    = "episodes.npy"

N_SHARDS   = 20        # ~2000 episodes. the full 10M frames is ~70GB decompressed.
SRC_SIZE   = 84        # native resolution
DST_SIZE   = 64        # 84 doesn't divide by patch 8; 64 does
T          = 16        # frames per clip
BATCH      = 16


# ── reading ArrayRecord ───────────────────────────────────────────────────
def open_records(paths):
    """grain is what Jasmine uses; array_record is the lighter fallback."""
    try:
        import grain
        return grain.sources.ArrayRecordDataSource(paths)
    except ImportError:
        from array_record.python.array_record_data_source import ArrayRecordDataSource
        return ArrayRecordDataSource(paths)


def preprocess(raw_dir=RAW_DIR, n_shards=N_SHARDS):
    """ArrayRecord -> (frames [N,C,64,64] uint8, episodes [n_eps,2])."""
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.array_record")))[:n_shards]
    if not paths:
        raise FileNotFoundError(f"no .array_record files under {raw_dir}")
    print(f"reading {len(paths)} shards")

    source = open_records(paths)
    frames_out, episodes, cursor, channels = [], [], 0, None

    for i in range(len(source)):
        el = pickle.loads(source[i])
        n  = el["sequence_length"]

        # derive C from the buffer length instead of assuming grayscale
        c = len(el["raw_video"]) // (n * SRC_SIZE * SRC_SIZE)
        if channels is None:
            channels = c
            print(f"detected {c} channel(s) -> patch_dim will be {8*8*c}")
        assert c == channels, f"channel count changed: {channels} -> {c}"

        ep = np.frombuffer(el["raw_video"], dtype=np.uint8)
        ep = ep.reshape(n, SRC_SIZE, SRC_SIZE, c).copy()      # frombuffer is read-only

        t = torch.from_numpy(ep).permute(0, 3, 1, 2).float()  # [n, C, 84, 84]
        t = F.interpolate(t, size=(DST_SIZE, DST_SIZE), mode="area")
        t = t.round().clamp(0, 255).to(torch.uint8)           # [n, C, 64, 64]

        frames_out.append(t.numpy())
        episodes.append((cursor, n))
        cursor += n

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(source)} episodes, {cursor} frames")

    frames   = np.concatenate(frames_out, axis=0)
    episodes = np.array(episodes, dtype=np.int64)

    np.save(FRAMES_NPY, frames)
    np.save(EPS_NPY, episodes)
    print(f"saved {frames.shape} ({frames.nbytes/1e9:.2f} GB), "
          f"{len(episodes)} episodes")
    return frames, episodes


# ── dataset ───────────────────────────────────────────────────────────────
class ClipDataset(Dataset):
    """Contiguous T-frame clips. Every window lies inside a single episode."""

    def __init__(self, frames, episodes, T=T):
        self.frames, self.T = frames, T

        starts = []
        for ep_start, ep_len in episodes:
            if ep_len >= T:                       # skip episodes too short to slice
                starts.extend(range(ep_start, ep_start + ep_len - T + 1))
        self.starts = np.array(starts, dtype=np.int64)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        clip = self.frames[s : s + self.T]                    # [T, C, 64, 64] uint8
        return torch.from_numpy(clip.copy()).float().div_(255.0)


def build_loader(batch_size=BATCH, num_workers=4):
    if not (os.path.exists(FRAMES_NPY) and os.path.exists(EPS_NPY)):
        preprocess()
    frames   = np.load(FRAMES_NPY)
    episodes = np.load(EPS_NPY)

    ds = ClipDataset(frames, episodes)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True, drop_last=True)
    return ds, loader


# ── sanity check ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    ds, loader = build_loader()

    print(f"\nframes:   {ds.frames.shape}  {ds.frames.dtype}")
    print(f"windows:  {len(ds):,}")

    clip = next(iter(loader))
    print(f"batch:    {tuple(clip.shape)}  {clip.dtype}")
    print(f"range:    [{clip.min():.3f}, {clip.max():.3f}]")

    # every window must sit inside one episode
    bounds = {int(s): int(s + l) for s, l in np.load(EPS_NPY)}
    ends = sorted(bounds.values())
    for s in ds.starts[:10000]:
        nxt = ends[np.searchsorted(ends, s, side="right")]
        assert s + T <= nxt, f"window at {s} crosses an episode boundary"
    print("boundary check passed")