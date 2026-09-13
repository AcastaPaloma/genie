"""DOOM video clips for both Genie encoders: float32 (B,T,H,W,3) in [0,1].

Supply frames.npy (uint8 NHWC RGB frames) and episodes.npy (start,length rows)
under data/doom. Each episode must have a constant source FPS; pass source_fps
when it differs from 10. Never place a clip across deaths/resets/scene cuts.
This loader does not download data or assume that Atari recordings are DOOM.
"""

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from config import (CONTENT_SIZE, IMAGE_SIZE, SEQUENCE_LENGTH, FPS,
                    DYNAMICS_TRAINING)

FRAMES_NPY = "data/doom/frames.npy"
EPS_NPY = "data/doom/episodes.npy"
T = SEQUENCE_LENGTH
BATCH = DYNAMICS_TRAINING.global_batch_size


def prepare_video(frames, *, layout="THWC", content_size=CONTENT_SIZE,
                  image_size=IMAGE_SIZE):
    """Resize to 160x90, replicate-pad bottom to 96, return normalized RGB.

    layout must be explicit for legacy channels-first arrays. Padding and area
    resizing are implementation choices. The source frame rate is handled by
    ClipDataset, not by this spatial preprocessing function.
    """
    if layout not in ("THWC", "TCHW"):
        raise ValueError("layout must be THWC or TCHW")
    x = torch.as_tensor(np.array(frames, copy=True))
    if x.ndim != 4 or x.dtype != torch.uint8:
        raise ValueError("Expected uint8 video (T,H,W,C) or (T,C,H,W)")
    if layout == "THWC":
        x = x.permute(0, 3, 1, 2)
    if x.shape[1] == 1:
        x = x.expand(-1, 3, -1, -1)
    elif x.shape[1] != 3:
        raise ValueError("Expected one or three input channels")
    h, w = image_size
    ch, cw = content_size
    if min(h, w, ch, cw) <= 0 or h < ch or w < cw:
        raise ValueError("Image size must contain the positive content size")
    x = F.interpolate(x.float() / 255.0, size=content_size, mode="area")
    x = F.pad(x, (0, w - cw, 0, h - ch), mode="replicate")
    return x.permute(0, 2, 3, 1).contiguous()


class ClipDataset(Dataset):
    """Sample T frames at 10 FPS within each episode, without boundary crossing."""

    def __init__(self, frames, episodes, T=T, *, source_fps=FPS, layout="THWC",
                 content_size=CONTENT_SIZE, image_size=IMAGE_SIZE):
        if T < 1 or not np.isfinite(source_fps) or source_fps < FPS:
            raise ValueError("T must be positive and source_fps must be at least 10")
        if layout not in ("THWC", "TCHW"):
            raise ValueError("layout must be THWC or TCHW")
        if frames.ndim != 4 or frames.dtype != np.uint8:
            raise ValueError("frames must be a 4-D uint8 array")
        episodes = np.asarray(episodes)
        if episodes.ndim != 2 or episodes.shape[1] != 2 or not np.issubdtype(episodes.dtype, np.integer):
            raise ValueError("episodes must contain integer (start,length) rows")
        self.frames, self.T, self.layout = frames, T, layout
        self.content_size, self.image_size = content_size, image_size
        self.offsets = np.floor(np.arange(T) * source_fps / FPS).astype(np.int64)
        span = int(self.offsets[-1]) + 1
        starts, counts = [], []
        previous_end = 0
        for start, length in episodes:
            if start < previous_end or length < 1 or start + length > len(frames):
                raise ValueError("Episodes must be sorted, nonoverlapping, positive and in bounds")
            previous_end = int(start + length)
            count = max(0, int(length) - span + 1)
            if count:
                starts.append(start)
                counts.append(count)
        # Prefix sums avoid storing one Python integer for every video window.
        self.episode_starts = np.asarray(starts, dtype=np.int64)
        self.cumulative_counts = np.cumsum(counts, dtype=np.int64)

    def __len__(self):
        return int(self.cumulative_counts[-1]) if len(self.cumulative_counts) else 0

    def __getitem__(self, i):
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(i)
        episode = np.searchsorted(self.cumulative_counts, i, side="right")
        previous = self.cumulative_counts[episode - 1] if episode else 0
        start = self.episode_starts[episode] + i - previous
        return prepare_video(self.frames[start + self.offsets], layout=self.layout,
                             content_size=self.content_size, image_size=self.image_size)


def build_loader(batch_size=BATCH, num_workers=4, *, frames_path=FRAMES_NPY,
                 episodes_path=EPS_NPY, source_fps=FPS, layout="THWC", T=T,
                 drop_last=True):
    """batch_size is local to this loader; distributed callers set microbatch size.

    Paper global batch = local microbatch * world size * accumulation steps.
    No hardware-dependent model downsizing is performed here.
    """
    for path in (frames_path, episodes_path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Supply DOOM data at {path}; see HYPERPARAMETERS.md")
    frames = np.load(frames_path, mmap_mode="r", allow_pickle=False)
    episodes = np.load(episodes_path, allow_pickle=False)
    ds = ClipDataset(frames, episodes, T=T, source_fps=source_fps, layout=layout)
    if not len(ds):
        raise ValueError("No episodes are long enough for a complete clip")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                        drop_last=drop_last)
    return ds, loader


if __name__ == "__main__":
    # Inspect one clip; this is not a change to the global training batch.
    dataset, loader = build_loader(batch_size=1, num_workers=0, drop_last=False)
    clip = next(iter(loader))
    print(f"{len(dataset)} windows; batch {tuple(clip.shape)}; {clip.dtype}")
