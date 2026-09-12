"""Minimal loader for one Atari Breakout ArrayRecord episode."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

DATASET_REPO = "p-doom/atari-breakout-dataset"
DEFAULT_SAMPLE_FILE = "test/data_0000.array_record"


def download_breakout_sample(*, filename: str = DEFAULT_SAMPLE_FILE, local_dir: str | Path | None = None) -> Path:
    """Download exactly one ArrayRecord shard and return its local path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError("Install dependencies with `pip install -r requirements.txt`.") from error
    return Path(hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset", filename=filename, local_dir=None if local_dir is None else str(local_dir)))


def load_breakout_episode(record_path: str | Path, *, episode_index: int = 0) -> tuple[np.ndarray, np.ndarray | None]:
    """Decode one episode as uint8 ``(T, 84, 84, C)`` frames and actions."""
    try:
        from array_record.python.array_record_module import ArrayRecordReader
    except ImportError as error:
        raise ImportError("Install the `array-record` package to read this dataset.") from error

    reader = ArrayRecordReader(str(record_path))
    try:
        if not 0 <= episode_index < reader.num_records():
            raise IndexError(f"episode_index must be in [0, {reader.num_records()}); got {episode_index}")
        episode = pickle.loads(reader.read([episode_index])[0])
    finally:
        reader.close()

    sequence_length = int(episode["sequence_length"])
    raw_video = np.frombuffer(episode["raw_video"], dtype=np.uint8)
    pixels_per_frame = 84 * 84
    if raw_video.size % (sequence_length * pixels_per_frame):
        raise ValueError("raw_video does not contain an integral number of 84x84 frames")
    channels = raw_video.size // (sequence_length * pixels_per_frame)
    frames = raw_video.reshape(sequence_length, 84, 84, channels)
    actions = np.asarray(episode["actions"], dtype=np.int64) if "actions" in episode else None
    return frames, actions


def load_breakout_sequence(*, sequence_length: int = 8, episode_index: int = 0, start_frame: int = 0, local_dir: str | Path | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Download one shard and return a deterministic contiguous frame sequence."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    frames, actions = load_breakout_episode(download_breakout_sample(local_dir=local_dir), episode_index=episode_index)
    stop_frame = start_frame + sequence_length
    if start_frame < 0 or stop_frame > len(frames):
        raise ValueError(f"Requested frames [{start_frame}:{stop_frame}] from an episode of {len(frames)} frames")
    return frames[start_frame:stop_frame], None if actions is None else actions[start_frame:stop_frame]
