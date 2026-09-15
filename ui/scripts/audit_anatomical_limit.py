"""Optimistic whole-neuron image fit, never a neural playback implementation.

Solve min ||H a - y||^2, 0 <= a <= 1, using every neuron's complete
projected arbor. Report a primal/dual gap so an unfinished optimizer is not
mistaken for an anatomical impossibility result. No dynamics are enforced.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def fit(h, target, iterations, report_every=100, lower=None, upper=None):
    support = np.asarray(h.sum(axis=1)).ravel() > 0
    y = np.where(support, target, 0)
    count = int(support.sum())
    # H is nonnegative. diag(H.T @ H @ 1) majorizes H.T @ H.
    diagonal = np.maximum(1e-12, h.T @ (h @ np.ones(h.shape[1])))
    lower = np.broadcast_to(0 if lower is None else lower, (h.shape[1],)).copy()
    upper = np.broadcast_to(1 if upper is None else upper, (h.shape[1],)).copy()
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or np.any(lower > upper):
        raise ValueError('Invalid activity bounds')
    a = np.clip(np.zeros(h.shape[1]), lower, upper)
    extrapolated = a.copy()
    momentum = 1.0
    best = a.copy()
    best_primal = float("inf")
    best_dual = 0.0
    history = []
    started = time.monotonic()
    for step in range(iterations + 1):
        if step % report_every == 0 or step == iterations:
            residual = h @ a - y
            gradient = h.T @ residual
            primal = float(residual @ residual)
            # Fenchel dual at u=Ha-y, with the exact support function of [l,h].
            dual = float(-residual @ residual - 2 * y @ residual
                         + 2 * np.where(gradient >= 0, lower * gradient, upper * gradient).sum())
            if primal < best_primal:
                best_primal, best = primal, a.copy()
            best_dual = max(best_dual, dual)
            record = {"iteration": step, "supportedMseUpperBound": best_primal / count,
                      "supportedMseLowerBound": best_dual / count,
                      "gap": (best_primal - best_dual) / count,
                      "elapsedSeconds": time.monotonic() - started}
            history.append(record)
            print(json.dumps(record), flush=True)
        if step == iterations:
            break
        new = np.clip(extrapolated - (h.T @ (h @ extrapolated - y)) / diagonal, lower, upper)
        next_momentum = (1 + np.sqrt(1 + 4 * momentum * momentum)) / 2
        extrapolated = new + (momentum - 1) / next_momentum * (new - a)
        a, momentum = new, next_momentum
    return best, h @ best, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--clip", action="append", choices=["map01", "map02", "map03"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cache = root / ".cache/full-brain"
    h = sp.load_npz(cache / "whole-arbor-320.npz").astype(np.float64).tocsr()
    meta = json.loads((root / "public/data/full-brain-783/whole-arbor-320.json").read_text())
    assert h.shape == (320 * 240, 139255) and h.nnz == 24403052
    for clip in args.clip or ["map01", "map02", "map03"]:
        source = json.loads((cache / f"validation/event-observed-image-v1-{clip}.json").read_text())
        print(clip, flush=True)
        target = np.asarray(source["target"], dtype=np.float64)
        a, prediction, history = fit(h, target, args.iterations)
        output = cache / f"validation/anatomical-limit-{clip}"
        np.savez_compressed(output.with_suffix(".npz"), target=target, activities=a, prediction=prediction)
        output.with_suffix(".json").write_text(json.dumps({
            "scope": "Optimistic 320x240 complete-arbor box-constrained fit; not simulated spikes",
            "anatomySha256": meta["sourceAnatomySha256"], "clip": clip,
            "sourceTime": source["metrics"]["time"], "neurons": h.shape[1],
            "entries": h.nnz, "history": history,
            "interpretation": "The converged optimum constrains this fixed-camera, one-brightness-per-neuron optical model only. It does not prove a bound on every possible display model."
        }, indent=2) + "\n")
