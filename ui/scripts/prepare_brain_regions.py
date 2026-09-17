"""Measure where each FlyWire super_class sits in the released geometry and emit
`regions.json`, the manifest the viewer uses to show one anatomical region.

This selects published neurons by their published annotation. It does not move,
resample, invent or recolour any geometry, and it does not alter the complete
`brain.json` import: every neuron remains in the archive and in the full-brain
route. Run from `ui/`:

    ../.venv/bin/python scripts/prepare_brain_regions.py

Reads   public/data/full-brain-783/brain.json and the 270 skeleton chunks.
Writes  public/data/full-brain-783/regions.json
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

DIRECTORY = pathlib.Path("public/data/full-brain-783")

# FlyWire v783 super_class -> the optic lobes, measured below and reported in the
# manifest. 'sensory' is split by cell type: the photoreceptors (R1-6, R7, R8)
# terminate in the optic lobe, while the bristle and Johnston's-organ afferents
# do not, so the split is by published cell_type prefix, not by a coordinate cut.
OPTIC_SUPER_CLASSES = {"optic"}
PHOTORECEPTOR_PREFIXES = ("R1-6", "R7", "R8", "ocellar retinula")


def is_photoreceptor(neuron: dict) -> bool:
    return neuron["superClass"] == "sensory" and str(neuron["cellType"]).startswith(PHOTORECEPTOR_PREFIXES)


def digest(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    manifest_path = DIRECTORY / "brain.json"
    if not manifest_path.exists():
        sys.exit(f"Missing complete anatomy manifest: {manifest_path}")
    meta = json.loads(manifest_path.read_text())
    if not meta["complete"] or meta["missingRootIds"]:
        sys.exit("The anatomy import is incomplete; regions must be measured on the complete release.")
    neurons = meta["neurons"]
    count = meta["neuronCount"]
    assert len(neurons) == count

    order = np.empty(count, dtype=np.int64)
    super_class = np.empty(count, dtype=object)
    optic_side = np.zeros(count, dtype=bool)
    for neuron in neurons:
        index = neuron["modelIndex"]
        order[index] = index
        super_class[index] = neuron["superClass"]
        optic_side[index] = neuron["superClass"] in OPTIC_SUPER_CLASSES or is_photoreceptor(neuron)

    # Accumulate exact per-neuron vertex bounds and centroids over every chunk.
    lo = np.full((count, 3), np.inf, dtype=np.float64)
    hi = np.full((count, 3), -np.inf, dtype=np.float64)
    total = np.zeros((count, 3), dtype=np.float64)
    seen = np.zeros(count, dtype=np.int64)
    edges_per_neuron = np.zeros(count, dtype=np.int64)

    # Vertex-weighted per-axis histograms, split by optic membership, so a region
    # can be framed on where its cable actually is. The absolute bounding box is
    # set by a handful of long projections and barely differs between regions.
    span_lo = np.array(meta["bounds"][0], dtype=np.float64)
    span_hi = np.array(meta["bounds"][1], dtype=np.float64)
    BINS = 1024
    histogram = np.zeros((2, 3, BINS), dtype=np.int64)

    for position, chunk in enumerate(meta["chunks"], 1):
        path = DIRECTORY / pathlib.PurePosixPath(chunk["file"]).name
        raw = np.fromfile(path, dtype=np.uint8)
        version, vertices, edge_count, isolated = np.frombuffer(raw[:16], dtype=np.uint32)
        if version != 785 or vertices != chunk["vertices"] or edge_count != chunk["edges"]:
            sys.exit(f"Incomplete anatomy chunk: {path.name}")
        packed = np.frombuffer(raw, dtype=np.float32, count=vertices * 5, offset=16).reshape(vertices, 5)
        index = packed[:, 3].astype(np.int64)
        xyz = packed[:, :3].astype(np.float64)
        for axis in range(3):
            column = xyz[:, axis]
            np.minimum.at(lo[:, axis], index, column)
            np.maximum.at(hi[:, axis], index, column)
            total[:, axis] += np.bincount(index, weights=column, minlength=count)
        seen += np.bincount(index, minlength=count)
        side = optic_side[index]
        for axis in range(3):
            bin_index = np.clip(((xyz[:, axis] - span_lo[axis]) / (span_hi[axis] - span_lo[axis]) * BINS).astype(np.int64), 0, BINS - 1)
            for group in (0, 1):
                selected = bin_index[side == bool(group)]
                if selected.size:
                    histogram[group, axis] += np.bincount(selected, minlength=BINS)
        child = np.frombuffer(raw, dtype=np.uint32, count=edge_count * 2, offset=16 + vertices * 20).reshape(-1, 2)[:, 0]
        edges_per_neuron += np.bincount(index[child.astype(np.int64)], minlength=count)
        print(f"  measured chunk {position}/{len(meta['chunks'])}", flush=True)

    if int(seen.sum()) != meta["vertexCount"] or int(edges_per_neuron.sum()) != meta["edgeCount"]:
        sys.exit("Measured geometry does not reconcile with the released counts.")
    centroid = total / np.maximum(seen, 1)[:, None]

    def framing(counts: np.ndarray, fraction: float = 0.01) -> list[list[float]]:
        """Per-axis box holding all but `fraction` of this region's vertices at each
        end, from the measured histograms. Used only to aim the camera; no geometry
        is clipped to it."""
        box = []
        for axis in range(3):
            column = counts[axis].astype(np.float64)
            cumulative = np.cumsum(column) / column.sum()
            first = int(np.searchsorted(cumulative, fraction))
            last = int(np.searchsorted(cumulative, 1.0 - fraction))
            width = (span_hi[axis] - span_lo[axis]) / BINS
            box.append([span_lo[axis] + first * width, span_lo[axis] + (last + 1) * width])
        return [[round(box[a][0], 4) for a in range(3)], [round(box[a][1], 4) for a in range(3)]]

    def summarise(mask: np.ndarray, counts: np.ndarray) -> dict:
        present = mask & (seen > 0)
        if not present.any():
            sys.exit("A region selected no released geometry.")
        return dict(
            neurons=int(present.sum()),
            vertices=int(seen[present].sum()),
            edges=int(edges_per_neuron[present].sum()),
            bounds=[lo[present].min(axis=0).round(4).tolist(), hi[present].max(axis=0).round(4).tolist()],
            framingBounds=framing(counts),
            centroid=centroid[present].mean(axis=0).round(4).tolist(),
        )

    both = histogram.sum(axis=0)
    classes = {}
    for name in sorted({str(value) for value in super_class}):
        selected = super_class == name
        group = histogram[1] if optic_side[selected].all() else histogram[0] if not optic_side[selected].any() else both
        classes[name] = summarise(selected, group)

    everything = np.ones(count, dtype=bool)
    regions = {
        "central": dict(
            label="Central brain",
            description="Every released neuron except the optic lobes: super_class 'optic' and the "
            "photoreceptor afferents (R1-6, R7, R8, ocellar retinula cells) are withheld. "
            "Published annotation only; no coordinate cut, resampling or added geometry.",
            excludedSuperClasses=sorted(OPTIC_SUPER_CLASSES),
            excludedCellTypePrefixes=list(PHOTORECEPTOR_PREFIXES),
            **summarise(~optic_side, histogram[0]),
        ),
        "optic": dict(
            label="Optic lobes",
            description="The withheld complement of the central-brain region.",
            **summarise(optic_side, histogram[1]),
        ),
        "all": dict(
            label="Complete brain",
            description="Every released neuron, unchanged. This is the original full-brain import.",
            **summarise(everything, both),
        ),
    }
    # Every neuron belongs to exactly one of central / optic.
    assert regions["central"]["neurons"] + regions["optic"]["neurons"] == regions["all"]["neurons"]
    assert regions["central"]["vertices"] + regions["optic"]["vertices"] == meta["vertexCount"]
    assert regions["central"]["edges"] + regions["optic"]["edges"] == meta["edgeCount"]

    record = dict(
        version=1,
        materialization=meta["materialization"],
        neuronCount=count,
        sourceAnatomySha256=digest(manifest_path),
        selection="published FlyWire super_class and cell_type annotations",
        coordinates=meta["coordinates"],
        regions=regions,
        superClasses=classes,
        # modelIndex membership, run-length encoded as [start, length] pairs so the
        # viewer rebuilds the exact mask without a per-neuron list.
        opticRuns=runs(optic_side),
    )
    out = DIRECTORY / "regions.json"
    out.write_text(json.dumps(record, indent=1))
    print(json.dumps({name: {k: v for k, v in value.items() if k in ("neurons", "vertices", "bounds", "framingBounds")}
                      for name, value in regions.items()}, indent=1))
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


def runs(mask: np.ndarray) -> list[list[int]]:
    """Run-length encode the True positions of a boolean mask."""
    padded = np.r_[False, mask, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [[int(a), int(b - a)] for a, b in zip(starts, ends)]


if __name__ == "__main__":
    raise SystemExit(main())
