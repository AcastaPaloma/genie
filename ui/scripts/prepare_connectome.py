"""Package the published Shiu-model v783 connection table for browser simulation.

Run from the repo root: .venv/bin/python ui/scripts/prepare_connectome.py
Requires numpy, pyarrow, certifi. Downloads are cached; no neuron/edge sampling.
"""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from prepare_assets import fetch, ANNOTATIONS, CACHE, OUT

REV = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
REPO = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/" + REV
cache = CACHE / "research"
cache.mkdir(parents=True, exist_ok=True)


def main():
    paths = {}
    for name in ("Connectivity_783.parquet", "Completeness_783.csv", "model.py", "LICENSE"):
        path = cache / ("Drosophila_brain_model-" + name)
        fetch(REPO + "/" + name, path)
        paths[name] = path
    with paths["Completeness_783.csv"].open() as handle:
        rows = list(csv.DictReader(handle))
    ids = [row[""] for row in rows]
    index = {root: i for i, root in enumerate(ids)}
    n = len(ids)
    table = pq.read_table(paths["Connectivity_783.parquet"], columns=[
        "Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity", "Connectivity"])
    pre = table["Presynaptic_Index"].to_numpy().astype(np.uint32)
    post = table["Postsynaptic_Index"].to_numpy().astype(np.uint32)
    weight = table["Excitatory x Connectivity"].to_numpy()
    assert pre.max() < n and post.max() < n
    assert abs(weight).max() < 32768
    # Stable presynaptic sorting builds event-driven outgoing adjacency lists.
    order = np.argsort(pre, kind="stable")
    counts = np.bincount(pre, minlength=n)
    pointers = np.concatenate(([0], np.cumsum(counts))).astype("<u4")
    destination = post[order].astype("<u4")
    signed = weight[order].astype("<i2")
    payload = struct.pack("<III", 783, n, len(post)) + pointers.tobytes() + destination.tobytes() + signed.tobytes()
    with (OUT / "connectome.bin.gz").open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, compresslevel=6) as packed:
            packed.write(payload)
    raw_annotations = fetch(ANNOTATIONS, CACHE / "flywire_annotations.tsv")
    annotations = list(csv.DictReader(raw_annotations.decode().splitlines(), delimiter="\t"))
    # These are optical stimulation targets, not a reconstructed retina.
    # Excitatory visual projection cells allow a zero-baseline LIF model to
    # transmit drive; graded retinal transduction is outside this model.
    inputs = [r for r in annotations if r["root_id"] in index and r["super_class"] == "visual_projection"
              and (r["known_nt"] or r["top_nt"]) == "acetylcholine"]
    inputs.sort(key=lambda r: int(r["root_id"]))
    # Deterministically spread stimulation sites over the annotated population.
    if len(inputs) > 1024:
        rng = np.random.default_rng(783)
        inputs = [inputs[i] for i in sorted(rng.choice(len(inputs), 1024, replace=False))]
    positions = np.array([[float(r["pos_x"]), float(r["pos_y"])] for r in inputs])
    lo, hi = np.percentile(positions, [2, 98], axis=0)
    uv = np.clip((positions-lo)/(hi-lo), 0, .9999)
    input_nodes = [dict(index=index[r["root_id"]], rootId=r["root_id"], cellType=r["cell_type"], side=r["side"],
                        u=round(float(uv[i, 0]), 5), v=round(float(uv[i, 1]), 5)) for i, r in enumerate(inputs)]
    output_types = {"DNa01", "DNa02", "DNg13", "DNp09", "MDN"}
    outputs = [dict(index=index[r["root_id"]], rootId=r["root_id"], cellType=r["cell_type"], side=r["side"])
               for r in annotations if r["root_id"] in index and r["cell_type"] in output_types]
    brain = json.loads((OUT / "brain.json").read_text())
    render_map = [index.get(r["rootId"], -1) for r in brain["neurons"]]
    metadata = dict(version=783, neuronCount=n, edgeCount=len(post), anatomicalSynapseCount=int(table["Connectivity"].to_numpy().sum()),
                    source="https://github.com/philshiu/Drosophila_brain_model", revision=REV,
                    sourceHashes={name: hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()},
                    dataLicense="CC-BY-4.0 (FlyWire)", codeLicense="MIT (Shiu model)",
                    binary="gzip; little-endian u32 [783,N,E], u32 offsets[N+1], u32 targets[E], i16 signedCounts[E]",
                    sha256=hashlib.sha256((OUT/"connectome.bin.gz").read_bytes()).hexdigest(),
                    model="Shiu-style leaky integrate-and-fire, published signed connection counts; no learned weights",
                    inputEncoding="Experimental artificial stimulation: frame luminance at normalized anatomical XY sets Poisson rate of selected cholinergic visual projection cells. Not biological retinotopy or a validated visual model.",
                    inputs=input_nodes, outputs=outputs, renderMap=render_map,
                    missingRenderedNeurons=sum(i<0 for i in render_map),
                    parameters=dict(restMv=-52, thresholdMv=-45, resetMv=-52, membraneMs=20, synapseMs=5,
                                    refractoryMs=2.2, delayMs=1.8, synapseWeightMv=.275, timestepMs=.2, opticalPulseMv=68.75))
    (OUT / "connectome.json").write_text(json.dumps(metadata, separators=(",", ":")))
    (OUT / "shiu-model-license.txt").write_text(paths["LICENSE"].read_text())
    print(json.dumps({k:metadata[k] for k in ["neuronCount","edgeCount","anatomicalSynapseCount","missingRenderedNeurons"]}))
    print(f"{len(input_nodes)} optical stimulation targets, {len(outputs)} identified output cells; {(OUT/'connectome.bin.gz').stat().st_size/1e6:.1f} MB compressed")


if __name__ == "__main__":
    main()
