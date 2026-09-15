"""An optimistic structural bound for the complete subthreshold-gate circuit.

A neuron cannot spike from rest under a subthreshold external bias unless it is
reachable from a suprathreshold seed through positive measured connections.
Reachability ignores timing and inhibition: it is necessary, never sufficient.
"""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
data = ROOT / "public/data/full-brain-783"
cache = ROOT / ".cache/full-brain"
meta = json.loads((data / "connectome.json").read_text())
brain = json.loads((data / "brain.json").read_text())
basis = json.loads((data / "whole-arbor-320.json").read_text())
raw = gzip.decompress((data / "connectome.bin.gz").read_bytes())
if hashlib.sha256(raw).hexdigest() != meta["uncompressedSha256"]:
    raise ValueError("Connectivity checksum differs")
version, n, edges = map(int, np.frombuffer(raw, dtype="<u4", count=3))
assert version == 783 and n == 139255
pointers = np.frombuffer(raw, dtype="<u4", count=n + 1, offset=12)
targets = np.frombuffer(raw, dtype="<u4", count=edges, offset=12 + (n + 1) * 4)
weights = np.frombuffer(raw, dtype="<i2", count=edges, offset=12 + (n + 1) * 4 + edges * 4)
positive = sp.csr_matrix((np.maximum(weights, 0), targets, pointers), shape=(n, n))
positive.eliminate_zeros()
negative = sp.csr_matrix((np.minimum(weights, 0), targets, pointers), shape=(n, n))
negative.eliminate_zeros()
h = sp.load_npz(cache / "whole-arbor-320.npz").tocsr()
assert h.shape == (320 * 240, n)
importance = np.asarray(h.sum(axis=0)).ravel()
outgoing = np.asarray(positive.sum(axis=1)).ravel()
eligible = np.flatnonzero((outgoing > 0) & (np.asarray(negative.sum(axis=1)).ravel() == 0))
motor = {item["index"] for item in meta["outputs"]}
eligible = np.array([i for i in eligible if i not in motor])
scores = np.sqrt(outgoing[eligible]) / np.maximum(.05, importance[eligible])
seeds = eligible[np.lexsort((eligible, -scores))[:700]]
reachable = np.zeros(n, dtype=bool)
reachable[seeds] = True
rounds = 0
while True:
    grown = reachable | (positive.T @ reachable.astype("float32") > 0)
    if np.array_equal(grown, reachable):
        break
    reachable = grown
    rounds += 1
support = np.asarray(h.sum(axis=1)).ravel() > 0
maximum = np.asarray(h @ reachable.astype("float32")).ravel()
sensory = np.zeros(n, dtype=bool)
sensory[meta["visualSensoryIds"]] = True
no_excitation = np.asarray(positive.sum(axis=0)).ravel() == 0
groups = {}
for name, mask in [("reachable", reachable), ("unreachable", ~reachable),
                   ("visualSensory", sensory), ("noExcitatoryInput", no_excitation)]:
    light = np.asarray(h @ mask.astype("float32")).ravel()
    groups[name] = {"neurons": int(mask.sum()), "meanProjectedOpticalShare": float(light[support].mean())}
frames = {}
for clip in ["map01", "map02", "map03"]:
    record = json.loads((cache / "validation" / f"direct-image-v1-{clip}.json").read_text())
    wanted = np.array(record["target"])
    frames[clip] = {"sourceTime": record["metrics"]["time"],
                    "optimisticSupportedMseLowerBound": float(np.square(np.maximum(0, wanted - maximum))[support].mean()),
                    "fractionExceedingReachabilityCeiling": float(np.mean(wanted[support] > maximum[support]))}
result = {"scope": "All official neurons and the complete 320x240 cable operator",
          "interpretation": "Necessary reachability bound only; does not establish attainable rates, image recognition or controllability",
          "anatomySha256": basis["sourceAnatomySha256"], "connectivitySha256": meta["uncompressedSha256"],
          "neurons": n, "connections": edges, "seedIds": seeds.tolist(), "propagationRounds": rounds,
          "groups": groups, "unreachableVisualSensory": int((sensory & ~reachable).sum()), "frames": frames}
output = cache / "validation" / "spike-reachability.json"
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"groups": groups, "frames": frames, "output": str(output)}, indent=2))
