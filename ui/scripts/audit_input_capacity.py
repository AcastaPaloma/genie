"""Conditional anatomical capacity under two explicitly defined input sets.

This is an optimistic offline fit, not playback. Unreachable neurons remain in
the complete optical matrix, with activity constrained to zero. Constant seed
light is fixed to its measured snapshot. All other reachable cells are free,
so the result does not establish dynamical attainability.
"""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from audit_anatomical_limit import fit

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--clip', choices=['map01', 'map02', 'map03'], default='map02')
parser.add_argument('--iterations', type=int, default=1200)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
cache = root / '.cache/full-brain'
records = cache / 'validation'
meta = json.loads((root / 'public/data/full-brain-783/connectome.json').read_text())
summary = json.loads((records / 'event-visual-input-v1-summary.json').read_text())['summary']
baseline = json.loads((records / 'spike-reachability.json').read_text())
seeds = np.array(summary['seedIds'], dtype=int)
if set(seeds) != set(baseline['seedIds']):
    raise ValueError('The actual controller seeds differ from the structural audit')
visual = np.array(summary['visualInputIds'], dtype=int)
if not set(visual).issubset(meta['visualSensoryIds']):
    raise ValueError('Direct visual inputs lack the published anatomical annotation')
raw = gzip.decompress((root / 'public/data/full-brain-783/connectome.bin.gz').read_bytes())
if hashlib.sha256(raw).hexdigest() != meta['uncompressedSha256']:
    raise ValueError('Connectivity checksum differs')
version, n, edges = map(int, np.frombuffer(raw, '<u4', count=3))
if version != 783 or n != 139255:
    raise ValueError('Complete graph required')
pointer = np.frombuffer(raw, '<u4', count=n + 1, offset=12)
target = np.frombuffer(raw, '<u4', count=edges, offset=12 + (n + 1) * 4)
weight = np.frombuffer(raw, '<i2', count=edges, offset=12 + (n + 1) * 4 + edges * 4)
positive = sp.csr_matrix(((weight > 0).astype(np.float32), target, pointer), shape=(n, n))
positive.eliminate_zeros()
h = sp.load_npz(cache / 'whole-arbor-320.npz').astype(np.float64).tocsr()
if h.shape != (320 * 240, n) or h.nnz != 24403052:
    raise ValueError('Complete optical matrix required')
target_record = json.loads((records / f'event-visual-input-v1-{args.clip}.json').read_text())
y = np.array(target_record['target'])
rates = np.frombuffer(base64.b64decode(target_record['neuronRatesBase64']), '<f4').reshape(2, n)[1]
seed_light = np.clip(rates[seeds] / summary['parameters']['rateScale'], 0, 1)
results = []
for name, additional in [('seeds', np.array([], dtype=int)), ('visual', visual)]:
    reachable = np.zeros(n, dtype=bool)
    reachable[seeds] = True
    reachable[additional] = True
    while True:
        grown = reachable | (positive.T @ reachable.astype(np.float32) > 0)
        if np.array_equal(grown, reachable):
            break
        reachable = grown
    lower, upper = np.zeros(n), reachable.astype(float)
    lower[seeds] = seed_light
    upper[seeds] = seed_light
    print(name, 'reachable', int(reachable.sum()), flush=True)
    activities, prediction, history = fit(h, y, args.iterations, report_every=200, lower=lower, upper=upper)
    output = records / f'input-capacity-{name}-{args.clip}'
    np.savez_compressed(output.with_suffix('.npz'), activities=activities, prediction=prediction, target=y, reachable=reachable)
    result = dict(inputSet=name, clip=args.clip, completeNeurons=n, reachableNeurons=int(reachable.sum()),
                  minimumFixedSeedLight=float(seed_light.min()), maximumFixedSeedLight=float(seed_light.max()), history=history,
                  interpretation='Conditional optimistic optical fit: seed brightness fixed at the recorded snapshot, unreachable cells dark, all remaining rates free. Ignores inhibition, timing, current limits and spike variability; not a neural reconstruction.')
    output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    results.append({k: v for k, v in result.items() if k != 'history'} | {'certificate': history[-1]})
print(json.dumps(results, indent=2))
