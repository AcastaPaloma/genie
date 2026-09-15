"""Choose one real branch observation site per neuron, independently of video.

Static bipartite matching balances anatomical coverage. It never moves anatomy,
creates neurons, or assigns independent activity to parts of the same neuron.
This is a point observation view, not an illuminated whole-arbor display.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, save_npz
from scipy.sparse.csgraph import maximum_bipartite_matching

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--width',type=int,default=160);parser.add_argument('--directory',default='large');parser.add_argument('--angle',type=float,default=-15);args=parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'public/data' / args.directory
meta = json.loads((OUT / 'brain.json').read_text())
points = np.fromfile(OUT / 'brain-points.bin', '<f4').reshape(-1, 4)
normal = np.array([np.sin(np.deg2rad(args.angle)), 0, np.cos(np.deg2rad(args.angle))])
right = np.cross([0, 1, 0], normal)
xy = points[:, :3] @ np.stack([right, [0, 1, 0]], axis=1)
uv = (xy - [0, 25]) / [300, 225] + .5
valid = ((uv >= 0) & (uv < 1)).all(axis=1)
width, height, n = args.width, args.width*3//4, meta['neuronCount']
first_sample = np.flatnonzero(np.r_[True, np.diff(points[:, 3]) != 0])
assert np.array_equal(points[first_sample, 3], np.arange(n))
indices = np.flatnonzero(valid)
cell = (uv[valid] * [width, height]).astype(int)
rows = (height - 1 - cell[:, 1]) * width + cell[:, 0]
cols = points[valid, 3].astype(int)
basis = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(width * height, n)).tocsr()
assignment = maximum_bipartite_matching(basis, perm_type='column')
selection = np.full(n, -1, int)
population = np.zeros(width * height, int)
lookup = {}
for i, row, col in zip(indices, rows, cols):
    lookup.setdefault((int(row), int(col)), int(i))
for row, col in enumerate(assignment):
    if col >= 0:
        selection[col] = lookup[row, int(col)]
        population[row] += 1
by_neuron = basis.tocsc()
for col in np.flatnonzero(selection < 0):
    candidates = by_neuron.indices[by_neuron.indptr[col]:by_neuron.indptr[col + 1]]
    if len(candidates):
        row = candidates[np.argmin(population[candidates])]
        selection[col] = lookup[int(row), int(col)]
        population[row] += 1
    else:
        # Real off-footprint anatomy remains visible when the camera is orbited.
        selection[col] = first_sample[col]
sites = points[selection]
site_uv = (sites[:, :3] @ np.stack([right, [0, 1, 0]], axis=1) - [0, 25]) / [300, 225] + .5
inside = ((site_uv >= 0) & (site_uv < 1)).all(axis=1)
cell = (site_uv[inside] * [width, height]).astype(int)
row = (height - 1 - cell[:, 1]) * width + cell[:, 0]
column = np.flatnonzero(inside)
H = coo_matrix((1 / population[row], (row, column)), shape=(width * height, n)).tocoo()
raw = np.array([784, width * height, n, H.nnz], '<u4').tobytes() + H.row.astype('<u4').tobytes() + H.col.astype('<u4').tobytes() + H.data.astype('<f4').tobytes()
(OUT / f'sites-{width}.bin').write_bytes(raw)
sites.astype('<f4').tofile(OUT / f'observation-points-{width}.bin')
np.asarray(selection, '<u4').tofile(OUT / f'observation-source-indices-{width}.bin')
save_npz(ROOT / f'.cache/large-control/{"" if args.directory=="large" else args.directory+"-"}sites-{width}.npz', H.tocsr())
record = dict(resolution=[width, height], normal=normal.tolist(), center=[0, 25], footprintMicrometers=[300, 225], cellMicrometers=300 / width,
              pixelSupport=float((population > 0).mean()), neurons=n, points=len(sites),
              operatorSha256=hashlib.sha256(raw).hexdigest(), sitesSha256=hashlib.sha256(sites.astype('<f4').tobytes()).hexdigest(),
              sourceAnatomySha256=meta['assetSha256']['brain-points.bin'],
              method='One immutable actual skeleton sample per neuron; video-independent maximum matching for coverage. Brightness is the owning neuron spike trace. Remaining branches are not illuminated in this observation-site view.')
(OUT / f'sites-{width}.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2), flush=True)
