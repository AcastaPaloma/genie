"""Precompute the anatomy/connectivity-only matrix for a bounded ADMM controller.

The inverse maps *current errors* to upstream control updates. It contains no
video frames, desired brightnesses, or recorded neural outputs. The LIF model
still produces every displayed spike using the full measured connection graph.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix, load_npz, save_npz
from scipy.linalg import cho_factor, cho_solve

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--width',type=int,default=128)
parser.add_argument('--rho',type=float,default=.1)
parser.add_argument('--ridge',type=float,default=.001)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
out=root/'public/data/large';cache=root/'.cache/large-control'
started=time.monotonic()
raw=(out/'controller-2.json').read_bytes();meta=json.loads(raw)
h=load_npz(cache/f'sites-{args.width}.npz').astype(np.float32)
b=coo_matrix((np.asarray(meta['weight'],np.float32),(meta['row'],meta['column'])),shape=(len(meta['outputIds']),len(meta['inputIds']))).tocsr()
a=(h@b).tocsr();a.sum_duplicates();save_npz(cache/f'current-A-{args.width}.npz',a)
print('A',a.shape,a.nnz,round(time.monotonic()-started,2),flush=True)
p=(a@a.T).toarray(order='F');p.flat[::len(p)+1]+=args.rho+args.ridge
print('Formed A Aᵀ',round(time.monotonic()-started,2),flush=True)
factor=cho_factor(p,overwrite_a=True,check_finite=False)
print('Cholesky',round(time.monotonic()-started,2),flush=True)
inverse=cho_solve(factor,np.eye(a.shape[0],dtype=np.float32,order='F'),overwrite_b=True,check_finite=False)
print('Inverted',round(time.monotonic()-started,2),flush=True)
# Independently verify several random right-hand sides through the sparse A.
rng=np.random.default_rng(341);rhs=rng.standard_normal((a.shape[0],4)).astype(np.float32)
x=inverse@rhs;residual=a@(a.T@x)+(args.rho+args.ridge)*x-rhs
relative=float(np.linalg.norm(residual)/np.linalg.norm(rhs))
if not np.isfinite(inverse).all() or relative>2e-3:raise RuntimeError(f'Inaccurate inverse: {relative}')
file=out/f'control-inverse-{args.width}.bin';inverse.astype('<f4').tofile(file)
record=dict(rows=a.shape[0],columns=a.shape[1],edges=a.nnz,rho=args.rho,ridge=args.ridge,relativeResidual=relative,seconds=time.monotonic()-started,
 sourceControllerSha256=hashlib.sha256(raw).hexdigest(),sourceOperatorSha256=hashlib.sha256((out/f'sites-{args.width}.bin').read_bytes()).hexdigest(),
 inverseSha256=hashlib.sha256(file.read_bytes()).hexdigest(),file=file.name,method='(A A^T + (rho + ridge) I)^-1, A = H B; anatomical observation operator and measured presynaptic weights only; no video-dependent precomputation.')
(out/f'control-inverse-{args.width}.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2),flush=True)
