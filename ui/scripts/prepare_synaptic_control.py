"""Prepare a broad, disjoint presynaptic controller experiment.

No displayed neuron is a stimulation target. Connections keep published signs
and contact counts. This file prepares controls and desired neuron activities;
the activities are targets, never passed to the live renderer as observations.
"""
import json
import os
import gzip
import hashlib
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, load_npz, save_npz

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '.cache/control-goal'
OUT.mkdir(parents=True,exist_ok=True)
circuit_meta=json.loads((ROOT/'public/data/connectome.json').read_text())
if not (OUT/'incoming.npz').exists():
    raw=gzip.decompress((ROOT/'public/data/connectome.bin.gz').read_bytes())
    _,n,e=np.frombuffer(raw,dtype='<u4',count=3)
    offsets=np.frombuffer(raw,dtype='<u4',count=n+1,offset=12)
    post=np.frombuffer(raw,dtype='<u4',count=e,offset=12+4*(n+1))
    weights=np.frombuffer(raw,dtype='<i2',count=e,offset=12+4*(n+1)+4*e)
    outputs=np.array(circuit_meta['renderMap']);lookup=np.full(n,-1,np.int32);lookup[outputs]=np.arange(len(outputs))
    selected=np.flatnonzero((lookup[post]>=0)&(weights!=0))
    pre=np.searchsorted(offsets,selected,side='right')-1
    allowed=lookup[pre]<0;selected=selected[allowed];pre=pre[allowed]
    inputs=np.unique(pre);columns=np.searchsorted(inputs,pre)
    incoming=coo_matrix((weights[selected].astype(np.float64),(lookup[post[selected]],columns)),shape=(len(outputs),len(inputs))).tocsr()
    save_npz(OUT/'incoming.npz',incoming);np.save(OUT/'input-ids.npy',inputs);np.save(OUT/'output-ids.npy',outputs)
A = load_npz(OUT/'incoming.npz').tocsr()
ids = np.load(OUT/'input-ids.npy')
output_ids = np.load(OUT/'output-ids.npy')
energy = np.asarray(abs(A).sum(axis=0)).ravel()
top_k = int(os.environ.get('CONTROL_INPUTS_PER_SIGN', '3'))
chosen = set()
for row in range(A.shape[0]):
    lo, hi = A.indptr[row:row+2]
    cols, weights = A.indices[lo:hi], A.data[lo:hi]
    for sign in (1, -1):
        candidates = np.flatnonzero(weights*sign > 0)
        quality = weights[candidates]**2 / energy[cols[candidates]]
        chosen.update(cols[candidates[np.argsort(quality)[-top_k:]]].tolist())
chosen = np.array(sorted(chosen))
forbidden=set(output_ids.tolist())|{n['index'] for n in circuit_meta['outputs']}
chosen=np.array([i for i in chosen if int(ids[i]) not in forbidden])
B = A[:, chosen].tocoo()
strength = np.asarray(abs(A[:, chosen]).max(axis=0).toarray()).ravel()
normalized = B.data/strength[B.col]
# Safe step for projected gradient on ||Bq-current||^2, using ||B||1 ||B||inf.
col_norm = np.bincount(B.col, weights=abs(normalized), minlength=len(chosen))
row_norm = np.bincount(B.row, weights=abs(normalized), minlength=A.shape[0])
step = 1/(col_norm.max()*row_norm.max())
metadata = dict(inputIds=ids[chosen].tolist(), outputIds=output_ids.tolist(),
                row=B.row.tolist(), column=B.col.tolist(), weight=normalized.tolist(),
                strength=strength.tolist(), step=float(step),
                method='Broad optical control of real presynaptic neurons; displayed and motor-output neurons never directly stimulated',
                source=circuit_meta['source'],revision=circuit_meta['revision'],
                connectomeSha256=hashlib.sha256((ROOT/'public/data/connectome.bin.gz').read_bytes()).hexdigest(),
                inputsPerSign=top_k,maximumOpticalInputHz=4000,displayRateScaleHz=60,spikeTraceMs=100,
                caveat='Strong artificial optical control in a simplified LIF model, with zero refractory period on optical targets. Not a validated protocol for a living fly.')
(OUT/f'presynaptic-controller-{top_k}.json').write_text(json.dumps(metadata,separators=(',',':')))
if top_k==3:
    (OUT/'presynaptic-controller.json').write_text(json.dumps(metadata,separators=(',',':')))
    (ROOT/'public/data/presynaptic-controller.json').write_text(json.dumps(metadata,separators=(',',':')))

if os.environ.get('CONTROL_ONLY_INPUTS'):
    print(f'{len(chosen)} controls prepared.'); raise SystemExit
raw = (ROOT/'public/data/image-operator.bin').read_bytes()
P,N,E = np.frombuffer(raw,dtype='<u4',count=3)
pixel = np.frombuffer(raw,dtype='<u2',count=E,offset=12)
neuron = np.frombuffer(raw,dtype='<u2',count=E,offset=12+2*E)
weight = np.frombuffer(raw,dtype='<f4',count=E,offset=12+4*E)
H = np.zeros((P,N)); H[pixel,neuron]=weight
supported = H.sum(axis=1)>0
targets=json.loads((ROOT/'public/data/controller-targets.json').read_text())
records=[]
for target in targets:
    y=np.array(target['pixels']); h=H[supported]; b=y[supported]
    def loss(a):
        residual=h@a-b
        return np.mean(residual**2),2*h.T@residual/len(b)
    result=minimize(loss,np.full(N,b.mean()),jac=True,bounds=[(0,1)]*N,method='L-BFGS-B',
                    options={'maxiter':600,'ftol':1e-12,'gtol':1e-7})
    records.append(dict(label=target['label'],pixels=target['pixels'],desiredActivity=result.x.tolist()))
    print(target['label'],float(result.fun),flush=True)
(OUT/'activity-targets.json').write_text(json.dumps(records,separators=(',',':')))
print(f'{len(chosen)} controls, {len(B.data)} connections to {A.shape[0]} displayed neurons, optimizer step {step:.5f}')
