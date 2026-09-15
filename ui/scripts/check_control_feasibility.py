"""Freeze one measured controller request and solve it to convergence offline.
This diagnostic is never used as a playback source.
"""
import json,time
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from scipy.optimize import minimize
root=Path(__file__).resolve().parents[1];cache=root/'.cache/large-control'
snapshot=json.loads((cache/'control-snapshot.json').read_text())['snapshot']
meta=json.loads((root/'public/data/large/controller-2.json').read_text())
a=load_npz(cache/'current-A-128.npz').astype(float);y=np.array(snapshot['target']);q=np.array(snapshot['q']).reshape(-1,4)[:,0];upper=np.array(meta['strength'])*4000*.001375
ridge=.001+.0001/(.001375*np.array(meta['strength']))**2
scale=1/np.sqrt(np.asarray(a.power(2).sum(axis=0)).ravel()+ridge)
def stats(q):
 r=a@q-y;return dict(rms=float(np.sqrt(np.mean(r*r))),medianAbsolute=float(np.median(abs(r))),maxAbsolute=float(max(abs(r))))
start=time.monotonic();calls=0
print('Online',stats(q),flush=True)
def objective(z):
 global calls
 q=z*scale;r=a@q-y;g=np.clip(r,-50,50)
 loss=np.where(abs(r)<=50,.5*r*r,50*abs(r)-1250).sum()+.5*np.sum(ridge*q*q)
 gradient=(a.T@g+ridge*q)*scale
 calls+=1
 if calls%100==0:print(calls,round(time.monotonic()-start,1),stats(q),float(loss),flush=True)
 return loss,gradient
result=minimize(objective,q/scale,jac=True,bounds=list(zip(np.zeros(len(q)),upper/scale)),method='L-BFGS-B',options=dict(maxiter=1500,maxcor=16,ftol=1e-10,gtol=1e-5))
solved=result.x*scale
record=dict(before=stats(q),after=stats(solved),success=bool(result.success),message=str(result.message),iterations=int(result.nit),seconds=time.monotonic()-start)
print(json.dumps(record,indent=2),flush=True)
(cache/'control-feasibility.json').write_text(json.dumps(record,indent=2));np.save(cache/'control-feasible-q.npy',solved)
