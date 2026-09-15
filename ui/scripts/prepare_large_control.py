"""Build a higher-resolution anatomical basis and real presynaptic controls.

Geometry fitting is an optimistic diagnostic only. These targets never replace
simulated spikes in the live renderer.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix,save_npz
from scipy.optimize import minimize
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'public/data';OUT=DATA/'large';CACHE=ROOT/'.cache/large-control';CACHE.mkdir(parents=True,exist_ok=True)
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--width',type=int,default=320);parser.add_argument('--inputs-per-sign',type=int,default=2);parser.add_argument('--only',choices=['operator','controls']);args=parser.parse_args()
meta=json.loads((OUT/'brain.json').read_text());outputs=np.array([n['modelIndex'] for n in meta['neurons']]);count=len(outputs)
if args.only!='controls':
    points=np.fromfile(OUT/'brain-points.bin',dtype='<f4').reshape(-1,4)
    normal=np.array([-np.sin(np.pi/12),0,np.cos(np.pi/12)]);right=np.cross([0,1,0],normal);up=np.cross(normal,right)
    xy=points[:,:3]@np.stack([right,up],axis=1);uv=(xy-[0,25])/[300,225]+.5
    valid=((uv>=0)&(uv<1)).all(axis=1);width=args.width;height=width*3//4
    cell=(uv[valid]*[width,height]).astype(int);rows=(height-1-cell[:,1])*width+cell[:,0];cols=points[valid,3].astype(int)
    H=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(width*height,count)).tocsr();density=np.asarray(H.sum(axis=1)).ravel();H=H.multiply(1/np.maximum(density[:,None],1)).tocsr()
    coo=H.tocoo();raw=np.array([784,H.shape[0],H.shape[1],H.nnz],dtype='<u4').tobytes()+coo.row.astype('<u4').tobytes()+coo.col.astype('<u4').tobytes()+coo.data.astype('<f4').tobytes()
    (OUT/f'operator-{width}.bin').write_bytes(raw);save_npz(CACHE/f'operator-{width}.npz',H)
    view=dict(resolution=[width,height],normal=normal.tolist(),center=[0,25],footprintMicrometers=[300,225],cellMicrometers=300/width,pixelSupport=float((density>0).mean()),neurons=count,operatorSha256=hashlib.sha256(raw).hexdigest(),anatomySha256=meta['assetSha256']['brain-points.bin'],method='Actual sampled skeleton branches, one scalar per whole neuron, projected density normalization')
    (OUT/f'operator-{width}.json').write_text(json.dumps(view,indent=2)+'\n')
    print(f'{width}x{height} basis: {count} neurons, {H.nnz} contributions, {view["pixelSupport"]:.2%} coverage',flush=True)
    targets=[]
    for name in ['frame-8.png','frame-24.png']:
        path=ROOT/'.cache/native-doom'/name
        if path.exists():
            rgb=np.asarray(Image.open(path).convert('RGB').resize((width,height),Image.Resampling.BOX)).astype(float)/255
            targets.append((name,rgb@np.array([.2126,.7152,.0722])))
    panel=Image.new('RGB',(width*2+48,(height+48)*len(targets)+40),'#f8f9fa');draw=ImageDraw.Draw(panel);records=[]
    draw.text((16,12),'Geometry capacity only: target / fitted whole-neuron brightness',fill='#242c34')
    for j,(name,target) in enumerate(targets):
        y=target.ravel()
        def objective(a):
            residual=H@a-y;return np.mean(residual**2),2*H.T@residual/len(y)
        result=minimize(objective,np.full(count,y.mean()),method='L-BFGS-B',jac=True,bounds=[(0,1)]*count,options={'maxiter':180,'ftol':1e-10,'gtol':1e-8,'maxcor':8})
        predicted=H@result.x;mse=float(np.mean((predicted-y)**2));record=dict(frame=name,mse=mse,psnr=float(-10*np.log10(mse)),iterations=result.nit)
        records.append(record);print(record,flush=True)
        for i,image in enumerate([target,predicted.reshape(height,width)]):panel.paste(Image.fromarray(np.clip(image*255,0,255).astype('uint8')).convert('RGB'),(16+i*(width+16),40+j*(height+48)))
        draw.text((16,44+j*(height+48)+height),f'{name}: {record["psnr"]:.2f} dB. No neural dynamics in this upper-bound test.',fill='#242c34')
    if targets:panel.save(CACHE/f'geometry-fit-{width}.png')
    (CACHE/f'geometry-fit-{width}.json').write_text(json.dumps(records,indent=2))
if args.only!='operator':
    circuit=json.loads((DATA/'connectome.json').read_text());raw=gzip.decompress((DATA/'connectome.bin.gz').read_bytes())
    _,n,e=np.frombuffer(raw,dtype='<u4',count=3);offsets=np.frombuffer(raw,dtype='<u4',count=n+1,offset=12);post=np.frombuffer(raw,dtype='<u4',count=e,offset=12+4*(n+1));w=np.frombuffer(raw,dtype='<i2',count=e,offset=12+4*(n+1)+4*e)
    lookup=np.full(n,-1,np.int32);lookup[outputs]=np.arange(count);edges=np.flatnonzero((lookup[post]>=0)&(w!=0));pre=np.searchsorted(offsets,edges,side='right')-1
    forbidden=set(outputs.tolist())|{x['index'] for x in circuit['outputs']};allowed=~np.isin(pre,list(forbidden));pre=pre[allowed];edges=edges[allowed]
    inputs=np.unique(pre);A=coo_matrix((w[edges].astype(float),(lookup[post[edges]],np.searchsorted(inputs,pre))),shape=(count,len(inputs))).tocsr()
    energy=np.asarray(abs(A).sum(axis=0)).ravel();chosen=set();k=args.inputs_per_sign
    for row in range(count):
        lo,hi=A.indptr[row:row+2];col=A.indices[lo:hi];weight=A.data[lo:hi]
        for sign in [1,-1]:
            candidates=np.flatnonzero(weight*sign>0);score=weight[candidates]**2/energy[col[candidates]];chosen.update(col[candidates[np.argsort(score)[-k:]]].tolist())
    chosen=np.array(sorted(chosen));B=A[:,chosen].tocoo();strength=np.asarray(abs(A[:,chosen]).max(axis=0).toarray()).ravel()
    control=dict(inputIds=inputs[chosen].tolist(),outputIds=outputs.tolist(),row=B.row.tolist(),column=B.col.tolist(),weight=(B.data/strength[B.col]).tolist(),strength=strength.tolist(),source=circuit['source'],revision=circuit['revision'],connectomeSha256=hashlib.sha256((DATA/'connectome.bin.gz').read_bytes()).hexdigest(),inputsPerSign=k,maximumOpticalInputHz=4000,displayRateScaleHz=60,spikeTraceMs=100,caveat='Strong artificial optical feedback in a simplified LIF model; not a validated biological stimulation protocol. Displayed and motor-output neurons excluded from direct stimulation.')
    (OUT/f'controller-{k}.json').write_text(json.dumps(control,separators=(',',':')))
    print(f'{len(chosen)} real control neurons, {len(B.data)} connections into {count} displayed neurons.',flush=True)
