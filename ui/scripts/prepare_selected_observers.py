"""Select one independently controlled real neuron per supported image cell.

Selection depends only on fixed anatomical sites and measured signed incoming
contact strengths. No video enters this preparation. Nonselected neurons are
simulated but not displayed in this view; displayed cells cannot be actuators.
"""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix,save_npz
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--width',type=int,default=160);p.add_argument('--directory',default='large');p.add_argument('--score',choices=['strength','external-selectivity'],default='strength');args=p.parse_args()
root=Path(__file__).resolve().parents[1];data=root/'public/data';large=data/args.directory;prefix='selected-' if args.directory=='large' else f'selected-{args.directory}-';prefix=prefix if args.score=='strength' else prefix+'selective-';out=data/f'{prefix}{args.width}';out.mkdir(exist_ok=True)
meta=json.loads((large/'brain.json').read_text());view=json.loads((large/f'sites-{args.width}.json').read_text());op=(large/f'sites-{args.width}.bin').read_bytes();_,pixels,count,entries=np.frombuffer(op,'<u4',4)
row=np.frombuffer(op,'<u4',entries,16);column=np.frombuffer(op,'<u4',entries,16+4*entries)
sitefile=large/f'observation-points-{args.width}.bin'
if not sitefile.exists():sitefile=large/'observation-points.bin'
assert hashlib.sha256(sitefile.read_bytes()).hexdigest()==view['sitesSha256']
sites=np.fromfile(sitefile,'<f4').reshape(-1,4)
network=json.loads((data/'connectome.json').read_text());raw=gzip.decompress((data/'connectome.bin.gz').read_bytes());_,n,e=np.frombuffer(raw,'<u4',3)
off=np.frombuffer(raw,'<u4',n+1,12);post=np.frombuffer(raw,'<u4',e,12+4*(n+1));weight=np.frombuffer(raw,'<i2',e,12+4*(n+1)+4*e)
positive=np.zeros(n);negative=np.zeros(n);np.maximum.at(positive,post,weight);np.maximum.at(negative,post,-weight.astype(np.int32))
model=np.array([x['modelIndex'] for x in meta['neurons']]);score=np.sqrt(positive[model]*negative[model])
if args.score=='external-selectivity':
 # Favor existing inputs with strong, localized influence. Exclude the entire
 # candidate observer pool from this scoring assumption, so an inhibitory
 # edge from another possible observer is never mistaken for an actuator.
 candidate=np.zeros(n,bool);candidate[model]=True
 target_edges=np.flatnonzero(candidate[post]);source=np.searchsorted(off,target_edges,side='right')-1
 usable=~candidate[source]&~np.isin(source,[x['index'] for x in network['outputs']])
 target_edges=target_edges[usable];source=source[usable];values=weight[target_edges].astype(float)
 energy=np.bincount(source,weights=values**2,minlength=n)
 selective=values**2/np.sqrt(np.maximum(energy[source],1))
 positive=np.zeros(n);negative=np.zeros(n)
 np.maximum.at(positive,post[target_edges],np.where(values>0,selective,0))
 np.maximum.at(negative,post[target_edges],np.where(values<0,selective,0))
 score=np.sqrt(positive[model]*negative[model])
forbidden={x['index'] for x in network['outputs']};chosen={}
for pixel,col in zip(row,column):
 if model[col] in forbidden:continue
 if int(pixel) not in chosen or score[col]>score[chosen[int(pixel)]]:chosen[int(pixel)]=int(col)
selected=np.array([chosen[k] for k in sorted(chosen)]);rows=np.array(sorted(chosen));outputs=model[selected];newsites=sites[selected].copy();newsites[:,3]=np.arange(len(selected));newsites.tofile(out/'observation-points.bin')
H=coo_matrix((np.ones(len(selected)),(rows,np.arange(len(selected)))),shape=(pixels,len(selected)))
operator=np.array([784,pixels,len(selected),len(selected)],'<u4').tobytes()+rows.astype('<u4').tobytes()+np.arange(len(selected),dtype='<u4').tobytes()+np.ones(len(selected),'<f4').tobytes();(out/'operator.bin').write_bytes(operator)
save_npz(root/f'.cache/large-control/{prefix}{args.width}-H.npz',H.tocsr())
lookup=np.full(n,-1,np.int32);lookup[outputs]=np.arange(len(outputs));edges=np.flatnonzero((lookup[post]>=0)&(weight!=0));pre=np.searchsorted(off,edges,side='right')-1
allow=~np.isin(pre,np.r_[outputs,list(forbidden)]);edges=edges[allow];pre=pre[allow];inputs=np.unique(pre)
B=coo_matrix((weight[edges].astype(float),(lookup[post[edges]],np.searchsorted(inputs,pre))),shape=(len(outputs),len(inputs))).tocsr();strength=np.asarray(abs(B).max(axis=0).toarray()).ravel();B=B.tocoo()
control=dict(inputIds=inputs.tolist(),outputIds=outputs.tolist(),row=B.row.tolist(),column=B.col.tolist(),weight=(B.data/strength[B.col]).tolist(),strength=strength.tolist(),source=network['source'],revision=network['revision'],connectomeSha256=network['sha256'],maximumOpticalInputHz=4000,caveat='Ideal upstream voltage clamps; displayed and motor outputs are never clamped. All full-graph connections remain simulated.')
(out/'controller.json').write_text(json.dumps(control,separators=(',',':')))
# Retain the same actual branch samples for the whole-arbor comparison.
branch=np.fromfile(large/'brain-points.bin','<f4').reshape(-1,4)
remap=np.full(meta['neuronCount'],-1,int);remap[selected]=np.arange(len(selected))
branch=branch[remap[branch[:,3].astype(int)]>=0].copy();branch[:,3]=remap[branch[:,3].astype(int)];branch.tofile(out/'brain-points.bin')
newmeta={**meta,'neuronCount':len(selected),'pointCount':len(branch),'lineVertexCount':0,'neurons':[meta['neurons'][i] for i in selected],'selection':f'One real, fixed branch observation site per supported pixel, chosen by {args.score} of signed incoming contacts independently of video. Unselected neurons are not displayed.','sourceVertexCount':sum(meta['neurons'][i]['vertices'] for i in selected),'sourceEdgeCount':sum(meta['neurons'][i]['edges'] for i in selected),'rendering':'One real observation site per displayed neuron; full-arbor option uses the unchanged parent branch samples.','parentAnatomySha256':meta['assetSha256']['brain-points.bin'],'assetSha256':{'observation-points.bin':hashlib.sha256(newsites.tobytes()).hexdigest(),'brain-points.bin':hashlib.sha256(branch.tobytes()).hexdigest()},'parentIndices':selected.tolist()}
(out/'brain.json').write_text(json.dumps(newmeta,separators=(',',':')))
view={**view,'neurons':len(selected),'points':len(selected),'pixelSupport':len(selected)/pixels,'operatorSha256':hashlib.sha256(operator).hexdigest(),'sitesSha256':newmeta['assetSha256']['observation-points.bin'],'method':newmeta['selection']}
(out/'view.json').write_text(json.dumps(view,indent=2))
print(json.dumps(dict(displayed=len(selected),actuators=len(inputs),controlEdges=len(B.data),resolution=view['resolution'],support=view['pixelSupport'])),flush=True)
