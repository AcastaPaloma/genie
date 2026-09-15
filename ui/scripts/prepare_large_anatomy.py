"""Expand the real, identity-preserving anatomy without replacing the baseline.

Downloads the documented FlyWire v783 skeletons and retains the original rigid
coordinate convention. Only true skeletal edges are sampled. No inferred edges.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import httpx
import prepare_assets
from prepare_assets import APP, CACHE, BASE, skeleton

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--neurons',type=int,default=30000)
parser.add_argument('--samples',type=int,default=256)
parser.add_argument('--add-central',type=int,default=0)
parser.add_argument('--model-ids',type=Path)
parser.add_argument('--manifest',type=Path,help='Reproduce the exact ordered parent anatomy recorded by an earlier build')
parser.add_argument('--output',default='large')
parser.add_argument('--workers',type=int,default=8)
parser.add_argument('--max-neurons',type=int)
parser.add_argument('--cached-only',action='store_true')
args=parser.parse_args()
# Reuse TLS/HTTP2 connections during bulk acquisition. urllib's per-request
# connections dominate a many-thousand-skeleton download.
client=httpx.Client(http2=True,verify=prepare_assets.CONTEXT,timeout=30,
                    limits=httpx.Limits(max_connections=args.workers,max_keepalive_connections=args.workers))
def pooled_fetch(url,path):
    if path.exists():return path.read_bytes()
    for attempt in range(3):
        try:
            response=client.get(url);response.raise_for_status();data=response.content
            path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_suffix('.partial');temporary.write_bytes(data);temporary.replace(path)
            return data
        except Exception:
            if attempt==2:raise
            time.sleep(attempt+1)
prepare_assets.fetch=pooled_fetch
out=APP/'public/data'/args.output
out.mkdir(parents=True,exist_ok=True)
baseline=json.loads((APP/'public/data/brain.json').read_text())
center=np.array(baseline['sourceCenterNm'])
with (CACHE/'research/Drosophila_brain_model-Completeness_783.csv').open() as f:
    ids=[r[''] for r in csv.DictReader(f)]
mapping={root:i for i,root in enumerate(ids)}
raw=(CACHE/'flywire_annotations.tsv').read_bytes()
rows=list(csv.DictReader(raw.decode().splitlines(),delimiter='\t'))
eligible={r['root_id']:r for r in rows if r['flow']=='intrinsic' and r['root_id'] in mapping}
retained=[r['rootId'] for r in baseline['neurons'] if r['rootId'] in eligible]
remaining=sorted(set(eligible)-set(retained),key=int)
rng=np.random.default_rng(783)
chosen=retained+[remaining[i] for i in rng.choice(len(remaining),args.neurons-len(retained),replace=False)]
if args.add_central:
    retained=[r['rootId'] for r in json.loads((out/'brain.json').read_text())['neurons']]
    central=sorted([root for root,row in eligible.items() if row['super_class']=='central' and root not in retained],key=int)
    chosen=retained+[central[i] for i in rng.choice(len(central),args.add_central,replace=False)]
    args.neurons=len(chosen)
if args.model_ids:
    selected_model_ids=set(np.load(args.model_ids).tolist())
    chosen=sorted([root for root in eligible if mapping[root] in selected_model_ids],key=int)
    if args.cached_only:chosen=[root for root in chosen if (CACHE/'skeletons'/root).exists()]
    if args.max_neurons:
        cached=[root for root in chosen if (CACHE/'skeletons'/root).exists()]
        cached_set=set(cached)
        missing=[root for root in chosen if root not in cached_set]
        missing.sort(key=lambda root:(eligible[root]['super_class']!='central',int(root)))
        chosen=(cached+missing)[:args.max_neurons]
    args.neurons=len(chosen)
manifest=None
if args.manifest:
    manifest=json.loads(args.manifest.read_text())
    assert hashlib.sha256(raw).hexdigest()==manifest['annotationsSha256'], 'Annotation source changed'
    chosen=[n['rootId'] for n in manifest['neurons']]
    assert all(root in eligible for root in chosen), 'Manifest roots no longer match annotations/model'
    assert all(mapping[n['rootId']]==n['modelIndex'] for n in manifest['neurons']), 'Model index mapping changed'
    assert manifest['pointCount']==len(chosen)*args.samples, 'Sample count differs from the manifest'
    center=np.array(manifest['sourceCenterNm']);args.neurons=len(chosen)
    expected_source={n['rootId']:n['sha256'] for n in manifest['neurons']}
neurons=[];failures=[];points=[];lines=[];start=time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
    futures={pool.submit(skeleton,eligible[root]):root for root in chosen}
    # Completion order never defines neuron identities: sort below before packing.
    fetched={}
    for j,future in enumerate(concurrent.futures.as_completed(futures)):
        root=futures[future]
        try:
            row,xyz,links,checksum=future.result()
            if manifest and checksum!=expected_source[root]:raise RuntimeError('Skeleton hash changed: '+root)
            fetched[root]=(row,xyz,links,checksum)
        except Exception as error:
            failures.append(dict(rootId=root,error=str(error)))
        if (j+1)%250==0:
            print(f'{j+1}/{len(chosen)} skeletons; {len(failures)} failures; {time.monotonic()-start:.1f}s',flush=True)
    for root in chosen:
        if root not in fetched:continue
        row,xyz,links,checksum=fetched.pop(root)
        a,b=xyz[links[:,0]],xyz[links[:,1]]
        length=np.linalg.norm(b-a,axis=1).astype(float);valid=length>0
        a,b,length=a[valid],b[valid],length[valid]
        if not len(length):continue
        local=np.random.default_rng(int(root)%2**32)
        take=local.choice(len(a),size=args.samples,p=length/length.sum())
        t=local.random((len(take),1));sampled=a[take]*(1-t)+b[take]*t
        index=len(neurons)
        points.append(np.column_stack((sampled,np.full(args.samples,index))))
        take=local.choice(len(a),size=min(48,len(a)),replace=False)
        segment=np.stack((a[take],b[take]),axis=1).reshape(-1,3)
        lines.append(np.column_stack((segment,np.full(len(segment),index))))
        neurons.append(dict(rootId=root,modelIndex=mapping[root],cellType=row['cell_type'],
                            superClass=row['super_class'],side=row['side'],sha256=checksum,
                            vertices=len(xyz),edges=len(links)))
if manifest and len(neurons)!=len(chosen):raise RuntimeError('Exact-manifest rebuild was incomplete')
if len(neurons)<args.neurons*.9:raise RuntimeError(f'Insufficient skeletons: {len(neurons)}; {failures[:3]}')
point_array=np.concatenate(points).astype('<f4');del points
line_array=np.concatenate(lines).astype('<f4');del lines
for array in (point_array,line_array):
    array[:,:3]=(array[:,:3]-center)/1000;array[:,1]*=-1
point_array.tofile(out/'brain-points.bin');line_array.tofile(out/'brain-lines.bin')
meta={**baseline,'requestedNeurons':args.neurons,'neuronCount':len(neurons),'pointCount':len(point_array),
      'lineVertexCount':len(line_array),'sourceVertexCount':sum(n['vertices'] for n in neurons),
      'sourceEdgeCount':sum(n['edges'] for n in neurons),'neurons':neurons,'failedDownloads':failures,
      'selection':('Intrinsic neurons selected by model IDs in '+str(args.model_ids)+'; cached anatomy first, then central neurons, when a size limit is supplied') if args.model_ids else f'Retain prior anatomy and add {args.add_central} central neurons with seed 783' if args.add_central else 'Retain the original 1500 intrinsic neurons; seed 783 sample of remaining model-mapped intrinsic neurons',
      'annotationsSha256':hashlib.sha256(raw).hexdigest(),'sourceUrl':BASE,
      'bounds':[np.percentile(point_array[:,:3],.1,axis=0).tolist(),np.percentile(point_array[:,:3],99.9,axis=0).tolist()],
      'rendering':f'{args.samples} length-weighted points along true edges and up to 48 original edges per neuron',
      'assetSha256':{name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['brain-points.bin','brain-lines.bin']}}
(out/'brain.json').write_text(json.dumps(meta,separators=(',',':')))
if manifest:
    for name,checksum in manifest['assetSha256'].items():
        assert meta['assetSha256'][name]==checksum, 'Rebuilt anatomy hash differs: '+name
print(f'Packaged {len(neurons)} real neurons, {len(point_array)} samples in {time.monotonic()-start:.1f}s',flush=True)
