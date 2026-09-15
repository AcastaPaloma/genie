"""Choose a reference view using anatomy occupancy only, never video content."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',default='inhibitory-now');p.add_argument('--width',type=int,default=128);p.add_argument('--angles',type=float,nargs='+',default=[-60,-45,-30,-15,0,15,30,45,60]);p.add_argument('--footprints',type=float,nargs='+',default=[275,300,350,400]);args=p.parse_args()
root=Path(__file__).resolve().parents[1];out=root/'public/data'/args.directory
meta=json.loads((out/'brain.json').read_text());points=np.fromfile(out/'brain-points.bin','<f4').reshape(-1,4);n=meta['neuronCount'];width=args.width;height=width*3//4;start=time.monotonic();records=[]
for angle in args.angles:
 normal=np.array([np.sin(np.deg2rad(angle)),0,np.cos(np.deg2rad(angle))]);right=np.cross([0,1,0],normal);xy=points[:,:3]@np.stack([right,[0,1,0]],axis=1)
 for footprint in args.footprints:
  uv=(xy-[0,25])/[footprint,footprint*.75]+.5;inside=((uv>=0)&(uv<1)).all(axis=1);cell=(uv[inside]*[width,height]).astype(int);rows=(height-1-cell[:,1])*width+cell[:,0];columns=points[inside,3].astype(int)
  h=coo_matrix((np.ones(len(rows),np.int8),(rows,columns)),shape=(width*height,n)).tocsr();h.data[:]=1;match=maximum_bipartite_matching(h,perm_type='column');record=dict(angle=angle,footprint=footprint,matched=int(sum(match>=0)),support=float(np.mean(match>=0)),candidateNeurons=int(len(np.unique(columns))))
  records.append(record);print(json.dumps(record),flush=True)
records.sort(key=lambda r:r['support'],reverse=True);(root/f'.cache/large-control/{args.directory}-views-{width}.json').write_text(json.dumps(records,indent=2));print('Best',records[0],time.monotonic()-start,flush=True)
