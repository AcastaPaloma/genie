"""Pack a complete anatomical operator into bounded GPU CSR blocks.

This is an immutable geometry cache, independent of video and neural activity.
Each stored weight integrates released anatomical cable inside one image pixel.
"""
import json
import numpy as np
from prepare_full_brain import OUT
from download_full_brain import digest

width = 1280
record = json.loads((OUT / f"whole-arbor-{width}.json").read_text())
path = OUT / f"whole-arbor-{width}.bin"
assert digest(path, "sha256") == record["operatorSha256"]
raw = np.memmap(path, mode="r", dtype=np.uint8)
version, rows, neurons, entries = map(int, np.frombuffer(raw, "<u4", count=4))
assert version == 784 and neurons == 139255
row = np.frombuffer(raw, "<u4", count=entries, offset=16)
column = np.frombuffer(raw, "<u4", count=entries, offset=16+entries*4)
weight = np.frombuffer(raw, "<f4", count=entries, offset=16+entries*8)
assert (row[1:] >= row[:-1]).all()
pointer = np.r_[0, np.cumsum(np.bincount(row, minlength=rows))]
directory = OUT / "render-1280"
directory.mkdir(exist_ok=True)
chunks = []
start = 0
while start < rows:
    end = min(rows, int(np.searchsorted(pointer, pointer[start]+8_000_000, side="right"))-1)
    end = max(start+1, end)
    begin, stop = map(int, (pointer[start], pointer[end]))
    packed = np.empty((stop-begin, 2), dtype="<u4")
    packed[:, 0] = column[begin:stop]
    packed.view("<f4")[:, 1] = weight[begin:stop]
    output = directory / f"block-{len(chunks):03d}.bin"
    with output.open("wb") as stream:
        np.array([786, start, end-start, stop-begin], "<u4").tofile(stream)
        (pointer[start:end+1]-begin).astype("<u4").tofile(stream)
        packed.tofile(stream)
    chunks.append(dict(file="render-1280/"+output.name, sha256=digest(output, "sha256"),
                       start=start, rows=end-start, entries=stop-begin))
    start = end
assert sum(c["entries"] for c in chunks) == entries
record.update(chunks=chunks, sourceOperatorSha256=record.pop("operatorSha256"),
              format="GPU CSR blocks: 4 uint32 header words, row pointers, (neuron uint32, weight float32) entries.")
(OUT / "render-1280.json").write_text(json.dumps(record, indent=2)+"\n")
print(json.dumps({"resolution":record["resolution"],"chunks":len(chunks),"entries":entries,
                  "neurons":neurons,"branches":record["edges"]}), flush=True)
