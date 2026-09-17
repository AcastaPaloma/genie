/** Fetches one released skeleton chunk and verifies it before it reaches the
 * renderer. The version, count and SHA-256 checks are the same ones the main
 * thread used to run inline; running them here leaves the main thread free for
 * GPU upload while several chunks are fetched and verified at once. The chunk
 * loop is main-thread bound, not I/O bound: a one-chunk prefetch on the main
 * thread saved 3%, while moving this work into workers scales with the pool.
 */
interface ChunkRequest{id:number;dir:string;file:string;vertices:number;edges:number;isolatedVertices:number;sha256:string;}
const post=(message:object,transfer?:Transferable[])=>(self as unknown as Worker).postMessage(message,transfer??[]);
self.onmessage=async(event:MessageEvent<ChunkRequest>)=>{
  const chunk=event.data;
  try{
    const response=await fetch(chunk.dir+chunk.file);
    if(!response.ok)throw Error(`Full anatomy chunk could not be loaded (${response.status}): ${chunk.file}`);
    const raw=await response.arrayBuffer();
    const [version,vertices,edges,isolated]=new Uint32Array(raw,0,4);
    if(version!==785||vertices!==chunk.vertices||edges!==chunk.edges||isolated!==chunk.isolatedVertices||raw.byteLength!==16+vertices*20+edges*8+isolated*4)
      throw Error(`Incomplete full-anatomy chunk: ${chunk.file}`);
    const checksum=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',raw)),n=>n.toString(16).padStart(2,'0')).join('');
    if(checksum!==chunk.sha256)throw Error(`Full-anatomy checksum differs: ${chunk.file}`);
    post({id:chunk.id,raw},[raw]);
  }catch(error){post({id:chunk.id,message:error instanceof Error?error.message:String(error)});}
};
