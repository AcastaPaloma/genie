/** Largest aspect-preserving target window meeting a geometric support floor.
 * This only places the controller's target. It never crops neurons, changes
 * their coordinates, or sends image pixels to the anatomical renderer.
 */
export interface AnatomicalAperture {x:number;y:number;width:number;height:number;support:number}
export function anatomicalAperture(mask:Uint8Array,width:number,height:number,aspect=4/3,minimumSupport=.95):AnatomicalAperture{
 if(mask.length!==width*height||!(aspect>0)||!(minimumSupport>0&&minimumSupport<=1))throw Error('Invalid anatomical support request.');
 const stride=width+1,integral=new Uint32Array(stride*(height+1));
 for(let y=0;y<height;y++){let row=0;for(let x=0;x<width;x++){row+=mask[y*width+x]?1:0;integral[(y+1)*stride+x+1]=integral[y*stride+x+1]+row;}}
 for(let w=Math.min(width,Math.floor(height*aspect));w>=2;w--){
  const h=Math.max(1,Math.round(w/aspect));if(h>height)continue;
  let best=-1,bx=0,by=0;
  for(let y=0;y<=height-h;y++)for(let x=0;x<=width-w;x++){
   const sum=integral[(y+h)*stride+x+w]-integral[y*stride+x+w]-integral[(y+h)*stride+x]+integral[y*stride+x];
   if(sum>best){best=sum;bx=x;by=y;}
  }
  if(best/(w*h)>=minimumSupport)return {x:bx,y:by,width:w,height:h,support:best/(w*h)};
 }
 throw Error('No image window meets the requested anatomical support.');
}
