/** What part of the brain the play page draws, and where its camera looks.
 *
 * Two independent choices, both display-only:
 * - region: which published anatomical region is drawn (see brain-regions.ts).
 *   Withheld neurons keep spiking and keep their connections.
 * - focus: camera framing inside that region. 'bottom' aims the orthographic
 *   front camera at the ventral part of the region's own framing box, around the
 *   midline (gnathal ganglia, the neck exit of the descending and ascending
 *   neurons, and the lower central neuropil). It frames; it never cuts: every
 *   neuron the region draws is still drawn and still lit, including any cable
 *   outside the frame.
 */
import {regionMask,type RegionManifest,type RegionName} from './brain-regions';
import type {FullBrainView} from './full-brain';

export const FOCUS_NAMES=['whole','bottom'] as const;
export type FocusName=(typeof FOCUS_NAMES)[number];

export function isFocusName(value:string|null|undefined):value is FocusName{
  return !!value&&(FOCUS_NAMES as readonly string[]).includes(value);
}

/** Share of the region's framing box the bottom focus keeps: the lowest 45% of
 * its height (screen up is +y) and the middle 70% of its width. */
const BOTTOM_HEIGHT=.45,BOTTOM_WIDTH=.7;

export function focusBounds(framing:[number[],number[]],focus:FocusName):[number[],number[]]{
  if(focus==='whole')return framing;
  const [low,high]=framing;
  const centreX=(low[0]+high[0])/2,halfWidth=(high[0]-low[0])*BOTTOM_WIDTH/2;
  return [[centreX-halfWidth,low[1],low[2]],[centreX+halfWidth,low[1]+(high[1]-low[1])*BOTTOM_HEIGHT,high[2]]];
}

export function applyBrainFocus(view:FullBrainView,manifest:RegionManifest,region:RegionName,focus:FocusName){
  const summary=manifest.regions[region];
  view.setRegion(region,regionMask(manifest,region),focusBounds(summary.framingBounds,focus),summary);
}

/** The interface must state the drawn region and how many neurons it draws. */
export function describeFocus(manifest:RegionManifest,region:RegionName,focus:FocusName){
  const summary=manifest.regions[region],total=manifest.neuronCount;
  const drawn=summary.neurons===total?`all ${total.toLocaleString()} neurons drawn`:`${summary.neurons.toLocaleString()} of ${total.toLocaleString()} neurons drawn, all simulated`;
  return `${summary.label} · ${drawn}${focus==='bottom'?' · bottom of the brain':''} · light from simulated spikes`;
}
