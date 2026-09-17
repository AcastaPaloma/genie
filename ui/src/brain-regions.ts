/** Anatomical region selection for the complete FlyWire v783 import.
 *
 * The complete archive is still downloaded, checksummed and uploaded: a region
 * only decides which released neurons are *drawn* and which contribute light.
 * Membership comes from `regions.json`, measured offline from the published
 * super_class and cell_type annotations by `scripts/prepare_brain_regions.py`.
 * No coordinate cut, resampling, or substituted geometry is involved, and the
 * per-neuron simulation is unchanged — withheld neurons keep spiking, they are
 * simply not rendered.
 */
export const REGION_NAMES = ["central", "optic", "all"] as const;
export type RegionName = (typeof REGION_NAMES)[number];

export interface RegionSummary {
  label: string;
  description: string;
  neurons: number;
  vertices: number;
  edges: number;
  /** Absolute min/max of every vertex in the region, micrometres. */
  bounds: [number[], number[]];
  /** Box holding all but the outermost 1% of the region's vertices per axis.
   * The camera is aimed at this; nothing is clipped to it. */
  framingBounds: [number[], number[]];
  centroid: number[];
  excludedSuperClasses?: string[];
  excludedCellTypePrefixes?: string[];
}

export interface RegionManifest {
  version: 1;
  materialization: number;
  neuronCount: number;
  sourceAnatomySha256: string;
  selection: string;
  coordinates: string;
  regions: Record<RegionName, RegionSummary>;
  superClasses: Record<string, RegionSummary>;
  /** modelIndex membership of the optic lobes, run-length encoded as [start, length]. */
  opticRuns: [number, number][];
}

export function isRegionName(
  value: string | null | undefined,
): value is RegionName {
  return !!value && (REGION_NAMES as readonly string[]).includes(value);
}

export async function fetchRegionManifest(
  directory: string,
  neuronCount: number,
): Promise<RegionManifest> {
  const response = await fetch(directory + "regions.json");
  if (!response.ok)
    throw Error(
      `The anatomical region manifest is unavailable (${response.status}).`,
    );
  const manifest = (await response.json()) as RegionManifest;
  if (manifest.version !== 1 || manifest.neuronCount !== neuronCount)
    throw Error("The region manifest does not describe this anatomy import.");
  for (const name of REGION_NAMES)
    if (!manifest.regions[name])
      throw Error(`The region manifest is missing "${name}".`);
  const optic = manifest.opticRuns.reduce((sum, [, length]) => sum + length, 0);
  if (optic !== manifest.regions.optic.neurons)
    throw Error(
      "The region manifest membership does not match its own optic-lobe count.",
    );
  if (manifest.regions.central.neurons + optic !== neuronCount)
    throw Error(
      "The region manifest does not partition every released neuron.",
    );
  return manifest;
}

/** 1 for every neuron the region draws, 0 for the rest. `null` means every
 * neuron is drawn, so the caller can skip masking entirely. */
export function regionMask(
  manifest: RegionManifest,
  region: RegionName,
): Float32Array | null {
  if (region === "all") return null;
  const keepOptic = region === "optic";
  const mask = new Float32Array(manifest.neuronCount).fill(keepOptic ? 0 : 1);
  for (const [start, length] of manifest.opticRuns) {
    if (start < 0 || start + length > manifest.neuronCount)
      throw Error("A region run falls outside the anatomy.");
    mask.fill(keepOptic ? 1 : 0, start, start + length);
  }
  const drawn = mask.reduce((sum, value) => sum + value, 0);
  if (drawn !== manifest.regions[region].neurons)
    throw Error(
      "The rebuilt region mask does not match the manifest neuron count.",
    );
  return mask;
}
