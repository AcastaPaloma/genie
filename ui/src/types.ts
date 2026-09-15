export interface BrainMetadata {
  dataset: string;
  materialization: number;
  sourceUrl: string;
  archiveUrl: string;
  neuronCount: number;
  pointCount: number;
  lineVertexCount: number;
  sourceVertexCount: number;
  bounds: [number[], number[]];
  neurons: { rootId: string; cellType: string; superClass: string; side: string }[];
}

export interface Clip {
  id: string;
  name: string;
  file: string;
  cover: string;
  frames: number;
  columns: number;
  width: number;
  height: number;
  originalWidth: number;
  originalHeight: number;
  recordIndex: number;
  split: string;
}

export interface ClipManifest {
  dataset: string;
  sourceUrl: string;
  displayFps: number;
  sourceFps: number | null;
  clips: Clip[];
}

export async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url} (${response.status}).`);
  return response.json();
}

export function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Could not load ${url}.`));
    image.src = url;
  });
}
