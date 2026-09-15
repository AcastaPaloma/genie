import type * as THREE from 'three';
export function loadScene(options: { assetsDir:string; xmlName?:string; onStage?:(text:string)=>void }): Promise<{mj:any;model:any;data:any;meta:any}>;
export function buildMeshes(model:any,meta:any):THREE.Group;
export function syncMeshes(group:THREE.Group,data:any):void;
