import * as THREE from 'three';
import {restoreCamera} from './camera-restore.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {mergeGeometries} from 'three/addons/utils/BufferGeometryUtils.js';
import {assembleCameras} from './camera-data.js';
const V=THREE.Vector3,M=THREE.Matrix4;
const white=new THREE.MeshStandardMaterial({color:'#dcdcdc',roughness:1,flatShading:true}),surface=new THREE.MeshStandardMaterial({color:'#969da0',roughness:1,side:THREE.DoubleSide});
const edgeGeometry=new THREE.BoxGeometry(1,1,1),edgeMaterial=new THREE.MeshBasicMaterial({color:'#e52c35',depthTest:true,depthWrite:true,toneMapped:false});
export const PRESETS={white:{default:'white',semantic:{},entities:{}},bbox:{default:'bbox',semantic:{},entities:{}},mixed:{default:'bbox',interactive:'white',semantic:{},entities:{}}};
const visible=node=>{for(let n=node;n;n=n.parent)if(!n.visible)return false;return true;};
const base64=bytes=>{let s='';for(let i=0;i<bytes.length;i+=32768)s+=String.fromCharCode(...bytes.subarray(i,i+32768));return btoa(s);};
const assetHash=async bytes=>[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');
export async function readJSONL(url,onBytes){const response=await fetch(url);if(!response.ok)throw Error('Missing frames: '+url);const bytes=await response.arrayBuffer();if(onBytes)await onBytes(bytes);const header=new Uint8Array(bytes),text=header[0]===0x1f&&header[1]===0x8b?await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text():new TextDecoder().decode(bytes);return text.trim().split('\n').map(JSON.parse);}
export async function loadEpisode(url){
  const root=new URL(url,location.href),response=await fetch(new URL('manifest.json',root));if(!response.ok)throw Error('Missing manifest');const manifestBytes=await response.arrayBuffer(),manifest=JSON.parse(new TextDecoder().decode(manifestBytes)),assetHashes={manifest:await assetHash(manifestBytes)};
  if(manifest.schemaVersion!=='generative-scene.v1')throw Error('Unsupported benchmark schema');
  const cameraURL=new URL(manifest.cameras,root);
  const cameraData=fetch(cameraURL).then(async response=>{if(!response.ok)throw Error('Missing camera index');return assembleCameras(await response.json(),async path=>{const r=await fetch(new URL(path,cameraURL));if(!r.ok)throw Error('Missing camera track: '+path);return r.json();});});
  const [frames,cameras,asset]=await Promise.all([readJSONL(new URL(manifest.states,root),async bytes=>{assetHashes.states=await assetHash(bytes);}),cameraData,fetch(new URL(manifest.assets,root)).then(r=>{if(!r.ok)throw Error('Missing GLB asset');return r.arrayBuffer();})]);
  assetHashes.geometry=await assetHash(asset);assetHashes.cameras=await assetHash(new TextEncoder().encode(JSON.stringify(cameras)));
  const gltf=await new GLTFLoader().parseAsync(asset,root.href),replay=new Replay(manifest,frames,cameras,gltf.scene);replay.assetHashes=assetHashes;return replay;
}
export class Replay{
  constructor(manifest,frames,cameras,canonical){
    Object.assign(this,{manifest,frames,cameras,canonical});this.nodes=new Map();canonical.traverse(n=>{const id=n.userData.nodeId??(/^n\d+$/.test(n.name)?n.name:null);if(id)this.nodes.set(id,n);});
    for(const d of manifest.nodes)if(!this.nodes.has(d.id))throw Error('Missing GLB node '+d.id);
    this.entityMap=new Map(manifest.entities.map(e=>[e.id,e]));this.modes={};this.batches=new Map();this.frameIndex=-1;this.dynamic=new Set(manifest.dynamicNodeIds);
    this.interactive=new Set(['player']);for(const f of frames){for(const r of f.interaction.relations){this.interactive.add(r.subject);this.interactive.add(r.object);}for(const c of f.interaction.contacts)if(c.targetEntityId)this.interactive.add(c.targetEntityId);}
    if(manifest.sceneId==='swimming')for(const e of manifest.entities)if(e.semanticLabel==='water')this.interactive.add(e.id);
    for(const e of manifest.entities)if(['fishing-line','bobber','fish','reins'].includes(e.semanticLabel))this.interactive.add(e.id);
    for(const [id,s] of Object.entries(manifest.initial.nodes))this.applyNode(id,s);
    canonical.updateMatrixWorld(true);
    this.scene=new THREE.Scene();const env=manifest.environment;this.scene.background=new THREE.Color(...env.background);if(env.fog)this.scene.fog=new THREE.Fog(new THREE.Color(...env.fog.color),env.fog.near,env.fog.far);
    for(const d of env.lights){let light;if(d.type==='HemisphereLight')light=new THREE.HemisphereLight(new THREE.Color(...d.color),new THREE.Color(...d.groundColor),d.intensity);else if(d.type==='DirectionalLight')light=new THREE.DirectionalLight(new THREE.Color(...d.color),d.intensity);else if(d.type==='PointLight')light=new THREE.PointLight(new THREE.Color(...d.color),d.intensity,d.distance,d.decay);else continue;light.position.fromArray(d.position);if(light.target&&d.target){light.target.position.fromArray(d.target);this.scene.add(light.target);}this.scene.add(light);}
    this.camera=new THREE.PerspectiveCamera();this.seek(0);
  }
  applyNode(id,s){const n=this.nodes.get(id);n.position.fromArray(s.position);n.quaternion.fromArray(s.quaternion);n.scale.fromArray(s.scale);n.visible=s.visible;if(s.positions){n.geometry.attributes.position.copyArray(s.positions);n.geometry.attributes.position.needsUpdate=true;n.geometry.computeBoundingBox();}}
  entityState(id){return this.frames[this.frameIndex].entities[id]??this.manifest.initial.entities[id];}
  seek(index){if(!Number.isInteger(index)||index<0||index>=this.frames.length)throw Error('Frame out of range');this.frameIndex=index;for(const [id,s] of Object.entries(this.frames[index].nodes))this.applyNode(id,s);this.canonical.updateMatrixWorld(true);for(const b of this.batches.values())b.dirty=true;}
  cameraAt(track){const s=this.cameras.tracks[track]?.[this.frameIndex];if(!s)throw Error('Unknown camera track '+track);return restoreCamera(this.camera,s);}
  modeFor(entity,config){return config.entities?.[entity.id]??config.semantic?.[entity.semanticLabel]??(entity.surface?'surface':config.interactive&&this.interactive.has(entity.id)?config.interactive:config.default);}
  configure(name,config){
    if(!config||(!['white','bbox','surface'].includes(config.default)&&!this.adapters?.has(config.default)))throw Error('Invalid default representation');
    for(const id of Object.keys(config.entities??{}))if(!this.entityMap.has(id))throw Error('Unknown entity '+id);
    for(const value of [...Object.values(config.entities??{}),...Object.values(config.semantic??{}),...(config.interactive?[config.interactive]:[])])if(!['white','bbox','surface'].includes(value)&&!this.adapters?.has(value))throw Error('Unknown representation '+value);
    this.modes[name]=JSON.parse(JSON.stringify(config));const old=this.batches.get(name);if(old){this.scene.remove(old.root);this.batches.delete(name);}return this;
  }
  registerAdapter(name,factory){if(typeof factory!=='function')throw Error('Adapter must be a factory');this.adapters??=new Map();this.adapters.set(name,factory);return this;}
  buildBatch(name){
    const config=this.modes[name]??PRESETS[name];if(!config)throw Error('Unknown representation preset');
    const root=new THREE.Group(),statics=new Map(),dynamic=[],boxes=[],custom=[];
    const moving=node=>{for(let n=node;n;n=n.parent)if(this.dynamic.has(n.userData.nodeId??n.name))return true;return false;};
    for(const e of this.manifest.entities){const mode=this.modeFor(e,config);if(mode==='bbox'){boxes.push(e);continue;}
      if(!['white','surface'].includes(mode)){const adapter=this.adapters.get(mode)({entity:e,nodes:this.nodes,assets:this.canonical});root.add(adapter.object);custom.push({entity:e,adapter});continue;}
      const mat=mode==='surface'?surface:white;
      for(const id of e.parts){const node=this.nodes.get(id);if(!node.geometry||this.manifest.nodes.find(d=>d.id===id)?.canonicalHidden)continue;
        // Authored vehicle windows retain their recorded transparency in white replay.
        const partMat=mode==='white'&&e.canonicalGeometry==='authored'&&node.material?.transparent?node.material:mat;
        if(node.isMesh&&!moving(node)&&visible(node)&&e.semanticLabel!=='person'){
          const g=node.geometry.clone();g.applyMatrix4(node.matrixWorld);for(const k of Object.keys(g.attributes))if(!['position','normal'].includes(k))g.deleteAttribute(k);if(!g.attributes.normal)g.computeVertexNormals();const flat=g.index?g.toNonIndexed():g;
          if(!statics.has(partMat))statics.set(partMat,[]);statics.get(partMat).push(flat);
        }else{const copy=node.isLine?(node.isLineSegments?new THREE.LineSegments(node.geometry,new THREE.LineBasicMaterial({color:'#dcdcdc'})):new THREE.Line(node.geometry,new THREE.LineBasicMaterial({color:'#dcdcdc'}))):new THREE.Mesh(node.geometry,partMat);copy.matrixAutoUpdate=false;copy.frustumCulled=false;root.add(copy);dynamic.push({source:node,copy});}
      }
    }
    for(const [mat,geometries] of statics){const merged=mergeGeometries(geometries,false);root.add(new THREE.Mesh(merged,mat));for(const g of geometries)g.dispose();}
    const bboxMaterial=config.bboxColor?edgeMaterial.clone():edgeMaterial;if(config.bboxColor)bboxMaterial.color.set(config.bboxColor);
    const bbox=new THREE.InstancedMesh(edgeGeometry,bboxMaterial,Math.max(1,boxes.length*12));bbox.count=boxes.length*12;bbox.frustumCulled=false;root.add(bbox);
    const batch={root,config,dynamic,boxes,bbox,custom,dirty:true};root.visible=false;this.scene.add(root);this.batches.set(name,batch);return batch;
  }
  setRepresentation(name){
    const batch=this.batches.get(name)??this.buildBatch(name);for(const b of this.batches.values())b.root.visible=b===batch;
    if(batch.dirty){for(const {source,copy} of batch.dynamic){copy.matrix.copy(source.matrixWorld);copy.visible=visible(source);}
      let slot=0;for(const e of batch.boxes){const state=this.entityState(e.id);for(const matrix of bboxMatrices(state)){batch.bbox.setMatrixAt(slot++,matrix);}}batch.bbox.instanceMatrix.needsUpdate=true;
      for(const {entity,adapter} of batch.custom)adapter.update({state:this.entityState(entity.id),frame:this.frames[this.frameIndex],nodes:this.nodes});batch.dirty=false;
    }this.currentRepresentation=name;return batch;
  }
  draw(renderer,track='follow',representation='white'){
    this.setRepresentation(representation);this.cameraAt(track);
    const env=this.modes[representation]?.environment;
    if(env){this.scene.background.set(env.background);if(env.fog===false)this.scene.fog=null;renderer.toneMapping=env.toneMapping;renderer.toneMappingExposure=env.exposure;}
    this.withCameraVisibility(track,this.scene,()=>renderer.render(this.scene,this.camera));
  }
  // Visibility is a camera render policy, not a change to recorded state.
  // Restore even on a failed draw so a subsequent third-person pass is intact.
  withCameraVisibility(track,scene,render){
    const hidden=new Set(this.cameras.tracks[track]?.[this.frameIndex]?.hiddenEntityIds??[]);
    if(!hidden.size)return render();
    const changed=[],instances=[];
    const hide=node=>{changed.push([node,node.visible]);node.visible=false;};
    if(scene===this.canonical){
      // Hide owned geometry only; child entities (e.g. a held item) remain visible.
      scene.traverse(node=>{if(node.geometry&&hidden.has(node.userData.entityId))hide(node);});
    }else{
      const batch=this.batches.get(this.currentRepresentation);
      for(const {source,copy} of batch.dynamic)if(hidden.has(source.userData.entityId))hide(copy);
      for(const {entity,adapter} of batch.custom)if(hidden.has(entity.id))hide(adapter.object);
      for(let i=0;i<batch.boxes.length;i++)if(hidden.has(batch.boxes[i].id))for(let k=0;k<12;k++){
        const slot=i*12+k,matrix=new M();batch.bbox.getMatrixAt(slot,matrix);instances.push([slot,matrix]);batch.bbox.setMatrixAt(slot,new M().makeScale(0,0,0));
      }
      if(instances.length)batch.bbox.instanceMatrix.needsUpdate=true;
    }
    try{return render();}finally{
      for(const [node,wasVisible] of changed)node.visible=wasVisible;
      if(instances.length){const batch=this.batches.get(this.currentRepresentation);for(const [slot,matrix] of instances)batch.bbox.setMatrixAt(slot,matrix);batch.bbox.instanceMatrix.needsUpdate=true;}
    }
  }
  verify(){
    let maxPositionError=0,maxQuaternionError=0,maxMatrixError=0,maxJointPositionError=0,maxContactError=0;
    for(const e of this.manifest.entities){const n=this.nodes.get(e.rootNodeId),s=this.entityState(e.id);maxPositionError=Math.max(maxPositionError,n.getWorldPosition(new V()).distanceTo(new V(...s.position)));n.matrixWorld.elements.forEach((v,i)=>{maxMatrixError=Math.max(maxMatrixError,Math.abs(v-s.worldMatrix[i]));});}
    for(const [id,s] of Object.entries(this.frames[this.frameIndex].nodes)){const n=this.nodes.get(id),q=new THREE.Quaternion(...s.quaternion);maxQuaternionError=Math.max(maxQuaternionError,n.quaternion.angleTo(q));}
    for(const [id,s] of Object.entries(this.frames[this.frameIndex].joints??{})){const n=this.nodes.get(id);maxJointPositionError=Math.max(maxJointPositionError,n.getWorldPosition(new V()).distanceTo(new V(...s.position)));maxQuaternionError=Math.max(maxQuaternionError,n.getWorldQuaternion(new THREE.Quaternion()).angleTo(new THREE.Quaternion(...s.quaternion)));}
    for(const c of this.frames[this.frameIndex].interaction.contacts){if(!c.actorNodeId||!c.targetNodeId)continue;const a=new V(...c.actorLocal).applyMatrix4(this.nodes.get(c.actorNodeId).matrixWorld),b=new V(...c.targetLocal).applyMatrix4(this.nodes.get(c.targetNodeId).matrixWorld);maxContactError=Math.max(maxContactError,Math.abs(a.distanceTo(b)-c.distance));}
    return {frame:this.frameIndex,maxPositionError,maxQuaternionError,maxMatrixError,maxJointPositionError,maxContactError};
  }
}
export function bboxMatrices(state){
  const min=new V(...state.bbox.min),max=new V(...state.bbox.max),center=min.clone().add(max).multiplyScalar(.5),size=max.clone().sub(min),world=new M().fromArray(state.worldMatrix),object=new THREE.Object3D(),result=[];
  for(let axis=0;axis<3;axis++){const others=[0,1,2].filter(x=>x!==axis);for(const a of [-1,1])for(const b of [-1,1]){object.position.copy(center);object.position.setComponent(others[0],center.getComponent(others[0])+a*size.getComponent(others[0])/2);object.position.setComponent(others[1],center.getComponent(others[1])+b*size.getComponent(others[1])/2);const thickness=Math.max(.012,Math.min(.045,size.length()*.007));object.scale.setScalar(thickness);object.scale.setComponent(axis,Math.max(thickness,size.getComponent(axis)));if(!state.visible)object.scale.setScalar(0);object.updateMatrix();result.push(world.clone().multiply(object.matrix));}}
  return result;
}
export {visible,base64};
