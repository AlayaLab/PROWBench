import * as THREE from 'three';

export const REFERENCE_PALETTE={sky:'#6b91db',ground:'#3c3d4d',vegetation:'#50934c',structure:'#050505',person:'#34aaea',bbox:'#d74535',water:'#42678b'};
export const REFERENCE_STYLE_ID='representation-ref-v3';
export const REFERENCE_MIXED_PROXIES={
  person:{adapter:'fixed-blue-person',pose:'neutral-standing-v1',height:1.78,anchor:'entity-root-at-feet',animateJoints:false},
  vehicle:{adapter:'tapered-vehicle-frame',roofWidthRatio:.64,roofLengthRatio:.60,dimensions:'initial-entity-local-bounds'},
};
const box=new THREE.BoxGeometry(1,1,1),cylinder=new THREE.CylinderGeometry(.5,.5,1,16),sphere=new THREE.SphereGeometry(.5,12,8),up=new THREE.Vector3(0,1,0);
const bodyMaterial=new THREE.MeshStandardMaterial({color:'#f4f4f4',roughness:1,metalness:0});
const blueMaterial=new THREE.MeshBasicMaterial({color:REFERENCE_PALETTE.person,toneMapped:false,fog:false});
const redMaterial=new THREE.MeshBasicMaterial({color:REFERENCE_PALETTE.bbox,toneMapped:false,fog:false});

function fixedSegment(object,from,to,width,material,name){
  const a=new THREE.Vector3(...from),b=new THREE.Vector3(...to),delta=b.sub(a),mesh=new THREE.Mesh(cylinder,material);
  mesh.name=name;mesh.position.copy(a).addScaledVector(delta,.5);mesh.quaternion.setFromUnitVectors(up,delta.clone().normalize());mesh.scale.set(width,delta.length(),width);
  mesh.updateMatrix();object.add(mesh);return mesh;
}
function followEntity(object){
  object.matrixAutoUpdate=false;
  return {object,update({state}){object.visible=state.visible;object.matrix.fromArray(state.worldMatrix);}};
}
export function fixedPersonAdapter(){
  // A single local-space silhouette, never sampled from the animation rig or
  // resized to the per-frame pose bbox. Only the entity transform is replayed.
  const object=new THREE.Group();object.userData.proxy={...REFERENCE_MIXED_PROXIES.person};
  const points={pelvis:[0,.83,0],chest:[0,1.31,0],neck:[0,1.46,0],
    leftShoulder:[-.14,1.30,0],leftElbow:[-.205,1.04,0],leftHand:[-.22,.78,0],
    rightShoulder:[.14,1.30,0],rightElbow:[.205,1.04,0],rightHand:[.22,.78,0],
    leftHip:[-.09,.83,0],leftKnee:[-.09,.44,0],leftAnkle:[-.09,.06,0],leftToe:[-.09,.045,-.10],
    rightHip:[.09,.83,0],rightKnee:[.09,.44,0],rightAnkle:[.09,.06,0],rightToe:[.09,.045,-.10]};
  const segment=(a,b,width)=>fixedSegment(object,points[a],points[b],width,blueMaterial,a+'-'+b);
  segment('pelvis','chest',.115);segment('chest','neck',.08);
  for(const side of ['left','right']){
    segment('chest',side+'Shoulder',.095);segment(side+'Shoulder',side+'Elbow',.085);segment(side+'Elbow',side+'Hand',.078);
    segment('pelvis',side+'Hip',.10);segment(side+'Hip',side+'Knee',.085);segment(side+'Knee',side+'Ankle',.08);segment(side+'Ankle',side+'Toe',.08);
  }
  for(const [name,position] of Object.entries(points)){
    const mesh=new THREE.Mesh(sphere,blueMaterial);mesh.name=name;mesh.position.fromArray(position);mesh.scale.setScalar(name.endsWith('Toe')?.09:.085);mesh.updateMatrix();object.add(mesh);
  }
  const head=new THREE.Mesh(sphere,blueMaterial);head.name='head';head.position.set(0,1.60,0);head.scale.set(.25,.36,.25);head.updateMatrix();object.add(head);
  return followEntity(object);
}
export function taperedVehicleAdapter(initialState){
  const object=new THREE.Group(),config=REFERENCE_MIXED_PROXIES.vehicle;
  object.userData.proxy={...config};
  const min=new THREE.Vector3(...initialState.bbox.min),max=new THREE.Vector3(...initialState.bbox.max),size=max.clone().sub(min),center=min.clone().add(max).multiplyScalar(.5);
  const width=Math.max(.012,Math.min(.045,size.length()*.007)),half=width/2,corners=[];
  // Two rectangular rings with different dimensions and four sloping pillars:
  // this is a tapered proxy in 3D, not a rectangular box seen in perspective.
  for(const roof of [false,true])for(const [x,z] of [[-1,-1],[1,-1],[1,1],[-1,1]])corners.push([
    center.x+x*(size.x/2-half)*(roof?config.roofWidthRatio:1),
    roof?max.y-half:min.y+half,
    center.z+z*(size.z/2-half)*(roof?config.roofLengthRatio:1),
  ]);
  for(let i=0;i<4;i++){
    fixedSegment(object,corners[i],corners[(i+1)%4],width,redMaterial,'base-'+i);
    fixedSegment(object,corners[i+4],corners[(i+1)%4+4],width,redMaterial,'roof-'+i);
    fixedSegment(object,corners[i],corners[i+4],width,redMaterial,'pillar-'+i);
  }
  return followEntity(object);
}

export function humanAdapter({entity,nodes},skeleton=false){
  const object=new THREE.Group(),bones=new Map(entity.joints.map(j=>[j.name,nodes.get(j.nodeId)])),updates=[];
  const material=skeleton?blueMaterial:bodyMaterial;
  const point=(name,offset)=>{const n=bones.get(name);if(!n)throw Error('Missing human joint '+name);return n.localToWorld(new THREE.Vector3(...(offset??[0,0,0])));};
  const mesh=(geometry)=>{const m=new THREE.Mesh(geometry,material);m.frustumCulled=false;object.add(m);return m;};
  function segment(from,to,width,depth=width,fromOffset,toOffset){
    const m=mesh(skeleton?cylinder:box);m.name=from+'-'+to;
    updates.push(()=>{const a=point(from,fromOffset),b=point(to,toOffset),delta=b.clone().sub(a);m.position.copy(a).add(b).multiplyScalar(.5);m.quaternion.setFromUnitVectors(up,delta.clone().normalize());m.scale.set(width,delta.length()+(skeleton?width*.45:.025),depth);});
  }
  function attached(name,geometry,scale,offset=[0,0,0]){
    const m=mesh(geometry);m.name=name;m.scale.fromArray(scale);
    updates.push(()=>{m.position.copy(point(name,offset));bones.get(name).getWorldQuaternion(m.quaternion);});
  }
  attached('head',skeleton?sphere:cylinder,skeleton?[.27,.36,.27]:[.285,.285,.285],[0,.045,0]);
  segment('chest','head',skeleton?.075:.095,skeleton?.075:.095,[0,.035,0],[0,-.08,0]);
  if(skeleton){
    segment('pelvis','chest',.075);
    segment('left-shoulder','chest',.065);segment('right-shoulder','chest',.065);
  }else{
    segment('pelvis','chest',.43,.255,[0,-.035,0],[0,.1,0]);
    attached('pelvis',box,[.34,.16,.245],[0,-.025,0]);
  }
  for(const side of ['left','right']){
    if(skeleton)segment('pelvis',side+'-hip',.07);
    segment(side+'-shoulder',side+'-elbow',skeleton?.066:.18,skeleton?.066:.21);
    segment(side+'-elbow',side+'-wrist',skeleton?.064:.16,skeleton?.064:.19);
    segment(side+'-hip',side+'-knee',skeleton?.073:.205,skeleton?.073:.225);
    segment(side+'-knee',side+'-ankle',skeleton?.070:.18,skeleton?.070:.205);
    if(skeleton){segment(side+'-wrist',side+'-palm',.066);segment(side+'-ankle',side+'-toe',.062);}
    else{attached(side+'-palm',box,[.13,.095,.09],[0,-.014,0]);attached(side+'-foot-ball',box,[.18,.065,.22],[0,0,-.025]);}
  }
  return {object,update({state}){object.visible=state.visible;if(state.visible)for(const update of updates)update();}};
}

function broadFloor(entity,nodes){
  if(entity.semanticLabel!=='geometry')return false;
  const bounds=new THREE.Box3();for(const id of entity.parts){const node=nodes.get(id);if(node.isMesh)bounds.expandByObject(node);}
  const s=bounds.getSize(new THREE.Vector3());return s.y<=.08&&s.x>=2&&s.z>=2;
}
function semanticColor(entity,nodes){
  if(entity.semanticLabel==='effect')return '#e9aa47';
  if(['tree','vegetation','plant','planter'].includes(entity.semanticLabel))return REFERENCE_PALETTE.vegetation;
  if(entity.semanticLabel==='water')return REFERENCE_PALETTE.water;
  if(entity.surface||['terrain','ground','road','path','pier','rug'].includes(entity.semanticLabel)||broadFloor(entity,nodes))return REFERENCE_PALETTE.ground;
  return REFERENCE_PALETTE.structure;
}

export function applyReferenceStyle(replay){
  if(replay.referenceStyle)return replay;
  replay.referenceStyle=REFERENCE_STYLE_ID;
  replay.registerAdapter('block-person',options=>humanAdapter(options,false));
  replay.registerAdapter('fixed-blue-person',()=>fixedPersonAdapter());
  replay.registerAdapter('tapered-vehicle-frame',({entity})=>taperedVehicleAdapter(replay.manifest.initial.entities[entity.id]));
  const colors=new Map();
  const whiteFloor=new THREE.MeshStandardMaterial({color:'#555858',roughness:1,side:THREE.DoubleSide});
  function solidAdapter({entity,nodes},whiteSurface=false){
    const object=new THREE.Group(),copies=[],color=semanticColor(entity,nodes);
    if(!colors.has(color))colors.set(color,new THREE.MeshBasicMaterial({color,side:THREE.DoubleSide,toneMapped:false,fog:false}));
    const material=whiteSurface?whiteFloor:colors.get(color),lineMaterial=new THREE.LineBasicMaterial({color:whiteSurface?'#555858':color,toneMapped:false,fog:false});
    for(const id of entity.parts){
      const source=nodes.get(id);if(!source.geometry||replay.manifest.nodes.find(n=>n.id===id)?.canonicalHidden)continue;
      const copy=source.isLine?(source.isLineSegments?new THREE.LineSegments(source.geometry,lineMaterial):new THREE.Line(source.geometry,lineMaterial)):new THREE.Mesh(source.geometry,material);
      copy.matrixAutoUpdate=false;copy.frustumCulled=false;object.add(copy);copies.push({source,copy});
    }
    return {object,update({state}){object.visible=state.visible;for(const {source,copy} of copies){copy.matrix.copy(source.matrixWorld);copy.visible=true;for(let n=source;n;n=n.parent)if(!n.visible){copy.visible=false;break;}}}};
  }
  replay.registerAdapter('semantic-solid',options=>solidAdapter(options));
  replay.registerAdapter('white-surface',options=>solidAdapter(options,true));
  const semantic=Object.fromEntries(replay.manifest.entities.map(e=>[e.semanticLabel,'semantic-solid']));
  semantic.person='fixed-blue-person';
  semantic.vehicle='tapered-vehicle-frame';
  for(const kind of ['boat','horse','crate','door','ball','jetpack','handle'])semantic[kind]='bbox';
  const whiteEntities=Object.fromEntries(replay.manifest.entities.filter(e=>e.surface||['pier','rug'].includes(e.semanticLabel)||broadFloor(e,replay.nodes)).map(e=>[e.id,'white-surface']));
  replay.configure('white',{default:'white',semantic:{person:'block-person'},entities:whiteEntities,environment:{background:'#aaadb0',fog:false,toneMapping:THREE.NoToneMapping,exposure:1}});
  replay.configure('mixed',{default:'semantic-solid',semantic,bboxColor:REFERENCE_PALETTE.bbox,environment:{background:REFERENCE_PALETTE.sky,fog:false,toneMapping:THREE.NoToneMapping,exposure:1}});
  // This profile is opt-in; archived recordings and their source lighting stay
  // unchanged. Neutral lights make the white proxy's volumes legible.
  replay.scene.traverse(n=>{if(n.isLight)n.visible=false;});
  replay.scene.add(new THREE.HemisphereLight('#ffffff','#8b9098',1.35));
  const key=new THREE.DirectionalLight('#ffffff',2.1);key.position.set(-6,12,8);replay.scene.add(key);
  return replay;
}
