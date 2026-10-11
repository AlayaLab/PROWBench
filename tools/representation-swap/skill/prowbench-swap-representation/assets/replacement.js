import * as THREE from 'three';

// This is a display adapter. Never modify frame.nodes, frame.entities or the camera here.
export function installReplacement(replay, {entity = 'player', color = '#ffcc22', height = 1.78, radius = 0.26} = {}) {
  if (!replay.entityMap.has(entity)) throw Error('Unknown entity: ' + entity);
  if (!(radius > 0 && height > 2 * radius)) throw Error('Capsule height must be greater than twice its radius');
  const adapterName = 'example-yellow-capsule';
  replay.registerAdapter(adapterName, () => {
    const object = new THREE.Group();
    object.matrixAutoUpdate = false;
    object.userData.replacement = {entity, adapterName, height, radius, color, footAnchor: [0, 0, 0]};
    const material = new THREE.MeshBasicMaterial({color, depthTest: true, depthWrite: true, toneMapped: false});
    const body = new THREE.Mesh(new THREE.CapsuleGeometry(radius, height - 2 * radius, 8, 16), material);
    body.name = 'capsule-body';
    body.position.y = height / 2; // CapsuleGeometry is centred: its bottom now coincides with the entity origin.
    body.updateMatrix();
    object.add(body);
    const marker = new THREE.Mesh(new THREE.SphereGeometry(radius * .17, 10, 6),
      new THREE.MeshBasicMaterial({color: '#ffffff', depthTest: true, depthWrite: true, toneMapped: false}));
    marker.name = 'local-minus-z-marker';
    marker.position.set(0, height * .73, -radius);
    marker.updateMatrix();
    object.add(marker);
    return {object, update({state}) {
      object.matrix.fromArray(state.worldMatrix);
      object.visible = state.visible;
    }};
  });
  const config = structuredClone(replay.modes.mixed);
  config.entities = {...config.entities, [entity]: adapterName};
  replay.configure('capsule-example', config);
  return {baseline: 'mixed', replacement: 'capsule-example', entity, adapterName, height, radius, color};
}
