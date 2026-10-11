// Preserve asymmetric frusta and view offsets; fov/aspect alone cannot reproduce them.
const matrix = value => Array.isArray(value) && value.length === 16 && value.every(Number.isFinite);
export function cameraRecord(camera) {
  camera.updateWorldMatrix(true, false);
  return {position:camera.getWorldPosition(camera.position.clone()).toArray(),quaternion:camera.getWorldQuaternion(camera.quaternion.clone()).toArray(),
    fov:camera.fov,aspect:camera.aspect,near:camera.near,far:camera.far,
    cameraToWorld:camera.matrixWorld.toArray(),worldToCamera:camera.matrixWorldInverse.toArray(),
    projection:camera.projectionMatrix.toArray(),projectionInverse:camera.projectionMatrixInverse.toArray()};
}
export function restoreCamera(camera, state) {
  for (const key of ['fov','aspect','near','far']) if (Number.isFinite(state[key])) camera[key]=state[key];
  if (state.cameraToWorld !== undefined || state.projection !== undefined) {
    if (!matrix(state.cameraToWorld) || !matrix(state.projection)) throw Error('Recorded camera needs finite cameraToWorld and projection matrices');
    camera.matrixAutoUpdate=false;
    camera.matrix.fromArray(state.cameraToWorld);
    if(camera.parent){camera.parent.updateWorldMatrix(true,false);camera.matrix.premultiply(camera.parent.matrixWorld.clone().invert());}
    camera.matrix.decompose(camera.position,camera.quaternion,camera.scale);
    camera.updateMatrixWorld(true);
    camera.projectionMatrix.fromArray(state.projection);
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  } else {
    camera.matrixAutoUpdate=true;camera.scale.set(1,1,1);
    camera.position.fromArray(state.position);camera.quaternion.fromArray(state.quaternion);
    camera.clearViewOffset();camera.updateProjectionMatrix();camera.updateMatrixWorld(true);
  }
  return camera;
}
