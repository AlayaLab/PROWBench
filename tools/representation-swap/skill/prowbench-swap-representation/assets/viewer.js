import * as THREE from 'three';
import {loadEpisode} from './runtime/replay.js';
import {applyReferenceStyle} from './runtime/reference-style.js';
import {readCanvasRenderer} from './runtime/renderer-info.js';
import {installReplacement} from './replacement.js';

const canvas = document.querySelector('canvas');
const renderer = new THREE.WebGLRenderer({canvas, antialias: true, preserveDrawingBuffer: true, powerPreference: 'high-performance'});
renderer.setPixelRatio(1);
renderer.outputColorSpace = THREE.SRGBColorSpace;
let replay, setup, cameraName, initialShape;
const maxError = (a, b) => Math.max(0, ...a.map((v, i) => Math.abs(v - b[i])));
const shape = object => JSON.stringify(object.children.map(c => ({name: c.name, matrix: c.matrix.toArray(),
  vertices: Array.from(c.geometry.attributes.position.array), indices: c.geometry.index ? Array.from(c.geometry.index.array) : []})));

window.representationDemo = {
  async load(options) {
    replay = await loadEpisode('/episode/');
    applyReferenceStyle(replay);
    setup = installReplacement(replay, options);
    cameraName = options.camera;
    if (!replay.cameras.tracks[cameraName]) throw Error('Selected camera not loaded');
    renderer.setSize(replay.manifest.width, replay.manifest.height, false);
    return {sceneId: replay.manifest.sceneId, frames: replay.frames.length, fps: replay.manifest.fps,
      width: canvas.width, height: canvas.height, setup, assetHashes: replay.assetHashes, renderer: readCanvasRenderer()};
  },
  renderPair(index, {png = false, jpeg = true} = {}) {
    if (renderer.getContext().isContextLost()) throw Error('WebGL context lost');
    replay.seek(index);
    const stateBefore = JSON.stringify(replay.frames[index]);
    const record = replay.cameras.tracks[cameraName][index];
    const variants = {}, images = {};
    for (const representation of [setup.baseline, setup.replacement]) {
      const started = performance.now();
      replay.draw(renderer, cameraName, representation);
      const submitted = performance.now();
      images[representation] = {
        ...(jpeg ? {jpeg: canvas.toDataURL('image/jpeg', .96).split(',')[1]} : {}),
        ...(png ? {png: canvas.toDataURL('image/png').split(',')[1]} : {}),
      };
      const finished = performance.now();
      const check = replay.verify();
      const cameraWorldError = maxError(replay.camera.matrixWorld.elements, record.cameraToWorld);
      const cameraProjectionError = maxError(replay.camera.projectionMatrix.elements, record.projection);
      const sourceStateUnchanged = JSON.stringify(replay.frames[index]) === stateBefore;
      const worldMatrices = Object.fromEntries(replay.manifest.entities.map(e => [e.id, replay.nodes.get(e.rootNodeId).matrixWorld.toArray()]));
      variants[representation] = {verification: check, sourceStateUnchanged, cameraWorldError, cameraProjectionError,
        cameraToWorld: replay.camera.matrixWorld.toArray(), projection: replay.camera.projectionMatrix.toArray(),
        entityWorldMatrices: worldMatrices, drawSubmitMs: submitted - started, readbackEncodeMs: finished - submitted};
      if (!sourceStateUnchanged) throw Error('A representation modified recorded state');
      for (const [name, value] of Object.entries(check)) {
        if (name !== 'frame' && (!Number.isFinite(value) || value > 1e-4)) throw Error('Replay mismatch: ' + name + '=' + value);
      }
      if (cameraWorldError > 1e-10 || cameraProjectionError > 1e-10) throw Error('Camera mismatch');
      if (representation === setup.replacement) {
        const entry = replay.batches.get(representation).custom.find(x => x.entity.id === setup.entity);
        const object = entry.adapter.object, state = replay.entityState(setup.entity);
        const proxyMatrixError = maxError(object.matrix.elements, state.worldMatrix);
        const currentShape = shape(object);
        initialShape ??= currentShape;
        const localGeometryUnchanged = currentShape === initialShape;
        const visibleMatchesRecord = object.visible === state.visible;
        Object.assign(variants[representation], {proxyWorldMatrix: object.matrixWorld.toArray(), proxyMatrixError,
          localGeometryUnchanged, visibleMatchesRecord, footAnchor: object.userData.replacement.footAnchor});
        if (proxyMatrixError > 1e-10 || !localGeometryUnchanged || !visibleMatchesRecord) throw Error('Replacement adapter invariant failed');
      }
    }
    if (renderer.getContext().isContextLost()) throw Error('WebGL context lost after render');
    return {index, time: replay.frames[index].time, variants, images};
  },
  rendererInfo: () => readCanvasRenderer(),
};
