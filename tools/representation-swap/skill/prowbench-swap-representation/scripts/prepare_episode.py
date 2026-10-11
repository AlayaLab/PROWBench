#!/usr/bin/env python3
"""Mount a materialized PROWBench recording into the replay layout without changing it."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode', type=Path, required=True)
    p.add_argument('--camera', default='follow')
    p.add_argument('--entity', default='player')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    source, out = a.episode.resolve(), a.out.resolve()
    if out.exists() and any(out.iterdir()):
        p.error('--out must be empty; keep each preparation in a separate directory')
    manifest_path = source / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schemaVersion') != 'generative-scene.v1':
        p.error('expected a materialized generative-scene.v1 episode')
    if a.entity not in {e['id'] for e in manifest['entities']}:
        p.error(f'unknown entity {a.entity}')
    assets = (source / manifest['assets']).resolve()
    states = (source / manifest['states']).resolve()
    index_path = (source / manifest['cameras']).resolve()
    index = json.loads(index_path.read_text())
    if index.get('schemaVersion') != 'generative-camera-index.v1':
        p.error('this helper expects the released split camera index; adapt an older inline index explicitly')
    if a.camera not in index['tracks']:
        p.error(f'camera {a.camera} not present in the index')
    camera_path = (index_path.parent / index['tracks'][a.camera]['path']).resolve()
    camera = json.loads(camera_path.read_text())
    assert camera['track'] == a.camera
    assert len(camera['frames']) == manifest['frameCount']
    for c in camera['frames']:
        assert len(c['cameraToWorld']) == len(c['projection']) == 16
    count = 0
    opener = gzip.open if states.suffix == '.gz' else open
    with opener(states, 'rt') as f:
        for line in f:
            frame = json.loads(line)
            assert frame['index'] == count
            assert 'nodes' in frame and 'entities' in frame
            count += 1
    assert count == manifest['frameCount']
    assert manifest['coordinates']['up'] == 'Y'
    assert manifest['coordinates']['matrices'] == 'column-major'
    out.mkdir(parents=True, exist_ok=True)
    for name, src in [('scene.glb', assets), ('frames.jsonl.gz', states), ('camera.json', camera_path)]:
        (out / name).symlink_to(src)
    derived_manifest = dict(manifest, assets='scene.glb', states='frames.jsonl.gz', cameras='cameras.json')
    (out / 'manifest.json').write_text(json.dumps(derived_manifest, ensure_ascii=False) + '\n')
    selected_index = dict(index)
    selected_index['tracks'] = {a.camera: dict(index['tracks'][a.camera], path='camera.json')}
    (out / 'cameras.json').write_text(json.dumps(selected_index, ensure_ascii=False, indent=2) + '\n')
    source_files = {name: {'path': str(path), 'sha256': digest(path)} for name, path in {
        'manifest': manifest_path, 'states': states, 'geometry': assets,
        'camera_index': index_path, 'camera_track': camera_path,
    }.items()}
    record = {
        'schema': 'prowbench.representation-input.v1', 'source': str(source),
        'prepared': str(out), 'scene_id': manifest['sceneId'], 'camera': a.camera,
        'entity': a.entity, 'fps': manifest['fps'], 'frame_count': count,
        'width': manifest['width'], 'height': manifest['height'], 'source_files': source_files,
        'derived_changes': ['Only assets/states/cameras path strings in the prepared manifest are changed.',
                            'The prepared camera index includes only the selected, actually released track.',
                            'GLB, state stream and selected camera file are symlinks to unchanged source bytes.'],
        'note': 'Historical sourceHashes can refer to the pre-materialization recording. This record hashes the actual current input files.',
    }
    (out / 'input-provenance.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
