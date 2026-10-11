#!/usr/bin/env python3
"""Independently compare exported transforms with source records, decode videos and make a comparison sheet."""
import argparse
import datetime
import gzip
import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distance(a, b):
    assert len(a) == len(b)
    return max([0.0] + [abs(x-y) for x, y in zip(a, b)])


def media_info(video, out):
    args = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
            'stream=width,height,avg_frame_rate,nb_frames,duration', '-of', 'json', str(video)]
    rec = json.loads(subprocess.check_output(args, text=True))['streams'][0]
    decoded = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-an', '-f', 'framemd5', '-'],
                             check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (out / (video.parent.name + '-framemd5.txt')).write_text(decoded.stdout)
    frames = [line.split(',')[-1].strip() for line in decoded.stdout.splitlines() if line and not line.startswith('#')]
    return dict(rec, decoded_frames=len(frames), sha256=sha(video), decoder_errors=decoded.stderr), frames


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True, help='prepared input directory')
    ap.add_argument('--render', type=Path, required=True, help='completed full-video render directory')
    ap.add_argument('--out', type=Path, required=True, help='new verification directory')
    a = ap.parse_args()
    out = a.out.resolve()
    if out.exists() and any(out.iterdir()):
        ap.error('--out must be empty')
    out.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((a.input / 'input-provenance.json').read_text())
    result = json.loads((a.render / 'render-result.json').read_text())
    assert result['status'] == 'completed' and result['mode'] == 'full-video'
    actual_hashes = {}
    for name, record in provenance['source_files'].items():
        actual_hashes[name] = sha(Path(record['path']))
        assert actual_hashes[name] == record['sha256'], ('source changed', name)
    source_manifest = json.loads(Path(provenance['source_files']['manifest']['path']).read_text())
    camera = json.loads(Path(provenance['source_files']['camera_track']['path']).read_text())['frames']
    state_path = Path(provenance['source_files']['states']['path'])
    opener = gzip.open if state_path.suffix == '.gz' else open
    with opener(state_path, 'rt') as f:
        frames = [json.loads(line) for line in f]
    checks = json.loads((a.render / 'frame-checks.json').read_text())
    assert len(checks) == len(frames) == len(camera) == provenance['frame_count']
    assert result['loaded']['assetHashes']['states'] == actual_hashes['states']
    assert result['loaded']['assetHashes']['geometry'] == actual_hashes['geometry']
    setup = result['loaded']['setup']
    before, after, entity = setup['baseline'], setup['replacement'], setup['entity']
    maxima = dict(camera_world=0.0, camera_projection=0.0, entity_world=0.0, replacement_world=0.0)
    root_checks = 0
    for i, check in enumerate(checks):
        assert check['index'] == frames[i]['index'] == i
        assert check['time'] == frames[i]['time']
        expected_entities = dict(source_manifest['initial']['entities'], **frames[i]['entities'])
        assert set(check['variants']) == {before, after}
        for mode, measured in check['variants'].items():
            assert measured['sourceStateUnchanged']
            maxima['camera_world'] = max(maxima['camera_world'], distance(measured['cameraToWorld'], camera[i]['cameraToWorld']))
            maxima['camera_projection'] = max(maxima['camera_projection'], distance(measured['projection'], camera[i]['projection']))
            assert set(measured['entityWorldMatrices']) == set(expected_entities)
            for entity_id, matrix in measured['entityWorldMatrices'].items():
                maxima['entity_world'] = max(maxima['entity_world'], distance(matrix, expected_entities[entity_id]['worldMatrix']))
                root_checks += 1
            if mode == after:
                assert measured['localGeometryUnchanged'] and measured['visibleMatchesRecord']
                maxima['replacement_world'] = max(maxima['replacement_world'], distance(measured['proxyWorldMatrix'], expected_entities[entity]['worldMatrix']))
    assert max(maxima.values()) <= 1e-10, maxima
    media, decoded = {}, {}
    for mode in (before, after):
        media[mode], decoded[mode] = media_info(a.render / mode / 'video.mp4', out)
        info = media[mode]
        assert int(info['width']) == provenance['width'] and int(info['height']) == provenance['height']
        assert float(Fraction(info['avg_frame_rate'])) == provenance['fps']
        assert info['decoded_frames'] == provenance['frame_count']
        assert not info['decoder_errors']
        assert abs(float(info['duration']) - provenance['frame_count']/provenance['fps']) < 1e-3
    changed_video_frames = sum(x != y for x, y in zip(decoded[before], decoded[after]))
    assert changed_video_frames > 0

    indices = [0, len(frames)//2, len(frames)-1]
    font_file = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font = ImageFont.truetype(str(font_file), 19) if font_file.exists() else ImageFont.load_default()
    sheet = Image.new('RGB', (1280, len(indices)*398), '#111827')
    draw = ImageDraw.Draw(sheet)
    differences = []
    for row, index in enumerate(indices):
        images = [Image.open(a.render / mode / 'frames' / f'{index:04d}.png').convert('RGB') for mode in (before, after)]
        diff = ImageChops.difference(*images)
        maximum = ImageChops.lighter(ImageChops.lighter(diff.getchannel('R'), diff.getchannel('G')), diff.getchannel('B'))
        mask = maximum.point(lambda x: 255 if x else 0)
        differences.append({'frame': index, 'time': frames[index]['time'], 'changed_pixels': mask.histogram()[255], 'bbox_xyxy': mask.getbbox()})
        assert differences[-1]['changed_pixels'] > 0
        for column, (mode, im) in enumerate(zip((before, after), images)):
            label = ('Mixed / before' if column == 0 else 'Yellow capsule / after') + f' | frame {index} | {index/provenance["fps"]:.2f}s'
            draw.text((column*640+12, row*398+8), label, fill='white', font=font)
            sheet.paste(im.resize((640,360)), (column*640, row*398+34))
    sheet.save(out / 'comparison.jpg', quality=94)

    ffargs = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
              '-i', str(a.render / before / 'video.mp4'), '-i', str(a.render / after / 'video.mp4'),
              '-filter_complex',
              '[0:v]scale=960:540,pad=iw:ih+36:0:36:color=0x111827,drawtext=text=Mixed / before:x=16:y=7:fontsize=22:fontcolor=white[left];'
              '[1:v]scale=960:540,pad=iw:ih+36:0:36:color=0x111827,drawtext=text=Yellow capsule / after:x=16:y=7:fontsize=22:fontcolor=white[right];'
              '[left][right]hstack=inputs=2[v]',
              '-map', '[v]', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
              '-threads', '2', '-movflags', '+faststart', str(out / 'comparison.mp4')]
    compare = subprocess.run(ffargs, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / 'comparison-ffmpeg.log').write_text(compare.stdout)
    assert compare.returncode == 0, compare.stdout
    summary = {
        'status': 'passed', 'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'frame_pairs': len(checks), 'rendered_frames_total': 2*len(checks), 'entity_matrix_checks': root_checks,
        'max_errors_against_original_recording': maxima, 'source_files_unchanged': True,
        'source_sha256': actual_hashes, 'source_state_unchanged_during_render': True,
        'replacement_local_geometry_unchanged': True, 'media': media,
        'decoded_frames_that_differ': changed_video_frames, 'png_differences': differences,
        'comparison_command': ffargs, 'renderer': json.loads((a.render / 'render-backend.json').read_text())['webgl'],
        'scope': 'Same source state, camera, frame index and time; only the selected entity display adapter changes. No model inference or new depth/ID/normal annotations.',
    }
    (out / 'verification.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
