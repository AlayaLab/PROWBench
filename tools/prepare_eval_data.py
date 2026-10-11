#!/usr/bin/env python3
"""Prepare the Hugging Face download of PROWBench for evaluation.

The Hugging Face dataset stores one directory per episode under `episodes/<group>/<name>/` (scene, cameras, logs,
annotations, and `views/<camera>/<representation>/` with `video.mp4`, `prompt.txt` and `sample.json`). The evaluation
package reads one directory per dataset instead:

  <out>/<dataset>/samples.jsonl, index.json
  <out>/<dataset>/episodes/<episode id>/{manifest.json, frames.jsonl.gz, events.json, cameras.json, scene.glb, ...}
  <out>/<dataset>/episodes/<episode id>/<camera>/{camera.json, engine.description.txt, <representation>/video.mp4, ...}

This script writes that structure to a new directory, which refers to the files of the download; the download must
remain in place. `samples.jsonl`, `index.json` and the per-episode `frames.jsonl.gz` are
written. Episode ids (dv200-114-city-push, ...) are read from `views/views.json` of each episode and checked against every
`sample.json`.

Groups: verified_first_frame -> verified_ff_40, unverified_first_frame -> unverified_ff_90, long_video -> long_horizon,
multi_view -> multi4, gunfight -> gunfight.

Benchmark phases. The download keeps the engine's own per-frame `interaction.phase`. The ISR events of
eval/protocol/<dataset>/events.json refer to the benchmark phase timeline, which is derived from it as follows (the
engine value is kept as `interaction.sourcePhase`):
  verified_ff_40, unverified_ff_90: phase runs shorter than 3 frames are merged into the following run (the previous
      one at the end of the clip); a recording left with a single phase is split at frame n // 2 and the second half is
      renamed '<phase>-continued'.
  multi4: a recording with a single engine phase is split at frame n // 2; the two halves take the phase names of the
      episode's event groups in eval/protocol/multi4/events.json, in order.
  long_horizon, gunfight: the engine phases are used as recorded.

`visibility.json` (per-frame participant visibility, read by the multi-view metrics) is included where the download
provides it, i.e. for the 40 multi4 views; the other datasets do not use it. Not in the download and therefore not prepared: depth
maps, `files.jsonl`, and a few per-view files listed in the report. `<out>/report.json`
lists the files absent from the download, the mapping from download directories to episode ids, and the phase
rewrites.

Prompts: `prompt.txt` of a five-second view becomes `video.h3-ref2va.txt`; the long-horizon prompts are linked as
`white/prompts.c2r.json`, `mixed/prompts.cwm.json` (`prompts.cwm.READY.json` in the download) and `colored-obb/prompts/`.

Usage:
  python3 tools/prepare_eval_data.py --download <local copy of the dataset> --out <empty directory>
      [--protocol eval/protocol]
  then evaluate with  DATA=<out>, --dataset $DATA/<dataset>  (see eval/USAGE.md).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path

GROUPS = {'verified_first_frame': 'verified_ff_40', 'unverified_first_frame': 'unverified_ff_90',
          'long_video': 'long_horizon', 'multi_view': 'multi4', 'gunfight': 'gunfight'}
EPISODE_FILES = {'manifest.json': 'manifest.json', 'events.json': 'logs/events.json', 'cameras.json': 'cameras/cameras.json',
                 'scene.glb': 'scene/scene.glb', 'world-metadata.json': 'scene/world-metadata.json'}
EPISODE_DIRS = ('cameras', 'logs', 'annotations')   # paths written inside manifest.json are relative to the episode root
VIEW_CONTEXT = ['engine.description.txt', 'engine.context.json', 'appearance.json', 'style.json', 'first_frame.json', 'first_frame.png']
OBB_FILES = ['color-map.json', 'obb-frames.jsonl.gz', 'render.json']
MIN_PHASE_FRAMES = 3
REPO = Path(__file__).resolve().parent.parent


def link(src: Path, dst: Path, absent: dict, kind: str) -> bool:
    if not src.exists():
        absent[kind].append(str(dst.relative_to(link.out)))
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(os.path.relpath(src, dst.parent), dst)
    return True


def link_prompts(src: Path, dst: Path, absent: dict) -> list[str]:
    """Link the prompt files of one representation; returns their names relative to `dst` (empty: none found)."""
    files = []
    if (src / 'prompt.txt').exists():
        link(src / 'prompt.txt', dst / 'video.h3-ref2va.txt', absent, 'prompt'); files.append('video.h3-ref2va.txt')
    for name, target in (('prompts.c2r.json', 'prompts.c2r.json'), ('prompts.cwm.READY.json', 'prompts.cwm.json'),
                         ('prompts.cwm.json', 'prompts.cwm.json')):
        if (src / name).exists() and target not in files:
            link(src / name, dst / target, absent, 'prompt'); files.append(target)
    if (src / 'prompts').is_dir():
        for f in sorted((src / 'prompts').iterdir()):
            link(f, dst / 'prompts' / f.name, absent, 'prompt'); files.append(f'prompts/{f.name}')
    if not files:
        absent['prompt'].append(str((dst / 'video.h3-ref2va.txt').relative_to(link.out)))
    return files


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def segments(frames: list[dict]) -> list[list]:
    out = []
    for fr in frames:
        ph = (fr.get('interaction') or {}).get('phase')
        if out and out[-1][0] == ph:
            out[-1][2] = fr['index'] + 1
        else:
            out.append([ph, fr['index'], fr['index'] + 1])
    return out


def set_phase(fr: dict, phase: str) -> None:
    it = fr.setdefault('interaction', {})
    it.setdefault('sourcePhase', it.get('phase'))
    it['phase'] = phase


def benchmark_phases(frames: list[dict], dataset: str, protocol_phases: list[str] | None) -> list[str]:
    """Rewrite interaction.phase in place with the benchmark timeline; returns what was changed (empty: unchanged)."""
    rec = []
    if dataset in ('verified_ff_40', 'unverified_ff_90'):
        changed = True
        while changed:
            changed = False
            segs = segments(frames)
            for i, (ph, a, b) in enumerate(segs):
                if b - a < MIN_PHASE_FRAMES and len(segs) > 1:
                    target = segs[i + 1][0] if i + 1 < len(segs) else segs[i - 1][0]
                    for fr in frames[a:b]:
                        set_phase(fr, target)
                    rec.append(f'merged {ph} [{a}, {b}) into {target}')
                    changed = True
                    break
        segs = segments(frames)
        if len(segs) == 1:
            p, half = segs[0][0], len(frames) // 2
            for fr in frames:
                it = fr.setdefault('interaction', {})
                it.setdefault('sourcePhase', it.get('phase'))
                if fr['index'] >= half:
                    it['phase'] = f'{p}-continued'
            rec.append(f'split {p} at frame {half}')
    elif dataset == 'multi4':
        segs = segments(frames)
        if len(segs) == 1:
            if not protocol_phases or len(protocol_phases) != 2:
                raise SystemExit(f'multi4: expected two protocol phases, got {protocol_phases}')
            half = len(frames) // 2
            for fr in frames:
                set_phase(fr, protocol_phases[0] if fr['index'] < half else protocol_phases[1])
            rec.append(f'split {segs[0][0]} at frame {half} into {protocol_phases[0]} / {protocol_phases[1]}')
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--download', required=True, help='local copy of the Hugging Face dataset (the directory that holds episodes/)')
    ap.add_argument('--out', required=True, help='new or empty directory for the prepared data')
    ap.add_argument('--protocol', default=str(REPO / 'eval' / 'protocol'), help='eval/protocol directory (default: this repository)')
    a = ap.parse_args(argv)
    download = Path(a.download).resolve()
    episodes_root = download / 'episodes' if (download / 'episodes').is_dir() else download
    out = Path(a.out).resolve()
    if out.exists() and not out.is_dir():
        raise SystemExit(f'{out} exists and is not a directory')
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} is not empty')
    out.mkdir(parents=True, exist_ok=True)
    link.out = out
    mv = json.load(open(Path(a.protocol) / 'multi4' / 'events.json'))
    mv = mv.get('episodes', mv)
    absent = collections.defaultdict(list)
    names, rewrites = {}, {}
    for group, ds in GROUPS.items():
        if not (episodes_root / group).is_dir():
            raise SystemExit(f'{episodes_root / group} not found')
        root = out / ds
        rows, eps = [], []
        for epdir in sorted((episodes_root / group).iterdir()):
            if not (epdir / 'views' / 'views.json').exists():   # an episode stored under versions/<name>/
                versions = sorted((epdir / 'versions').iterdir()) if (epdir / 'versions').is_dir() else []
                if len(versions) != 1:
                    raise SystemExit(f'{epdir}: no views/views.json and not exactly one versions/ entry')
                epdir = versions[0]
            views = json.load(open(epdir / 'views' / 'views.json'))
            ep = views['episodeId']
            eps.append(ep)
            names[str(epdir.relative_to(episodes_root))] = f'{ds}/{ep}'
            E = root / 'episodes' / ep
            for dst, src in EPISODE_FILES.items():
                link(epdir / src, E / dst, absent, dst)
            for sub in EPISODE_DIRS:
                if (epdir / sub).is_dir():
                    link(epdir / sub, E / sub, absent, sub)
            for f in sorted((epdir / 'scene').iterdir()):
                if f.name != 'frames.jsonl.gz':
                    link(f, E / 'scene' / f.name, absent, 'scene/' + f.name)
            frames = [json.loads(line) for line in gzip.open(epdir / 'scene' / 'frames.jsonl.gz', 'rt', encoding='utf-8')]
            protocol_phases = [p for g in mv[ep] for p in g['phases']] if ds == 'multi4' else None
            rec = benchmark_phases(frames, ds, protocol_phases)
            if rec:
                rewrites[f'{ds}/{ep}'] = rec
            (E / 'scene').mkdir(parents=True, exist_ok=True)
            with gzip.open(E / 'scene' / 'frames.jsonl.gz', 'wt', encoding='utf-8') as fo:
                for fr in frames:
                    fo.write(json.dumps(fr, ensure_ascii=False, separators=(',', ':')) + '\n')
            link(E / 'scene' / 'frames.jsonl.gz', E / 'frames.jsonl.gz', absent, 'frames.jsonl.gz')
            contexts = sorted((epdir / 'views').glob('*/prompt-context/scene-context.json'))
            if contexts:
                link(contexts[0], E / 'scene-context.json', absent, 'scene-context.json')
            else:
                absent['scene-context.json'].append(str((E / 'scene-context.json').relative_to(out)))
            for cam in views['cameras']:
                V, P = E / cam, epdir / 'views' / cam
                link(epdir / 'cameras' / cam / 'camera.json', V / 'camera.json', absent, 'camera.json')
                for f in VIEW_CONTEXT:
                    if (P / 'prompt-context' / f).exists() or f == 'engine.description.txt':
                        link(P / 'prompt-context' / f, V / f, absent, f)
                if (P / 'visibility.json').exists():
                    link(P / 'visibility.json', V / 'visibility.json', absent, 'visibility.json')
                elif ds == 'multi4':
                    absent['visibility.json'].append(str((V / 'visibility.json').relative_to(out)))
                for rep in views['representations']:
                    Rd = P / rep
                    s = json.load(open(Rd / 'sample.json'))
                    if (s['episodeId'], s['camera'], s['representation']) != (ep, cam, rep):
                        raise SystemExit(f'{Rd / "sample.json"}: identity {s["episodeId"]}/{s["camera"]}/{s["representation"]} '
                                         f'does not match {ep}/{cam}/{rep}')
                    link(Rd / 'video.mp4', V / rep / 'video.mp4', absent, 'video.mp4')
                    prompts = link_prompts(Rd, V / rep, absent)
                    main = next((f for f in ('video.h3-ref2va.txt', 'prompts/win0.txt') if f in prompts), prompts[0] if prompts else None)
                    if rep == 'colored-obb':
                        for f in OBB_FILES:
                            link(Rd / f, V / rep / f, absent, f)
                    video = Rd / 'video.mp4'
                    rows.append(dict(dataset=ds, episode=ep, camera=cam, perspective=s.get('perspective'), representation=rep,
                                     video=f'episodes/{ep}/{cam}/{rep}/video.mp4', sha256=sha256(video),
                                     width=s['width'], height=s['height'], fps=float(s['fps']), frames=s['frameCount'],
                                     durationSeconds=s['durationSeconds'], bytes=video.stat().st_size,
                                     prompt=f'episodes/{ep}/{cam}/{rep}/{main}' if main else None,
                                     promptFiles=[f'episodes/{ep}/{cam}/{rep}/{f}' for f in prompts],
                                     firstFrame=(f'episodes/{ep}/{cam}/first_frame.png'
                                                 if (P / 'prompt-context' / 'first_frame.png').exists() else None)))
        root.mkdir(parents=True, exist_ok=True)
        (root / 'samples.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        (root / 'index.json').write_text(json.dumps(dict(dataset=ds, episodes=eps, samples=rows), indent=1) + '\n')
        print(f'{ds}: {len(eps)} episodes, {len(rows)} samples', flush=True)
    report = dict(absentFromDownload={k: dict(count=len(v), paths=v) for k, v in sorted(absent.items())},
                  episodeIds=names, phaseRewrites=rewrites)
    (out / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    print('absent from the download:', {k: len(v) for k, v in sorted(absent.items())})
    print(f'phase rewrites: {len(rewrites)} episodes; report: {out / "report.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
