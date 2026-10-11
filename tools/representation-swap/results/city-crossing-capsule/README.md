# City crossing: mixed human to yellow capsule

These results were rendered again using the skill **inside this example package**,
with a fresh local npm dependency installation. The replay runtime is bundled with
the skill; the original engine project was not used as a runtime dependency.

## Outputs and checks

- [Side-by-side video](checks/comparison.mp4) and [three-frame comparison](checks/comparison.jpg).
- [Mixed video](render/mixed/video.mp4) and [capsule video](render/capsule-example/video.mp4):
  1280 × 720, 30 fps, 150 frames, 5 seconds each.
- [Independent verification](checks/verification.json): passed; all 150 paired frames
  checked against the source records; 4,800 entity root matrix comparisons.
- Maximum error for camera world, camera projection, entity world and replacement
  world matrices: **0**. All five original source files have unchanged SHA256 hashes.
- Both videos decode completely without errors; all 150 decoded frame pairs differ.
  In the inspected PNGs at frames 0, 75 and 149, the visible change is confined to
  the central player: a blue human proxy becomes a yellow capsule.
- [Per-frame matrices](render/frame-checks.json), [render summary](render/render-result.json),
  [device evidence](render/render-backend.json) and [source provenance](input-provenance.json).

## Reproduction record

| Item | Recorded value |
| --- | --- |
| PROWBench base commit | `a70c54c6f769458de783b04c624a707b7b51cdb4` |
| Dataset | `AlayaLab/PROWBench` |
| Dataset revision | `9d731e1554949a1875d79ca86023bc26a796695b` |
| Episode / camera / entity | `verified_first_frame/city-crossing` / `follow` / `player` |
| Capsule | Height 1.78 m, radius 0.26 m, `#ffcc22`; Y up; base at entity origin |
| GPU | NVIDIA H200, ANGLE Vulkan, WebGL2 |
| Browser | Chromium headless shell 141.0.7390.37 |
| Dependencies | Three.js 0.181.0, Vite 7.1.5, Playwright 1.56.1 |
| Two-video render time | 13.233 s on this host |

The render time includes server/browser startup, scene loading, GPU drawing/readback
and CPU H.264 encoding. It excludes installation, preparation and downstream video
verification; it is one measured run, not a performance benchmark.

[workflow.json](workflow.json) records the exact commands, selected browser cache,
start/end times and exit codes. All three stages exited 0. See
[environment.json](environment.json) for the recorded versions and
[prepare](logs/prepare.log), [render](logs/render.log) and [verify](logs/verify.log) logs.

This host used Node 22.11.0, which causes an npm engine warning for Vite 7.1.5.
The no-config Vite path used here completed successfully. For a supported setup,
use Node 22.12+ as specified in the [installation instructions](../../README.md).

Historical absolute paths in the JSON and logs record where this run happened.
They are not installation requirements. To reproduce in another location, follow
the relative package entry points in the instructions and prepare new input so
that its provenance points to your local dataset. No prepared input symlinks,
browser binaries or npm dependencies are included in the distributed package.

## Interpretation

The source states, object roots and camera trajectories remain fixed while the
visible representation changes. Small canonical quaternion roundoff is reported
separately (`maxQuaternionError` approximately 7.9e-8 radians); the four matrix
errors above are exactly zero in this run.

This experiment generates RGB proxy inputs. It does not test video model inference
or benchmark scores. Capsule surfaces, silhouettes and occlusions differ from the
human proxy. Canonical contact checks do not establish contact of every new surface,
and this run does not produce new depth, normal or instance-ID annotations.
