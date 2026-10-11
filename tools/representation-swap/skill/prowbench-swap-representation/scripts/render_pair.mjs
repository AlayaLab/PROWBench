#!/usr/bin/env node
import {createServer} from 'vite';
import {mkdir, writeFile, readFile, readdir} from 'node:fs/promises';
import {resolve, relative, dirname, extname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {parseArgs} from 'node:util';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {createHash} from 'node:crypto';
import {launchRenderBrowser, verifyRenderPage} from './render-backend.mjs';

const {values: args} = parseArgs({options: {
  input: {type: 'string'}, out: {type: 'string'}, camera: {type: 'string'}, entity: {type: 'string'},
  color: {type: 'string', default: '#ffcc22'}, height: {type: 'string', default: '1.78'},
  radius: {type: 'string', default: '0.26'}, frames: {type: 'string'},
  port: {type: 'string', default: '4391'}, 'render-backend': {type: 'string', default: 'gpu'},
}});
if (!args.input || !args.out) throw Error('Use --input <prepared episode> --out <new output directory>');
const skillRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const input = resolve(args.input), out = resolve(args.out);
const provenance = JSON.parse(await readFile(join(input, 'input-provenance.json'), 'utf8'));
const camera = args.camera ?? provenance.camera, entity = args.entity ?? provenance.entity;
const config = {camera, entity, color: args.color, height: Number(args.height), radius: Number(args.radius)};
try { if ((await readdir(out)).length) throw Error('Output directory must be empty: ' + out); }
catch (e) { if (e.code !== 'ENOENT') throw e; }
await mkdir(out, {recursive: true});
await mkdir(join(out, 'logs'), {recursive: true});
const started = performance.now(), records = [], errors = [], encoders = [];
let browser, launched, server, loaded;
const sha = b => createHash('sha256').update(b).digest('hex');
const implementation = {};
for (const name of ['assets/replacement.js', 'assets/viewer.js', 'assets/runtime/replay.js',
                    'assets/runtime/reference-style.js', 'assets/runtime/camera-restore.js',
                    'scripts/render_pair.mjs', 'scripts/render-backend.mjs']) {
  implementation[name] = sha(await readFile(join(skillRoot, name)));
}
const workflow = {command: [process.execPath, ...process.argv.slice(1)], cwd: process.cwd(),
  startedAt: new Date().toISOString(), input, out, config, implementation};
await writeFile(join(out, 'command.json'), JSON.stringify(workflow, null, 2));

function encoder(name, fps) {
  const path = join(out, name, 'video.mp4');
  const command = ['-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-f', 'image2pipe', '-vcodec', 'mjpeg',
    '-framerate', String(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
    '-pix_fmt', 'yuv420p', '-threads', '2', '-movflags', '+faststart', path];
  const proc = spawn(process.env.FFMPEG ?? 'ffmpeg', command, {stdio: ['pipe', 'ignore', 'pipe']});
  let stderr = '';
  proc.stderr.on('data', b => { stderr += b.toString(); });
  proc.stdin.on('error', () => {});
  const completed = new Promise((yes, no) => {
    proc.once('error', no);
    proc.once('close', async code => {
      await writeFile(join(out, 'logs', name + '-ffmpeg.log'), stderr);
      if (code === 0) yes(); else no(Error('FFmpeg failed: ' + name + ' exit=' + code + '\n' + stderr));
    });
  });
  completed.catch(() => {});
  return {proc, completed, command, path,
    async push(jpeg) {
      if (proc.exitCode !== null) throw Error('Encoder exited before all frames were supplied');
      if (!proc.stdin.write(Buffer.from(jpeg, 'base64'))) {
        await Promise.race([once(proc.stdin, 'drain'), completed.then(() => {throw Error('Encoder ended while writing');})]);
      }
    }};
}

try {
  const mount = {name: 'raw-episode', configureServer(vite) {
    vite.middlewares.use(async (req, res, next) => {
      const path = decodeURIComponent(new URL(req.url, 'http://local').pathname);
      if (!path.startsWith('/episode/')) return next();
      const file = resolve(input, path.slice('/episode/'.length));
      if (relative(input, file).startsWith('..')) { res.statusCode = 403; return res.end(); }
      try {
        res.setHeader('Content-Type', ({'.json': 'application/json', '.glb': 'model/gltf-binary', '.gz': 'application/gzip'})[extname(file)] ?? 'application/octet-stream');
        // Keep gzip bytes intact: the replay loader decompresses them exactly once.
        res.end(await readFile(file));
      } catch { res.statusCode = 404; res.end(); }
    });
  }};
  server = await createServer({configFile: false, root: join(skillRoot, 'assets'), plugins: [mount],
    logLevel: 'error', server: {host: '127.0.0.1', port: Number(args.port), strictPort: true, hmr: false, watch: null}});
  await server.listen();
  launched = await launchRenderBrowser(args['render-backend']);
  browser = launched.browser;
  await writeFile(join(out, 'render-backend.json'), JSON.stringify(launched.report, null, 2));
  const page = await browser.newPage({viewport: {width: provenance.width, height: provenance.height}});
  page.on('pageerror', e => { errors.push(e.message); console.error('PAGE_ERROR', e.message); });
  page.on('requestfailed', r => console.error('REQUEST_FAILED', r.url(), r.failure()?.errorText));
  page.on('response', r => { if (r.status() >= 400) console.error('HTTP_ERROR', r.status(), r.url()); });
  await page.goto(`http://127.0.0.1:${args.port}/viewer.html`, {waitUntil: 'networkidle'});
  await page.waitForFunction(() => window.representationDemo);
  await verifyRenderPage(page, launched.report);
  loaded = await page.evaluate(config => window.representationDemo.load(config), config);
  const modes = [loaded.setup.baseline, loaded.setup.replacement];
  for (const name of modes) await mkdir(join(out, name, 'frames'), {recursive: true});
  const selection = args.frames ? args.frames.split(',').map(Number) : Array.from({length: loaded.frames}, (_, i) => i);
  if (!selection.length || new Set(selection).size !== selection.length || selection.some(i => !Number.isInteger(i) || i < 0 || i >= loaded.frames)) throw Error('Invalid frame selection');
  const stills = new Set(args.frames ? selection : [0, Math.floor(loaded.frames / 2), loaded.frames - 1]);
  if (!args.frames) for (const name of modes) encoders.push(encoder(name, loaded.fps));
  console.log('LOADED', JSON.stringify({scene: loaded.sceneId, camera, entity, frames: selection.length, mode: args.frames ? 'preview' : 'full-video'}));
  const renderStart = performance.now();
  for (const index of selection) {
    const frame = await page.evaluate(({index, png, jpeg}) => window.representationDemo.renderPair(index, {png, jpeg}),
      {index, png: stills.has(index), jpeg: !args.frames});
    for (const [i, name] of modes.entries()) {
      const image = frame.images[name];
      if (image.png) await writeFile(join(out, name, 'frames', String(index).padStart(4, '0') + '.png'), Buffer.from(image.png, 'base64'));
      if (encoders[i]) await encoders[i].push(image.jpeg);
    }
    delete frame.images;
    records.push(frame);
    if (args.frames || (index + 1) % 25 === 0 || index === loaded.frames - 1) console.log('FRAME', index, '/', loaded.frames - 1);
  }
  const renderAndPipeMs = performance.now() - renderStart;
  for (const e of encoders) e.proc.stdin.end();
  await Promise.all(encoders.map(e => e.completed));
  await verifyRenderPage(page, launched.report);
  if (errors.length) throw Error(errors.join('\n'));
  const maxima = {};
  for (const record of records) for (const variant of Object.values(record.variants)) {
    for (const [k, v] of Object.entries(variant.verification)) if (k !== 'frame') maxima[k] = Math.max(maxima[k] ?? 0, v);
    for (const k of ['cameraWorldError', 'cameraProjectionError', 'proxyMatrixError']) if (variant[k] !== undefined) maxima[k] = Math.max(maxima[k] ?? 0, variant[k]);
  }
  const final = {status: 'completed', mode: args.frames ? 'preview' : 'full-video', completedAt: new Date().toISOString(),
    frames: records.length, loaded, maxima, sourceStateUnchanged: true, replacementGeometryUnchanged: true,
    renderAndPipeMs, elapsedMs: performance.now() - started, browserProbeMs: launched.startupAndProbeMs,
    timingNote: 'Elapsed includes server/browser startup, loading, two representations, readback, encoding and checks; excludes source preparation, development and downstream video verification.',
    encoders: encoders.map(e => ({path: e.path, command: ['ffmpeg', ...e.command]}))};
  await writeFile(join(out, 'frame-checks.json'), JSON.stringify(records));
  await writeFile(join(out, 'render-result.json'), JSON.stringify(final, null, 2));
  console.log('RENDER_COMPLETE', JSON.stringify({frames: records.length, maxima, elapsedSeconds: final.elapsedMs / 1000}));
} catch (e) {
  await writeFile(join(out, 'failure.json'), JSON.stringify({status: 'failed', error: String(e), stack: e.stack, completedFrames: records.length}, null, 2));
  if (records.length) await writeFile(join(out, 'partial-frame-checks.json'), JSON.stringify(records));
  throw e;
} finally {
  for (const e of encoders) if (e.proc.exitCode === null) e.proc.kill('SIGTERM');
  await browser?.close();
  await server?.close();
}
