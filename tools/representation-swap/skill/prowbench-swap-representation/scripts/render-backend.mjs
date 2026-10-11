import {existsSync} from 'node:fs';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {delimiter} from 'node:path';
// Use Playwright's standard cache or an explicit PLAYWRIGHT_BROWSERS_PATH.
// The skill can be copied anywhere; browser discovery must not depend on its location.
import {readCanvasRenderer} from '../assets/runtime/renderer-info.js';

const software=/swiftshader|swangle|llvmpipe|softpipe|swrast|software|microsoft basic render/i;
const driverKeys=['VK_DRIVER_FILES','VK_ICD_FILENAMES','__EGL_VENDOR_LIBRARY_FILENAMES'];
export function backendOption(argv=process.argv.slice(2)){
  const value=argv.find(a=>a.startsWith('--render-backend='))?.slice('--render-backend='.length)??'gpu';
  if(!['gpu','cpu'].includes(value))throw Error('--render-backend must be gpu or cpu');
  return value;
}
export function validateBackend(requested,webgl,featureStatus,devices=[]){
  if(!['gpu','cpu'].includes(requested))throw Error('Unknown render backend');
  if(!webgl?.renderer||!webgl?.vendor)throw Error('Unmasked WebGL renderer evidence missing');
  const isSoftware=software.test(webgl.renderer+' '+webgl.vendor);
  if(requested==='gpu'&&(isSoftware||!/^enabled/.test(featureStatus?.webgl2??'')||devices.some(d=>software.test(d.driverVendor??'')))){
    throw Error('GPU rendering required, but hardware WebGL2 was not verified: '+webgl.renderer+'. Check GPU device access and Vulkan/EGL drivers; CPU requires explicit --render-backend=cpu.');
  }
  if(requested==='cpu'&&!isSoftware)throw Error('CPU control unexpectedly uses a hardware renderer: '+webgl.renderer);
  return isSoftware?'cpu':'gpu';
}
export function assertSameRenderer(expected,actual){
  for(const key of ['renderer','vendor','version'])if(actual?.[key]!==expected?.[key])throw Error('Render canvas backend changed: '+key);
}

export async function launchRenderBrowser(requested=backendOption()){
  if(!['gpu','cpu'].includes(requested))throw Error('Unknown render backend');
  const started=performance.now(),env={...process.env};
  if(requested==='cpu'){
    // A forced NVIDIA ICD can otherwise make a nominal SwiftShader control use NVIDIA.
    for(const key of driverKeys)delete env[key];
  }else if(process.platform==='linux'&&existsSync('/dev/nvidiactl')){
    // Container images sometimes mount the driver libraries without their discovery JSON.
    // These manifests only locate existing libraries; no system drivers are installed/changed.
    env.VK_DRIVER_FILES??=env.VK_ICD_FILENAMES??fileURLToPath(new URL('../assets/gpu/nvidia-vulkan.json',import.meta.url));
    env.VK_ICD_FILENAMES??=env.VK_DRIVER_FILES;
    env.__EGL_VENDOR_LIBRARY_FILENAMES??=fileURLToPath(new URL('../assets/gpu/nvidia-egl.json',import.meta.url));
  }
  const driverManifests={};
  for(const key of driverKeys)if(env[key])driverManifests[key]=await Promise.all(env[key].split(delimiter).map(async path=>({path,sha256:createHash('sha256').update(await readFile(path)).digest('hex')})));
  const launchArgs=['--no-sandbox','--disable-dev-shm-usage',...(requested==='gpu'?['--enable-gpu','--use-angle=vulkan','--enable-features=Vulkan','--disable-vulkan-surface','--ignore-gpu-blocklist','--disable-software-rasterizer']:['--use-angle=swiftshader','--enable-unsafe-swiftshader'])];
  // Use Playwright's Chromium headless shell. The full Chromium binary is a different backend.
  const {chromium}=await import('playwright');
  const browser=await chromium.launch({headless:true,args:launchArgs,env});
  try{
    const page=await browser.newPage({viewport:{width:1280,height:720}});
    const {smokePixel,...webgl}=await page.evaluate(readCanvasRenderer,{probe:true});
    if(JSON.stringify(smokePixel)!=='[255,0,0,255]')throw Error('WebGL render/readback smoke check failed');
    const cdp=await browser.newBrowserCDPSession(),{gpu}=await cdp.send('SystemInfo.getInfo');
    const actual=validateBackend(requested,webgl,gpu.featureStatus,gpu.devices);
    const report={schemaVersion:'render-backend.v1',requested,actual,browserEngine:'chromium-headless-shell',browserVersion:browser.version(),launchArgs,driverManifests,webgl,devices:gpu.devices,featureStatus:gpu.featureStatus};
    await cdp.detach();await page.close();
    console.log('RENDER_BACKEND',actual,webgl.renderer);
    return {browser,report,startupAndProbeMs:performance.now()-started};
  }catch(error){await browser.close();throw error;}
}

export async function verifyRenderPage(page,report){
  const actual=await page.evaluate(readCanvasRenderer);
  assertSameRenderer(report.webgl,actual);
  validateBackend(report.requested,actual,report.featureStatus,report.devices);
  return actual;
}
