// Self-contained so Playwright can evaluate this in the actual canvas page.
export function readCanvasRenderer({probe=false}={}){
  const canvas=probe?document.createElement('canvas'):document.querySelector('canvas');
  if(!canvas)throw Error('No render canvas');
  if(probe){canvas.width=1280;canvas.height=720;document.body.append(canvas);}
  const gl=canvas.getContext('webgl2',{antialias:true,preserveDrawingBuffer:true,powerPreference:'high-performance'});
  if(!gl||gl.isContextLost())throw Error('WebGL2 unavailable or context lost');
  const debug=gl.getExtension('WEBGL_debug_renderer_info');
  if(!debug)throw Error('Cannot verify renderer: WEBGL_debug_renderer_info unavailable');
  let smokePixel;
  if(probe){
    gl.clearColor(1,0,0,1);gl.clear(gl.COLOR_BUFFER_BIT);
    const pixel=new Uint8Array(4);gl.readPixels(0,0,1,1,gl.RGBA,gl.UNSIGNED_BYTE,pixel);smokePixel=[...pixel];
  }
  return {vendor:gl.getParameter(debug.UNMASKED_VENDOR_WEBGL),renderer:gl.getParameter(debug.UNMASKED_RENDERER_WEBGL),version:gl.getParameter(gl.VERSION),shadingLanguageVersion:gl.getParameter(gl.SHADING_LANGUAGE_VERSION),contextAttributes:gl.getContextAttributes(),width:gl.drawingBufferWidth,height:gl.drawingBufferHeight,...(probe?{smokePixel}:{})};
}
