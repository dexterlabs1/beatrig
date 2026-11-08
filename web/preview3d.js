// beatrig 3D preview renderer: plain WebGL2, no libraries, no build step.
// It draws what app.js hands it each frame: glow sprites (lamp cores, halos, pools of light on the floor, markers),
// instanced boxes (material-driven meshes, with their transforms), instanced beam cones (spot lights) and lines
// (ground grid, camera glyph, drop lines). The scene is drawn in linear light into a multisampled half-float buffer
// (supersampled when the screen is not high-DPI), resolved, given a bloom (a soft-threshold bright pass, a chain of
// half-resolution blurs added back up), then tone-mapped (1 - e^-x, so overlapping light rolls off to white instead of
// clipping) and sRGB-encoded onto the canvas. Distance fog gives depth. Positions are Unreal centimetres used as is:
// the view matrix is built from the camera's own forward / right / up vectors (beatrig's rotBasis), so the picture is
// what the editor shows, not a mirror of it. Nothing is allocated per frame: instance data goes into Float32Arrays that
// are grown only when the scene gets bigger.
"use strict";
(function () {
  const SPRITE_FLOATS = 8;            // x, y, z, size | r, g, b, shape
  const BOX_FLOATS = 16;              // origin | x axis | y axis | z axis (each scaled) | rgb, albedo (boxes and cones)
  const LINE_FLOATS = 7;              // x, y, z, r, g, b, a
  const HALO = 0, CORE = 1, POOL = 2, MARKER = 3;   // sprite shapes (the .a of the second vec4)
  const CONE_SEGMENTS = 32;
  const BLOOM_LEVELS = 5;
  const MAX_TARGET_PX = 4.2e6;        // supersampling stops at about this many pixels in the offscreen buffer

  const FOG = `
uniform vec3 uEye; uniform float uFog;
float fogAt(vec3 p) { float d = distance(p, uEye) * uFog; return 1.0 - exp(-d * d); }`;
  const VS_SPRITE = `#version 300 es
layout(location=0) in vec2 aCorner;
layout(location=1) in vec4 iPosSize;
layout(location=2) in vec4 iColShape;
uniform mat4 uVP; uniform vec2 uProj; uniform vec2 uViewport; uniform float uScreen; uniform float uDpr;
${FOG}
out vec2 vUV; out vec3 vCol; flat out float vShape;
void main() {
  vUV = aCorner; vShape = iColShape.a;
  float size = iPosSize.w;
  if (uScreen > 0.5) {                       // dome: xy are already in clip space, size is in pixels
    vCol = iColShape.rgb;
    gl_Position = vec4(iPosSize.xy + aCorner * size * 2.0 / uViewport, 0.0, 1.0);
    return;
  }
  vCol = iColShape.rgb * (1.0 - fogAt(iPosSize.xyz));
  if (vShape > 1.5 && vShape < 2.5) {        // pool: a disc lying on the floor
    gl_Position = uVP * vec4(iPosSize.xyz + vec3(aCorner * size, 0.0), 1.0);
    return;
  }
  vec4 c = uVP * vec4(iPosSize.xyz, 1.0);
  if (vShape > 2.5) {                        // marker: a fixed size on screen
    c.xy += aCorner * size * 2.0 / uViewport * c.w;
  } else {                                   // halo / core: faces the camera, never smaller than a few pixels
    float px = size * uProj.y / max(c.w, 1e-4) * uViewport.y * 0.5;
    float k = max(px, (vShape < 0.5 ? 4.0 : 2.2) * uDpr) / max(px, 1e-6);
    c.xy += aCorner * uProj * size * k;
  }
  gl_Position = c;
}`;
  const FS_SPRITE = `#version 300 es
precision highp float;
in vec2 vUV; in vec3 vCol; flat in float vShape;
out vec4 o;
void main() {
  float r2 = dot(vUV, vUV);
  if (r2 >= 1.0) discard;
  float a;
  if (vShape < 0.5) a = (exp(-5.0 * r2) - 0.006738) / 0.993262;      // halo: a gaussian that reaches 0 at the edge
  else if (vShape < 1.5) a = smoothstep(1.0, 0.45, sqrt(r2));        // core: a soft-edged disc
  else if (vShape < 2.5) { float k = 1.0 - r2; a = k * k * k; }      // pool of light on the floor
  else a = smoothstep(1.0, 0.35, sqrt(r2));                          // marker
  o = vec4(vCol * a, 1.0);
}`;
  const VS_BOX = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;
layout(location=2) in vec3 iPos;
layout(location=3) in vec3 iX;
layout(location=4) in vec3 iY;
layout(location=5) in vec3 iZ;
layout(location=6) in vec4 iCol;
uniform mat4 uVP;
out vec3 vN; out vec4 vCol; out vec3 vW;
void main() {
  vN = iX * aNrm.x + iY * aNrm.y + iZ * aNrm.z;
  vCol = iCol;
  vW = iPos + iX * aPos.x + iY * aPos.y + iZ * aPos.z;
  gl_Position = uVP * vec4(vW, 1.0);
}`;
  const FS_BOX = `#version 300 es
precision highp float;
in vec3 vN; in vec4 vCol; in vec3 vW;
uniform vec3 uAmb; uniform vec3 uBg;
${FOG}
out vec4 o;
void main() {
  vec3 n = normalize(vN);
  float key = max(dot(n, normalize(vec3(-0.35, -0.5, 1.0))), 0.0), fill = max(dot(n, normalize(vec3(0.6, 0.4, -0.2))), 0.0);
  vec3 c = vCol.a * (0.012 + 0.07 * key + 0.02 * fill) + uAmb * 0.03 + vCol.rgb;
  o = vec4(mix(c, uBg, fogAt(vW)), 1.0);
}`;
  // beam cone: apex at the lamp, unit cone along +X (the spot's forward); aPos.x is the distance along the beam (0..1)
  const VS_CONE = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;
layout(location=2) in vec3 iPos;
layout(location=3) in vec3 iX;
layout(location=4) in vec3 iY;
layout(location=5) in vec3 iZ;
layout(location=6) in vec4 iCol;
uniform mat4 uVP;
${FOG}
out vec3 vN; out vec3 vW; out vec3 vCol; out float vT;
void main() {
  vW = iPos + iX * aPos.x + iY * aPos.y + iZ * aPos.z;
  vN = normalize(iX * aNrm.x / max(dot(iX, iX), 1e-6) + iY * aNrm.y / max(dot(iY, iY), 1e-6) + iZ * aNrm.z / max(dot(iZ, iZ), 1e-6));
  vT = aPos.x;
  vCol = iCol.rgb * (1.0 - fogAt(vW));
  gl_Position = uVP * vec4(vW, 1.0);
}`;
  const FS_CONE = `#version 300 es
precision highp float;
in vec3 vN; in vec3 vW; in vec3 vCol; in float vT;
uniform vec3 uEye;
out vec4 o;
void main() {
  vec3 v = normalize(uEye - vW);
  float face = abs(dot(normalize(vN), v));                   // seen edge-on, the beam's surface fades: a soft volume
  float along = (1.0 - vT) * (1.0 - vT) * smoothstep(0.0, 0.06, vT);
  o = vec4(vCol * along * pow(face, 1.6), 1.0);
}`;
  const VS_LINE = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aCol;
uniform mat4 uVP;
${FOG}
out vec4 vCol; out vec2 vXY;
void main() { vCol = vec4(aCol.rgb * (1.0 - fogAt(aPos)), aCol.a); vXY = aPos.xy; gl_Position = uVP * vec4(aPos, 1.0); }`;
  const FS_LINE = `#version 300 es
precision highp float;
in vec4 vCol; in vec2 vXY;
uniform vec3 uFade;        // grid centre x, y and radius: lines fade out towards the edge
uniform vec3 uAmb;         // ambient (MPC outputs): lifts the lines a little
uniform float uGain;       // thin lines lose weight in a supersampled buffer: this puts it back
out vec4 o;
void main() {
  float f = uFade.z > 0.0 ? 1.0 - smoothstep(uFade.z * 0.3, uFade.z, distance(vXY, uFade.xy)) : 1.0;
  o = vec4(vCol.rgb * (0.75 + uAmb) * vCol.a * f * uGain, 1.0);
}`;
  const VS_FULL = `#version 300 es
const vec2 P[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
out vec2 vUV;
void main() { vec2 p = P[gl_VertexID]; vUV = p * 0.5 + 0.5; gl_Position = vec4(p, 0.0, 1.0); }`;
  // bloom: 4-tap box downsample (the first pass also keeps only what is brighter than the threshold, with a soft knee)
  const FS_DOWN = `#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uSrc; uniform vec2 uTexel; uniform float uPrefilter;
out vec4 o;
void main() {
  vec3 c = (texture(uSrc, vUV + uTexel * vec2(-1.0, -1.0)).rgb + texture(uSrc, vUV + uTexel * vec2(1.0, -1.0)).rgb +
            texture(uSrc, vUV + uTexel * vec2(-1.0, 1.0)).rgb + texture(uSrc, vUV + uTexel * vec2(1.0, 1.0)).rgb) * 0.25;
  c = min(max(c, 0.0), vec3(64.0));
  if (uPrefilter > 0.5) {
    float br = max(c.r, max(c.g, c.b)), knee = 0.5, thr = 0.8;
    float soft = clamp(br - thr + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee + 1e-4);
    c *= max(soft, br - thr) / max(br, 1e-4);
  }
  o = vec4(c, 1.0);
}`;
  // 9-tap tent upsample, added onto the next larger level
  const FS_UP = `#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uSrc; uniform vec2 uTexel;
out vec4 o;
void main() {
  vec2 d = uTexel;
  vec3 c = texture(uSrc, vUV).rgb * 4.0;
  c += (texture(uSrc, vUV + vec2(d.x, 0.0)).rgb + texture(uSrc, vUV - vec2(d.x, 0.0)).rgb +
        texture(uSrc, vUV + vec2(0.0, d.y)).rgb + texture(uSrc, vUV - vec2(0.0, d.y)).rgb) * 2.0;
  c += texture(uSrc, vUV + d).rgb + texture(uSrc, vUV - d).rgb + texture(uSrc, vUV + vec2(d.x, -d.y)).rgb + texture(uSrc, vUV - vec2(d.x, -d.y)).rgb;
  o = vec4(c / 16.0, 1.0);
}`;
  const FS_COMPOSITE = `#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uTex; uniform sampler2D uBloom; uniform float uBloomK;
out vec4 o;
vec3 srgb(vec3 c) { return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c)); }
void main() {
  vec3 c = max(texture(uTex, vUV).rgb, 0.0);
  if (uBloomK > 0.0) c += texture(uBloom, vUV).rgb * uBloomK;
  o = vec4(srgb(clamp(1.0 - exp(-c), 0.0, 1.0)), 1.0);
}`;

  // unit cube, 36 vertices: position (-1..1) and face normal
  function cubeData() {
    const out = [];
    for (const [ax, s] of [[0, 1], [0, -1], [1, 1], [1, -1], [2, 1], [2, -1]]) {
      const u = (ax + 1) % 3, v = (ax + 2) % 3;
      for (const [a, b] of [[-1, -1], [1, -1], [1, 1], [-1, -1], [1, 1], [-1, 1]]) {
        const p = [0, 0, 0], n = [0, 0, 0];
        p[ax] = s; p[u] = a * s; p[v] = b; n[ax] = s;
        out.push(p[0], p[1], p[2], n[0], n[1], n[2]);
      }
    }
    return new Float32Array(out);
  }
  // open unit cone along +X: apex at the origin, a circle of radius 1 at x = 1; position + outward surface normal
  function coneData() {
    const out = [], k = Math.SQRT1_2;
    for (let i = 0; i < CONE_SEGMENTS; i++) {
      const a0 = (i / CONE_SEGMENTS) * Math.PI * 2, a1 = ((i + 1) / CONE_SEGMENTS) * Math.PI * 2, am = (a0 + a1) / 2;
      const P = (a) => [1, Math.cos(a), Math.sin(a)], N = (a) => [-k, Math.cos(a) * k, Math.sin(a) * k];
      out.push(0, 0, 0, ...N(am), ...P(a0), ...N(a0), ...P(a1), ...N(a1));
    }
    return new Float32Array(out);
  }

  // Shaders compile in the GPU process while the page goes on loading: nothing here asks for their status, which
  // would wait for them. link() does that once, at the first draw (by then the page's data has arrived).
  function compile(gl, vs, fs) {
    const mk = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; };
    const p = gl.createProgram(), v = mk(gl.VERTEX_SHADER, vs), f = mk(gl.FRAGMENT_SHADER, fs);
    gl.attachShader(p, v); gl.attachShader(p, f);
    gl.linkProgram(p);
    return { p, v, f, u: null };
  }
  function link(gl, prog) {
    if (prog.u) return;
    if (!gl.getProgramParameter(prog.p, gl.LINK_STATUS) && !gl.isContextLost()) {
      throw new Error(gl.getShaderInfoLog(prog.v) || gl.getShaderInfoLog(prog.f) || gl.getProgramInfoLog(prog.p) || "shader link failed");
    }
    const u = {}, n = gl.getProgramParameter(prog.p, gl.ACTIVE_UNIFORMS) || 0;
    for (let i = 0; i < n; i++) { const info = gl.getActiveUniform(prog.p, i); u[info.name] = gl.getUniformLocation(prog.p, info.name); }
    prog.u = u;
  }
  function instanced(gl, floats, attrBase) {    // the 16-float origin/axes/colour layout shared by boxes and cones
    for (let i = 0; i < 4; i++) { gl.enableVertexAttribArray(attrBase + i); gl.vertexAttribPointer(attrBase + i, 3, gl.FLOAT, false, floats * 4, i * 12); gl.vertexAttribDivisor(attrBase + i, 1); }
    gl.enableVertexAttribArray(attrBase + 4); gl.vertexAttribPointer(attrBase + 4, 4, gl.FLOAT, false, floats * 4, 48); gl.vertexAttribDivisor(attrBase + 4, 1);
  }
  function texRT(gl, fmt, w, h, filter) {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texStorage2D(gl.TEXTURE_2D, 1, fmt, w, h);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
    const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    return { t, fb, w, h, ok, texel: new Float32Array([1 / w, 1 / h]) };
  }

  class Renderer {
    constructor(canvas) {
      this.canvas = canvas;
      this.gl = canvas.getContext("webgl2", { antialias: false, alpha: false, depth: true, premultipliedAlpha: false, powerPreference: "high-performance" });
      if (!this.gl) return;
      this.vp = new Float32Array(16); this.proj = new Float32Array(2); this.dpr = 1; this.eye = new Float32Array(3);
      this.sprites = new Float32Array(256 * SPRITE_FLOATS); this.nSprites = 0;
      this.boxes = new Float32Array(64 * BOX_FLOATS); this.nBoxes = 0;
      this.cones = new Float32Array(16 * BOX_FLOATS); this.nCones = 0;
      this.dyn = new Float32Array(128 * LINE_FLOATS); this.nDyn = 0;     // per-frame lines (drop lines to the floor)
      this.nLines = 0; this.nGrid = 0;
      this.fade = new Float32Array(3); this.amb = new Float32Array(3);    // amb: ambient light colour x level (0..1)
      this.bg = new Float32Array([0.005, 0.0056, 0.006]); this.fogCol = new Float32Array(3);
      this.fog = 0;                                                        // fog density per cm (0 = off)
      this.bloom = 0.55;                                                   // bloom strength (0 = off)
      this.clearRGBA = new Float32Array(4); this.clearZ = new Float32Array([1]);
      this.u1 = new Float32Array(3); this.gain1 = new Float32Array(1); this.zero1 = new Float32Array(1);                                       // fog, dpr, bloom as uniform vectors (no boxed floats)
      this.screen = false;
      this.w = 1; this.h = 1; this.ss = 1;
      this.lost = false;
      canvas.addEventListener("webglcontextlost", (e) => { e.preventDefault(); this.lost = true; });
      canvas.addEventListener("webglcontextrestored", () => { this.lost = false; this.init(); if (this.onrestore) this.onrestore(); });
      this.init();
    }
    init() {
      const gl = this.gl;
      this.linked = false;
      this.pSprite = compile(gl, VS_SPRITE, FS_SPRITE);
      this.pBox = compile(gl, VS_BOX, FS_BOX);
      this.pCone = compile(gl, VS_CONE, FS_CONE);
      this.pLine = compile(gl, VS_LINE, FS_LINE);
      this.pDown = compile(gl, VS_FULL, FS_DOWN);
      this.pUp = compile(gl, VS_FULL, FS_UP);
      this.pComp = compile(gl, VS_FULL, FS_COMPOSITE);
      // sprites: a quad (triangle strip) + per-instance data
      this.vaoSprite = gl.createVertexArray(); gl.bindVertexArray(this.vaoSprite);
      const quad = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, quad);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      this.bufSprite = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.bufSprite);
      gl.bufferData(gl.ARRAY_BUFFER, this.sprites.byteLength, gl.DYNAMIC_DRAW); this.capSprite = this.sprites.length;
      for (let i = 0; i < 2; i++) {
        gl.enableVertexAttribArray(1 + i); gl.vertexAttribPointer(1 + i, 4, gl.FLOAT, false, SPRITE_FLOATS * 4, i * 16); gl.vertexAttribDivisor(1 + i, 1);
      }
      // boxes and cones: a mesh + per-instance origin, axes and colour
      const mesh = (data) => {
        const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
        const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b);
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
        gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
        const inst = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, inst);
        instanced(gl, BOX_FLOATS, 2);
        return { vao, inst, cap: 0 };
      };
      this.mBox = mesh(cubeData());
      this.mCone = mesh(coneData());
      // lines: static (setLines) and per frame
      const lines = () => {
        const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
        const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b);
        gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, LINE_FLOATS * 4, 0);
        gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.FLOAT, false, LINE_FLOATS * 4, 12);
        return { vao, b, cap: 0 };
      };
      this.lStatic = lines();
      if (this.lineData) gl.bufferData(gl.ARRAY_BUFFER, this.lineData, gl.STATIC_DRAW);
      this.lDyn = lines();
      this.vaoFull = gl.createVertexArray();
      gl.bindVertexArray(null);
      this.hdr = !!gl.getExtension("EXT_color_buffer_float");
      this.samples = Math.min(4, gl.getParameter(gl.MAX_SAMPLES) || 0);
      this.maxRB = gl.getParameter(gl.MAX_RENDERBUFFER_SIZE) || 4096;
      this.fbW = 0; this.fbH = 0; this.fbo = null;
    }
    // Offscreen targets, rebuilt when the size changes: the multisampled scene (colour + depth), its resolve texture and
    // the bloom chain (half resolution and down), all half-float when the browser can render to it.
    targets(w, h) {
      if (this.fbo && this.fbW === w && this.fbH === h) return !!this.fbo.ok;
      const gl = this.gl, old = this.fbo;
      if (old) {
        gl.deleteFramebuffer(old.ms); gl.deleteRenderbuffer(old.col); gl.deleteRenderbuffer(old.dep);
        for (const r of [old.rs, ...old.bloom]) { gl.deleteFramebuffer(r.fb); gl.deleteTexture(r.t); }
      }
      const fmt = this.hdr ? gl.RGBA16F : gl.RGBA8;
      const f = { ms: gl.createFramebuffer(), col: gl.createRenderbuffer(), dep: gl.createRenderbuffer(), rs: null, bloom: [], ok: false };
      gl.bindRenderbuffer(gl.RENDERBUFFER, f.col); gl.renderbufferStorageMultisample(gl.RENDERBUFFER, this.samples, fmt, w, h);
      gl.bindRenderbuffer(gl.RENDERBUFFER, f.dep); gl.renderbufferStorageMultisample(gl.RENDERBUFFER, this.samples, gl.DEPTH_COMPONENT24, w, h);
      gl.bindFramebuffer(gl.FRAMEBUFFER, f.ms);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, f.col);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, f.dep);
      const okMs = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
      f.rs = texRT(gl, fmt, w, h, gl.LINEAR);
      if (this.hdr) {
        let bw = w, bh = h;
        for (let i = 0; i < BLOOM_LEVELS; i++) {
          bw = Math.max(1, bw >> 1); bh = Math.max(1, bh >> 1);
          const r = texRT(gl, fmt, bw, bh, gl.LINEAR);
          if (!r.ok) break;
          f.bloom.push(r);
          if (bw <= 8 || bh <= 8) break;
        }
      }
      f.ok = okMs && f.rs.ok;
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      this.fbo = f; this.fbW = w; this.fbH = h;
      return f.ok;
    }
    // capacity: grown (rarely, at scene load), never shrunk
    reserve(nSprites, nBoxes, nDyn, nCones) {
      if (nSprites * SPRITE_FLOATS > this.sprites.length) this.sprites = new Float32Array(Math.ceil(nSprites * 1.5) * SPRITE_FLOATS);
      if (nBoxes * BOX_FLOATS > this.boxes.length) this.boxes = new Float32Array(Math.ceil(nBoxes * 1.5) * BOX_FLOATS);
      if ((nCones || 0) * BOX_FLOATS > this.cones.length) this.cones = new Float32Array(Math.ceil(nCones * 1.5) * BOX_FLOATS);
      if (nDyn * 2 * LINE_FLOATS > this.dyn.length) this.dyn = new Float32Array(Math.ceil(nDyn * 1.5) * 2 * LINE_FLOATS);
    }
    sprite(x, y, z, size, r, g, b, shape) {
      const d = this.sprites, i = this.nSprites++ * SPRITE_FLOATS;
      d[i] = x; d[i + 1] = y; d[i + 2] = z; d[i + 3] = size; d[i + 4] = r; d[i + 5] = g; d[i + 6] = b; d[i + 7] = shape;
    }
    // F, R, U: the box's unit axes (UE forward, right, up); sx, sy, sz: half extents along them
    box(x, y, z, F, R, U, sx, sy, sz, r, g, b, albedo) { this.nBoxes = put(this.boxes, this.nBoxes, x, y, z, F, R, U, sx, sy, sz, r, g, b, albedo); }
    // a beam from (x, y, z) along F, length len, end radius rad, colour (linear, additive)
    cone(x, y, z, F, R, U, len, rad, r, g, b) { this.nCones = put(this.cones, this.nCones, x, y, z, F, R, U, len, rad, rad, r, g, b, 0); }
    // a per-frame line segment, grey level c
    line(x0, y0, z0, x1, y1, z1, c) {
      const d = this.dyn, i = this.nDyn * LINE_FLOATS;
      this.nDyn += 2;
      d[i] = x0; d[i + 1] = y0; d[i + 2] = z0; d[i + 3] = c; d[i + 4] = c; d[i + 5] = c; d[i + 6] = 1;
      d[i + 7] = x1; d[i + 8] = y1; d[i + 9] = z1; d[i + 10] = c; d[i + 11] = c; d[i + 12] = c; d[i + 13] = 1;
    }
    // static lines: Float32Array of [x, y, z, r, g, b, a] per vertex, pairs make segments (built once per scene)
    setLines(data) {
      this.lineData = data; this.nLines = data ? data.length / LINE_FLOATS : 0;
      const gl = this.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.lStatic.b);
      gl.bufferData(gl.ARRAY_BUFFER, data || new Float32Array(0), gl.STATIC_DRAW);
    }
    // perspective camera from an eye point and unit axes (UE forward / right / up), vertical fov in radians
    camera(eye, F, R, U, vfov, aspect, near, far) {
      const f = 1 / Math.tan(vfov / 2), A = (far + near) / (far - near), B = -2 * far * near / (far - near), m = this.vp;
      const tr = -(R[0] * eye[0] + R[1] * eye[1] + R[2] * eye[2]);
      const tu = -(U[0] * eye[0] + U[1] * eye[1] + U[2] * eye[2]);
      const tf = -(F[0] * eye[0] + F[1] * eye[1] + F[2] * eye[2]);
      const fx = f / aspect;
      this.proj[0] = fx; this.proj[1] = f;
      this.eye[0] = eye[0]; this.eye[1] = eye[1]; this.eye[2] = eye[2];
      // column-major: m[col * 4 + row]
      m[0] = fx * R[0]; m[4] = fx * R[1]; m[8] = fx * R[2]; m[12] = fx * tr;
      m[1] = f * U[0]; m[5] = f * U[1]; m[9] = f * U[2]; m[13] = f * tu;
      m[2] = A * F[0]; m[6] = A * F[1]; m[10] = A * F[2]; m[14] = A * tf + B;
      m[3] = F[0]; m[7] = F[1]; m[11] = F[2]; m[15] = tf;
      this.screen = false;
    }
    // world point -> canvas CSS pixels (out[0], out[1]); false when it is behind the camera
    project(x, y, z, out) {
      const m = this.vp, w = m[3] * x + m[7] * y + m[11] * z + m[15];
      if (this.screen || w <= 1e-6) return false;
      out[0] = ((m[0] * x + m[4] * y + m[8] * z + m[12]) / w * 0.5 + 0.5) * this.cssW;
      out[1] = (0.5 - (m[1] * x + m[5] * y + m[9] * z + m[13]) / w * 0.5) * this.cssH;
      return true;
    }
    // canvas size; the offscreen buffer is up to 2x that (supersampling) when the screen isn't high-DPI
    resize(cssW, cssH, dpr) {
      const w = Math.max(1, Math.round(cssW * dpr)), h = Math.max(1, Math.round(cssH * dpr));
      if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
      this.w = w; this.h = h; this.cssW = cssW; this.cssH = cssH; this.dpr = dpr;
      let ss = Math.max(1, Math.min(2, 2 / dpr, Math.sqrt(MAX_TARGET_PX / (w * h))));
      ss = Math.min(ss, this.maxRB / w, this.maxRB / h);
      this.ss = Math.max(1, ss);
    }
    // Draw everything set up for this frame. lines: 0 none, 1 the first nGrid line vertices (the ground grid), 2 all of
    // them (grid + camera glyph); perspective modes only.
    render(lines) {
      const gl = this.gl;
      if (this.lost || this.broken || gl.isContextLost()) return false;
      if (!this.linked) {
        try { for (const p of [this.pSprite, this.pBox, this.pCone, this.pLine, this.pDown, this.pUp, this.pComp]) link(gl, p); this.linked = true; }
        catch (e) { console.warn("beatrig: WebGL2 preview off:", e); this.broken = true; return false; }
      }
      const w = this.w, h = this.h, tw = Math.round(w * this.ss), th = Math.round(h * this.ss), off = this.targets(tw, th);
      gl.bindFramebuffer(gl.FRAMEBUFFER, off ? this.fbo.ms : null);
      gl.viewport(0, 0, off ? tw : w, off ? th : h);
      const bg = this.bg, a = this.amb, cc = this.clearRGBA, fc = this.fogCol, U1 = this.u1;
      U1[0] = this.screen ? 0 : this.fog; U1[1] = this.dpr; U1[2] = this.bloom;
      const fog = U1.subarray(0, 1);
      cc[0] = bg[0] + a[0] * 0.006; cc[1] = bg[1] + a[1] * 0.006; cc[2] = bg[2] + a[2] * 0.006; cc[3] = 1;
      fc[0] = cc[0]; fc[1] = cc[1]; fc[2] = cc[2];
      gl.depthMask(true);
      gl.clearBufferfv(gl.COLOR, 0, cc);                 // typed-array clears: no boxed float arguments per frame
      gl.clearBufferfv(gl.DEPTH, 0, this.clearZ);
      gl.disable(gl.CULL_FACE);
      // boxes: opaque, depth-tested and written
      if (this.nBoxes && !this.screen) {
        gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.disable(gl.BLEND);
        const u = this.pBox.u;
        gl.useProgram(this.pBox.p);
        gl.uniformMatrix4fv(u.uVP, false, this.vp);
        gl.uniform3fv(u.uAmb, this.amb); gl.uniform3fv(u.uBg, fc); gl.uniform3fv(u.uEye, this.eye); gl.uniform1fv(u.uFog, fog);
        this.drawMesh(this.mBox, this.boxes, this.nBoxes, 36);
      }
      // everything else adds light: depth-tested against the boxes, never written
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE); gl.blendEquation(gl.FUNC_ADD);
      gl.depthMask(false);
      if (this.screen) gl.disable(gl.DEPTH_TEST); else gl.enable(gl.DEPTH_TEST);
      const lineGain = this.ss > 1.5 ? 2 : 1;
      if (lines && this.nLines && !this.screen) {
        this.lineProgram(this.fade, fog, lineGain);
        gl.bindVertexArray(this.lStatic.vao);
        gl.drawArrays(gl.LINES, 0, lines >= 2 ? this.nLines : Math.min(this.nGrid, this.nLines));
      }
      if (this.nDyn && !this.screen) {
        this.lineProgram(null, fog, lineGain);
        gl.bindVertexArray(this.lDyn.vao);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.lDyn.b);
        if (this.dyn.length > this.lDyn.cap) { gl.bufferData(gl.ARRAY_BUFFER, this.dyn.byteLength, gl.DYNAMIC_DRAW); this.lDyn.cap = this.dyn.length; }
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.dyn, 0, this.nDyn * LINE_FLOATS);
        gl.drawArrays(gl.LINES, 0, this.nDyn);
      }
      if (this.nCones && !this.screen) {
        const u = this.pCone.u;
        gl.useProgram(this.pCone.p);
        gl.uniformMatrix4fv(u.uVP, false, this.vp);
        gl.uniform3fv(u.uEye, this.eye); gl.uniform1fv(u.uFog, fog);
        this.drawMesh(this.mCone, this.cones, this.nCones, CONE_SEGMENTS * 3);
      }
      if (this.nSprites) {
        const u = this.pSprite.u;
        gl.useProgram(this.pSprite.p);
        gl.uniformMatrix4fv(u.uVP, false, this.vp);
        gl.uniform2fv(u.uProj, this.proj); gl.uniform1fv(u.uDpr, U1.subarray(1, 2));
        gl.uniform2f(u.uViewport, w, h);                // sprite sizes are in canvas pixels, whatever the buffer size
        gl.uniform1f(u.uScreen, this.screen ? 1 : 0);
        gl.uniform3fv(u.uEye, this.eye); gl.uniform1fv(u.uFog, fog);
        gl.bindVertexArray(this.vaoSprite);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.bufSprite);
        if (this.sprites.length > this.capSprite) { gl.bufferData(gl.ARRAY_BUFFER, this.sprites.byteLength, gl.DYNAMIC_DRAW); this.capSprite = this.sprites.length; }
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.sprites, 0, this.nSprites * SPRITE_FLOATS);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.nSprites);
      }
      gl.depthMask(true);
      gl.disable(gl.BLEND); gl.disable(gl.DEPTH_TEST);
      if (off) {
        const F = this.fbo;
        // resolve the samples
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, F.ms);
        gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, F.rs.fb);
        gl.blitFramebuffer(0, 0, tw, th, 0, 0, tw, th, gl.COLOR_BUFFER_BIT, gl.NEAREST);
        gl.bindVertexArray(this.vaoFull);
        gl.activeTexture(gl.TEXTURE0);
        // bloom: bright pass + downsample chain, then tent upsamples added back up the chain
        const B = F.bloom, bloomOn = this.bloom > 0 && B.length > 1;
        if (bloomOn) {
          gl.useProgram(this.pDown.p);
          const du = this.pDown.u;
          gl.uniform1i(du.uSrc, 0);
          let src = F.rs;
          for (let i = 0; i < B.length; i++) {
            gl.bindFramebuffer(gl.FRAMEBUFFER, B[i].fb); gl.viewport(0, 0, B[i].w, B[i].h);
            gl.bindTexture(gl.TEXTURE_2D, src.t);
            gl.uniform2fv(du.uTexel, src.texel);
            gl.uniform1f(du.uPrefilter, i === 0 ? 1 : 0);
            gl.drawArrays(gl.TRIANGLES, 0, 3);
            src = B[i];
          }
          gl.useProgram(this.pUp.p);
          const uu = this.pUp.u;
          gl.uniform1i(uu.uSrc, 0);
          gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
          for (let i = B.length - 1; i > 0; i--) {
            gl.bindFramebuffer(gl.FRAMEBUFFER, B[i - 1].fb); gl.viewport(0, 0, B[i - 1].w, B[i - 1].h);
            gl.bindTexture(gl.TEXTURE_2D, B[i].t);
            gl.uniform2fv(uu.uTexel, B[i].texel);
            gl.drawArrays(gl.TRIANGLES, 0, 3);
          }
          gl.disable(gl.BLEND);
        }
        // tone-map onto the canvas (the linear filter also takes the supersampled buffer down to the canvas size)
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, w, h);
        const cu = this.pComp.u;
        gl.useProgram(this.pComp.p);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, F.rs.t); gl.uniform1i(cu.uTex, 0);
        gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, bloomOn ? B[0].t : F.rs.t); gl.uniform1i(cu.uBloom, 1);
        gl.uniform1fv(cu.uBloomK, bloomOn ? U1.subarray(2, 3) : this.zero1);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.activeTexture(gl.TEXTURE0);
      }
      gl.bindVertexArray(null);
      return true;
    }
    drawMesh(m, data, n, verts) {
      const gl = this.gl;
      gl.bindVertexArray(m.vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, m.inst);
      if (data.length > m.cap) { gl.bufferData(gl.ARRAY_BUFFER, data.byteLength, gl.DYNAMIC_DRAW); m.cap = data.length; }
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, data, 0, n * BOX_FLOATS);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, verts, n);
    }
    lineProgram(fade, fog, gain) {
      const gl = this.gl, u = this.pLine.u;
      gl.useProgram(this.pLine.p);
      gl.uniformMatrix4fv(u.uVP, false, this.vp);
      if (fade) gl.uniform3fv(u.uFade, fade); else gl.uniform3f(u.uFade, 0, 0, 0);
      this.gain1[0] = gain;
      gl.uniform3fv(u.uAmb, this.amb); gl.uniform3fv(u.uEye, this.eye); gl.uniform1fv(u.uFog, fog); gl.uniform1fv(u.uGain, this.gain1);
    }
  }
  // one instance in the origin / axes / colour layout: axes F, R, U scaled by sx, sy, sz
  function put(d, n, x, y, z, F, R, U, sx, sy, sz, r, g, b, a) {
    const i = n * BOX_FLOATS;
    d[i] = x; d[i + 1] = y; d[i + 2] = z;
    d[i + 3] = F[0] * sx; d[i + 4] = F[1] * sx; d[i + 5] = F[2] * sx;
    d[i + 6] = R[0] * sy; d[i + 7] = R[1] * sy; d[i + 8] = R[2] * sy;
    d[i + 9] = U[0] * sz; d[i + 10] = U[1] * sz; d[i + 11] = U[2] * sz;
    d[i + 12] = r; d[i + 13] = g; d[i + 14] = b; d[i + 15] = a;
    return n + 1;
  }

  window.BeatrigGL = {
    HALO, CORE, POOL, MARKER, LINE_FLOATS,
    // a renderer for this canvas, or null when the browser has no WebGL2 (the page then keeps its 2D views)
    create(canvas) {
      try { const r = new Renderer(canvas); return r.gl ? r : null; } catch (e) { console.warn("beatrig: WebGL2 preview off:", e); return null; }
    },
  };
})();
