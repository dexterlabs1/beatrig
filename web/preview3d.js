// beatrig 3D preview renderer: plain WebGL2, no libraries, no build step.
// It draws what app.js hands it each frame: glow sprites (lights, halos, light pools on the floor, markers), instanced
// boxes (material-driven meshes, with their transforms) and static lines (ground grid, camera glyph). The scene is
// drawn in linear light into a multisampled half-float buffer, then tone-mapped (1 - e^-x, so overlapping glows roll
// off to white instead of clipping) and sRGB-encoded onto the canvas. Positions are Unreal centimetres used as is:
// the view matrix is built from the camera's own forward / right / up vectors (beatrig's rotBasis), so the picture is
// what the editor shows, not a mirror of it. Nothing is allocated per frame: instance data goes into Float32Arrays that
// are grown only when the scene gets bigger.
"use strict";
(function () {
  const SPRITE_FLOATS = 8;            // x, y, z, size | r, g, b, shape
  const BOX_FLOATS = 16;              // centre | x axis | y axis | z axis (each scaled by the half extent) | rgb emissive, albedo
  const LINE_FLOATS = 7;              // x, y, z, r, g, b, a
  // sprite shapes (the .a of the second vec4)
  const HALO = 0, CORE = 1, POOL = 2, MARKER = 3;

  const VS_SPRITE = `#version 300 es
layout(location=0) in vec2 aCorner;
layout(location=1) in vec4 iPosSize;
layout(location=2) in vec4 iColShape;
uniform mat4 uVP; uniform vec2 uProj; uniform vec2 uViewport; uniform float uScreen; uniform float uDpr;
out vec2 vUV; out vec3 vCol; flat out float vShape;
void main() {
  vUV = aCorner; vCol = iColShape.rgb; vShape = iColShape.a;
  float size = iPosSize.w;
  if (uScreen > 0.5) {                       // dome: xy are already in clip space, size is in pixels
    gl_Position = vec4(iPosSize.xy + aCorner * size * 2.0 / uViewport, 0.0, 1.0);
    return;
  }
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
out vec3 vN; out vec4 vCol;
void main() {
  vN = iX * aNrm.x + iY * aNrm.y + iZ * aNrm.z;
  vCol = iCol;
  gl_Position = uVP * vec4(iPos + iX * aPos.x + iY * aPos.y + iZ * aPos.z, 1.0);
}`;
  const FS_BOX = `#version 300 es
precision highp float;
in vec3 vN; in vec4 vCol;
uniform vec3 uAmb;
out vec4 o;
void main() {
  vec3 n = normalize(vN);
  float key = max(dot(n, normalize(vec3(-0.35, -0.5, 1.0))), 0.0), fill = max(dot(n, normalize(vec3(0.6, 0.4, -0.2))), 0.0);
  o = vec4(vCol.a * (0.012 + 0.07 * key + 0.02 * fill) + uAmb * 0.03 + vCol.rgb, 1.0);
}`;
  const VS_LINE = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aCol;
uniform mat4 uVP;
out vec4 vCol; out vec2 vXY;
void main() { vCol = aCol; vXY = aPos.xy; gl_Position = uVP * vec4(aPos, 1.0); }`;
  const FS_LINE = `#version 300 es
precision highp float;
in vec4 vCol; in vec2 vXY;
uniform vec3 uFade;        // grid centre x, y and radius: lines fade out towards the edge
uniform vec3 uAmb;         // ambient (MPC outputs): lifts the lines a little
out vec4 o;
void main() {
  float f = uFade.z > 0.0 ? 1.0 - smoothstep(uFade.z * 0.3, uFade.z, distance(vXY, uFade.xy)) : 1.0;
  o = vec4(vCol.rgb * (0.75 + uAmb) * vCol.a * f, 1.0);
}`;
  const VS_FULL = `#version 300 es
const vec2 P[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
out vec2 vUV;
void main() { vec2 p = P[gl_VertexID]; vUV = p * 0.5 + 0.5; gl_Position = vec4(p, 0.0, 1.0); }`;
  const FS_COMPOSITE = `#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uTex;
out vec4 o;
vec3 srgb(vec3 c) { return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c)); }
void main() {
  vec3 c = max(texture(uTex, vUV).rgb, 0.0);
  o = vec4(srgb(clamp(1.0 - exp(-c), 0.0, 1.0)), 1.0);
}`;

  // unit cube, 36 vertices: position (-1..1) and face normal
  function cubeData() {
    const out = [];
    const faces = [[0, 1], [0, -1], [1, 1], [1, -1], [2, 1], [2, -1]];
    for (const [ax, s] of faces) {
      const u = (ax + 1) % 3, v = (ax + 2) % 3;
      const c = [[-1, -1], [1, -1], [1, 1], [-1, -1], [1, 1], [-1, 1]];
      for (const [a, b] of c) {
        const p = [0, 0, 0], n = [0, 0, 0];
        p[ax] = s; p[u] = a * s; p[v] = b; n[ax] = s;
        out.push(p[0], p[1], p[2], n[0], n[1], n[2]);
      }
    }
    return new Float32Array(out);
  }

  function compile(gl, vs, fs) {
    const mk = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS) && !gl.isContextLost()) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, mk(gl.VERTEX_SHADER, vs)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS) && !gl.isContextLost()) throw new Error(gl.getProgramInfoLog(p));
    const u = {};
    const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS) || 0;
    for (let i = 0; i < n; i++) { const info = gl.getActiveUniform(p, i); u[info.name] = gl.getUniformLocation(p, info.name); }
    return { p, u };
  }

  class Renderer {
    constructor(canvas) {
      this.canvas = canvas;
      this.gl = canvas.getContext("webgl2", { antialias: false, alpha: false, depth: true, premultipliedAlpha: false, powerPreference: "low-power" });
      if (!this.gl) return;
      this.vp = new Float32Array(16); this.proj = new Float32Array(2); this.dpr = 1;
      this.sprites = new Float32Array(256 * SPRITE_FLOATS); this.nSprites = 0;
      this.boxes = new Float32Array(64 * BOX_FLOATS); this.nBoxes = 0;
      this.dyn = new Float32Array(128 * LINE_FLOATS); this.nDyn = 0;     // per-frame lines (drop lines to the floor)
      this.nLines = 0; this.nGrid = 0;
      this.fade = new Float32Array(3); this.amb = new Float32Array(3);    // amb: ambient light colour x level (0..1)
      this.bg = new Float32Array([0.005, 0.0056, 0.006]);
      this.screen = false;
      this.w = 1; this.h = 1;
      this.lost = false;
      canvas.addEventListener("webglcontextlost", (e) => { e.preventDefault(); this.lost = true; });
      canvas.addEventListener("webglcontextrestored", () => { this.lost = false; this.init(); if (this.onrestore) this.onrestore(); });
      this.init();
    }
    init() {
      const gl = this.gl;
      this.pSprite = compile(gl, VS_SPRITE, FS_SPRITE);
      this.pBox = compile(gl, VS_BOX, FS_BOX);
      this.pLine = compile(gl, VS_LINE, FS_LINE);
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
      // boxes: a cube + per-instance data
      this.vaoBox = gl.createVertexArray(); gl.bindVertexArray(this.vaoBox);
      const cube = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, cube);
      gl.bufferData(gl.ARRAY_BUFFER, cubeData(), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
      gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
      this.bufBox = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.bufBox);
      gl.bufferData(gl.ARRAY_BUFFER, this.boxes.byteLength, gl.DYNAMIC_DRAW); this.capBox = this.boxes.length;
      for (let i = 0; i < 4; i++) {
        gl.enableVertexAttribArray(2 + i); gl.vertexAttribPointer(2 + i, 3, gl.FLOAT, false, BOX_FLOATS * 4, i * 12); gl.vertexAttribDivisor(2 + i, 1);
      }
      gl.enableVertexAttribArray(6); gl.vertexAttribPointer(6, 4, gl.FLOAT, false, BOX_FLOATS * 4, 48); gl.vertexAttribDivisor(6, 1);
      // lines: static, replaced by setLines()
      this.vaoLine = gl.createVertexArray(); gl.bindVertexArray(this.vaoLine);
      this.bufLine = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.bufLine);
      gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, LINE_FLOATS * 4, 0);
      gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.FLOAT, false, LINE_FLOATS * 4, 12);
      if (this.lineData) gl.bufferData(gl.ARRAY_BUFFER, this.lineData, gl.STATIC_DRAW);
      this.vaoDyn = gl.createVertexArray(); gl.bindVertexArray(this.vaoDyn);
      this.bufDyn = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.bufDyn);
      gl.bufferData(gl.ARRAY_BUFFER, this.dyn.byteLength, gl.DYNAMIC_DRAW); this.capDyn = this.dyn.length;
      gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, LINE_FLOATS * 4, 0);
      gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.FLOAT, false, LINE_FLOATS * 4, 12);
      this.vaoFull = gl.createVertexArray();
      gl.bindVertexArray(null);
      this.hdr = !!gl.getExtension("EXT_color_buffer_float");
      this.samples = Math.min(4, gl.getParameter(gl.MAX_SAMPLES) || 0);
      this.fbW = 0; this.fbH = 0; this.fbo = null;
    }
    // The offscreen targets follow the canvas size: a multisampled colour + depth buffer, resolved into a texture.
    targets(w, h) {
      if (this.fbo && this.fbW === w && this.fbH === h) return !!this.fbo.ok;
      const gl = this.gl;
      if (this.fbo) {
        gl.deleteFramebuffer(this.fbo.ms); gl.deleteFramebuffer(this.fbo.rs);
        gl.deleteRenderbuffer(this.fbo.col); gl.deleteRenderbuffer(this.fbo.dep); gl.deleteTexture(this.fbo.tex);
      }
      const fmt = this.hdr ? gl.RGBA16F : gl.RGBA8;
      const f = { ms: gl.createFramebuffer(), rs: gl.createFramebuffer(), col: gl.createRenderbuffer(), dep: gl.createRenderbuffer(), tex: gl.createTexture(), ok: false };
      gl.bindRenderbuffer(gl.RENDERBUFFER, f.col); gl.renderbufferStorageMultisample(gl.RENDERBUFFER, this.samples, fmt, w, h);
      gl.bindRenderbuffer(gl.RENDERBUFFER, f.dep); gl.renderbufferStorageMultisample(gl.RENDERBUFFER, this.samples, gl.DEPTH_COMPONENT24, w, h);
      gl.bindFramebuffer(gl.FRAMEBUFFER, f.ms);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, f.col);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, f.dep);
      const okMs = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
      gl.bindTexture(gl.TEXTURE_2D, f.tex);
      gl.texStorage2D(gl.TEXTURE_2D, 1, fmt, w, h);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.bindFramebuffer(gl.FRAMEBUFFER, f.rs);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, f.tex, 0);
      f.ok = okMs && gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      this.fbo = f; this.fbW = w; this.fbH = h;
      return f.ok;
    }
    // capacity helpers: grow (rarely, at scene load), never shrink
    reserve(nSprites, nBoxes, nDyn) {
      if (nSprites * SPRITE_FLOATS > this.sprites.length) this.sprites = new Float32Array(Math.ceil(nSprites * 1.5) * SPRITE_FLOATS);
      if (nBoxes * BOX_FLOATS > this.boxes.length) this.boxes = new Float32Array(Math.ceil(nBoxes * 1.5) * BOX_FLOATS);
      if (nDyn * 2 * LINE_FLOATS > this.dyn.length) this.dyn = new Float32Array(Math.ceil(nDyn * 1.5) * 2 * LINE_FLOATS);
    }
    // a per-frame line segment, grey level c
    line(x0, y0, z0, x1, y1, z1, c) {
      const d = this.dyn, i = this.nDyn * LINE_FLOATS;
      this.nDyn += 2;
      d[i] = x0; d[i + 1] = y0; d[i + 2] = z0; d[i + 3] = c; d[i + 4] = c; d[i + 5] = c; d[i + 6] = 1;
      d[i + 7] = x1; d[i + 8] = y1; d[i + 9] = z1; d[i + 10] = c; d[i + 11] = c; d[i + 12] = c; d[i + 13] = 1;
    }
    sprite(x, y, z, size, r, g, b, shape) {
      const d = this.sprites, i = this.nSprites++ * SPRITE_FLOATS;
      d[i] = x; d[i + 1] = y; d[i + 2] = z; d[i + 3] = size; d[i + 4] = r; d[i + 5] = g; d[i + 6] = b; d[i + 7] = shape;
    }
    // F, R, U: the box's unit axes (UE forward, right, up); sx, sy, sz: half extents along them
    box(x, y, z, F, R, U, sx, sy, sz, r, g, b, albedo) {
      const d = this.boxes, i = this.nBoxes++ * BOX_FLOATS;
      d[i] = x; d[i + 1] = y; d[i + 2] = z;
      d[i + 3] = F[0] * sx; d[i + 4] = F[1] * sx; d[i + 5] = F[2] * sx;
      d[i + 6] = R[0] * sy; d[i + 7] = R[1] * sy; d[i + 8] = R[2] * sy;
      d[i + 9] = U[0] * sz; d[i + 10] = U[1] * sz; d[i + 11] = U[2] * sz;
      d[i + 12] = r; d[i + 13] = g; d[i + 14] = b; d[i + 15] = albedo;
    }
    // static lines: Float32Array of [x, y, z, r, g, b, a] per vertex, pairs make segments (built once per scene)
    setLines(data) {
      this.lineData = data; this.nLines = data ? data.length / LINE_FLOATS : 0;
      const gl = this.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.bufLine);
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
    resize(cssW, cssH, dpr) {
      const w = Math.max(1, Math.round(cssW * dpr)), h = Math.max(1, Math.round(cssH * dpr));
      if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
      this.w = w; this.h = h; this.cssW = cssW; this.cssH = cssH; this.dpr = dpr;
    }
    // Draw everything set up for this frame. lines: 0 none, 1 the first nGrid line vertices (the ground grid), 2 all of
    // them (grid + camera glyph); perspective modes only.
    render(lines) {
      const gl = this.gl;
      if (this.lost || gl.isContextLost()) return false;
      const w = this.w, h = this.h, off = this.targets(w, h);
      gl.bindFramebuffer(gl.FRAMEBUFFER, off ? this.fbo.ms : null);
      gl.viewport(0, 0, w, h);
      const bg = this.bg, a = this.amb;
      gl.clearColor(bg[0] + a[0] * 0.006, bg[1] + a[1] * 0.006, bg[2] + a[2] * 0.006, 1);
      gl.clearDepth(1);
      gl.depthMask(true);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.disable(gl.CULL_FACE);
      // boxes: opaque, depth-tested and written
      if (this.nBoxes && !this.screen) {
        gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.disable(gl.BLEND);
        gl.useProgram(this.pBox.p);
        gl.uniformMatrix4fv(this.pBox.u.uVP, false, this.vp);
        gl.uniform3fv(this.pBox.u.uAmb, this.amb);
        gl.bindVertexArray(this.vaoBox);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.bufBox);
        if (this.boxes.length > this.capBox) { gl.bufferData(gl.ARRAY_BUFFER, this.boxes.byteLength, gl.DYNAMIC_DRAW); this.capBox = this.boxes.length; }
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.boxes, 0, this.nBoxes * BOX_FLOATS);
        gl.drawArraysInstanced(gl.TRIANGLES, 0, 36, this.nBoxes);
      }
      // everything else adds light: depth-tested against the boxes, never written
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE); gl.blendEquation(gl.FUNC_ADD);
      gl.depthMask(false);
      if (this.screen) gl.disable(gl.DEPTH_TEST); else gl.enable(gl.DEPTH_TEST);
      if (lines && this.nLines && !this.screen) {
        gl.useProgram(this.pLine.p);
        gl.uniformMatrix4fv(this.pLine.u.uVP, false, this.vp);
        gl.uniform3fv(this.pLine.u.uFade, this.fade);
        gl.uniform3fv(this.pLine.u.uAmb, this.amb);
        gl.bindVertexArray(this.vaoLine);
        gl.drawArrays(gl.LINES, 0, lines >= 2 ? this.nLines : Math.min(this.nGrid, this.nLines));
      }
      if (this.nDyn && !this.screen) {
        gl.useProgram(this.pLine.p);
        gl.uniformMatrix4fv(this.pLine.u.uVP, false, this.vp);
        gl.uniform3f(this.pLine.u.uFade, 0, 0, 0);
        gl.uniform3fv(this.pLine.u.uAmb, this.amb);
        gl.bindVertexArray(this.vaoDyn);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.bufDyn);
        if (this.dyn.length > this.capDyn) { gl.bufferData(gl.ARRAY_BUFFER, this.dyn.byteLength, gl.DYNAMIC_DRAW); this.capDyn = this.dyn.length; }
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.dyn, 0, this.nDyn * LINE_FLOATS);
        gl.drawArrays(gl.LINES, 0, this.nDyn);
      }
      if (this.nSprites) {
        gl.useProgram(this.pSprite.p);
        gl.uniformMatrix4fv(this.pSprite.u.uVP, false, this.vp);
        gl.uniform2fv(this.pSprite.u.uProj, this.proj); gl.uniform1f(this.pSprite.u.uDpr, this.dpr);
        gl.uniform2f(this.pSprite.u.uViewport, w, h);
        gl.uniform1f(this.pSprite.u.uScreen, this.screen ? 1 : 0);
        gl.bindVertexArray(this.vaoSprite);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.bufSprite);
        if (this.sprites.length > this.capSprite) { gl.bufferData(gl.ARRAY_BUFFER, this.sprites.byteLength, gl.DYNAMIC_DRAW); this.capSprite = this.sprites.length; }
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.sprites, 0, this.nSprites * SPRITE_FLOATS);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.nSprites);
      }
      gl.depthMask(true);
      gl.disable(gl.BLEND); gl.disable(gl.DEPTH_TEST);
      if (off) {
        // resolve the samples, then tone-map onto the canvas
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.fbo.ms);
        gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this.fbo.rs);
        gl.blitFramebuffer(0, 0, w, h, 0, 0, w, h, gl.COLOR_BUFFER_BIT, gl.NEAREST);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, w, h);
        gl.useProgram(this.pComp.p);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.fbo.tex);
        gl.uniform1i(this.pComp.u.uTex, 0);
        gl.bindVertexArray(this.vaoFull);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      }
      gl.bindVertexArray(null);
      return true;
    }
  }

  window.BeatrigGL = {
    HALO, CORE, POOL, MARKER, LINE_FLOATS,
    // a renderer for this canvas, or null when the browser has no WebGL2 (the page then keeps its 2D views)
    create(canvas) {
      try { const r = new Renderer(canvas); return r.gl ? r : null; } catch (e) { console.warn("beatrig: WebGL2 preview off:", e); return null; }
    },
  };
})();
