// beatrig preview page. Plain script, no build, no dependencies.
// One shared time axis (seconds of the audio file; the rig's window is audio_in_s .. audio_out_s) across every lane. Curve data arrives as Float32Array
// buffers (/api/curves) and is drawn per pixel column from a min/max pyramid, so a redraw costs about the same at any
// zoom and any show length. Only lanes inside the scroll viewport are drawn. Canvases redraw only when something
// changes; requestAnimationFrame runs only while audio plays; server changes arrive over /api/events (no polling).
"use strict";

const $ = (s) => document.querySelector(s);
const audio = $("#audio");
const timeline = $("#timeline");
const lanesEl = $("#lanes");
const playheadEl = $("#playhead");
// The pop-out window (/preview, or ?view=preview) is this page in a view-only mode: the preview alone, no audio, no
// edits and no writes to the editor. It follows the main page over a BroadcastChannel and edits over the event stream.
const POPOUT = location.pathname.replace(/\/+$/, "") === "/preview" || new URLSearchParams(location.search).get("view") === "preview";

// Colours and the UI font come from the CSS tokens (app.css :root), so the canvases and the chrome stay one system.
const css = getComputedStyle(document.documentElement);
const tok = (name, d) => css.getPropertyValue(name).trim() || d;
const C = {
  bg: tok("--bg", "#101112"), panel: tok("--surface", "#17181a"), panel2: tok("--surface-2", "#1c1d20"),
  line: tok("--line", "#26282b"), line2: tok("--line-2", "#33363a"),
  text: tok("--text", "#d6d8dc"), dim: tok("--dim", "#80858d"), faint: tok("--faint", "#50545a"), wave: tok("--wave", "#6d727a"),
  accent: tok("--accent", "#e3a33b"), err: tok("--err", "#d8695b"),
  note: tok("--note", "#aaaeb5"), pat: tok("--pat", "#8b97a8"),
  font: tok("--font", "system-ui, sans-serif"),
  warm: [1.0, 0.62, 0.3],            // linear RGB of a lamp with no colour output
};
function alpha(hex, a) {              // "#rrggbb" + alpha -> rgba(); built once at load, not per draw
  const n = parseInt(String(hex).replace("#", ""), 16) || 0x808080;
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}
const F10 = `10px ${C.font}`, F11 = `11px ${C.font}`;
const K = {
  patSpan: alpha(C.pat, 0.07), patStrip: alpha(C.pat, 0.28), patLine: alpha(C.pat, 0.6), patBody: alpha(C.pat, 0.16),
  selStrip: alpha(C.accent, 0.42), selBody: alpha(C.accent, 0.22), rows: alpha(C.accent, 0.55),
  noteBody: alpha(C.note, 0.12), flash: alpha(C.err, 0.22),
};
const MAX_CANVAS_PX = 4096;
const BEATS_PER_BAR = 4;             // design.md has no meter: 4/4 for downbeats, bar numbers and the metronome accent
const STRIP = 10;                    // pattern strip height at the top of a note lane
// Domemaster orientation (OWL convention): the fisheye centre is the camera actor's up axis, so pitch tilts the dome.
// The camera's forward axis lands at the bottom of the circle; set -1 to put it at the top. A camera that is not a
// domemaster is shown as a fisheye along its forward axis instead (up at the top).
const DOME_FORWARD_DOWN = 1;

const S = {
  rig: null, notes: null, bake: null, bands: [], analysis: {}, scene: null, live: {}, flags: null, warnings: [],
  dirty: false, bakeMs: null, bakeId: 0, server: null,
  tStart: 0, duration: 1, fps: 30, offset: 0,     // the audio window in file seconds (duration = its end)
  view: { t0: 0, pps: 50 }, width: 600,
  sel: new Set(),                    // selected objects: hits, pattern hits, pattern copies, sections
  owner: new WeakMap(),              // object -> { track } | { pattern }
  ex: {},                            // track -> { ex: [expanded hits], refs: [pattern copies], t: Float64Array }
  collapsed: new Set(), stacked: new Set(), seenOutputs: new Set(),
  baking: false, queued: null, rev: null, seenRev: 0,
  lastGood: { rig: null, notes: null },
  series: new Map(), peaks: null, peaksKey: "",
  tempo: null, ptempo: null, ops: null,
  keymap: {}, hoverTrack: null, take: null, playStartT: 0, pausedT: 0, audioStale: false,
  undo: [], redo: [],
  view3: "orbit", geo: null, actorMap: new Map(), schemF: null,
  link: false, lastPagePush: 0,
  perf: { flush: [], tick: [], schem: [], maxFlush: 0, drawn: 0, skipped: 0, stateMs: null, firstPaint: null },
};

// ------------------------------------------------------------------------------------------- utils
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const fmt = (v, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : "-");
const deep = (o) => JSON.parse(JSON.stringify(o));
const enc = encodeURIComponent;
const round9 = (v) => Math.round(v * 1e9) / 1e9;
function msg(text, isErr = false) {
  const m = $("#msg");
  m.textContent = text || "";
  m.classList.toggle("err", !!isErr);
}
function lsGet(k, d) { try { const v = localStorage.getItem("beatrig:" + k); return v == null ? d : JSON.parse(v); } catch { return d; } }
function lsSet(k, v) { try { localStorage.setItem("beatrig:" + k, JSON.stringify(v)); } catch { /* private window */ } }
function lowerBound(a, v) { let lo = 0, hi = a.length; while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] < v) lo = m + 1; else hi = m; } return lo; }
const xOf = (t) => (t - S.view.t0) * S.view.pps;
const tOf = (x) => S.view.t0 + x / S.view.pps;
// frame f starts at file time tStart + (f - offset) / fps (frame 0 = audio_in_s, as the engine maps it)
const frameOf = (t) => Math.floor((t - S.tStart) * S.fps + 1e-6) + S.offset;   // the frame showing at time t
const tOfFrame = (f) => (f - S.offset) / S.fps + S.tStart;
const fxOf = (x) => (tOf(x) - S.tStart) * S.fps + S.offset;                     // fractional frame at pixel x
const xOfFrame = (f) => xOf(tOfFrame(f));
const hitFrame = (t) => Math.ceil(Math.round((t - S.tStart) * S.fps * 1e6) / 1e6) + S.offset;   // first frame at or after t
function perfPush(arr, v) { arr.push(v); if (arr.length > 300) arr.shift(); }

// linear -> sRGB byte; HDR colours are scaled down by their largest component so the hue survives
const SRGB = new Uint8Array(4096);
for (let i = 0; i < 4096; i++) { const v = i / 4095; SRGB[i] = Math.round(255 * (v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055)); }
function toSRGB(r, g, b, out, k = 1) {
  r *= k; g *= k; b *= k;
  const m = Math.max(r, g, b);
  if (m > 1) { r /= m; g /= m; b /= m; }
  out[0] = SRGB[(clamp(r, 0, 1) * 4095) | 0]; out[1] = SRGB[(clamp(g, 0, 1) * 4095) | 0]; out[2] = SRGB[(clamp(b, 0, 1) * 4095) | 0];
  return out;
}
const rgbCss = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;

// ------------------------------------------------------------------------------------------- tempo
// Port of beatrig.tempo.Tempo: piecewise-constant bpm map in seconds, beat 0 at the grid offset. Pattern copies are
// placed with timeAt(), so it must agree with the engine to the bit.
class Tempo {
  constructor(segs, gridOffset) {
    segs = (segs || []).map(([t, b]) => [+t, +b]).sort((a, b) => a[0] - b[0]);
    if (!segs.length) segs = [[0, 120]];
    this.t = segs.map((s) => s[0]);
    this.bpm = segs.map((s) => s[1]);
    this.B = [0];
    for (let i = 1; i < segs.length; i++) this.B.push(this.B[i - 1] + (this.t[i] - this.t[i - 1]) * this.bpm[i - 1] / 60);
    this.b0 = this._raw(+gridOffset || 0);
  }
  _seg(t) { let lo = 0, hi = this.t.length; while (lo < hi) { const m = (lo + hi) >> 1; if (this.t[m] <= t) lo = m + 1; else hi = m; } return clamp(lo - 1, 0, this.t.length - 1); }
  _raw(t) { const i = this._seg(t); return this.B[i] + (t - this.t[i]) * this.bpm[i] / 60; }
  beatAt(t) { return this._raw(t) - this.b0; }
  timeAt(b) {
    const raw = b + this.b0;
    let lo = 0, hi = this.B.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (this.B[m] <= raw) lo = m + 1; else hi = m; }
    const i = clamp(lo - 1, 0, this.B.length - 1);
    return this.t[i] + (raw - this.B[i]) * 60 / this.bpm[i];
  }
  bpmAt(t) { return this.bpm[this._seg(t)]; }
}
// Grid tempo: the notes' bpm map, else the analysis map. Pattern tempo: the notes' bpm map, else 120 bpm from the
// notes' grid_offset, which is how beatrig.notes expands placements (the two are the same whenever notes.json has a map).
function rebuildTempo() {
  const n = S.notes || {}, a = S.analysis || {}, nm = n.bpm_map && n.bpm_map.length;
  S.ptempo = nm ? new Tempo(n.bpm_map.map((b) => [b.time, b.bpm]), n.grid_offset || 0) : new Tempo([[0, 120]], n.grid_offset || 0);
  if (nm) S.tempo = S.ptempo;
  else if (a.bpm_map && a.bpm_map.length) S.tempo = new Tempo(a.bpm_map.map((b) => [b.time, b.bpm]), a.grid_offset || 0);
  else S.tempo = S.ptempo;
}
S.tempo = S.ptempo = new Tempo([[0, 120]], 0);   // until the first state arrives (a resize can draw before it)
const beatAt = (t) => S.tempo.beatAt(t);
const timeAt = (b) => S.tempo.timeAt(b);
const pBeatAt = (t) => S.ptempo.beatAt(t);
const pTimeAt = (b) => S.ptempo.timeAt(b);
const gridSize = () => parseFloat($("#grid").value);
const snapOn = () => $("#snap").checked;
const snapBeat = (b) => { const g = gridSize(); return Math.round(b / g) * g; };
const snapT = (t) => (snapOn() ? timeAt(snapBeat(beatAt(t))) : t);

// Visible grid lines: [{x, beat, level}] level 2 = bar, 1 = beat, 0 = subdivision.
function gridLines(w, sub) {
  const out = [];
  const b0 = Math.floor(beatAt(tOf(0))), b1 = Math.ceil(beatAt(tOf(w)));
  const pxPerBeat = S.view.pps * 60 / S.tempo.bpmAt(tOf(w / 2));
  if (pxPerBeat <= 6) {
    let every = 1;
    while (pxPerBeat * BEATS_PER_BAR * every < 10) every *= 2;
    const span = BEATS_PER_BAR * every;
    for (let b = Math.floor(b0 / span) * span; b <= b1; b += span) out.push({ x: xOf(timeAt(b)), beat: b, level: 2 });
    return out;
  }
  const step = sub && pxPerBeat * sub > 7 ? sub : 1;
  const n = Math.round((b1 - b0) / step);
  for (let i = 0; i <= n; i++) {
    const b = b0 + i * step;
    const isBeat = Math.abs(b - Math.round(b)) < 1e-9;
    const level = isBeat ? (((Math.round(b) % BEATS_PER_BAR) + BEATS_PER_BAR) % BEATS_PER_BAR === 0 ? 2 : 1) : 0;
    out.push({ x: xOf(timeAt(b)), beat: b, level });
  }
  return out;
}
function drawGrid(ctx, w, h, sub, strong = 1) {
  const c2 = `rgba(150,155,165,${0.30 * strong})`, c1 = `rgba(150,155,165,${0.13 * strong})`, c0 = `rgba(150,155,165,${0.06 * strong})`;
  for (const g of gridLines(w, sub)) {
    if (g.x < -1 || g.x > w + 1) continue;
    ctx.fillStyle = g.level === 2 ? c2 : g.level === 1 ? c1 : c0;
    ctx.fillRect(Math.round(g.x), 0, 1, h);
  }
}
const sections = () => (S.notes && S.notes.sections) || [];
function drawSectionLines(ctx, w, h) {
  ctx.fillStyle = "rgba(205,208,213,0.17)";
  for (const s of sections()) { const x = Math.round(xOf(s.time)); if (x >= 0 && x <= w) ctx.fillRect(x, 0, 1, h); }
}
function shadeEnd(ctx, w, h) {
  ctx.fillStyle = "rgba(0,0,0,0.35)";
  const xe = xOf(S.duration), xs = xOf(S.tStart);
  if (xe < w) ctx.fillRect(Math.max(0, xe), 0, w - Math.max(0, xe), h);
  if (xs > 0) ctx.fillRect(0, 0, Math.min(w, xs), h);
}

// ------------------------------------------------------------------------------------------- binary data
// /api/curves and /api/peaks: uint32 LE header length, JSON header (4-byte aligned), then the data.
async function fetchBin(url) {
  const res = await fetch(url);
  if (!res.ok) { let e = `HTTP ${res.status}`; try { e = (await res.json()).error || e; } catch { /* not JSON */ } throw new Error(e); }
  const buf = await res.arrayBuffer();
  const H = new DataView(buf).getUint32(0, true);
  return { hdr: JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, H))), buf, off: 4 + H };
}
const MN = new Float32Array(MAX_CANVAS_PX), MX = new Float32Array(MAX_CANVAS_PX), RGB = new Float32Array(MAX_CANVAS_PX * 3);

// One output (or band) at full frame rate: values [C][N] or colours [C][N][4], plus a lazily built pyramid per
// channel (min/max, or mean colour) in steps of 4 frames, so a pixel column never reads more than ~16 numbers.
class Series {
  constructor(hdr, buf, off) {
    this.id = hdr.id; this.hash = hdr.hash; this.bakeId = hdr.bake_id;
    this.C = hdr.channels; this.N = hdr.frames; this.comps = hdr.components;
    this.S0 = hdr.start_frame || 0;          // series index i = sequence frame S0 + i (bake.start_frame, v0.3)
    this.d = new Float32Array(buf, off, this.C * this.N * this.comps);
    this.pyr = new Array(this.C);
    this.max = null;
  }
  at(c, f) { f -= this.S0; f = f < 0 ? 0 : f >= this.N ? this.N - 1 : f | 0; return this.d[c * this.N + f]; }
  rgba(c, f, out) {
    f -= this.S0; f = f < 0 ? 0 : f >= this.N ? this.N - 1 : f | 0;
    const i = (c * this.N + f) * 4, d = this.d;
    out[0] = d[i]; out[1] = d[i + 1]; out[2] = d[i + 2]; out[3] = d[i + 3];
    return out;
  }
  peak() {               // largest value in channel 0 (band lanes scale to it)
    if (this.max == null) { let m = 0; for (let i = 0; i < this.N; i++) if (this.d[i] > m) m = this.d[i]; this.max = m || 1; }
    return this.max;
  }
  levels(c) {
    if (this.pyr[c]) return this.pyr[c];
    const lv = [], N = this.N, d = this.d;
    if (this.comps === 1) {
      const base = c * N;
      let src = null;
      for (let B = 4; ; B *= 4) {
        const M = Math.ceil(N / B), mn = new Float32Array(M), mx = new Float32Array(M);
        for (let m = 0; m < M; m++) {
          let lo = Infinity, hi = -Infinity;
          if (!src) { for (let i = base + m * 4, e = base + Math.min(N, m * 4 + 4); i < e; i++) { const v = d[i]; if (v < lo) lo = v; if (v > hi) hi = v; } }
          else { for (let i = m * 4, e = Math.min(src.mn.length, m * 4 + 4); i < e; i++) { if (src.mn[i] < lo) lo = src.mn[i]; if (src.mx[i] > hi) hi = src.mx[i]; } }
          mn[m] = lo; mx[m] = hi;
        }
        src = { B, mn, mx }; lv.push(src);
        if (M <= 1) break;
      }
    } else {
      let src = null;
      for (let B = 4; ; B *= 4) {
        const M = Math.ceil(N / B), a = new Float32Array(M * 3);
        for (let m = 0; m < M; m++) {
          let r = 0, g = 0, b = 0, n = 0;
          if (!src) { for (let f = m * 4, e = Math.min(N, m * 4 + 4); f < e; f++, n++) { const i = (c * N + f) * 4; r += d[i]; g += d[i + 1]; b += d[i + 2]; } }
          else { for (let i = m * 4, e = Math.min(src.a.length / 3, m * 4 + 4); i < e; i++, n++) { r += src.a[i * 3]; g += src.a[i * 3 + 1]; b += src.a[i * 3 + 2]; } }
          a[m * 3] = r / n; a[m * 3 + 1] = g / n; a[m * 3 + 2] = b / n;
        }
        src = { B, a }; lv.push(src);
        if (M <= 1) break;
      }
    }
    return (this.pyr[c] = lv);
  }
  // pick the coarsest level whose bucket is at most a quarter of a pixel column
  level(c, fpp) {
    if (fpp < 16) return null;
    let L = null;
    for (const l of this.levels(c)) { if (l.B * 4 <= fpp) L = l; else break; }
    return L;
  }
  // min and max per pixel column over [0, w); NaN where the column has no frames
  cols(c, w, mn, mx) {
    const N = this.N, d = this.d, base = c * N;
    const fpp = S.fps / S.view.pps, f0 = fxOf(0) - this.S0, L = this.level(c, fpp);
    for (let x = 0; x < w; x++) {
      // each column covers the frames whose start falls inside it (at least one, so zoomed in it repeats a frame)
      const fa = Math.floor(f0 + x * fpp);
      let fb = Math.floor(f0 + (x + 1) * fpp); if (fb <= fa) fb = fa + 1;
      const a = fa < 0 ? 0 : fa, b = fb > N ? N : fb;
      if (a >= b) { mn[x] = NaN; mx[x] = NaN; continue; }
      let lo = Infinity, hi = -Infinity;
      if (!L) { for (let i = base + a, e = base + b; i < e; i++) { const v = d[i]; if (v < lo) lo = v; if (v > hi) hi = v; } }
      else { const B = L.B, lm = L.mn, lx = L.mx; for (let m = (a / B) | 0, e = ((b - 1) / B) | 0; m <= e; m++) { if (lm[m] < lo) lo = lm[m]; if (lx[m] > hi) hi = lx[m]; } }
      mn[x] = lo; mx[x] = hi;
    }
  }
  // mean colour per pixel column (RGB triplets); NaN where the column has no frames
  colsRGB(c, w, out) {
    const N = this.N, d = this.d;
    const fpp = S.fps / S.view.pps, f0 = fxOf(0) - this.S0, L = this.level(c, fpp);
    for (let x = 0; x < w; x++) {
      const fa = Math.floor(f0 + x * fpp);
      let fb = Math.floor(f0 + (x + 1) * fpp); if (fb <= fa) fb = fa + 1;
      const a = fa < 0 ? 0 : fa, b = fb > N ? N : fb;
      if (a >= b) { out[x * 3] = NaN; continue; }
      let r = 0, g = 0, bl = 0, n = 0;
      if (!L) { for (let f = a; f < b; f++, n++) { const i = (c * N + f) * 4; r += d[i]; g += d[i + 1]; bl += d[i + 2]; } }
      else { const B = L.B, A = L.a; for (let m = (a / B) | 0, e = ((b - 1) / B) | 0; m <= e; m++, n++) { r += A[m * 3]; g += A[m * 3 + 1]; bl += A[m * 3 + 2]; } }
      out[x * 3] = r / n; out[x * 3 + 1] = g / n; out[x * 3 + 2] = bl / n;
    }
  }
}

// Pyramids are built lazily on first use; this warms them in idle time (one channel per slice) right after a series
// arrives, so the first zoom-out after a load or a re-bake does not stall on ~100 ms of building.
const idle = window.requestIdleCallback || ((f) => setTimeout(() => f({ timeRemaining: () => 8 }), 1));
function warmPyramids(s) {
  let c = 0;
  const step = (dl) => {
    while (c < s.C && dl.timeRemaining() > 2) { if (S.series.get((s.kindKey)) !== s) return; s.levels(c++); }
    if (c < s.C) idle(step);
  };
  idle(step);
}

// Waveform peaks: uint8 max |mono| at 200 per second, with a max pyramid in steps of 4.
class Peaks {
  constructor(hdr, buf, off) {
    this.rate = hdr.rate; this.d = new Uint8Array(buf, off, hdr.count);
    this.lv = [];
    let src = this.d;
    for (let B = 4; src.length > 1; B *= 4) {
      const M = Math.ceil(src.length / 4), a = new Uint8Array(M);
      for (let m = 0; m < M; m++) { let v = 0; for (let i = m * 4, e = Math.min(src.length, m * 4 + 4); i < e; i++) if (src[i] > v) v = src[i]; a[m] = v; }
      this.lv.push({ B, a }); src = a;
    }
  }
  cols(w, out) {
    const n = this.d.length, per = this.rate / S.view.pps;  // peaks per pixel
    let L = null;
    if (per >= 16) for (const l of this.lv) { if (l.B * 4 <= per) L = l; else break; }
    const i0 = tOf(0) * this.rate;
    for (let x = 0; x < w; x++) {
      const ia = Math.floor(i0 + x * per);
      let ib = Math.floor(i0 + (x + 1) * per); if (ib <= ia) ib = ia + 1;
      const a = ia < 0 ? 0 : ia, b = ib > n ? n : ib;
      if (a >= b) { out[x] = -1; continue; }
      let v = 0;
      if (!L) { for (let i = a; i < b; i++) if (this.d[i] > v) v = this.d[i]; }
      else { const B = L.B, A = L.a; for (let m = (a / B) | 0, e = ((b - 1) / B) | 0; m <= e; m++) if (A[m] > v) v = A[m]; }
      out[x] = v;
    }
  }
}

// ------------------------------------------------------------------------------------------- canvases
// Redraws are batched into one animation frame. Canvases scrolled out of view are skipped and drawn when they
// scroll back in, so the cost of a redraw follows what's on screen, not how many lanes the rig has.
const canvases = new Set();
function setupCanvas(cv) {
  const r = cv.getBoundingClientRect();
  const w = Math.max(1, Math.round(r.width)), h = Math.max(1, Math.round(r.height));
  const dpr = Math.min(window.devicePixelRatio || 1, MAX_CANVAS_PX / w);
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}
let redrawPending = false;
const dirtyCanvases = new Set(), staleCanvases = new Set();
function schedule(cv) {
  if (cv) dirtyCanvases.add(cv); else for (const c of canvases) dirtyCanvases.add(c);
  requestFlush();
}
function requestFlush() {
  if (redrawPending) return;
  redrawPending = true;
  requestAnimationFrame(flush);
}
function flush() {
  redrawPending = false;
  const t0 = performance.now();
  const vr = timeline.getBoundingClientRect();
  let drawn = 0;
  for (const c of dirtyCanvases) {
    if (!c.isConnected || c.offsetParent === null) continue;
    const r = c.getBoundingClientRect();
    if (r.bottom < vr.top || r.top > vr.bottom) { staleCanvases.add(c); continue; }
    staleCanvases.delete(c);
    drawCanvas(c);
    drawn++;
  }
  dirtyCanvases.clear();
  placePlayhead();
  const ms = performance.now() - t0;
  if (drawn) {
    perfPush(S.perf.flush, ms);
    S.perf.maxFlush = Math.max(S.perf.maxFlush, ms);
    S.perf.drawn = drawn;
    S.perf.skipped = staleCanvases.size;
  }
  // first paint = the first frame drawn once every curve and the waveform are in (lanes below the fold excepted)
  if (S.perf.firstPaint == null && S.perf.dataReady != null) S.perf.firstPaint = performance.now();
}
timeline.addEventListener("scroll", () => { if (staleCanvases.size) { for (const c of staleCanvases) dirtyCanvases.add(c); requestFlush(); } }, { passive: true });
function scheduleKind(sel) { document.querySelectorAll(sel).forEach((c) => schedule(c)); }
function drawCanvas(cv) {
  const { ctx, w, h } = setupCanvas(cv);
  const k = cv.dataset.kind;
  if (k === "ruler") drawRuler(ctx, w, h);
  else if (k === "sections") drawSections(ctx, w, h);
  else if (k === "wave") drawWave(ctx, w, h);
  else if (k === "band") drawBand(ctx, w, h, cv.dataset.id);
  else if (k === "notes") drawNotes(ctx, w, h, cv.dataset.track);
  else if (k === "layer") drawLayer(ctx, w, h, cv.dataset.id);
  else if (k === "curve") drawCurve(ctx, w, h, cv.dataset.id);
}

// ------------------------------------------------------------------------------------------- ruler, sections, waveform, bands
function niceStep(minPx) {
  const steps = [1 / S.fps, 2 / S.fps, 5 / S.fps, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
  for (const s of steps) if (s * S.view.pps >= minPx) return s;
  return 1200;
}
function tLabel(t, step) {
  if (step < 1) {
    const f = Math.round(t * S.fps);
    return step < 0.25 ? `${f}f` : `${t.toFixed(2)}`;
  }
  const m = Math.floor(t / 60), s = t - m * 60;
  return m ? `${m}:${String(Math.floor(s)).padStart(2, "0")}` : `${Math.round(s)}s`;
}
function drawRuler(ctx, w, h) {
  ctx.fillStyle = C.panel; ctx.fillRect(0, 0, w, h);
  const step = niceStep(70);
  const minor = step / 5;
  const t0 = Math.floor(tOf(0) / minor) * minor;
  ctx.font = F11;
  ctx.textBaseline = "top";
  for (let t = t0, i = 0; t <= tOf(w) && i < 2000; t += minor, i++) {
    const x = Math.round(xOf(t));
    const major = Math.abs(t / step - Math.round(t / step)) < 1e-6;
    ctx.fillStyle = major ? C.dim : C.faint;
    ctx.fillRect(x, major ? h - 9 : h - 4, 1, major ? 9 : 4);
    if (major && t >= -1e-9) { ctx.fillStyle = C.dim; ctx.fillText(tLabel(Math.max(0, t), step), x + 4, 4); }
  }
  // flash-safety windows from every output, as ticks along the ruler
  if (S.flags) {
    ctx.fillStyle = C.err;
    for (const k in S.flags) if (k[0] !== "_") for (const [a, b] of S.flags[k]) { const x0 = xOf(a), x1 = xOf(b); if (x1 >= 0 && x0 <= w) ctx.fillRect(x0, h - 3, Math.max(1, x1 - x0), 3); }
  }
  shadeEnd(ctx, w, h);
}
const secLabelW = new Map();
function secBoxes(ctx) {
  const out = [];
  ctx.font = F11;
  for (const s of sections()) {
    let tw = secLabelW.get(s.name);
    if (tw == null) { tw = ctx.measureText(s.name || "").width; secLabelW.set(s.name, tw); }
    out.push({ s, x: xOf(s.time), tw });
  }
  return out;
}
function drawSections(ctx, w, h) {
  ctx.fillStyle = C.panel2; ctx.fillRect(0, 0, w, h);
  const boxes = secBoxes(ctx);
  ctx.textBaseline = "middle";
  for (let i = 0; i < boxes.length; i++) {
    const b = boxes[i], next = i + 1 < boxes.length ? boxes[i + 1].x : xOf(S.duration);
    if (next < 0 || b.x > w) continue;
    if (i % 2) { ctx.fillStyle = "rgba(255,255,255,0.025)"; ctx.fillRect(b.x, 0, next - b.x, h); }
    const sel = S.sel.has(b.s);
    ctx.fillStyle = sel ? C.accent : C.dim;
    ctx.fillRect(Math.round(b.x), 0, 1, h);
    ctx.save();
    ctx.beginPath(); ctx.rect(b.x + 1, 0, Math.max(0, next - b.x - 3), h); ctx.clip();
    ctx.fillStyle = sel ? C.accent : C.text;
    ctx.fillText(b.s.name || "", b.x + 4, h / 2 + 0.5);
    ctx.restore();
  }
  if (!boxes.length) { ctx.fillStyle = C.faint; ctx.fillText("double-click to add a section (M at the playhead)", 8, h / 2 + 0.5); }
  shadeEnd(ctx, w, h);
}
function drawWave(ctx, w, h) {
  ctx.fillStyle = C.bg; ctx.fillRect(0, 0, w, h);
  drawGrid(ctx, w, h, gridSize(), 1.2);
  drawSectionLines(ctx, w, h);
  ctx.font = F10; ctx.textBaseline = "top"; ctx.fillStyle = C.faint;
  let lastX = -1e9;
  for (const g of gridLines(w, 0)) if (g.level === 2 && g.x - lastX > 28 && g.x >= 0) {
    ctx.fillText(String(Math.round(g.beat / BEATS_PER_BAR) + 1), g.x + 3, 2); lastX = g.x;
  }
  const mid = h / 2 + 4, amp = h / 2 - 8;
  if (S.peaks) {
    S.peaks.cols(w, MN);
    let xa = 0; while (xa < w && MN[xa] < 0) xa++;
    let xb = w - 1; while (xb >= xa && MN[xb] < 0) xb--;
    if (xb >= xa) {
      ctx.beginPath();
      ctx.moveTo(xa, mid - Math.max(0.5, MN[xa] / 255 * amp));
      for (let x = xa; x <= xb; x++) { const y = mid - Math.max(0.5, MN[x] / 255 * amp); ctx.lineTo(x, y); ctx.lineTo(x + 1, y); }
      for (let x = xb; x >= xa; x--) { const y = mid + Math.max(0.5, MN[x] / 255 * amp); ctx.lineTo(x + 1, y); ctx.lineTo(x, y); }
      ctx.closePath(); ctx.fillStyle = C.wave; ctx.fill();
    }
  } else {
    ctx.fillStyle = C.faint; ctx.font = F11;
    ctx.fillText("loading waveform", 8, h / 2);
  }
  // detected beats from the analysis: short ticks along the bottom edge (skipped when they would merge)
  const beats = S.analysis && S.analysis.beats;
  if (beats && beats.length) {
    const i0 = lowerBound(beats, tOf(0)), i1 = lowerBound(beats, tOf(w));
    if (i1 - i0 < w / 3) {
      ctx.fillStyle = C.accent; ctx.globalAlpha = 0.7;
      for (let i = i0; i < i1; i++) ctx.fillRect(Math.round(xOf(beats[i])), h - 5, 1, 5);
      ctx.globalAlpha = 1;
    }
  }
  shadeEnd(ctx, w, h);
}
// Value series in a box: a polyline through the frames when zoomed in, else a min/max band per pixel column.
// fill = area down to the box bottom (rows), otherwise a band between min and max at least 1 px thick (overlay).
function drawValues(ctx, s, c, w, top, height, lo, hi, color, fill) {
  const span = hi - lo || 1, base = top + height;
  const yOf = (v) => top + height - clamp((v - lo) / span, -0.03, 1.03) * height;
  const fpp = S.fps / S.view.pps;
  ctx.beginPath();
  if (fpp <= 2) {
    const S0 = s.S0 || 0;                                   // series index -> sequence frame S0 + index
    const f0 = Math.max(0, Math.floor(fxOf(0) - S0) - 1), f1 = Math.min(s.N - 1, Math.ceil(fxOf(w) - S0) + 1);
    if (f1 < f0) return;
    const off = c * s.N, d = s.d;
    if (fill) ctx.moveTo(xOfFrame(f0 + S0), base);
    for (let f = f0; f <= f1; f++) { const x = xOfFrame(f + S0), y = yOf(d[off + f]); if (f === f0 && !fill) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
    if (fill) { ctx.lineTo(xOfFrame(f1 + S0), base); ctx.closePath(); ctx.fillStyle = color; ctx.fill(); }
    else { ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.stroke(); }
    return;
  }
  s.cols(c, w, MN, MX);
  let xa = 0; while (xa < w && MN[xa] !== MN[xa]) xa++;
  let xb = w - 1; while (xb >= xa && MN[xb] !== MN[xb]) xb--;
  if (xb < xa) return;
  if (fill) {
    let top = -Infinity;
    for (let x = xa; x <= xb; x++) if (MX[x] > top) top = MX[x];
    if (top <= lo) return;                                   // flat at the floor across the whole view
    ctx.moveTo(xa, base);
    for (let x = xa; x <= xb; x++) ctx.lineTo(x + 0.5, yOf(MX[x]));
    ctx.lineTo(xb + 1, base);
  } else {
    for (let x = xa; x <= xb; x++) { const y = yOf(MX[x]); if (x === xa) ctx.moveTo(x, y); ctx.lineTo(x, y); ctx.lineTo(x + 1, y); }
    for (let x = xb; x >= xa; x--) { const y = Math.max(yOf(MN[x]), yOf(MX[x]) + 1); ctx.lineTo(x + 1, y); ctx.lineTo(x, y); }
  }
  ctx.closePath(); ctx.fillStyle = color; ctx.fill();
}
// Colour channel as a strip: each pixel column is the (mean) colour of its frames, built as one row of pixels.
// All channels go into one image (a pixel row each), drawn with one scaled drawImage; rows are then separated by lines.
const stripCv = document.createElement("canvas");
const stripCtx = stripCv.getContext("2d");
let stripImg = null;
const RGB8 = new Uint8Array(3);
function drawColors(ctx, s, w, h) {
  const n = s.C;
  if (!stripImg || stripImg.width < w || stripImg.height < n) {
    stripCv.width = Math.max(w, stripImg ? stripImg.width : 0); stripCv.height = Math.max(n, stripImg ? stripImg.height : 0);
    stripImg = stripCtx.createImageData(stripCv.width, stripCv.height);
  }
  const px = stripImg.data, W = stripImg.width;
  for (let c = 0; c < n; c++) {
    s.colsRGB(c, w, RGB);
    for (let x = 0, i = c * W * 4; x < w; x++, i += 4) {
      const r = RGB[x * 3];
      if (r !== r) { px[i + 3] = 0; continue; }
      toSRGB(r, RGB[x * 3 + 1], RGB[x * 3 + 2], RGB8);
      px[i] = RGB8[0]; px[i + 1] = RGB8[1]; px[i + 2] = RGB8[2]; px[i + 3] = 255;
    }
  }
  stripCtx.putImageData(stripImg, 0, 0, 0, 0, w, n);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(stripCv, 0, 0, w, n, 0, 0, w, h);
  const rowH = h / n;
  if (rowH >= 5) { ctx.fillStyle = C.bg; for (let c = 1; c < n; c++) ctx.fillRect(0, Math.round(c * rowH) - 1, w, 1); }
}
function drawBand(ctx, w, h, id) {
  ctx.fillStyle = C.bg; ctx.fillRect(0, 0, w, h);
  drawGrid(ctx, w, h, 0, 0.6);
  const s = S.series.get("b:" + id);
  if (s) drawValues(ctx, s, 0, w, 2, h - 3, 0, s.peak(), "rgba(150,155,165,0.45)", true);
  shadeEnd(ctx, w, h);
}

// ------------------------------------------------------------------------------------------- notes model
// A track holds plain hits ({time, vel}) and pattern copies ({pattern, at_beat, repeat}). Copies expand here with the
// same tempo the server uses; every expanded hit points back at the one pattern hit it came from, so selecting or
// editing it in any copy edits the pattern and every copy redraws.
const kindOf = (o) => (o.pattern != null ? "ref" : o.beat != null ? "phit" : o.name != null ? "section" : "hit");
function noteLen(b) { return b == null ? 0.25 : b; }
function buildEx(track) {
  const list = (S.notes.tracks || {})[track] || [];
  const pats = S.notes.patterns || {};
  const ex = [], refs = [];
  for (const h of list) {
    if (h && h.pattern != null) {
      S.owner.set(h, { track });
      const p = pats[h.pattern];
      if (!p) continue;
      const L = +p.length_beats || 4, rep = Math.max(1, (h.repeat | 0) || 1), ab = +h.at_beat || 0;
      refs.push({ obj: h, p: h.pattern, ab, L, rep, t0: pTimeAt(ab), t1: pTimeAt(ab + rep * L) });
      for (let r = 0; r < rep; r++) {
        for (const ph of p.hits || []) {
          const b = +ph.beat || 0;
          if (b < 0 || b >= L) continue;
          S.owner.set(ph, { pattern: h.pattern });
          const t = pTimeAt(ab + r * L + b);
          ex.push({ time: t, vel: ph.vel == null ? 1 : ph.vel, len: noteLen(ph.len_beats), src: ph, ref: h, track });
        }
      }
    } else if (h) {
      S.owner.set(h, { track });
      ex.push({ time: +h.time, vel: h.vel == null ? 1 : h.vel, len: noteLen(h.len_beats), src: h, ref: null, track });
    }
  }
  ex.sort((a, b) => a.time - b.time);
  refs.sort((a, b) => a.t0 - b.t0);
  const t = new Float64Array(ex.length);
  for (let i = 0; i < ex.length; i++) t[i] = ex[i].time;
  S.ex[track] = { ex, refs, t };
}
function buildAllEx() {
  S.ex = {};
  for (const tr of Object.keys((S.notes && S.notes.tracks) || {})) buildEx(tr);
  for (const s of sections()) S.owner.set(s, { sections: true });
}
function entryStart(h) { return h.pattern != null ? pTimeAt(+h.at_beat || 0) : +h.time; }
function sortTrack(track) { const l = S.notes.tracks[track]; if (l) l.sort((a, b) => entryStart(a) - entryStart(b)); }
function sortSections() { if (S.notes.sections) S.notes.sections.sort((a, b) => a.time - b.time); }
function noteDur(n) { return Math.max(0.001, timeAt(beatAt(n.time) + n.len) - n.time); }

function drawLinkGlyph(ctx, x, y) {      // two interlocked links: the marker for a linked pattern copy
  ctx.strokeRect(x + 0.5, y + 0.5, 6, 4);
  ctx.strokeRect(x + 4.5, y + 2.5, 6, 4);
}
function drawNotes(ctx, w, h, track) {
  ctx.fillStyle = C.bg; ctx.fillRect(0, 0, w, h);
  drawGrid(ctx, w, h, gridSize(), 1);
  drawSectionLines(ctx, w, h);
  const E = S.ex[track];
  if (!E) return;
  // pattern copies: shaded span, strip with the pattern name and a link marker on every repeat
  ctx.font = F10; ctx.textBaseline = "middle"; ctx.lineWidth = 1;
  for (const R of E.refs) {
    const x0 = xOf(R.t0), x1 = xOf(R.t1);
    if (x1 < 0 || x0 > w) continue;
    const sel = S.sel.has(R.obj);
    ctx.fillStyle = K.patSpan; ctx.fillRect(x0, 0, x1 - x0, h);
    ctx.fillStyle = sel ? K.selStrip : K.patStrip; ctx.fillRect(x0, 0, x1 - x0, STRIP);
    for (let r = 0; r < R.rep; r++) {
      const xa = xOf(pTimeAt(R.ab + r * R.L)), xb = xOf(pTimeAt(R.ab + (r + 1) * R.L));
      if (xb < 0 || xa > w) continue;
      ctx.fillStyle = sel ? C.accent : K.patLine; ctx.fillRect(Math.round(xa), 0, 1, h);
      if (xb - xa > 34) {
        ctx.save(); ctx.beginPath(); ctx.rect(xa, 0, xb - xa - 2, STRIP); ctx.clip();
        ctx.strokeStyle = sel ? C.bg : C.text; drawLinkGlyph(ctx, xa + 3, 1.5);
        ctx.fillStyle = sel ? C.bg : C.text; ctx.fillText(R.p + (r === 0 && R.rep > 1 ? `  x${R.rep}` : ""), xa + 17, STRIP / 2 + 0.5);
        ctx.restore();
      }
    }
    ctx.fillStyle = sel ? C.accent : K.patLine; ctx.fillRect(Math.round(x1) - 3, 0, 3, STRIP);   // repeat handle
  }
  // hits: only the visible slice; aggregated per pixel column when there are more hits than pixels
  const pad = 4, H = h - pad * 2;
  const i0 = lowerBound(E.t, tOf(0) - 4), i1 = lowerBound(E.t, tOf(w));
  if (i1 - i0 > w * 2) {
    MX.fill(0, 0, w);
    for (let i = i0; i < i1; i++) { const x = Math.floor(xOf(E.t[i])); if (x >= 0 && x < w && E.ex[i].vel > MX[x]) MX[x] = E.ex[i].vel; }
    ctx.fillStyle = C.note;
    for (let x = 0; x < w; x++) if (MX[x] > 0) { const vh = Math.max(2, MX[x] * H); ctx.fillRect(x, pad + H - vh, 1, vh); }
    return;
  }
  for (let i = i0; i < i1; i++) {
    const n = E.ex[i];
    const x = xOf(n.time);
    const wpx = Math.max(3, noteDur(n) * S.view.pps);
    if (x + wpx < 0 || x > w) continue;
    const sel = S.sel.has(n.src), pat = !!n.ref;
    const vh = Math.max(2, clamp(n.vel, 0, 1) * H);
    ctx.fillStyle = sel ? K.selBody : pat ? K.patBody : K.noteBody;
    ctx.fillRect(x, pad + H - vh, wpx, vh);
    ctx.fillStyle = sel ? C.accent : pat ? C.pat : C.note;
    ctx.fillRect(Math.round(x), pad + H - vh, 2, vh);           // onset stem, height = velocity
    ctx.fillRect(x, pad + H - vh, wpx, 1);                       // velocity line
  }
}
function hitNote(track, x) {
  const E = S.ex[track];
  if (!E) return null;
  const i0 = lowerBound(E.t, tOf(x - 4) - 4), i1 = lowerBound(E.t, tOf(x + 5));
  let best = null, bestD = 1e9;
  for (let i = i0; i < i1; i++) {
    const n = E.ex[i], nx = xOf(n.time), wpx = Math.max(6, noteDur(n) * S.view.pps);
    if (x >= nx - 4 && x <= nx + wpx) { const d = Math.abs(x - nx); if (d < bestD) { bestD = d; best = n; } }
  }
  return best;
}
function refAt(track, x) {
  const E = S.ex[track];
  if (!E) return null;
  const t = tOf(x);
  for (const R of E.refs) if (t >= R.t0 && t < R.t1) return R;
  return null;
}
function selectOnly(o, add) {
  if (add) { if (S.sel.has(o)) S.sel.delete(o); else S.sel.add(o); }
  else if (!S.sel.has(o)) { S.sel.clear(); S.sel.add(o); }
  selChanged();
}
function selChanged() {
  const n = { hit: 0, phit: 0, ref: 0, section: 0 };
  for (const o of S.sel) n[kindOf(o)]++;
  const parts = [];
  if (n.hit) parts.push(`${n.hit} hit${n.hit > 1 ? "s" : ""}`);
  if (n.phit) parts.push(`${n.phit} pattern hit${n.phit > 1 ? "s" : ""}`);
  if (n.ref) parts.push(`${n.ref} pattern cop${n.ref > 1 ? "ies" : "y"}`);
  if (n.section) parts.push(`${n.section} section${n.section > 1 ? "s" : ""}`);
  $("#sel-info").textContent = parts.join(", ");
  $("#mk-pat").disabled = !n.hit;
  $("#dup-pat").disabled = $("#unlink-pat").disabled = !n.ref;
  scheduleNotes();
  scheduleKind('canvas[data-kind="sections"]');
}
function attachNoteLane(cv, track) {
  cv.addEventListener("mousemove", (e) => {
    const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    const R = y <= STRIP + 1 ? refAt(track, x) : null;
    cv.style.cursor = R ? (x >= xOf(R.t1) - 5 ? "ew-resize" : "grab") : "crosshair";
  });
  cv.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    const R = refAt(track, x);
    if (R && y <= STRIP + 1) return dragRef(e, track, R, x);
    const n = hitNote(track, x);
    if (n) return dragHits(e, n, r, null);
    // empty space: add a hit (inside a pattern copy it goes into the pattern, so every copy gets it)
    const before = deep(S.notes);
    const t = Math.max(0, snapT(tOf(x)));
    const H = r.height - 8, vel = clamp(Math.round((1 - (y - 4) / H) * 100) / 100, 0.05, 1);
    let obj;
    if (R) {
      const p = S.notes.patterns[R.p], b = pBeatAt(t);
      const rr = clamp(Math.floor((b - R.ab) / R.L), 0, R.rep - 1);
      obj = { beat: round9(clamp(b - R.ab - rr * R.L, 0, R.L - 1e-6)), vel };
      (p.hits = p.hits || []).push(obj);
      rebuildEx();
    } else {
      obj = { time: +t.toFixed(4), vel };
      (S.notes.tracks[track] = S.notes.tracks[track] || []).push(obj);
      sortTrack(track); buildEx(track);
    }
    S.sel.clear(); S.sel.add(obj); selChanged();
    let added = null, bd = Infinity;     // the expanded copy of the new hit nearest the click anchors the drag
    for (const q of S.ex[track].ex) if (q.src === obj && Math.abs(q.time - t) < bd) { bd = Math.abs(q.time - t); added = q; }
    added = added || { time: t, src: obj };
    dragHits(e, added, r, before);
  });
  const lane = cv.parentElement;
  lane.addEventListener("mouseenter", () => { S.hoverTrack = track; lane.classList.add("hover"); });
  lane.addEventListener("mouseleave", () => { if (S.hoverTrack === track) S.hoverTrack = null; lane.classList.remove("hover"); });
}
// Move selected hits along time (pattern hits move inside their pattern) or change their velocity.
function dragHits(e, n, r, before) {
  const added = !!before;
  if (!added) {
    if (e.shiftKey) { selectOnly(n.src, true); return; }
    selectOnly(n.src, false);
    before = deep(S.notes);
  }
  const items = [...S.sel].filter((o) => { const k = kindOf(o); return k === "hit" || k === "phit"; }).map((o) => {
    if (kindOf(o) === "hit") return { o, time: +o.time, vel: o.vel == null ? 1 : o.vel };
    const p = (S.notes.patterns || {})[(S.owner.get(o) || {}).pattern];
    return { o, beat: +o.beat, vel: o.vel == null ? 1 : o.vel, L: p ? +p.length_beats || 4 : 4 };
  });
  const anchorT = n.time, anchorB = pBeatAt(anchorT);
  const x0 = e.clientX, y0 = e.clientY, H = r.height - 8;
  let axis = null, changed = added;
  const move = (ev) => {
    const dx = ev.clientX - x0, dy = ev.clientY - y0;
    if (!axis) { if (Math.abs(dx) < 3 && Math.abs(dy) < 3) return; axis = Math.abs(dx) >= Math.abs(dy) ? "x" : "y"; }
    if (axis === "x") {
      const T = snapT(anchorT + dx / S.view.pps), dt = T - anchorT, dB = pBeatAt(T) - anchorB;
      for (const s of items) {
        if (s.time != null) s.o.time = +Math.max(0, s.time + dt).toFixed(4);
        else s.o.beat = round9(clamp(s.beat + dB, 0, s.L - 1e-6));
      }
    } else {
      for (const s of items) s.o.vel = +clamp(s.vel - dy / H, 0.01, 1).toFixed(3);
      msg(`velocity ${(n.src.vel == null ? 1 : n.src.vel).toFixed(2)}`);
    }
    changed = true;
    rebuildEx();
    scheduleNotes();
  };
  const up = () => {
    window.removeEventListener("mousemove", move);
    window.removeEventListener("mouseup", up);
    if (changed) { for (const tr of Object.keys(S.notes.tracks)) sortTrack(tr); rebuildEx(); pushUndo("notes", before); postNotes(); }
    scheduleNotes();
  };
  window.addEventListener("mousemove", move);
  window.addEventListener("mouseup", up);
  scheduleNotes();
}
// Pattern strip: drag moves the copy (Alt drags out a new linked copy), the right edge sets the repeat count.
function dragRef(e, track, R, x) {
  const before = deep(S.notes);
  const edge = x >= xOf(R.t1) - 5;
  let obj = R.obj, changed = false;
  if (e.altKey && !edge) {
    obj = { pattern: R.obj.pattern, at_beat: R.obj.at_beat, repeat: R.obj.repeat || 1 };
    S.notes.tracks[track].push(obj);
    changed = true;
  }
  if (e.shiftKey && !changed) { selectOnly(obj, true); return; }
  S.sel.clear(); S.sel.add(obj); selChanged();
  const r = e.target.getBoundingClientRect();
  const b0 = pBeatAt(tOf(x)), ab0 = +obj.at_beat || 0, L = R.L;
  const move = (ev) => {
    const b = pBeatAt(tOf(ev.clientX - r.left));
    if (edge) {
      const rep = Math.max(1, Math.round((b - ab0) / L));
      if (rep !== (obj.repeat || 1)) { obj.repeat = rep; changed = true; }
    } else {
      let nab = ab0 + (b - b0);
      if (snapOn()) nab = snapBeat(nab);
      nab = round9(Math.max(pBeatAt(S.tStart), nab));
      if (nab !== obj.at_beat) { obj.at_beat = nab; changed = true; }
    }
    buildEx(track);
    scheduleNotes();
  };
  const up = () => {
    window.removeEventListener("mousemove", move);
    window.removeEventListener("mouseup", up);
    if (changed) { sortTrack(track); buildEx(track); pushUndo("notes", before); postNotes(); msg(edge ? `repeat ${obj.repeat}` : `${obj.pattern} at beat ${fmt(obj.at_beat, 3)}`); }
    scheduleNotes();
  };
  window.addEventListener("mousemove", move);
  window.addEventListener("mouseup", up);
  buildEx(track);
  scheduleNotes();
}
function rebuildEx() { buildAllEx(); }
function scheduleNotes() { scheduleKind('canvas[data-kind="notes"]'); }

// Selected plain hits (one lane) -> a pattern starting on the bar line before the first hit, whole bars long.
function makePattern() {
  const hits = [...S.sel].filter((o) => kindOf(o) === "hit");
  if (!hits.length) return msg("select hits in one lane first", true);
  const tracks = new Set(hits.map((o) => (S.owner.get(o) || {}).track));
  if (tracks.size !== 1) return msg("a pattern takes hits from one lane", true);
  const track = [...tracks][0];
  const before = deep(S.notes);
  hits.sort((a, b) => a.time - b.time);
  const beats = hits.map((h) => pBeatAt(h.time));
  const ab = Math.floor(beats[0] / BEATS_PER_BAR + 1e-9) * BEATS_PER_BAR;
  const L = Math.max(BEATS_PER_BAR, Math.ceil((beats[beats.length - 1] - ab + 1e-6) / BEATS_PER_BAR) * BEATS_PER_BAR);
  const pats = (S.notes.patterns = S.notes.patterns || {});
  let i = 1; while (pats[`${track}${i}`]) i++;
  const name = `${track}${i}`;
  pats[name] = {
    length_beats: L,
    hits: hits.map((h, j) => {
      const ph = { beat: round9(beats[j] - ab), vel: h.vel == null ? 1 : h.vel };
      if (h.len_beats != null) ph.len_beats = h.len_beats;
      if (h.shape != null) ph.shape = h.shape;
      return ph;
    }),
  };
  const drop = new Set(hits);
  const ref = { pattern: name, at_beat: ab, repeat: 1 };
  S.notes.tracks[track] = S.notes.tracks[track].filter((h) => !drop.has(h)).concat([ref]);
  sortTrack(track);
  S.sel.clear(); S.sel.add(ref);
  afterNotesEdit(before);
  msg(`pattern ${name}: ${hits.length} hits, ${L} beats. Ctrl+D places a linked copy`);
}
function copyPatterns() {
  const refs = [...S.sel].filter((o) => kindOf(o) === "ref");
  if (!refs.length) return msg("select a pattern copy (click its strip)", true);
  const before = deep(S.notes);
  S.sel.clear();
  for (const o of refs) {
    const track = (S.owner.get(o) || {}).track, p = (S.notes.patterns || {})[o.pattern];
    if (!track || !p) continue;
    const c = { pattern: o.pattern, at_beat: round9((+o.at_beat || 0) + (+p.length_beats || 4) * ((o.repeat | 0) || 1)), repeat: 1 };
    S.notes.tracks[track].push(c); sortTrack(track); S.sel.add(c);
  }
  afterNotesEdit(before);
  msg(`placed ${refs.length} linked cop${refs.length > 1 ? "ies" : "y"}`);
}
function unlinkPatterns() {
  const refs = [...S.sel].filter((o) => kindOf(o) === "ref");
  if (!refs.length) return msg("select a pattern copy (click its strip)", true);
  const before = deep(S.notes);
  S.sel.clear();
  for (const o of refs) {
    const track = (S.owner.get(o) || {}).track, p = (S.notes.patterns || {})[o.pattern];
    if (!track || !p) continue;
    const L = +p.length_beats || 4, hits = [];
    for (let r = 0; r < ((o.repeat | 0) || 1); r++) for (const ph of p.hits || []) {
      if (ph.beat < 0 || ph.beat >= L) continue;
      const h = { time: round9(pTimeAt((+o.at_beat || 0) + r * L + ph.beat)), vel: ph.vel == null ? 1 : ph.vel };
      if (ph.len_beats != null) h.len_beats = ph.len_beats;
      if (ph.shape != null) h.shape = ph.shape;
      hits.push(h); S.sel.add(h);
    }
    S.notes.tracks[track] = S.notes.tracks[track].filter((h) => h !== o).concat(hits);
    sortTrack(track);
  }
  afterNotesEdit(before);
  msg("unlinked: the hits are plain hits again");
}
function deleteSelected() {
  if (!S.sel.size) return;
  const before = deep(S.notes);
  let n = 0;
  for (const o of S.sel) {
    const k = kindOf(o), own = S.owner.get(o) || {};
    if ((k === "hit" || k === "ref") && own.track && S.notes.tracks[own.track]) {
      const l = S.notes.tracks[own.track], i = l.indexOf(o); if (i >= 0) { l.splice(i, 1); n++; }
    } else if (k === "phit" && own.pattern) {
      const l = ((S.notes.patterns || {})[own.pattern] || {}).hits || [], i = l.indexOf(o); if (i >= 0) { l.splice(i, 1); n++; }
    } else if (k === "section" && S.notes.sections) {
      const i = S.notes.sections.indexOf(o); if (i >= 0) { S.notes.sections.splice(i, 1); n++; }
    }
  }
  S.sel.clear();
  if (!n) return;
  afterNotesEdit(before);
  msg(`removed ${n} item${n > 1 ? "s" : ""}`);
}
function nudgeSelected(dir, beats) {
  const items = [...S.sel].filter((o) => { const k = kindOf(o); return k === "hit" || k === "phit" || k === "section" || k === "ref"; });
  if (!items.length) return false;
  const before = deep(S.notes), step = beats ? 1 : gridSize();
  for (const o of items) {
    const k = kindOf(o);
    if (k === "phit") {
      const p = (S.notes.patterns || {})[(S.owner.get(o) || {}).pattern];
      o.beat = round9(clamp(o.beat + dir * step, 0, (p ? +p.length_beats || 4 : 4) - 1e-6));
    } else if (k === "ref") o.at_beat = round9((+o.at_beat || 0) + dir * step);
    else o.time = +Math.max(0, timeAt(beatAt(o.time) + dir * step)).toFixed(4);
  }
  for (const tr of Object.keys(S.notes.tracks)) sortTrack(tr);
  sortSections();
  afterNotesEdit(before);
  return true;
}
function velSelected(d) {
  const items = [...S.sel].filter((o) => { const k = kindOf(o); return k === "hit" || k === "phit"; });
  if (!items.length) return;
  const before = deep(S.notes);
  for (const o of items) o.vel = +clamp((o.vel == null ? 1 : o.vel) + d, 0.01, 1).toFixed(3);
  afterNotesEdit(before);
  msg(`velocity ${items[0].vel.toFixed(2)}`);
}
// after any notes edit made in place: undo point, redraw, re-bake
function afterNotesEdit(before) {
  const tracksBefore = Object.keys(before.tracks || {}).join("\n"), tracksNow = Object.keys(S.notes.tracks || {}).join("\n");
  rebuildEx();
  if (tracksBefore !== tracksNow) buildNoteLanes();
  pushUndo("notes", before);
  selChanged();
  notesChangedViews();
  postNotes();
}
function notesChangedViews() {
  updateNoteCounts();
  schedule();
  $("#sec-count").textContent = sections().length ? String(sections().length) : "";
}

// ------------------------------------------------------------------------------------------- sections lane
function sectionAt(ctx, x) {
  for (const b of secBoxes(ctx).reverse()) if (x >= b.x - 3 && x <= b.x + b.tw + 8) return b.s;
  return null;
}
const secCv = document.querySelector('canvas[data-kind="sections"]');
secCv.addEventListener("mousedown", (e) => {
  if (e.button !== 0 || e.detail > 1) return;
  const r = secCv.getBoundingClientRect(), x = e.clientX - r.left;
  const s = sectionAt(secCv.getContext("2d"), x);
  if (!s) { S.sel.clear(); selChanged(); seek(tOf(x)); return; }
  if (e.shiftKey) { selectOnly(s, true); return; }
  selectOnly(s, false);
  const before = deep(S.notes), t0 = s.time, x0 = e.clientX;
  let changed = false;
  const move = (ev) => {
    if (!changed && Math.abs(ev.clientX - x0) < 3) return;
    s.time = +Math.max(0, snapT(t0 + (ev.clientX - x0) / S.view.pps)).toFixed(4);
    changed = true;
    notesChangedViews();
  };
  const up = () => {
    window.removeEventListener("mousemove", move);
    window.removeEventListener("mouseup", up);
    if (changed) { sortSections(); pushUndo("notes", before); postNotes(); msg(`${s.name} at ${fmt(s.time, 3)} s`); }
  };
  window.addEventListener("mousemove", move);
  window.addEventListener("mouseup", up);
});
secCv.addEventListener("dblclick", (e) => {
  const r = secCv.getBoundingClientRect(), x = e.clientX - r.left;
  const s = sectionAt(secCv.getContext("2d"), x);
  if (s) renameSection(s);
  else addSection(snapT(tOf(x)), true);
});
function addSection(t, rename) {
  const before = deep(S.notes);
  const secs = (S.notes.sections = S.notes.sections || []);
  let i = secs.length + 1; while (secs.some((s) => s.name === `section ${i}`)) i++;
  const s = { name: `section ${i}`, time: +clamp(t, S.tStart, S.duration).toFixed(4) };
  secs.push(s); sortSections();
  S.sel.clear(); S.sel.add(s);
  afterNotesEdit(before);
  if (rename) renameSection(s);
}
function renameSection(s) {
  const inp = $("#rename");
  const lr = secCv.getBoundingClientRect(), tr = timeline.getBoundingClientRect();
  inp.style.left = `${lr.left - tr.left + clamp(xOf(s.time), 0, lr.width - 140)}px`;
  inp.style.top = `${lr.top - tr.top + timeline.scrollTop + 1}px`;
  inp.value = s.name || "";
  inp.hidden = false; inp.focus(); inp.select();
  const done = (ok) => {
    inp.onkeydown = inp.onblur = null; inp.hidden = true;
    const name = inp.value.trim();
    if (!ok || !name || name === s.name) return;
    if (sections().some((o) => o !== s && o.name === name)) return msg(`a section is already called ${name}`, true);
    const before = deep(S.notes), old = s.name;
    // layer keys that name the section follow the rename, in the same bake (either half alone is invalid)
    const rig = deep(S.rig);
    let rigChanged = false;
    for (const L of rig.layers || []) for (const k of L.intensity || []) if (k.section === old) { k.section = name; rigChanged = true; }
    s.name = name;
    secLabelW.delete(name);
    if (!rigChanged) return afterNotesEdit(before);
    pushUndo("both", { rig: deep(S.rig), notes: before });
    S.rig = rig;
    rebuildEx(); selChanged(); notesChangedViews(); buildLayerLanes();
    queueBake({ rig: true, notes: true, note: `renamed ${old} to ${name}; layer keys follow` });
  };
  inp.onkeydown = (e) => { e.stopPropagation(); if (e.key === "Enter") done(true); else if (e.key === "Escape") done(false); };
  inp.onblur = () => done(true);
}

// ------------------------------------------------------------------------------------------- layer lanes
function layerKeys(L) {
  const secs = {};
  for (const s of sections()) secs[s.name] = s.time;
  const keys = [];
  for (const k of L.intensity || []) {
    if (k.frame != null) keys.push({ ...k, frame: +k.frame });
    else if (k.section != null && secs[k.section] != null) keys.push({ ...k, frame: hitFrame(secs[k.section]) });
  }
  return keys.sort((a, b) => a.frame - b.frame);
}
function ease(name, x) {
  x = clamp(x, 0, 1);
  if (name === "in") return x * x;
  if (name === "out") return 1 - (1 - x) * (1 - x);
  if (name === "linear") return x;
  return x * x * (3 - 2 * x);
}
// port of beatrig.bake.layer_curve at one frame: the last key at or before f fades from the key before it
function layerAt(keys, f) {
  if (!keys.length) return 1;
  let lo = 0, hi = keys.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (keys[m].frame <= f) lo = m + 1; else hi = m; }
  const j = lo - 1;
  if (j < 1) return +keys[0].value;
  const k = keys[j], cur = +keys[j - 1].value, fade = +k.fade_frames || 0;
  const x = fade > 0 ? (f - k.frame) / fade : 1;
  return cur + (+k.value - cur) * ease(k.ease || "in_out", x);
}
function drawLayer(ctx, w, h, id) {
  ctx.fillStyle = C.bg; ctx.fillRect(0, 0, w, h);
  drawGrid(ctx, w, h, 0, 0.5);
  drawSectionLines(ctx, w, h);
  const L = ((S.rig && S.rig.layers) || []).find((l) => l.id === id);
  if (!L) return;
  const keys = layerKeys(L), top = 3, H = h - 6;
  ctx.beginPath(); ctx.moveTo(0, top + H);
  for (let x = 0; x <= w; x++) ctx.lineTo(x, top + H - clamp(layerAt(keys, fxOf(x)), 0, 1) * H);
  ctx.lineTo(w, top + H); ctx.closePath();
  ctx.fillStyle = "rgba(205,208,213,0.10)"; ctx.fill();
  ctx.beginPath();
  for (let x = 0; x <= w; x++) { const y = top + H - clamp(layerAt(keys, fxOf(x)), 0, 1) * H; if (x) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
  ctx.strokeStyle = C.dim; ctx.lineWidth = 1; ctx.stroke();
  // key markers; keys tied to a section are drawn in the accent colour
  for (const k of keys) {
    const x0 = xOfFrame(k.frame), x1 = xOfFrame(k.frame + (+k.fade_frames || 0));
    if (x1 < -4 || x0 > w + 4) continue;
    ctx.fillStyle = k.section != null ? C.accent : C.text;
    ctx.fillRect(Math.round(x0) - 1, 0, 3, 3);
    if (x1 - x0 > 2) { ctx.globalAlpha = 0.5; ctx.fillRect(x0, 0, x1 - x0, 1); ctx.globalAlpha = 1; }
  }
  shadeEnd(ctx, w, h);
}

// ------------------------------------------------------------------------------------------- curve lanes
function outputById(id) { return S.bake && (S.bake.outputs || []).find((o) => o.id === id); }
function targetSummary(t) {
  if (!t) return "";
  const n = (t.actors || []).length, actors = n ? (n === 1 && String(t.actors[0])[0] === "@" ? t.actors[0] : `x${n}`) : "";
  if (t.kind === "light") return `light ${actors} ${t.property || "intensity"}`;
  if (t.kind === "mpc") return `mpc ${t.param || ""}`;
  if (t.kind === "material") return `material ${actors} ${t.param || ""}`;
  if (t.kind === "niagara") return `niagara ${actors} ${t.param || ""}`;
  if (t.kind === "transform") return `transform ${actors} ${t.property || ""}.${t.axis || ""}`;
  return `${t.kind || ""} ${t.param || t.property || ""}`;
}
const isColor = (o) => o && o.components === 4;
function curveHeight(o) {
  const n = o.channels.length;
  if (isColor(o)) return clamp(n * 14, 24, 220);
  if (S.stacked.has(o.id) && n > 1) return clamp(n * 13, 48, 260);
  return 78;
}
function fitLaneHeights() {
  const cvs = [...document.querySelectorAll(".lane.curve:not(.collapsed) canvas")];
  for (const cv of cvs) cv.style.height = curveHeight(outputById(cv.dataset.id)) + "px";
  if (!cvs.length) return;
  const content = $("#curve-lanes").getBoundingClientRect().bottom - lanesEl.getBoundingClientRect().top;
  const free = timeline.clientHeight - content - 1;
  if (free > 0) {
    const extra = free / cvs.length;
    for (const cv of cvs) {
      const o = outputById(cv.dataset.id), base = curveHeight(o);
      const cap = isColor(o) ? o.channels.length * 24 : S.stacked.has(o.id) && o.channels.length > 1 ? o.channels.length * 36 : 240;
      cv.style.height = Math.floor(Math.max(base, Math.min(cap, base + extra))) + "px";
    }
  }
  for (const cv of cvs) schedule(cv);
}
function drawCurve(ctx, w, h, id) {
  ctx.fillStyle = C.bg; ctx.fillRect(0, 0, w, h);
  const o = outputById(id);
  if (!o) return;
  const color = isColor(o);
  if (!color) drawGrid(ctx, w, h, 0, 0.7);
  drawSectionLines(ctx, w, h);
  const s = S.series.get("o:" + id);
  const n = o.channels.length;
  if (!s) { ctx.fillStyle = C.faint; ctx.font = F11; ctx.textBaseline = "middle"; ctx.fillText("loading", 8, h / 2); return; }
  const lo = o.min ?? 0, hi = o.max ?? 1;
  if (color) drawColors(ctx, s, w, h);
  else if (S.stacked.has(id) && n > 1) {
    const rowH = h / n;
    for (let i = 0; i < n; i++) {
      const top = i * rowH;
      ctx.fillStyle = C.line; ctx.fillRect(0, Math.round(top + rowH) - 1, w, 1);
      drawValues(ctx, s, i, w, top + 1.5, rowH - 3, lo, hi, K.rows, true);
    }
  } else {
    ctx.globalAlpha = n > 1 ? 0.75 : 1;
    for (let i = 0; i < n; i++) drawValues(ctx, s, i, w, 4, h - 8, lo, hi, C.accent, false);
    ctx.globalAlpha = 1;
  }
  // flash-safety windows on this output
  const fl = S.flags && S.flags[id];
  if (fl) for (const [a, b] of fl) {
    const x0 = xOf(a), x1 = xOf(b);
    if (x1 < 0 || x0 > w) continue;
    ctx.fillStyle = K.flash; ctx.fillRect(x0, 0, Math.max(1, x1 - x0), h);
    ctx.fillStyle = C.err; ctx.fillRect(x0, 0, Math.max(1, x1 - x0), 2);
  }
  shadeEnd(ctx, w, h);
}
function buildCurveLanes() {
  const host = $("#curve-lanes");
  host.textContent = "";
  for (const o of (S.bake && S.bake.outputs) || []) {
    // multi-channel outputs start as one row per channel (chases and waves read better that way)
    if (!S.seenOutputs.has(o.id)) { S.seenOutputs.add(o.id); if (o.channels.length > 1) S.stacked.add(o.id); }
    const lane = document.createElement("div");
    lane.className = "lane curve" + (S.collapsed.has(o.id) ? " collapsed" : "");
    const head = document.createElement("div");
    head.className = "lane-head";
    const nch = o.channels.length, keys = o.channels.reduce((a, c) => a + (c.key_count || 0), 0);
    head.innerHTML = `<div class="row"><button class="chev" title="Collapse"><svg viewBox="0 0 10 10" width="10" height="10"><path d="M1 3h8L5 8z"/></svg></button><span class="name"></span></div>
      <small class="dim tgt"></small>
      <div class="row"><small class="dim rng"></small>${nch > 1 && !isColor(o) ? '<button class="tog" title="Overlay or one row per channel">Rows</button>' : ""}</div>
      <small class="flag" hidden></small>`;
    head.querySelector(".name").textContent = o.id;
    head.querySelector(".tgt").textContent = targetSummary(o.target);
    head.querySelector(".rng").textContent = isColor(o) ? `${nch} ch colour` : `${nch} ch  ${o.min}–${o.max}`;
    head.title = `${o.id}: ${keys} keys after simplification`;
    const fl = S.flags && S.flags[o.id];
    if (fl) { const f = head.querySelector(".flag"); f.hidden = false; f.textContent = `${fl.length} flash window${fl.length > 1 ? "s" : ""}`; }
    const cv = document.createElement("canvas");
    cv.dataset.kind = "curve"; cv.dataset.id = o.id;
    cv.style.height = curveHeight(o) + "px";
    head.querySelector(".chev").onclick = () => {
      if (S.collapsed.has(o.id)) S.collapsed.delete(o.id); else S.collapsed.add(o.id);
      lane.classList.toggle("collapsed");
      fitLaneHeights();
    };
    const tog = head.querySelector(".tog");
    if (tog) {
      tog.classList.toggle("on", S.stacked.has(o.id));
      tog.onclick = () => {
        if (S.stacked.has(o.id)) S.stacked.delete(o.id); else S.stacked.add(o.id);
        tog.classList.toggle("on", S.stacked.has(o.id));
        fitLaneHeights();
      };
    }
    attachSeek(cv, false);
    lane.append(head, cv);
    host.append(lane);
    canvases.add(cv);
  }
  pruneCanvases();
  fitLaneHeights();
}
function buildNoteLanes() {
  const host = $("#note-lanes");
  host.textContent = "";
  const tracks = Object.keys((S.notes && S.notes.tracks) || {});
  for (const t of tracks) {
    const lane = document.createElement("div");
    lane.className = "lane notes";
    lane.dataset.track = t;
    const head = document.createElement("div");
    head.className = "lane-head";
    head.innerHTML = `<div class="row"><span class="name"></span><span class="keys" hidden></span></div><small class="dim cnt"></small>`;
    head.querySelector(".name").textContent = t;
    const cv = document.createElement("canvas");
    cv.dataset.kind = "notes"; cv.dataset.track = t;
    lane.append(head, cv);
    attachNoteLane(cv, t);
    host.append(lane);
    canvases.add(cv);
  }
  if (!tracks.length) host.innerHTML = '<div class="group-head dim">notes.json has no tracks</div>';
  pruneCanvases();
  updateNoteCounts();
  scheduleNotes();
}
function buildBandLanes() {
  const host = $("#band-lanes");
  const ids = (S.bands || []).map((b) => b.id).join("\n");
  if (host.dataset.ids === ids) return;
  host.dataset.ids = ids;
  host.textContent = "";
  for (const b of S.bands || []) {
    const lane = document.createElement("div");
    lane.className = "lane band";
    const head = document.createElement("div");
    head.className = "lane-head";
    head.innerHTML = `<div class="row"><span class="name"></span><small class="dim"></small></div>`;
    head.querySelector(".name").textContent = b.id;
    head.querySelector("small").textContent = b.lo != null ? `${b.lo}–${b.hi} Hz` : "";
    const cv = document.createElement("canvas");
    cv.dataset.kind = "band"; cv.dataset.id = b.id;
    attachSeek(cv, false);
    lane.append(head, cv);
    host.append(lane);
    canvases.add(cv);
  }
  pruneCanvases();
}
function buildLayerLanes() {
  const host = $("#layer-lanes");
  const layers = (S.rig && S.rig.layers) || [];
  $("#layer-group").hidden = !layers.length;
  const ids = layers.map((l) => l.id).join("\n");
  if (host.dataset.ids !== ids) {
    host.dataset.ids = ids;
    host.textContent = "";
    for (const L of layers) {
      const lane = document.createElement("div");
      lane.className = "lane layer";
      const head = document.createElement("div");
      head.className = "lane-head";
      head.innerHTML = `<span class="name"></span><small class="dim"></small>`;
      head.querySelector(".name").textContent = L.id;
      const cv = document.createElement("canvas");
      cv.dataset.kind = "layer"; cv.dataset.id = L.id;
      attachSeek(cv, false);
      lane.append(head, cv);
      host.append(lane);
      canvases.add(cv);
    }
    pruneCanvases();
  }
  host.querySelectorAll(".lane").forEach((lane, i) => {
    const L = layers[i], n = (L.intensity || []).length, ns = (L.intensity || []).filter((k) => k.section != null).length;
    const sm = lane.querySelector("small");
    sm.textContent = `${n} key${n === 1 ? "" : "s"}${ns ? `, ${ns} on sections` : ""}`;
    sm.title = sm.textContent;
  });
  scheduleKind('canvas[data-kind="layer"]');
}
function pruneCanvases() { for (const c of [...canvases]) if (!c.isConnected) canvases.delete(c); }
function updateNoteCounts() {
  document.querySelectorAll(".lane.notes").forEach((lane) => {
    const t = lane.dataset.track, E = S.ex[t];
    const nref = E ? E.refs.length : 0, nhit = E ? E.ex.length : 0;
    lane.querySelector(".cnt").textContent = `${nhit} hits${nref ? `, ${nref} pattern cop${nref > 1 ? "ies" : "y"}` : ""}`;
    const keys = [];
    for (let d = 1; d <= 9; d++) if (trackForKey(d) === t) keys.push(d);
    const k = lane.querySelector(".keys");
    k.hidden = !keys.length; k.textContent = keys.join(" ");
    k.title = `tap-record keys for ${t} (Shift+digit over a lane reassigns)`;
  });
}

// ------------------------------------------------------------------------------------------- preview: ring, plan, dome
const schem = $("#schem-canvas");
const CB = new Float32Array(4), C8 = new Uint8Array(3);
function buildActorMap() {
  const m = new Map();
  for (const o of (S.bake && S.bake.outputs) || []) {
    const t = o.target || {}, col = isColor(o);
    const bright = !col && ((t.kind === "light" && (t.property || "intensity") === "intensity") || t.kind === "material" || t.kind === "niagara");
    if (!col && !bright) continue;
    o.channels.forEach((ch, c) => {
      if (!ch.actor) return;
      const e = m.get(ch.actor) || { s: [], c: [] };
      (col ? e.c : e.s).push({ o, c });
      m.set(ch.actor, e);
    });
  }
  S.actorMap = m;
}
// display colour of an actor at frame f (sRGB bytes in C8); false when nothing drives it
function actorColor(a, f) {
  const e = S.actorMap.get(a);
  if (!e) return false;
  let b = -1;
  for (const { o, c } of e.s) {
    const s = S.series.get("o:" + o.id); if (!s) continue;
    const lo = o.min ?? 0, hi = o.max ?? 1;
    b = Math.max(b, clamp((s.at(c, f) - lo) / ((hi - lo) || 1), 0, 1));
  }
  let r = C.warm[0], g = C.warm[1], bl = C.warm[2], got = false;
  for (const { o, c } of e.c) {
    const s = S.series.get("o:" + o.id); if (!s) continue;
    s.rgba(c, f, CB); r = CB[0]; g = CB[1]; bl = CB[2]; got = true; break;
  }
  if (b < 0 && !got) return false;
  toSRGB(r, g, bl, C8, b < 0 ? 1 : 0.04 + 0.96 * b);
  return true;
}
function schemHeight() {
  const w = schem.getBoundingClientRect().width || 320;
  if (S.view3 !== "ring") return Math.round(Math.min(w, 340));
  let h = 8;
  for (const o of (S.bake && S.bake.outputs) || []) h += o.channels.some((c) => c.actor) && (o.target || {}).kind !== "transform" ? 150 : 26;
  return Math.max(60, h + 4);
}
const schemFrameEl = $("#schem-frame");
// The preview at the playhead: the WebGL2 scene when the browser has it, else the 2D views. force = false (playback,
// seeks) skips the redraw when the frame and the view haven't changed.
function drawSchem(force = true) {
  const t0 = performance.now();
  const f = frameOf(curT());
  if (f !== S.schemF) { S.schemF = f; schemFrameEl.textContent = `frame ${f}`; }
  if (P3.away) return;                                               // shown in the pop-out window instead
  if (P3.r) { if (!draw3D(f, force)) return; }
  else {
    const want = schemHeight() + "px";
    if (schem.style.height !== want) { schem.style.height = want; S.geo = null; }
    const { ctx, w, h } = setupCanvas(schem);
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, w, h);
    if (S.view3 === "ring") drawRing(ctx, w, f);
    else drawSpace(ctx, w, h, f, S.view3);
  }
  perfPush(S.perf.schem, performance.now() - t0);
}
function drawRing(ctx, w, f) {
  ctx.textBaseline = "middle";
  let y = 8;
  for (const o of (S.bake && S.bake.outputs) || []) {
    const s = S.series.get("o:" + o.id), col = isColor(o);
    const lo = o.min ?? 0, hi = o.max ?? 1, span = hi - lo || 1;
    const n = o.channels.length;
    if (o.channels.some((c) => c.actor) && (o.target || {}).kind !== "transform") {
      const cx = w / 2, cy = y + 74, R = 52;
      ctx.font = F11; ctx.fillStyle = C.dim; ctx.textAlign = "left";
      ctx.fillText(o.id, 10, y + 6);
      ctx.textAlign = "right"; ctx.fillStyle = C.faint;
      ctx.fillText(col ? "colour" : (o.target.property || o.target.param || ""), w - 10, y + 6);
      ctx.strokeStyle = C.line; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const a = -Math.PI / 2 + (i / n) * Math.PI * 2;
        const x = cx + Math.cos(a) * R, yy = cy + Math.sin(a) * R;
        if (!s) ctx.fillStyle = C.line2;
        else if (col) { s.rgba(i, f, CB); ctx.fillStyle = rgbCss(toSRGB(CB[0], CB[1], CB[2], C8)); }
        else {
          const v = s.at(i, f); sum += v;
          const b = clamp((v - lo) / span, 0, 1);
          ctx.fillStyle = rgbCss(toSRGB(C.warm[0], C.warm[1], C.warm[2], C8, 0.03 + 0.97 * b));
        }
        ctx.beginPath(); ctx.arc(x, yy, n > 24 ? 4 : 8, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = C.line2; ctx.stroke();
      }
      ctx.textAlign = "center"; ctx.fillStyle = C.dim; ctx.font = F11;
      if (s && !col) ctx.fillText(fmtVal(sum / (n || 1)), cx, cy);
      y += 150;
    } else {
      const label = (o.target && (o.target.param || o.target.property)) || o.id;
      ctx.font = F11; ctx.textAlign = "left"; ctx.fillStyle = C.dim;
      ctx.fillText(label, 10, y + 10);
      const bx = 84, bw = w - bx - 64;
      ctx.textAlign = "right"; ctx.fillStyle = C.text; ctx.font = F11;
      if (s && col) {
        s.rgba(0, f, CB);
        ctx.fillStyle = rgbCss(toSRGB(CB[0], CB[1], CB[2], C8)); ctx.fillRect(bx, y + 4, bw, 12);
        ctx.fillStyle = C.text; ctx.fillText(`${fmt(CB[0], 2)} ${fmt(CB[1], 2)} ${fmt(CB[2], 2)}`, w - 10, y + 10);
      } else {
        const v = s ? s.at(0, f) : lo, b = clamp((v - lo) / span, 0, 1);
        ctx.fillStyle = C.line; ctx.fillRect(bx, y + 6, bw, 8);
        ctx.fillStyle = C.accent; ctx.fillRect(bx, y + 6, Math.round(bw * b), 8);
        ctx.fillStyle = C.text; ctx.fillText(fmtVal(v), w - 10, y + 10);
      }
      y += 26;
    }
  }
  ctx.textAlign = "left";
}
function fmtVal(v) {
  const a = Math.abs(v);
  return a >= 1000 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : v.toFixed(3);
}
// UE rotator (pitch, yaw, roll in degrees; Z up, left-handed) -> forward, right, up unit vectors (FRotationMatrix)
function rotBasis(rot) {
  const [p, y, r] = (rot || [0, 0, 0]).map((d) => (+d || 0) * Math.PI / 180);
  const SP = Math.sin(p), CP = Math.cos(p), SY = Math.sin(y), CY = Math.cos(y), SR = Math.sin(r), CR = Math.cos(r);
  return {
    F: [CP * CY, CP * SY, SP],
    R: [SR * SP * CY - CR * SY, SR * SP * SY + CR * CY, -SR * CP],
    U: [-(CR * SP * CY + SR * SY), CY * SR - CR * SP * SY, CR * CP],
  };
}
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
// The fisheye's axes for a scene camera: centre (view direction), screen right, screen down.
function domeAxes(cam) {
  const b = rotBasis(cam.rotation);
  if (String(cam.projection || "").startsWith("dome")) return { c: b.U, x: b.R, y: b.F.map((v) => v * DOME_FORWARD_DOWN), dome: true };
  return { c: b.F, x: b.R, y: b.U.map((v) => -v), dome: false };
}
// Equidistant fisheye: radius proportional to the angle from the centre axis, the circle's edge at dome_angle / 2.
// Returns [x, y] in -1..1 units of the circle radius, or null when the point is outside the dome.
// domeProjectTo writes into out (no allocation, for the 3D view's frame loop); domeProject returns a new pair.
function domeProjectTo(lx, ly, lz, cam, ax, half, out) {
  const d0 = lx - cam.location[0], d1 = ly - cam.location[1], d2 = lz - cam.location[2];
  const len = Math.hypot(d0, d1, d2);
  if (len < 1e-6) return false;
  const x = (d0 * ax.x[0] + d1 * ax.x[1] + d2 * ax.x[2]) / len, y = (d0 * ax.y[0] + d1 * ax.y[1] + d2 * ax.y[2]) / len;
  const z = (d0 * ax.c[0] + d1 * ax.c[1] + d2 * ax.c[2]) / len;
  const theta = Math.acos(clamp(z, -1, 1));
  if (theta > half + 1e-9) return false;
  const rr = theta / half, q = Math.hypot(x, y) || 1;
  out[0] = rr * x / q; out[1] = rr * y / q;
  return true;
}
function domeProject(loc, cam, ax, half) {
  const out = [0, 0];
  return domeProjectTo(loc[0], loc[1], loc[2], cam, ax, half, out) ? out : null;
}
function sceneGeo(w, h, mode) {
  const g = S.geo;
  if (g && g.w === w && g.h === h && g.mode === mode && g.scene === S.scene && g.bake === S.bake) return g;
  const out = { w, h, mode, scene: S.scene, bake: S.bake, pts: [], note: "" };
  S.geo = out;
  const sc = S.scene, actors = (sc && sc.actors) || {};
  const names = Object.keys(actors).filter((a) => Array.isArray(actors[a].location));
  if (!sc) { out.note = "No scene.json. Scan reads it from the editor."; return out; }
  const cam = sc.camera;
  if (mode === "dome") {
    if (!cam || !Array.isArray(cam.location)) { out.note = "scene.json has no camera"; return out; }
    const ax = domeAxes(cam), R = Math.min(w, h) / 2 - 12;
    out.angle = clamp(+cam.dome_angle || 180, 10, 360);
    out.cx = w / 2; out.cy = h / 2; out.R = R; out.domeCam = ax.dome;
    for (const a of names) {
      const p = domeProject(actors[a].location, cam, ax, (out.angle / 2) * Math.PI / 180);
      if (p) out.pts.push({ a, x: out.cx + p[0] * R, y: out.cy + p[1] * R, driven: S.actorMap.has(a) });
    }
    out.label = `${cam.actor || "camera"}  ${fmt(out.angle, 0)}°  pitch ${fmt(+(cam.rotation || [0])[0] || 0, 0)}` +
      (ax.dome ? "" : "  (not a domemaster: fisheye along its forward axis)");
    return out;
  }
  // plan: top-down, X up the screen, Y to the right; fitted to the driven actors (or all) plus the camera
  const driven = names.filter((a) => S.actorMap.has(a));
  const fit = (driven.length ? driven : names).map((a) => actors[a].location);
  if (cam && Array.isArray(cam.location)) fit.push(cam.location);
  if (!fit.length) { out.note = "scene.json has no actor positions"; return out; }
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const p of fit) { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); }
  const span = Math.max(x1 - x0, y1 - y0, 1), k = (Math.min(w, h) - 36) / span;
  const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
  const P = (p) => [w / 2 + (p[1] - my) * k, h / 2 - (p[0] - mx) * k];
  for (const a of names) { const [x, y] = P(actors[a].location); out.pts.push({ a, x, y, driven: S.actorMap.has(a) }); }
  if (cam && Array.isArray(cam.location)) {
    const [x, y] = P(cam.location), yaw = (+(cam.rotation || [0, 0])[1] || 0) * Math.PI / 180;
    out.cam = { x, y, dx: Math.sin(yaw), dy: -Math.cos(yaw) };
  }
  out.k = k;
  out.label = `${names.length} actors, ${driven.length} driven`;
  return out;
}
function drawSpace(ctx, w, h, f, mode) {
  const g = sceneGeo(w, h, mode);
  ctx.font = F11; ctx.textBaseline = "middle";
  if (g.note) { ctx.fillStyle = C.dim; ctx.textAlign = "center"; ctx.fillText(g.note, w / 2, h / 2); ctx.textAlign = "left"; return; }
  if (mode === "dome") {
    ctx.strokeStyle = C.line2; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(g.cx, g.cy, g.R, 0, Math.PI * 2); ctx.stroke();
    const half = g.angle / 2;
    ctx.strokeStyle = C.line;
    for (const deg of [30, 60, 90, 120, 150]) if (deg < half) { ctx.beginPath(); ctx.arc(g.cx, g.cy, g.R * deg / half, 0, Math.PI * 2); ctx.stroke(); }
    ctx.beginPath(); ctx.moveTo(g.cx - g.R, g.cy); ctx.lineTo(g.cx + g.R, g.cy); ctx.moveTo(g.cx, g.cy - g.R); ctx.lineTo(g.cx, g.cy + g.R); ctx.stroke();
    ctx.fillStyle = C.faint; ctx.textAlign = "center";
    if (g.domeCam) ctx.fillText("front", g.cx, DOME_FORWARD_DOWN > 0 ? g.cy + g.R + 7 : g.cy - g.R - 7);
    ctx.textAlign = "left";
  } else {
    // 1 m grid when it is readable, else 10 m
    const step = g.k * 100 >= 10 ? 100 : 1000, px = step * g.k;
    if (px >= 6) {
      ctx.fillStyle = "rgba(150,155,165,0.06)";
      for (let x = (w / 2) % px; x < w; x += px) ctx.fillRect(Math.round(x), 0, 1, h);
      for (let y = (h / 2) % px; y < h; y += px) ctx.fillRect(0, Math.round(y), w, 1);
    }
    if (g.cam) {
      const c = g.cam;
      ctx.fillStyle = C.dim;
      ctx.beginPath(); ctx.moveTo(c.x + c.dx * 9, c.y + c.dy * 9); ctx.lineTo(c.x - c.dy * 5 - c.dx * 4, c.y + c.dx * 5 - c.dy * 4); ctx.lineTo(c.x + c.dy * 5 - c.dx * 4, c.y - c.dx * 5 - c.dy * 4); ctx.closePath(); ctx.fill();
    }
  }
  ctx.lineWidth = 1;
  for (const p of g.pts) {
    if (!p.driven || !actorColor(p.a, f)) { ctx.fillStyle = C.faint; ctx.fillRect(p.x - 1, p.y - 1, 2, 2); continue; }
    ctx.fillStyle = rgbCss(C8);
    ctx.beginPath(); ctx.arc(p.x, p.y, 4.5, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = C.line2; ctx.stroke();
  }
  ctx.fillStyle = C.faint; ctx.fillText(g.label || "", 8, 10);
}
// preview mode: orbit / camera / dome with WebGL2, ring / plan / dome without (an old stored mode maps to the first)
function setView3(v) {
  const views = P3.r ? VIEWS_GL : VIEWS_2D;
  if (!views.includes(v)) v = views[0];
  S.view3 = v;
  lsSet(POPOUT ? "view3:popout" : "view3", v);
  document.querySelectorAll("#schem .tabs button").forEach((b) => b.classList.toggle("on", b.dataset.view === v));
  S.geo = null;
  if (P3.r) { P3.dirty = true; P3.ovStale = true; $("#schem-tip").hidden = true; hud3(); }
  drawSchem();
}
function cycleView3() {
  const views = P3.r ? VIEWS_GL : VIEWS_2D;
  for (let k = 1; k <= views.length; k++) {
    const v = views[(views.indexOf(S.view3) + k) % views.length], b = document.querySelector(`#schem .tabs button[data-view="${v}"]`);
    if (b && !b.disabled) return setView3(v);
  }
}
document.querySelectorAll("#schem .tabs button").forEach((b) => { b.onclick = () => setView3(b.dataset.view); });
function sceneInfo() {
  const sc = S.scene, el = $("#scene-info");
  if (!sc) { el.textContent = "no scene.json"; el.title = ""; }
  else {
    const n = Object.keys(sc.actors || {}).length;
    el.textContent = `scene: ${n} actors${sc.camera ? `, camera ${sc.camera.actor || ""}` : ""}${sc.level ? `, ${sc.level.split("/").pop()}` : ""}`;
    el.title = sc.level || "";
  }
  const dl = $("#seq-list");
  dl.textContent = "";
  for (const s of (sc && sc.sequences) || []) { const o = document.createElement("option"); o.value = s; dl.append(o); }
}

// ------------------------------------------------------------------------------------------- preview: 3D (WebGL2)
// Every scene.json actor an output drives, at its real position: lights as glow sprites with a pool of light on the
// floor, material/Niagara-driven meshes as boxes that glow, transform outputs moving, turning and scaling them, MPC
// outputs as an ambient tint. Brightness and colour are the baked values at the playhead frame (no interpolation).
// Modes: orbit (drag, right/Shift-drag, wheel, double-click frames), camera (through scene.camera) and dome (the
// domemaster, placed with domeProjectTo: the same projection the 2D dome view uses). The model is rebuilt when the
// scene or the bake changes; the per-frame path writes into preallocated arrays and allocates nothing.
const GL = window.BeatrigGL;
const glCv = $("#schem-gl"), ovCv = $("#schem-overlay"), meterCv = $("#schem-meters"), schemView = $("#schem-view");
const KIND_OTHER = 0, KIND_LIGHT = 1, KIND_MESH = 2, KIND_FX = 3;
const P3 = {
  r: null, m: null, w: 0, h: 0, mw: 0, dpr: 1, lastF: -1, dirty: true, ovStale: true, meterStale: true, meterF: -1, meterRows: [], domeG: null,
  orbit: { yaw: 0, pitch: -30, dist: 2000, t: [0, 0, 0], user: false, scene: null },
  drag: false,
};
const VFOV = 40 * Math.PI / 180;
const F3 = new Float32Array(3), R3 = new Float32Array(3), U3 = new Float32Array(3), EYE = new Float32Array(3), PXY = new Float32Array(2);
const AXIS = { x: 0, y: 1, z: 2 }, ROT_AXIS = { pitch: 0, yaw: 1, roll: 2, y: 0, z: 1, x: 2 };   // scene rotation = [pitch, yaw, roll]
function gridStep(v) { for (const s of [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]) if (s >= v) return s; return 100000; }

function buildModel() {
  const sc = S.scene, actors = (sc && sc.actors) || {}, cam = sc && sc.camera && Array.isArray(sc.camera.location) ? sc.camera : null;
  const names = [], kinds = [], base = [], rot0 = [], cones = [], idx = new Map();
  for (const a of Object.keys(actors)) {
    const A = actors[a];
    if (!Array.isArray(A.location) || (cam && a === cam.actor)) continue;
    const ks = new Set((A.components || []).map((c) => c.kind));
    idx.set(a, names.length); names.push(a);
    kinds.push(ks.has("light") ? KIND_LIGHT : ks.has("niagara") ? KIND_FX : ks.has("mesh") ? KIND_MESH : KIND_OTHER);
    // a spot light: a beam along the actor's forward axis. The scan has no cone angle yet; read one if a scene file
    // carries it (outer_cone_angle / cone_angle, the half-angle in degrees), else Unreal's default outer cone, 44
    const spot = (A.components || []).find((c) => c.kind === "light" && /spot/i.test(String(c.class || "")));
    cones.push(spot ? clamp(+(spot.outer_cone_angle ?? spot.cone_angle ?? 44) || 44, 1, 80) : 0);
    base.push(+A.location[0] || 0, +A.location[1] || 0, +A.location[2] || 0);
    const r = A.rotation || [0, 0, 0];
    rot0.push(+r[0] || 0, +r[1] || 0, +r[2] || 0);
  }
  const n = names.length;
  const m = {
    scene: sc, bake: S.bake, cam, names, idx, n, kind: Uint8Array.from(kinds), base: Float32Array.from(base), rot0: Float32Array.from(rot0),
    coneTan: Float32Array.from(cones.map((d) => (d ? Math.tan(d * Math.PI / 180) : 0))),
    bri: names.map(() => []), col: names.map(() => []), xf: names.map(() => []), outs: names.map(() => []),
    mpcS: [], mpcC: [], driven: new Uint8Array(n),
    b: new Float32Array(n), rgb: new Float32Array(n * 3), pos: new Float32Array(n * 3), rot: new Float32Array(n * 3), scl: new Float32Array(n * 3),
    tint: new Float32Array(3), amb: 0,
  };
  for (const o of (S.bake && S.bake.outputs) || []) {
    const t = o.target || {}, col = isColor(o), key = "o:" + o.id, lo = o.min ?? 0, hi = o.max ?? 1, span = (hi - lo) || 1;
    if (t.kind === "mpc") { (col ? m.mpcC : m.mpcS).push({ key, lo, span }); continue; }
    o.channels.forEach((ch, c) => {
      const i = idx.get(ch.actor);
      if (i == null) return;
      m.outs[i].push({ o, c });
      if (t.kind === "transform") {
        const prop = { location: 0, rotation: 1, scale: 2 }[t.property], ax = (prop === 1 ? ROT_AXIS : AXIS)[String(t.axis || "").toLowerCase()];
        if (!col && prop != null && ax != null) m.xf[i].push({ key, c, prop, ax, add: t.additive !== false });
        return;
      }
      if (col) m.col[i].push({ key, c });
      else if ((t.kind === "light" && (t.property || "intensity") === "intensity") || t.kind === "material" || t.kind === "niagara") m.bri[i].push({ key, c, lo, span });
    });
  }
  // framing: the driven actors (all of them when nothing is driven) and the camera
  for (let i = 0; i < n; i++) m.driven[i] = m.bri[i].length || m.col[i].length || m.xf[i].length ? 1 : 0;
  const anyDriven = m.driven.some((d) => d), fit = [];
  for (let i = 0; i < n; i++) if (m.driven[i] || !anyDriven) fit.push([base[i * 3], base[i * 3 + 1], base[i * 3 + 2]]);
  if (cam) fit.push(cam.location.map((v) => +v || 0));
  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const p of fit) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], p[k]); hi[k] = Math.max(hi[k], p[k]); }
  if (!fit.length) { lo = [-500, -500, 0]; hi = [500, 500, 0]; }
  m.centre = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
  m.radius = 100;
  for (const p of fit) m.radius = Math.max(m.radius, Math.hypot(p[0] - m.centre[0], p[1] - m.centre[1], p[2] - m.centre[2]));
  m.unit = clamp(m.radius * 0.03, 6, 400);
  m.floorZ = lo[2];
  for (let i = 0; i < n; i++) m.floorZ = Math.min(m.floorZ, base[i * 3 + 2]);
  m.nDriven = 0;
  for (let i = 0; i < n; i++) m.nDriven += m.driven[i];
  if (cam) { m.domeAx = domeAxes(cam); m.domeAngle = clamp(+cam.dome_angle || 180, 10, 360); }
  P3.r.reserve(n * 3 + 8, n + 4, n + 4, n + 4);
  P3.r.setLines(sceneLines(m));
  return m;
}
// ground grid (fading out from the framed centre) + the scene camera glyph, as one static line buffer
function sceneLines(m) {
  const L = [], step = gridStep(m.radius / 5), half = Math.ceil(m.radius * 1.5 / step) * step, z = m.floorZ;
  const cx = Math.round(m.centre[0] / step) * step, cy = Math.round(m.centre[1] / step) * step;
  const push = (x0, y0, z0, x1, y1, z1, c, a) => L.push(x0, y0, z0, c, c, c, a, x1, y1, z1, c, c, c, a);
  for (let k = -half; k <= half + 1e-6; k += step) {
    const major = Math.abs(Math.round((cx + k) / step)) % 5 === 0, origin = Math.abs(cx + k) < 1e-6;
    push(cx + k, cy - half, z, cx + k, cy + half, z, 0.014, origin ? 2.6 : major ? 1.8 : 1);
  }
  for (let k = -half; k <= half + 1e-6; k += step) {
    const major = Math.abs(Math.round((cy + k) / step)) % 5 === 0, origin = Math.abs(cy + k) < 1e-6;
    push(cx - half, cy + k, z, cx + half, cy + k, z, 0.014, origin ? 2.6 : major ? 1.8 : 1);
  }
  m.nGrid = L.length / GL.LINE_FLOATS;
  m.gridFade = [cx, cy, half];
  const cam = m.cam;
  if (cam) {
    const b = rotBasis(cam.rotation), e = cam.location.map((v) => +v || 0), s = m.unit * 3, c = 0.06;
    const P = (f, r, u) => [e[0] + b.F[0] * f + b.R[0] * r + b.U[0] * u, e[1] + b.F[1] * f + b.R[1] * r + b.U[1] * u, e[2] + b.F[2] * f + b.R[2] * r + b.U[2] * u];
    const seg = (p, q) => push(p[0], p[1], p[2], q[0], q[1], q[2], c, 1);
    if (String(cam.projection || "").startsWith("dome")) {
      // dome camera: its rim (the horizon of the fisheye), the zenith (actor up) and the front (actor forward)
      for (let i = 0; i < 32; i++) {
        const a0 = (i / 32) * Math.PI * 2, a1 = ((i + 1) / 32) * Math.PI * 2;
        seg(P(Math.cos(a0) * s, Math.sin(a0) * s, 0), P(Math.cos(a1) * s, Math.sin(a1) * s, 0));
      }
      seg(P(0, 0, 0), P(0, 0, s * 1.4)); seg(P(0, 0, 0), P(s * 1.6, 0, 0));
    } else {
      const d = s * 1.6, w = s * 0.8, hh = s * 0.5, k = [P(d, -w, hh), P(d, w, hh), P(d, w, -hh), P(d, -w, -hh)];
      for (let i = 0; i < 4; i++) { seg(P(0, 0, 0), k[i]); seg(k[i], k[(i + 1) % 4]); }
    }
  }
  return new Float32Array(L);
}
function frameOrbit(m) {
  const o = P3.orbit, aspect = P3.w / Math.max(1, P3.h) || 4 / 3;
  const half = Math.min(VFOV, 2 * Math.atan(Math.tan(VFOV / 2) * aspect)) / 2;
  o.t = m.centre.slice();
  o.dist = m.radius / Math.sin(half) * 0.95;
  o.scene = m.scene;
}
function model3() {
  const m = P3.m;
  if (m && m.scene === S.scene && m.bake === S.bake) return m;
  P3.m = buildModel();
  if (!P3.orbit.user || P3.orbit.scene !== S.scene) { P3.orbit.user = false; frameOrbit(P3.m); }
  P3.dirty = true; P3.ovStale = true; P3.meterStale = true; P3.domeG = null;
  hud3();
  return P3.m;
}
// the baked values of every actor at frame f -> m.b (brightness 0..1, -1 = not driven), m.rgb (linear, scaled by
// brightness), m.pos / m.rot / m.scl (with transform outputs applied); MPC outputs -> m.amb, m.tint
function evalModel(m, f) {
  for (let i = 0, j = 0; i < m.n; i++, j += 3) {
    const B = m.bri[i], Cl = m.col[i], X = m.xf[i];
    let b = -1;
    for (let k = 0; k < B.length; k++) {
      const e = B[k], s = S.series.get(e.key);
      if (!s) continue;
      const v = (s.at(e.c, f) - e.lo) / e.span;
      if (v > b) b = v < 0 ? 0 : v > 1 ? 1 : v;
    }
    let r = C.warm[0], g = C.warm[1], bl = C.warm[2], got = false;
    for (let k = 0; k < Cl.length; k++) {
      const s = S.series.get(Cl[k].key);
      if (!s) continue;
      s.rgba(Cl[k].c, f, CB); r = CB[0] > 0 ? CB[0] : 0; g = CB[1] > 0 ? CB[1] : 0; bl = CB[2] > 0 ? CB[2] : 0; got = true; break;
    }
    if (b < 0 && !got) { m.b[i] = X.length ? 0 : -1; m.rgb[j] = m.rgb[j + 1] = m.rgb[j + 2] = 0; }
    else { const k = b < 0 ? 1 : 0.04 + 0.96 * b; m.b[i] = b < 0 ? 1 : b; m.rgb[j] = r * k; m.rgb[j + 1] = g * k; m.rgb[j + 2] = bl * k; }
    m.pos[j] = m.base[j]; m.pos[j + 1] = m.base[j + 1]; m.pos[j + 2] = m.base[j + 2];
    m.rot[j] = m.rot0[j]; m.rot[j + 1] = m.rot0[j + 1]; m.rot[j + 2] = m.rot0[j + 2];
    m.scl[j] = m.scl[j + 1] = m.scl[j + 2] = 1;
    for (let k = 0; k < X.length; k++) {
      const e = X[k], s = S.series.get(e.key);
      if (!s) continue;
      const arr = e.prop === 0 ? m.pos : e.prop === 1 ? m.rot : m.scl;
      // additive (the default) is an offset on the actor's own value (scale: on 1); otherwise the value itself
      if (e.add) arr[j + e.ax] += s.at(e.c, f); else arr[j + e.ax] = s.at(e.c, f);
    }
  }
  let a = 0, na = 0;
  for (let k = 0; k < m.mpcS.length; k++) {
    const e = m.mpcS[k], s = S.series.get(e.key);
    if (s) { const v = (s.at(0, f) - e.lo) / e.span; a += v < 0 ? 0 : v > 1 ? 1 : v; na++; }
  }
  m.amb = na ? a / na : 0;
  let tr = 1, tg = 1, tb = 1;
  for (let k = 0; k < m.mpcC.length; k++) {
    const s = S.series.get(m.mpcC[k].key);
    if (s) { s.rgba(0, f, CB); tr = CB[0]; tg = CB[1]; tb = CB[2]; if (!na) m.amb = 0.5; break; }
  }
  const amb = (m.mpcS.length || m.mpcC.length) ? m.amb : 0, tm = Math.max(tr, tg, tb, 1e-6);
  m.tint[0] = tr / tm * amb; m.tint[1] = tg / tm * amb; m.tint[2] = tb / tm * amb;
}
// UE rotator (pitch, yaw, roll in degrees) -> forward, right, up into F3, R3, U3 (rotBasis without allocating)
function basisTo(p, y, r) {
  p *= Math.PI / 180; y *= Math.PI / 180; r *= Math.PI / 180;
  const SP = Math.sin(p), CP = Math.cos(p), SY = Math.sin(y), CY = Math.cos(y), SR = Math.sin(r), CR = Math.cos(r);
  F3[0] = CP * CY; F3[1] = CP * SY; F3[2] = SP;
  R3[0] = SR * SP * CY - CR * SY; R3[1] = SR * SP * SY + CR * CY; R3[2] = -SR * CP;
  U3[0] = -(CR * SP * CY + SR * SY); U3[1] = CY * SR - CR * SP * SY; U3[2] = CR * CP;
}
function setupCamera(m, mode) {
  const r = P3.r, aspect = P3.w / Math.max(1, P3.h);
  if (mode === "camera" && m.cam) {
    const c = m.cam, rot = c.rotation || [0, 0, 0];
    basisTo(+rot[0] || 0, +rot[1] || 0, +rot[2] || 0);
    EYE[0] = +c.location[0] || 0; EYE[1] = +c.location[1] || 0; EYE[2] = +c.location[2] || 0;
    const hfov = clamp(+c.fov || 90, 5, 170) * Math.PI / 180;           // Unreal's field of view is horizontal
    const far = m.radius * 8 + Math.hypot(EYE[0] - m.centre[0], EYE[1] - m.centre[1], EYE[2] - m.centre[2]);
    r.camera(EYE, F3, R3, U3, 2 * Math.atan(Math.tan(hfov / 2) / aspect), aspect, Math.max(1, m.radius * 0.002), far);
    return;
  }
  const o = P3.orbit;
  basisTo(o.pitch, o.yaw, 0);
  EYE[0] = o.t[0] - F3[0] * o.dist; EYE[1] = o.t[1] - F3[1] * o.dist; EYE[2] = o.t[2] - F3[2] * o.dist;
  r.camera(EYE, F3, R3, U3, VFOV, aspect, Math.max(0.5, o.dist * 0.005), o.dist + m.radius * 8);
}
function draw3D(f, force) {
  const r = P3.r;
  if (!S.bake) return false;                                         // nothing to draw before the first state
  if (!P3.w) sizeView3();
  const m = model3(), mode = S.view3;
  if (!force && !P3.dirty && f === P3.lastF) return false;
  P3.lastF = f; P3.dirty = false;
  r.nSprites = 0; r.nBoxes = 0; r.nDyn = 0; r.nCones = 0;
  const empty = !m.n || (mode !== "orbit" && !m.cam);
  overlay3(m, mode, empty);
  if (!empty) {
    evalModel(m, f);
    r.amb[0] = m.tint[0]; r.amb[1] = m.tint[1]; r.amb[2] = m.tint[2];
    if (mode === "dome") fillDome(m);
    else {
      setupCamera(m, mode); fillPerspective(m);
      // fog: a depth cue that keeps its strength at any zoom (scaled by the viewing distance plus the scene's size)
      const vd = mode === "orbit" ? P3.orbit.dist : Math.hypot(EYE[0] - m.centre[0], EYE[1] - m.centre[1], EYE[2] - m.centre[2]);
      r.fog = 0.55 / (vd + 1.5 * m.radius);
    }
    r.fade[0] = m.gridFade[0]; r.fade[1] = m.gridFade[1]; r.fade[2] = m.gridFade[2];
  }
  r.nGrid = m.nGrid;
  // lines: orbit draws the grid and the camera glyph, camera mode the grid only, dome none
  if (!r.render(empty ? 0 : mode === "orbit" ? 2 : mode === "camera" ? 1 : 0) && r.broken) { use2D(); drawSchem(); return false; }
  drawMeters(f);
  return true;
}
// a lamp's core: 3x its emitted colour, but never darker than a quarter of the colour itself, so a light at its
// minimum still reads as a lamp of that colour (the halo and the floor pool carry the intensity)
function coreGain(b) { return Math.max(3, 0.25 / (0.04 + 0.96 * b)); }
function fillPerspective(m) {
  const r = P3.r, u = m.unit, dpr = P3.dpr, drop = S.view3 === "orbit";
  for (let i = 0, j = 0; i < m.n; i++, j += 3) {
    const x = m.pos[j], y = m.pos[j + 1], z = m.pos[j + 2], b = m.b[i];
    if (b < 0) { r.sprite(x, y, z, 2.2 * dpr, 0.03, 0.032, 0.035, GL.MARKER); continue; }
    const cr = m.rgb[j], cg = m.rgb[j + 1], cb = m.rgb[j + 2], k = m.kind[i];
    if (drop && z - m.floorZ > u) r.line(x, y, z, x, y, m.floorZ, 0.007);   // drop line to the floor: reads the height
    if (k === KIND_MESH) {
      basisTo(m.rot[j], m.rot[j + 1], m.rot[j + 2]);
      const h = u * 0.7;
      r.box(x, y, z, F3, R3, U3, h * m.scl[j], h * m.scl[j + 1], h * m.scl[j + 2], cr * 2, cg * 2, cb * 2, 1);
      if (cr + cg + cb > 0) r.sprite(x, y, z, u * (1.6 + 2.4 * b), cr * 0.3, cg * 0.3, cb * 0.3, GL.HALO);
      continue;
    }
    if (cr + cg + cb <= 0) { r.sprite(x, y, z, 3 * dpr, 0.06, 0.064, 0.07, GL.MARKER); continue; }
    const s = 0.5 + 0.5 * Math.max(m.scl[j], m.scl[j + 1], m.scl[j + 2]);
    const lamp = coreGain(b);
    r.sprite(x, y, z, u * (1.3 + 3.4 * b) * s, cr * 0.3, cg * 0.3, cb * 0.3, GL.HALO);   // the bloom adds the rest
    r.sprite(x, y, z, u * 0.42 * s, cr * lamp, cg * lamp, cb * lamp, GL.CORE);
    const tn = m.coneTan[i];
    if (tn > 0) {
      // spot: a beam down its forward axis, to the floor when it points at it; its pool lands where the beam does
      basisTo(m.rot[j], m.rot[j + 1], m.rot[j + 2]);
      const down = F3[2] < -0.05, len = down ? clamp((z - m.floorZ) / -F3[2], u * 4, m.radius * 3) : m.radius * 0.9, rad = len * tn;
      r.cone(x, y, z, F3, R3, U3, len, rad, cr * 0.16, cg * 0.16, cb * 0.16);
      if (down && z - m.floorZ > u) {
        const fall = 0.32 / (1 + (len / m.radius) * (len / m.radius));
        r.sprite(x + F3[0] * len, y + F3[1] * len, m.floorZ + 0.5, rad * 1.15, cr * fall, cg * fall, cb * fall, GL.POOL);
      }
      continue;
    }
    if (k === KIND_LIGHT) {
      const ht = z - m.floorZ;
      if (ht > u) {
        const rad = clamp(ht * 0.9, u * 2, m.radius * 0.8), q = ht / m.radius, fall = 0.16 / (1 + q * q);
        r.sprite(x, y, m.floorZ + 0.5, rad, cr * fall, cg * fall, cb * fall, GL.POOL);
      }
    }
  }
}
// dome: every actor placed on the domemaster circle with domeProjectTo; sizes are the angle each glow subtends
function fillDome(m) {
  const r = P3.r, u = m.unit, dpr = P3.dpr, g = domeGeo3(m), cam = m.cam, ax = m.domeAx, half = g.half;
  r.screen = true;
  const cx = +cam.location[0] || 0, cy = +cam.location[1] || 0, cz = +cam.location[2] || 0, k = g.R * dpr / half;
  for (let i = 0, j = 0; i < m.n; i++, j += 3) {
    const x = m.pos[j], y = m.pos[j + 1], z = m.pos[j + 2], b = m.b[i];
    if (!domeProjectTo(x, y, z, cam, ax, half, PXY)) continue;
    const nx = (g.cx + PXY[0] * g.R) / P3.w * 2 - 1, ny = 1 - (g.cy + PXY[1] * g.R) / P3.h * 2;
    const dist = Math.hypot(x - cx, y - cy, z - cz) || 1;
    if (b < 0) { r.sprite(nx, ny, 0, 2 * dpr, 0.03, 0.032, 0.035, GL.MARKER); continue; }
    const cr = m.rgb[j], cg = m.rgb[j + 1], cb = m.rgb[j + 2];
    if (cr + cg + cb <= 0) { r.sprite(nx, ny, 0, 3 * dpr, 0.06, 0.064, 0.07, GL.MARKER); continue; }
    // the angle a glow subtends, as pixels on the circle (with a floor, so every driven actor stays readable)
    r.sprite(nx, ny, 0, Math.max(7 * dpr, Math.atan(u * (1.3 + 3.4 * b) / dist) * k * 1.5), cr * 0.5, cg * 0.5, cb * 0.5, GL.HALO);
    const lamp = coreGain(b);
    r.sprite(nx, ny, 0, Math.max(3.5 * dpr, Math.atan(u * 0.42 / dist) * k), cr * lamp, cg * lamp, cb * lamp, GL.CORE);
  }
}
function domeGeo3(m) {
  const g = P3.domeG;
  if (g && g.w === P3.w && g.h === P3.h && g.m === m) return g;
  return (P3.domeG = { w: P3.w, h: P3.h, m, cx: P3.w / 2, cy: P3.h / 2, R: Math.min(P3.w, P3.h) / 2 - 12, half: (m.domeAngle / 2) * Math.PI / 180 });
}
// the 2D layer over the GL view: dome reticle and the empty-state note; redrawn only when the view changes
function overlay3(m, mode, empty) {
  if (!P3.ovStale) return;
  P3.ovStale = false;
  const dpr = P3.dpr;
  ovCv.width = Math.round(P3.w * dpr); ovCv.height = Math.round(P3.h * dpr);
  const ctx = ovCv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, P3.w, P3.h);
  const note = $("#schem-empty");
  note.hidden = !empty || !S.bake;                                  // nothing to say until the first state arrives
  if (empty) {
    note.querySelector("b").textContent = !S.scene ? "No scene.json" : !m.n ? "scene.json has no actor positions" : "scene.json has no camera";
    note.querySelector("span").textContent = !S.scene || !m.n ? "Scan reads actors, positions and the camera from the editor (it needs the live link)." : "Orbit shows the actors. Scan again once the level has a camera.";
    return;
  }
  if (mode !== "dome") return;
  const g = domeGeo3(m), half = m.domeAngle / 2;
  ctx.strokeStyle = C.line2; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(g.cx, g.cy, g.R, 0, Math.PI * 2); ctx.stroke();
  ctx.strokeStyle = C.line;
  for (const deg of [30, 60, 90, 120, 150]) if (deg < half) { ctx.beginPath(); ctx.arc(g.cx, g.cy, g.R * deg / half, 0, Math.PI * 2); ctx.stroke(); }
  ctx.beginPath(); ctx.moveTo(g.cx - g.R, g.cy); ctx.lineTo(g.cx + g.R, g.cy); ctx.moveTo(g.cx, g.cy - g.R); ctx.lineTo(g.cx, g.cy + g.R); ctx.stroke();
  if (m.domeAx.dome) {
    ctx.font = F11; ctx.fillStyle = C.faint; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText("front", g.cx, DOME_FORWARD_DOWN > 0 ? g.cy + g.R + 8 : g.cy - g.R - 8);
    ctx.textAlign = "left";
  }
}
function hud3() {
  const m = P3.m, el = $("#schem-label");
  if (!P3.r || !m) return;
  const mode = S.view3, cam = m.cam;
  let t = "";
  if (mode === "orbit") t = m.n ? `${m.n} actors, ${m.nDriven} driven` : "";
  else if (cam && mode === "camera") t = `${cam.actor || "camera"}  ${String(cam.projection || "").startsWith("dome") ? "forward" : "view"}  ${fmt(clamp(+cam.fov || 90, 5, 170), 0)}°`;
  else if (cam) t = `${cam.actor || "camera"}  ${fmt(m.domeAngle, 0)}°  pitch ${fmt(+(cam.rotation || [0])[0] || 0, 0)}` + (m.domeAx.dome ? "" : "  (not a domemaster: fisheye along its forward axis)");
  el.textContent = t;
  $("#schem-fit").hidden = mode !== "orbit" || !m.n;
  glCv.classList.toggle("orbit", mode === "orbit");
  document.querySelectorAll("#schem .tabs button").forEach((b) => { if (b.dataset.view === "camera" || b.dataset.view === "dome") b.disabled = !cam; });
}
// One row per output under the view: channel bars (or true colours) at the playhead, and the value. Row labels,
// series keys and the colour-row pixels are prepared when the bake or the size changes; a value's text is rebuilt
// only when the value changes.
const meterRowCv = document.createElement("canvas"), meterRowCtx = meterRowCv.getContext("2d");
let meterImg = null;
function drawMeters(f) {
  const outs = (S.bake && S.bake.outputs) || [], rowH = 18;
  if (P3.meterStale) {
    P3.meterStale = false;
    P3.meterRows = outs.slice(0, 12).map((o) => ({
      o, key: "o:" + o.id, n: o.channels.length, color: isColor(o), lo: o.min ?? 0, span: ((o.max ?? 1) - (o.min ?? 0)) || 1,
      label: o.id.length > 14 ? o.id.slice(0, 13) + "…" : o.id, v: NaN, txt: "-",
    }));
    const rows = P3.meterRows.length, H = rows ? rows * rowH + 8 : 0, maxN = Math.max(1, ...P3.meterRows.map((r) => r.n));
    meterCv.style.height = H + "px";
    meterCv.hidden = !H;
    meterCv.width = Math.round(P3.mw * P3.dpr); meterCv.height = Math.round(H * P3.dpr);
    if (!meterImg || meterImg.width < maxN) { meterRowCv.width = maxN; meterRowCv.height = 1; meterImg = meterRowCtx.createImageData(maxN, 1); }
    P3.meterF = null;
  }
  const rows = P3.meterRows;
  if (!rows.length || f === P3.meterF) return;
  P3.meterF = f;
  const H = rows.length * rowH + 8, ctx = meterCv.getContext("2d"), w = P3.mw, px = meterImg.data;
  ctx.setTransform(P3.dpr, 0, 0, P3.dpr, 0, 0);
  ctx.fillStyle = C.panel; ctx.fillRect(0, 0, w, H);
  ctx.font = F11; ctx.textBaseline = "middle";
  const x0 = 96, bw = w - 64 - x0, hh = rowH - 6;
  for (let i = 0; i < rows.length; i++) {
    const R = rows[i], y = 4 + i * rowH, mid = y + rowH / 2, s = S.series.get(R.key), n = R.n;
    const gap = n > 1 && bw / n > 4 ? 1 : 0, cw = (bw - gap * (n - 1)) / n;
    ctx.fillStyle = C.dim; ctx.textAlign = "left";
    ctx.fillText(R.label, 12, mid);
    if (R.color) {
      // one pixel per channel, stretched into the cells in one draw; the gaps are painted over after
      for (let c = 0, k = 0; c < n; c++, k += 4) {
        if (s) { s.rgba(c, f, CB); toSRGB(CB[0], CB[1], CB[2], C8); px[k] = C8[0]; px[k + 1] = C8[1]; px[k + 2] = C8[2]; }
        else { px[k] = px[k + 1] = px[k + 2] = 38; }
        px[k + 3] = 255;
      }
      meterRowCtx.putImageData(meterImg, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(meterRowCv, 0, 0, n, 1, x0, y + 3, bw, hh);
      if (gap) { ctx.fillStyle = C.panel; for (let c = 1; c < n; c++) ctx.fillRect(Math.round(x0 + (c * bw) / n) - 1, y + 3, 1, hh); }
      ctx.fillStyle = C.faint; ctx.textAlign = "right"; ctx.fillText("colour", w - 12, mid);
      continue;
    }
    let sum = 0;
    for (let c = 0; c < n; c++) {
      const v = s ? s.at(c, f) : R.lo, b = clamp((v - R.lo) / R.span, 0, 1), xa = x0 + c * (cw + gap);
      sum += v;
      ctx.fillStyle = C.line; ctx.fillRect(xa, y + 3, Math.max(1, cw), hh);
      ctx.fillStyle = C.accent;
      if (n > 1) { const bh = Math.max(b > 0 ? 1 : 0, Math.round(hh * b)); ctx.fillRect(xa, y + 3 + hh - bh, Math.max(1, cw), bh); }
      else ctx.fillRect(xa, y + 3, Math.round(cw * b), hh);
    }
    const v = s ? sum / n : NaN;
    if (v !== R.v && !(v !== v && R.v !== R.v)) { R.v = v; R.txt = s ? fmtVal(v) : "-"; }
    ctx.fillStyle = C.text; ctx.textAlign = "right";
    ctx.fillText(R.txt, w - 12, mid);
  }
  ctx.textAlign = "left";
}
function sizeView3() {
  const r = schemView.getBoundingClientRect();
  P3.w = Math.max(1, Math.round(r.width)); P3.h = Math.max(1, Math.round(r.height));
  P3.mw = Math.max(1, Math.round($("#schem").getBoundingClientRect().width));
  P3.dpr = Math.min(window.devicePixelRatio || 1, 2);
  P3.r.resize(P3.w, P3.h, P3.dpr);
  P3.dirty = true; P3.domeG = null; P3.ovStale = true; P3.meterStale = true;
}
let schemRaf = 0;
function requestSchem() {
  P3.dirty = true;
  if (schemRaf) return;
  schemRaf = requestAnimationFrame(() => { schemRaf = 0; drawSchem(false); });
}
// orbit: drag turns, right-drag / middle-drag / Shift-drag pans, the wheel zooms, a double-click frames
function attachOrbit() {
  glCv.addEventListener("contextmenu", (e) => e.preventDefault());
  glCv.addEventListener("mousedown", (e) => {
    if (S.view3 !== "orbit" || !P3.m || !P3.m.n) return;
    e.preventDefault();
    const pan = e.button !== 0 || e.shiftKey, o = P3.orbit, x0 = e.clientX, y0 = e.clientY;
    const yaw0 = o.yaw, pitch0 = o.pitch, t0 = o.t.slice();
    basisTo(o.pitch, o.yaw, 0);
    const R = [R3[0], R3[1], R3[2]], U = [U3[0], U3[1], U3[2]], k = 2 * o.dist * Math.tan(VFOV / 2) / Math.max(1, P3.h);
    P3.drag = true; glCv.classList.add("dragging"); $("#schem-tip").hidden = true;
    const move = (ev) => {
      const dx = ev.clientX - x0, dy = ev.clientY - y0;
      if (pan) for (let i = 0; i < 3; i++) o.t[i] = t0[i] - R[i] * dx * k + U[i] * dy * k;
      else { o.yaw = yaw0 + dx * 0.4; o.pitch = clamp(pitch0 - dy * 0.4, -89, 89); }
      o.user = true;
      requestSchem();
    };
    const up = () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); P3.drag = false; glCv.classList.remove("dragging"); };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  });
  glCv.addEventListener("wheel", (e) => {
    if (S.view3 !== "orbit" || !P3.m || !P3.m.n) return;
    e.preventDefault();
    const o = P3.orbit;
    o.dist = clamp(o.dist * Math.exp(e.deltaY * 0.0012), P3.m.radius * 0.05, P3.m.radius * 60);
    o.user = true;
    requestSchem();
  }, { passive: false });
  const reframe = () => { if (P3.m && P3.m.n) { frameOrbit(P3.m); P3.orbit.user = false; requestSchem(); } };
  glCv.addEventListener("dblclick", () => { if (S.view3 === "orbit") reframe(); });
  $("#schem-fit").onclick = reframe;
  glCv.addEventListener("mousemove", (e) => { if (!P3.drag) hover3(e); });
  glCv.addEventListener("mouseleave", () => { $("#schem-tip").hidden = true; });
}
// hover: the actor nearest the pointer (within 14 px) and what drives it, at the playhead
function hover3(e) {
  const m = P3.m, tip = $("#schem-tip");
  if (!m || !m.n || (S.view3 !== "orbit" && !m.cam)) { tip.hidden = true; return; }
  const rc = glCv.getBoundingClientRect(), x = e.clientX - rc.left, y = e.clientY - rc.top;
  const g = S.view3 === "dome" ? domeGeo3(m) : null;
  let best = -1, bd = 14 * 14;
  for (let i = 0, j = 0; i < m.n; i++, j += 3) {
    let sx, sy;
    if (g) {
      if (!domeProjectTo(m.pos[j], m.pos[j + 1], m.pos[j + 2], m.cam, m.domeAx, g.half, PXY)) continue;
      sx = g.cx + PXY[0] * g.R; sy = g.cy + PXY[1] * g.R;
    } else {
      if (!P3.r.project(m.pos[j], m.pos[j + 1], m.pos[j + 2], PXY)) continue;
      sx = PXY[0]; sy = PXY[1];
    }
    const d = (sx - x) * (sx - x) + (sy - y) * (sy - y);
    if (d < bd) { bd = d; best = i; }
  }
  if (best < 0) { tip.hidden = true; return; }
  const f = frameOf(curT()), lines = [m.names[best]];
  for (const { o, c } of m.outs[best]) {
    const s = S.series.get("o:" + o.id);
    if (!s) continue;
    if (isColor(o)) { s.rgba(c, f, CB); lines.push(`${o.id}  ${fmt(CB[0], 2)} ${fmt(CB[1], 2)} ${fmt(CB[2], 2)}`); }
    else lines.push(`${o.id}  ${fmtVal(s.at(c, f))}`);
  }
  if (lines.length === 1) lines.push("not driven");
  tip.textContent = lines.join("\n");
  tip.hidden = false;
  tip.style.left = `${Math.max(4, Math.min(x + 12, P3.w - tip.offsetWidth - 4))}px`;
  tip.style.top = `${Math.max(4, Math.min(y + 12, P3.h - tip.offsetHeight - 4))}px`;
}
// WebGL2 on, unless the browser lacks it or the page was opened with ?gl=0 (to check the 2D fallback)
function init3D() {
  const off = new URLSearchParams(location.search).get("gl") === "0";
  P3.r = GL && !off ? GL.create(glCv) : null;
  if (!P3.r) return use2D();
  glCv.hidden = ovCv.hidden = false;
  $("#schem-hud").hidden = false;
  schem.hidden = true;
  schemView.classList.add("gl");
  P3.r.onrestore = () => { P3.m = null; requestSchem(); };
  attachOrbit();
  new ResizeObserver(() => { sizeView3(); drawSchem(); }).observe(schemView);
}
// the 2D views (no WebGL2, ?gl=0, or shaders that failed to build): ring (outputs as rings and bars), plan, dome
function use2D() {
  const was = !!P3.r;
  P3.r = null;
  glCv.hidden = ovCv.hidden = meterCv.hidden = true;
  $("#schem-hud").hidden = true; $("#schem-empty").hidden = true; $("#schem-tip").hidden = true;
  schem.hidden = false;
  schemView.classList.remove("gl");
  const spec = [["ring", "Ring", "Outputs as rings and bars (V cycles)"], ["plan", "Plan", "Top-down plan of the scanned actors"], ["dome", "Dome", "Domemaster seen from the scene camera"]];
  document.querySelectorAll("#schem .tabs button").forEach((b, i) => { b.dataset.view = spec[i][0]; b.textContent = spec[i][1]; b.title = spec[i][2]; b.disabled = false; });
  if (was) setView3(S.view3 === "dome" ? "dome" : "ring");
}
const VIEWS_GL = ["orbit", "camera", "dome"], VIEWS_2D = ["ring", "plan", "dome"];

// ------------------------------------------------------------------------------------------- params + graph editing
const SKIP_KEYS = new Set(["id", "op", "in", "track", "signal", "target", "layer", "fn", "shape", "order", "preset", "direction", "value_type"]);
const INT_KEYS = new Set(["channels", "seed", "neighbours", "every", "subdivide", "audio_offset_frames", "frames"]);
function sliderRange(key, v) {
  const k = key.toLowerCase();
  if (INT_KEYS.has(k)) return k === "channels" ? [1, 64, 1] : k === "audio_offset_frames" ? [0, 300, 1] : [0, Math.max(16, v * 2), 1];
  if (k.endsWith("_ms") || k === "ms") return [0, Math.max(2000, v * 2), 1];
  if (k === "lo" || k === "hi" || k.endsWith("_hz") || k === "freq_hz") return k === "freq_hz" ? [0, 20, 0.01] : [20, 20000, 1];
  if (k.endsWith("_db")) return [0, 60, 0.5];
  if (k.endsWith("_beats") || k === "beats") return [0, Math.max(8, v * 2), 0.0625];
  if (k.endsWith("_pct")) return [0, 100, 0.5];
  if (k === "gamma") return [0.1, 4, 0.01];
  if (["damping", "falloff", "level", "threshold", "sustain", "strength", "amount", "phase", "phase_step", "min_swing"].includes(k)) return [0, 1, 0.001];
  if (k === "key_tolerance") return [0, 0.05, 0.0005];
  if (k === "min" || k === "max") { const m = Math.max(1, Math.abs(v) * 3); return [v < 0 ? -m : 0, m, m / 1000]; }
  const m = Math.max(1, Math.abs(v) * 2);
  return [v < 0 ? -m : 0, m, m / 1000];
}
const isLog = (key) => key === "lo" || key === "hi";
function toSlider(key, v, r) { return isLog(key) ? (Math.log(clamp(v, r[0], r[1]) / r[0]) / Math.log(r[1] / r[0])) * 1000 : v; }
function fromSlider(key, s, r) { return isLog(key) ? Math.round(r[0] * Math.pow(r[1] / r[0], s / 1000)) : s; }
function nodeInputs(n) {
  const v = n.in;
  return v == null ? [] : Array.isArray(v) ? v.filter((x) => typeof x === "string") : typeof v === "string" ? [v] : [];
}
function paramEntries() {
  const groups = [];
  const rig = S.rig;
  const g0 = { title: "rig", sub: "", rows: [] };
  if (typeof rig.audio_offset_frames === "number") g0.rows.push({ path: ["audio_offset_frames"], key: "audio_offset_frames" });
  if (rig.tempo && typeof rig.tempo.bpm === "number") g0.rows.push({ path: ["tempo", "bpm"], key: "bpm" });
  if (typeof rig.seed === "number") g0.rows.push({ path: ["seed"], key: "seed" });
  if (rig.bake && typeof rig.bake.key_tolerance === "number") g0.rows.push({ path: ["bake", "key_tolerance"], key: "key_tolerance" });
  (rig.nodes || []).forEach((n, ni) => {
    const ins = nodeInputs(n);
    const g = { title: n.id, sub: n.op + (ins.length ? ` ← ${ins.join(", ")}` : n.track ? ` ${n.track}` : ""), rows: [], node: n.id };
    for (const [k, v] of Object.entries(n)) if (!SKIP_KEYS.has(k) && typeof v === "number") g.rows.push({ path: ["nodes", ni, k], key: k });
    // params the rig leaves at the op's default (engine registry): shown dimmed, written into the node when changed
    const defs = (S.opParams || {})[n.op] || {};
    for (const [k, v] of Object.entries(defs)) if (!(k in n) && !SKIP_KEYS.has(k)) g.rows.push({ path: ["nodes", ni, k], key: k, def: v });
    groups.push(g);
  });
  (rig.outputs || []).forEach((o, oi) => {
    const g = { title: o.id, sub: `output ← ${o.signal} → ${targetSummary(o.target)}`, rows: [], output: o.id };
    for (const [k, v] of Object.entries(o)) if (!SKIP_KEYS.has(k) && typeof v === "number") g.rows.push({ path: ["outputs", oi, k], key: k });
    groups.push(g);
  });
  if (g0.rows.length) groups.push(g0);
  return groups;
}
const getPath = (o, p) => p.reduce((a, k) => a[k], o);
function setPath(o, p, v) { const last = p[p.length - 1]; getPath(o, p.slice(0, -1))[last] = v; }
function buildParams() {
  const host = $("#param-list");
  const scroll = host.scrollTop;
  host.textContent = "";
  for (const g of paramEntries()) {
    const box = document.createElement("div");
    box.className = "pgroup";
    const head = document.createElement("div");
    head.className = "pgroup-head";
    head.innerHTML = "<b></b><small class=dim></small>";
    head.querySelector("b").textContent = g.title;
    head.querySelector("small").textContent = g.sub;
    head.querySelector("small").title = g.sub;
    if (g.node || g.output) {
      const rm = document.createElement("button");
      rm.className = "rm"; rm.textContent = "×"; rm.title = g.node ? "Remove this node (refused while something uses it)" : "Remove this output";
      rm.onclick = () => (g.node ? removeNode(g.node) : removeOutput(g.output));
      head.append(rm);
    }
    box.append(head);
    for (const row of g.rows) box.append(paramRow(row));
    host.append(box);
  }
  host.scrollTop = scroll;
}
const valueOr = (path, def) => { const v = getPath(S.rig, path); return v === undefined ? def : v; };
function paramRow({ path, key, def }) {
  const v0 = valueOr(path, def);
  const r = sliderRange(key, v0);
  const el = document.createElement("div");
  el.className = "prow";
  el.innerHTML = `<label></label><input type="range"><input type="number">`;
  const lab = el.querySelector("label"), sl = el.querySelector("input[type=range]"), num = el.querySelector("input[type=number]");
  lab.textContent = key; lab.title = path.join(".") + (def !== undefined ? " (op default)" : "");
  if (def !== undefined) el.classList.add("def");
  if (isLog(key)) { sl.min = 0; sl.max = 1000; sl.step = 1; } else { sl.min = r[0]; sl.max = r[1]; sl.step = r[2]; }
  num.step = "any";
  const show = (v) => { sl.value = toSlider(key, v, r); num.value = +(+v).toPrecision(6); };
  show(v0);
  const commit = (v) => {
    if (!Number.isFinite(v)) { show(valueOr(path, def)); return; }
    if (INT_KEYS.has(key)) v = Math.round(v);
    if (v === getPath(S.rig, path)) return;
    el.classList.remove("def");
    const rig = deep(S.rig);
    setPath(rig, path, v);
    pushUndo("rig", deep(S.rig));
    postRig(rig, { el, path, show, def });
  };
  sl.addEventListener("input", () => { num.value = +(+fromSlider(key, +sl.value, r)).toPrecision(6); });
  sl.addEventListener("change", () => commit(+fromSlider(key, +sl.value, r)));
  num.addEventListener("change", () => commit(parseFloat(num.value)));
  num.addEventListener("keydown", (e) => { if (e.key === "Enter") num.blur(); e.stopPropagation(); });
  el._show = show;
  return el;
}
function usersOf(id) {
  const out = [];
  for (const n of S.rig.nodes || []) if (nodeInputs(n).includes(id)) out.push(n.id);
  for (const o of S.rig.outputs || []) if (o.signal === id) out.push(`output ${o.id}`);
  return out;
}
function removeNode(id) {
  const users = usersOf(id);
  if (users.length) return msg(`${id} is used by ${users.join(", ")}: remove or rewire those first`, true);
  const rig = deep(S.rig);
  rig.nodes = rig.nodes.filter((n) => n.id !== id);
  pushUndo("rig", deep(S.rig));
  postRig(rig, null, `removed node ${id}`);
}
function removeOutput(id) {
  const rig = deep(S.rig);
  rig.outputs = rig.outputs.filter((o) => o.id !== id);
  pushUndo("rig", deep(S.rig));
  postRig(rig, null, `removed output ${id}`);
}
const form = $("#graph-form");
function closeForm() { form.hidden = true; form.textContent = ""; }
function frow(label, el, hint) {
  const d = document.createElement("div");
  d.className = "frow";
  const l = document.createElement("label"); l.textContent = label;
  d.append(l, el);
  if (hint) el.title = hint;
  return d;
}
function mk(tag, props = {}, kids = []) { const e = document.createElement(tag); Object.assign(e, props); for (const k of kids) e.append(k); return e; }
function selectOf(values, value, labels) {
  const s = mk("select");
  values.forEach((v, i) => { const o = mk("option", { value: v, textContent: labels ? labels[i] : v }); s.append(o); });
  if (value != null) s.value = value;
  return s;
}
function uniqueId(base) { const ids = new Set([...(S.rig.nodes || []).map((n) => n.id), ...(S.rig.outputs || []).map((o) => o.id)]); let i = 1; while (ids.has(`${base}${i}`)) i++; return `${base}${i}`; }
function parseVal(s) { s = String(s).trim(); if (s === "") return undefined; try { return JSON.parse(s); } catch { return s; } }
function formActions(okText) {
  const err = mk("div", { className: "ferr" });
  const act = mk("div", { className: "actions" }, [mk("button", { type: "button", textContent: "Cancel", onclick: closeForm }), mk("button", { type: "submit", textContent: okText })]);
  return { err, act };
}
async function ensureOps() {
  if (!S.ops) { try { S.ops = (await api("/api/ops")).ops; } catch (e) { msg("ops: " + e.message, true); } }
  return S.ops || {};
}
async function openNodeForm() {
  const ops = await ensureOps();
  form.textContent = ""; form.hidden = false; form.dataset.kind = "node";
  const opSel = selectOf(Object.keys(ops), "envelope");
  const idIn = mk("input", { type: "text", spellcheck: false });
  const body = mk("div", { style: "display:grid;gap:4px" });
  const { err, act } = formActions("Add node");
  form.append(mk("div", { className: "hint", textContent: "New node. Optional params show up as sliders once it exists." }), frow("op", opSel), frow("id", idIn), body, err, act);
  const ids = (S.rig.nodes || []).map((n) => n.id);
  const fill = () => {
    const spec = ops[opSel.value] || { params: {}, required: [], inputs: null, enums: {} };
    idIn.value = uniqueId(opSel.value);
    opSel.title = spec.doc || "";
    body.textContent = "";
    if (spec.inputs === "one") body.append(frow("in", selectOf(ids, ids[ids.length - 1]), "input node"));
    else if (spec.inputs === "many") {
      const box = mk("div", { className: "checks" });
      for (const id of ids) box.append(mk("label", { className: "chk" }, [mk("input", { type: "checkbox", value: id }), id]));
      body.append(frow("in", box, "input nodes, in order"));
    } else if (spec.inputs === "ref") {
      const bands = (S.rig.nodes || []).filter((n) => n.op === "band").map((n) => n.id);
      body.append(frow("in", selectOf(["", ...bands], bands[0] || "", ["(full band)", ...bands]), "band node to detect onsets in"));
    }
    for (const k of spec.required || []) {
      let el;
      if (spec.enums && spec.enums[k]) el = selectOf(spec.enums[k]);
      else if (k === "track") el = selectOf(Object.keys(S.notes.tracks || {}));
      else el = mk("input", { type: "text", spellcheck: false, placeholder: "required" });
      el.dataset.param = k;
      body.append(frow(k, el));
    }
    for (const group of spec.one_of || []) {
      const el = mk("input", { type: "text", spellcheck: false, placeholder: `${group.join(" or ")} (one is needed)` });
      el.dataset.oneOf = group.join(",");
      body.append(frow(group.join(" / "), el, `a number for ${group[0]}, or prefix the name: ${group[1]}=2`));
    }
    for (const [k, vals] of Object.entries(spec.enums || {})) {
      if ((spec.required || []).includes(k)) continue;
      const el = selectOf(vals, spec.params[k]); el.dataset.param = k; el.dataset.optional = "1";
      body.append(frow(k, el));
    }
    // optional params without a numeric default get no slider later, so offer them here (blank keeps the default)
    const inOneOf = new Set((spec.one_of || []).flat());
    for (const [k, v] of Object.entries(spec.params || {})) {
      if (v !== null || (spec.required || []).includes(k) || inOneOf.has(k) || (spec.enums || {})[k]) continue;
      const el = mk("input", { type: "text", spellcheck: false, placeholder: "optional" });
      el.dataset.param = k; el.dataset.optional = "1";
      body.append(frow(k, el));
    }
  };
  opSel.onchange = fill; fill();
  form.onsubmit = (e) => {
    e.preventDefault();
    const spec = ops[opSel.value];
    const node = { id: idIn.value.trim(), op: opSel.value };
    if (!node.id) { err.textContent = "id is required"; return; }
    if (usersOf(node.id).length || (S.rig.nodes || []).some((n) => n.id === node.id) || (S.rig.outputs || []).some((o) => o.id === node.id)) { err.textContent = `${node.id} is taken`; return; }
    if (spec.inputs === "one") node.in = body.querySelector("select").value;
    else if (spec.inputs === "many") { node.in = [...body.querySelectorAll(".checks input:checked")].map((c) => c.value); if (node.in.length < 2) { err.textContent = "pick at least two inputs"; return; } }
    else if (spec.inputs === "ref") { const v = body.querySelector("select").value; if (v) node.in = v; }
    for (const el of body.querySelectorAll("[data-param]")) {
      const v = parseVal(el.value);
      if (v === undefined) { if (!el.dataset.optional) { err.textContent = `${el.dataset.param} is required`; return; } continue; }
      if (el.dataset.optional && v === spec.params[el.dataset.param]) continue;
      node[el.dataset.param] = v;
    }
    for (const el of body.querySelectorAll("[data-one-of]")) {
      const s = el.value.trim(); if (!s) { err.textContent = `${el.dataset.oneOf.replace(",", " or ")} is needed`; return; }
      const m = s.match(/^(\w+)\s*=\s*(.+)$/);
      const keys = el.dataset.oneOf.split(",");
      if (m && keys.includes(m[1])) node[m[1]] = parseVal(m[2]); else node[keys[0]] = parseVal(s);
    }
    const rig = deep(S.rig);
    rig.nodes = [...(rig.nodes || []), node];
    pushUndo("rig", deep(S.rig));
    postRig(rig, null, `added node ${node.id}`);
    closeForm();
  };
  idIn.focus();
}
const KIND_DEFAULTS = { light: [0, 10000], material: [0, 1], niagara: [0, 1], mpc: [0, 1], transform: [0, 360] };
function groupsOf() { return S.rig.groups && typeof S.rig.groups === "object" ? S.rig.groups : {}; }
function openOutputForm() {
  form.textContent = ""; form.hidden = false; form.dataset.kind = "output";
  const nodes = (S.rig.nodes || []).map((n) => n.id);
  const idIn = mk("input", { type: "text", spellcheck: false, value: uniqueId("out") });
  const sig = selectOf(nodes, nodes[nodes.length - 1]);
  const kind = selectOf(["light", "material", "mpc", "niagara", "transform"], "light");
  const body = mk("div", { style: "display:grid;gap:4px" });
  const lo = mk("input", { type: "number", step: "any", value: 0 }), hi = mk("input", { type: "number", step: "any", value: 10000 });
  const colour = mk("input", { type: "checkbox" });
  const { err, act } = formActions("Add output");
  form.append(mk("div", { className: "hint", textContent: "New output. Actors, parameters and MPCs come from the scanned scene." }),
    frow("id", idIn), frow("signal", sig), frow("target", kind), body,
    frow("colour", mk("label", { className: "chk" }, [colour, "colour signal (value_type color)"])),
    frow("min / max", mk("div", { style: "display:grid;grid-template-columns:1fr 1fr;gap:4px" }, [lo, hi])), err, act);
  const sc = S.scene || {}, actors = Object.keys(sc.actors || {}).sort(), groups = Object.keys(groupsOf());
  let actorSel = null, filter = null;
  const paramList = (sel) => {
    const out = new Set();
    for (const a of sel) for (const c of ((sc.actors || {})[a] || {}).components || []) for (const m of c.materials || []) for (const p of [...(m.scalars || []), ...(colour.checked ? m.vectors || [] : [])]) out.add(p);
    return [...out].sort();
  };
  const dl = mk("datalist", { id: "param-dl" });
  const fill = () => {
    const k = kind.value;
    body.textContent = "";
    const [a, b] = KIND_DEFAULTS[k]; lo.value = a; hi.value = b;
    if (k !== "mpc") {
      filter = mk("input", { type: "text", spellcheck: false, placeholder: actors.length ? "filter actors" : "actor names, comma separated" });
      actorSel = mk("select", { multiple: true, size: 6 });
      const fillActors = () => {
        const q = filter.value.trim().toLowerCase();
        actorSel.textContent = "";
        for (const g of groups) if (!q || g.toLowerCase().includes(q)) actorSel.append(mk("option", { value: "@" + g, textContent: `@${g} (${(groupsOf()[g] || []).length})` }));
        for (const x of actors) if (!q || x.toLowerCase().includes(q)) actorSel.append(mk("option", { value: x, textContent: x }));
      };
      filter.oninput = fillActors; fillActors();
      body.append(frow("actors", filter));
      if (actors.length || groups.length) body.append(frow("", actorSel, "Ctrl/Shift-click for several; channel i drives actor i"));
    }
    if (k === "light") body.append(frow("property", (() => { const s = selectOf(["intensity", "color"]); s.dataset.f = "property"; s.onchange = () => { colour.checked = s.value === "color"; }; return s; })()));
    if (k === "material" || k === "niagara") {
      const p = mk("input", { type: "text", spellcheck: false, placeholder: "parameter name" }); p.dataset.f = "param"; p.setAttribute("list", "param-dl");
      body.append(frow("param", p), dl);
      const upd = () => { dl.textContent = ""; for (const x of paramList([...actorSel.selectedOptions].map((o) => o.value))) dl.append(mk("option", { value: x })); };
      actorSel.onchange = upd; colour.onchange = upd;
    }
    if (k === "mpc") {
      const mpcs = Object.keys(sc.mpcs || {});
      const asset = mk("input", { type: "text", spellcheck: false, placeholder: "/Game/Path/MPC_Name" }); asset.dataset.f = "asset"; asset.setAttribute("list", "mpc-dl");
      const mdl = mk("datalist", { id: "mpc-dl" }); for (const m of mpcs) mdl.append(mk("option", { value: m }));
      const p = mk("input", { type: "text", spellcheck: false, placeholder: "parameter name" }); p.dataset.f = "param"; p.setAttribute("list", "param-dl");
      const upd = () => { dl.textContent = ""; const m = (sc.mpcs || {})[asset.value] || {}; for (const x of [...(m.scalars || []), ...(colour.checked ? m.vectors || [] : [])]) dl.append(mk("option", { value: x })); };
      asset.oninput = upd; colour.onchange = upd;
      if (mpcs.length) asset.value = mpcs[0];
      body.append(frow("asset", asset), mdl, frow("param", p), dl);
      upd();
    }
    if (k === "transform") {
      const prop = selectOf(["rotation", "location", "scale"]); prop.dataset.f = "property";
      let axis = selectOf(["yaw", "pitch", "roll"]); axis.dataset.f = "axis";
      prop.onchange = () => {
        const a = selectOf(prop.value === "rotation" ? ["yaw", "pitch", "roll"] : ["x", "y", "z"]);
        a.dataset.f = "axis"; axis.replaceWith(a); axis = a;
        const [lo0, hi0] = prop.value === "rotation" ? [0, 360] : prop.value === "scale" ? [1, 2] : [0, 100]; lo.value = lo0; hi.value = hi0;
      };
      body.append(frow("property", prop), frow("axis", axis));
    }
  };
  kind.onchange = fill; fill();
  form.onsubmit = (e) => {
    e.preventDefault();
    const o = { id: idIn.value.trim(), signal: sig.value, target: { kind: kind.value } };
    if (!o.id) { err.textContent = "id is required"; return; }
    if ((S.rig.outputs || []).some((x) => x.id === o.id)) { err.textContent = `${o.id} is taken`; return; }
    if (kind.value !== "mpc") {
      let list = actorSel ? [...actorSel.selectedOptions].map((x) => x.value) : [];
      if (!list.length && filter) list = filter.value.split(",").map((x) => x.trim()).filter(Boolean);
      if (!list.length) { err.textContent = "pick at least one actor or group"; return; }
      o.target.actors = list;
    }
    for (const el of body.querySelectorAll("[data-f]")) { const v = el.value.trim(); if (v) o.target[el.dataset.f] = v; }
    if ((kind.value === "material" || kind.value === "niagara" || kind.value === "mpc") && !o.target.param) { err.textContent = "param is required"; return; }
    if (kind.value === "mpc" && !o.target.asset) { err.textContent = "asset is required"; return; }
    if (kind.value === "transform") o.target.additive = true;
    if (colour.checked) o.value_type = "color";
    else { o.min = +lo.value; o.max = +hi.value; }
    const rig = deep(S.rig);
    rig.outputs = [...(rig.outputs || []), o];
    pushUndo("rig", deep(S.rig));
    postRig(rig, null, `added output ${o.id}`);
    closeForm();
  };
  idIn.focus();
}
$("#add-node").onclick = () => (!form.hidden && form.dataset.kind === "node" ? closeForm() : openNodeForm());
$("#add-out").onclick = () => (!form.hidden && form.dataset.kind === "output" ? closeForm() : openOutputForm());

// ------------------------------------------------------------------------------------------- server I/O + undo
async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  let data;
  try { data = await res.json(); } catch { data = { ok: false, error: `HTTP ${res.status}` }; }
  if (!res.ok || data.ok === false) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}
function setDirty(d) {
  S.dirty = d;
  $("#dirty").hidden = !d;
  $("#save").disabled = !d;
  document.title = (d ? "• " : "") + "beatrig";
}
function setBakeInfo(text) { $("#bake-info").textContent = text; }
function applyBake(data) {
  const prevIds = ((S.bake && S.bake.outputs) || []).map((o) => `${o.id}:${o.channels.length}:${o.components}`).join();
  S.bake = data.bake;
  S.bands = data.bands || [];
  S.bakeMs = data.bake_ms;
  S.bakeId = data.bake_id;
  S.flags = data.flags && !data.flags._error ? data.flags : null;
  if (data.rev != null) S.rev = data.rev;
  S.fps = S.bake.fps || S.fps;
  S.offset = (S.rig && S.rig.audio_offset_frames) || 0;
  rebuildTempo();
  setDirty(!!data.dirty);
  const warns = data.warnings || [];
  setBakeInfo(`bake ${fmt(S.bakeMs, 0)} ms` + (warns.length ? `  · ${warns.length} warning${warns.length > 1 ? "s" : ""}` : ""));
  $("#bake-info").title = warns.length ? "Warnings:\n" + warns.join("\n") : "";
  if (data.flags && data.flags._error) msg("flash check failed: " + data.flags._error, true);
  const nflash = S.flags ? Object.values(S.flags).reduce((a, l) => a + l.length, 0) : 0;
  const fb = $("#flash");
  fb.hidden = !nflash; fb.textContent = `${nflash} flash`;
  fb.title = nflash ? "Flash-safety windows (photosensitivity): click to jump to the next one\n" + Object.entries(S.flags).map(([k, l]) => `${k}: ${l.map(([a, b]) => `${fmt(a, 2)}-${fmt(b, 2)} s`).slice(0, 6).join(", ")}${l.length > 6 ? " ..." : ""}`).join("\n") : "";
  const ids = (S.bake.outputs || []).map((o) => `${o.id}:${o.channels.length}:${o.components}`).join();
  const fsig = S.flags ? Object.entries(S.flags).map(([k, l]) => `${k}:${l.length}`).join() : "";
  if (ids !== prevIds || fsig !== S._fsig) buildCurveLanes();
  S._fsig = fsig;
  buildBandLanes();
  buildLayerLanes();
  buildActorMap();
  S.geo = null;
  loadCurves();
  schedule();
  drawSchem();
}
let curveGen = 0;
async function loadCurves() {
  const gen = ++curveGen, want = [], keep = new Set();
  for (const o of S.bake.outputs || []) {
    const k = "o:" + o.id; keep.add(k);
    const s = S.series.get(k);
    if (!s || s.hash !== o.hash) want.push([k, `/api/curves?output=${enc(o.id)}&lod=0`]);
  }
  for (const b of POPOUT ? [] : S.bands || []) {                    // the pop-out has no band lanes
    const k = "b:" + b.id; keep.add(k);
    const s = S.series.get(k);
    if (!s || s.hash !== b.hash) want.push([k, `/api/curves?band=${enc(b.id)}&lod=0`]);
  }
  for (const k of [...S.series.keys()]) if (!keep.has(k)) S.series.delete(k);
  await Promise.all(want.map(async ([k, url]) => {
    try {
      const { hdr, buf, off } = await fetchBin(url);
      const cur = S.series.get(k);
      if (cur && cur.bakeId > hdr.bake_id) return;                       // a newer bake already landed
      const ser = new Series(hdr, buf, off);
      ser.kindKey = k;
      S.series.set(k, ser);
      warmPyramids(ser);
      const [kind, id] = [k[0], k.slice(2)];
      for (const cv of document.querySelectorAll(kind === "o" ? 'canvas[data-kind="curve"]' : 'canvas[data-kind="band"]')) if (cv.dataset.id === id) schedule(cv);
    } catch (e) { msg(`curves ${k.slice(2)}: ${e.message}`, true); }
  }));
  if (gen === curveGen) { drawSchem(); markReady(); }
}
// data ready = state, every curve and the waveform are in memory (first paint = this + one full redraw)
function markReady() {
  if (S.perf.dataReady == null && S.peaks && S.bake && S.bake.outputs.every((o) => S.series.has("o:" + o.id))) { S.perf.dataReady = performance.now(); schedule(); }
}
async function loadPeaks() {
  try {
    const { hdr, buf, off } = await fetchBin("/api/peaks?lod=0");
    S.peaks = new Peaks(hdr, buf, off);
    markReady();
  } catch (e) { S.peaks = null; msg("waveform: " + e.message, true); }
  scheduleKind('canvas[data-kind="wave"]');
}
// One re-bake at a time. Edits made meanwhile merge into one pending job that sends the rig and/or notes as they are
// when it goes out, so a quick notes edit followed by a slider is never dropped, and a rig + notes change that only
// makes sense together (a section rename a layer key follows) goes as one /api/edit.
function queueBake(job) {
  const q = S.queued || { rig: false, notes: false };
  S.queued = { rig: q.rig || !!job.rig, notes: q.notes || !!job.notes, ctx: job.ctx || (job.rig ? null : q.ctx), note: job.note || q.note };
  runQueue();
}
async function runQueue() {
  if (S.baking || !S.queued) return;
  const job = S.queued; S.queued = null;
  const rig = job.rig ? deep(S.rig) : null, notes = job.notes ? deep(S.notes) : null;
  const [path, body] = rig && notes ? ["/api/edit", { rig, notes }] : rig ? ["/api/rig", rig] : ["/api/notes", notes];
  S.baking = true;
  setBakeInfo("baking…");
  try {
    const data = await api(path, body);
    if (rig) S.lastGood.rig = rig;
    if (notes) S.lastGood.notes = notes;
    applyBake(data);
    clearError();
    if (job.ctx && job.ctx.el) job.ctx.el.classList.remove("bad");
    msg(job.note || (rig && notes ? "rig and notes re-baked" : rig ? "rig re-baked" : "notes re-baked"));
  } catch (err) {
    // the server kept its last good state; mirror that here
    if (rig) {
      S.rig = deep(S.lastGood.rig);
      if (job.ctx && job.ctx.show) { job.ctx.show(valueOr(job.ctx.path, job.ctx.def)); job.ctx.el.classList.add("bad"); }
      if (!job.ctx) buildParams();
      buildLayerLanes();
    }
    if (notes) { S.notes = deep(S.lastGood.notes); S.sel.clear(); rebuildTempo(); buildAllEx(); buildNoteLanes(); notesChangedViews(); }
    const what = job.ctx && job.ctx.path ? `${job.ctx.path.join(".")} reverted` : rig && notes ? "rig and notes reverted" : rig ? "rig change reverted" : "notes reverted";
    showError(`${err.message}\n(${what})`);
    setBakeInfo(`bake ${fmt(S.bakeMs, 0)} ms`);
  } finally {
    S.baking = false;
    if (S.pendingReload && !S.queued) { S.pendingReload = false; if (S.maxSeenRev > S.rev) reloadState(); }
    runQueue();
  }
}
function postRig(rig, ctx, note) {
  S.rig = rig;
  if (!ctx) buildParams();
  buildLayerLanes();
  queueBake({ rig: true, ctx, note });
}
let notesTimer = 0;
function postNotes(delay = 0) {
  clearTimeout(notesTimer);
  if (delay) notesTimer = setTimeout(() => queueBake({ notes: true }), delay); else queueBake({ notes: true });
}
function showError(text) { const e = $("#param-error"); e.textContent = text; e.hidden = false; msg(text.split("\n")[0], true); }
function clearError() { $("#param-error").hidden = true; document.querySelectorAll(".prow.bad").forEach((r) => r.classList.remove("bad")); }
async function save() {
  if (!S.dirty) return;
  try {
    const d = await api("/api/save", {});
    if (d.rev != null) S.rev = d.rev;
    setDirty(false);
    msg(`saved ${d.saved.map((p) => p.split(/[\\/]/).pop()).join(", ")} (.bak written)`);
  } catch (err) { msg("save failed: " + err.message, true); }
}
function pushUndo(kind, data) {
  S.undo.push({ kind, data });
  if (S.undo.length > 100) S.undo.shift();
  S.redo.length = 0;
  updUndo();
}
function updUndo() { $("#undo").disabled = !S.undo.length; $("#redo").disabled = !S.redo.length; }
function undoRedo(from, to, verb) {
  const u = from.pop();
  if (!u) return;
  const cur = { rig: deep(S.rig), notes: deep(S.notes) };
  to.push({ kind: u.kind, data: u.kind === "both" ? cur : cur[u.kind] });
  const notes = u.kind === "notes" ? u.data : u.kind === "both" ? u.data.notes : null;
  const rig = u.kind === "rig" ? u.data : u.kind === "both" ? u.data.rig : null;
  if (notes) { S.notes = notes; S.sel.clear(); rebuildTempo(); buildAllEx(); buildNoteLanes(); selChanged(); notesChangedViews(); }
  if (rig) { S.rig = rig; buildParams(); buildLayerLanes(); }
  queueBake({ rig: !!rig, notes: !!notes, note: verb });
  updUndo();
}

// ------------------------------------------------------------------------------------------- view + transport
function clampView() {
  const vis = S.width / S.view.pps;
  S.view.t0 = clamp(S.view.t0, S.tStart, Math.max(S.tStart, S.duration - vis * 0.5));
}
const fitPps = () => S.width / Math.max(0.1, S.duration - S.tStart);
const maxPps = () => S.fps * 60; // 60 px per frame
function syncZoomSlider() {
  const a = Math.log(fitPps()), b = Math.log(maxPps());
  $("#zoom").value = clamp((Math.log(S.view.pps) - a) / (b - a), 0, 1) * 1000;
}
function zoomTo(pps, anchorX) {
  const tA = tOf(anchorX);
  S.view.pps = clamp(pps, fitPps() * 0.5, maxPps());
  S.view.t0 = tA - anchorX / S.view.pps;
  clampView(); syncZoomSlider(); schedule();
}
function zoomFit() { S.view.pps = fitPps(); S.view.t0 = S.tStart; syncZoomSlider(); schedule(); }
// The playhead: the audio element's clock; in the pop-out, the main page's clock (or its own silent one), extrapolated
// from the last message so the pop-out draws at its own frame rate.
const Remote = { playing: false, t0: 0, at: 0, rate: 1, main: false, mainAt: 0 };
const playing = () => (POPOUT ? Remote.playing : !audio.paused);
const curT = () => (POPOUT
  ? (Remote.playing ? clamp(Remote.t0 + (performance.now() - Remote.at) * 0.001 * Remote.rate, S.tStart, S.duration) : S.pausedT)
  : (audio.paused ? S.pausedT : audio.currentTime));
let gutterW = 150;
function placePlayhead() {
  const x = xOf(curT());
  playheadEl.style.transform = `translateX(${Math.round(gutterW + x)}px)`;
  playheadEl.style.display = x < 0 || x > S.width ? "none" : "block";
}
const RO = { sec: $("#t-sec"), frame: $("#t-frame"), beat: $("#t-beat"), bpm: $("#bpm") };
function setText(el, s) { if (el.textContent !== s) el.textContent = s; }
// audio time as m:ss.mmm (minutes keep counting past 59, so a 75-minute show reads 75:03.250)
function clockText(t) {
  const ms = Math.max(0, Math.round(t * 1000)), m = Math.floor(ms / 60000), r = ms - m * 60000;
  return `${m}:${String(Math.floor(r / 1000)).padStart(2, "0")}.${String(r % 1000).padStart(3, "0")}`;
}
function updateReadouts() {
  const t = curT();
  setText(RO.sec, clockText(t));
  setText(RO.frame, String(frameOf(t)));
  const b = beatAt(t);
  if (b >= 0) {
    const bar = Math.floor(b / BEATS_PER_BAR) + 1, beat = Math.floor(b % BEATS_PER_BAR) + 1, six = Math.floor((b % 1) * 4) + 1;
    setText(RO.beat, `${bar}.${beat}.${six}`);
  } else setText(RO.beat, "0.0.0");
  setText(RO.bpm, fmt(S.tempo.bpmAt(t), 2));
}
function onTimeChanged() {
  updateReadouts();
  placePlayhead();
  drawSchem(false);
  if (POPOUT) popScrubShow();
}
function followTo(t) {
  if (!$("#follow").checked) return;
  const x = xOf(t);
  if (x > S.width * 0.92 || x < 0) { S.view.t0 = t - (S.width * 0.08) / S.view.pps; clampView(); schedule(); }
}
let raf = 0;
function tick() {
  raf = 0;
  if (!playing()) return;
  const t0 = performance.now();
  if (POPOUT) { if (!Remote.main && curT() >= S.duration) { popStop(S.duration); return; } }
  else {
    followTo(curT());
    if (t0 - Pop.sentAt > 250) bcTime();
  }
  onTimeChanged();
  perfPush(S.perf.tick, performance.now() - t0);
  raf = requestAnimationFrame(tick);
}
function setPlayGlyph(on) {
  $("#play").classList.toggle("on", on);
  $("#play-glyph").setAttribute("d", on ? "M4 2.5h3v11H4zM9 2.5h3v11H9z" : "M4 2.5v11l9-5.5z");
}
function togglePlay() {
  if (POPOUT) return popToggle();
  if (audio.paused) {
    metroCtx();
    if (S.audioStale || Math.abs(audio.currentTime - S.pausedT) > 1e-3) { audio.currentTime = S.pausedT; S.audioStale = false; }
    audio.play().catch((e) => msg("audio: " + e.message, true));
  } else audio.pause();
}
audio.addEventListener("play", () => {
  setPlayGlyph(true);
  S.playStartT = curT();
  metroStart();
  pushPlayhead();
  bcTime();
  if (!raf) raf = requestAnimationFrame(tick);
});
audio.addEventListener("pause", () => {
  S.pausedT = clamp(audio.currentTime, S.tStart, S.duration);
  setPlayGlyph(false);
  metroStop();
  endTake();
  pushPlayhead();
  bcTime();
  onTimeChanged();
});
audio.addEventListener("seeked", () => { if (!audio.paused) { metroStart(); bcTime(); } });
audio.addEventListener("error", () => msg("audio failed to load from /api/audio", true));
function seek(t, fromEditor) {
  if (POPOUT) return popSeek(t);
  t = clamp(t, S.tStart, S.duration);
  S.pausedT = t;
  if (fromEditor && audio.paused) S.audioStale = true;
  else { S.audioStale = false; try { audio.currentTime = t; } catch { S.audioStale = true; } }
  if (audio.paused) { onTimeChanged(); bcTime(); }
  if (!fromEditor) pushPlayhead();
}
// Seek on click; on the ruler and waveform also scrub while dragging.
function attachSeek(cv, scrub) {
  cv.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    const r = cv.getBoundingClientRect();
    seek(tOf(e.clientX - r.left));
    if (!scrub) return;
    const move = (ev) => seek(tOf(ev.clientX - r.left));
    const up = () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  });
}
// wheel: scroll horizontally over the time canvases; Ctrl+wheel zooms about the cursor.
// Over the lane names it scrolls the lanes vertically as usual.
timeline.addEventListener("wheel", (e) => {
  if (!(e.target instanceof HTMLCanvasElement)) return;
  e.preventDefault();
  const r = e.target.getBoundingClientRect();
  if (e.ctrlKey) { zoomTo(S.view.pps * Math.pow(1.0015, -e.deltaY), e.clientX - r.left); return; }
  const d = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
  S.view.t0 += d / S.view.pps;
  clampView(); schedule();
}, { passive: false });

// ------------------------------------------------------------------------------------------- metronome
// A lookahead scheduler: every 25 ms while playing, clicks for the beats in the next 120 ms are queued on the
// AudioContext clock, mapped from the <audio> element's time. Beat 0 of every bar is accented. Nothing runs while paused.
const Metro = { ctx: null, on: lsGet("click", false), vol: lsGet("clickVol", 0.5), timer: 0, next: 0 };
function metroCtx() {
  if (!Metro.on) return null;
  if (!Metro.ctx) { try { Metro.ctx = new AudioContext(); } catch { return null; } }
  if (Metro.ctx.state === "suspended") Metro.ctx.resume();
  return Metro.ctx;
}
function metroStart() {
  metroStop();
  if (!Metro.on || audio.paused || !metroCtx()) return;
  Metro.next = Math.ceil(beatAt(curT()) - 1e-6);
  Metro.timer = setInterval(metroTick, 25);
  metroTick();
}
function metroStop() { if (Metro.timer) clearInterval(Metro.timer); Metro.timer = 0; }
function metroTick() {
  const ctx = Metro.ctx;
  if (!ctx || audio.paused) return;
  const at = audio.currentTime, now = ctx.currentTime, rate = audio.playbackRate || 1;
  for (let guard = 0; guard < 16; guard++) {
    const tb = timeAt(Metro.next);
    if (tb > at + 0.12) break;
    if (tb >= at - 0.03) clickAt(Math.max(now, now + (tb - at) / rate), ((Metro.next % BEATS_PER_BAR) + BEATS_PER_BAR) % BEATS_PER_BAR === 0);
    Metro.next++;
  }
}
function clickAt(when, accent) {
  const ctx = Metro.ctx, o = ctx.createOscillator(), g = ctx.createGain();
  o.frequency.value = accent ? 1760 : 1175;
  const v = Metro.vol * (accent ? 1 : 0.55);
  g.gain.setValueAtTime(0.0001, when);
  g.gain.linearRampToValueAtTime(Math.max(0.0002, v), when + 0.001);
  g.gain.exponentialRampToValueAtTime(0.0001, when + 0.045);
  o.connect(g).connect(ctx.destination);
  o.start(when); o.stop(when + 0.05);
}
function setClick(on) {
  Metro.on = on; lsSet("click", on);
  $("#click").classList.toggle("on", on);
  if (on) { metroCtx(); metroStart(); } else metroStop();
}
$("#click").onclick = () => setClick(!Metro.on);
$("#click-vol").value = Metro.vol;
$("#click-vol").addEventListener("input", (e) => { Metro.vol = +e.target.value; lsSet("clickVol", Metro.vol); });

// ------------------------------------------------------------------------------------------- tap-record
// While playing, digit keys add hits at (audio time - latency), snapped when Snap is on. Existing hits are never
// removed unless Replace is on: then, when the take ends, the plain hits it played over are cleared from the lanes
// that got taps (the new ones stay). One take = one undo step.
function trackForKey(d) {
  const tracks = Object.keys((S.notes && S.notes.tracks) || {});
  const m = S.keymap[d];
  if (m && tracks.includes(m)) return m;
  const used = new Set(Object.values(S.keymap).filter((t) => tracks.includes(t)));
  return m === undefined && !used.has(tracks[d - 1]) ? tracks[d - 1] || null : null;
}
function assignKey(d) {
  const t = S.hoverTrack;
  if (!t) return msg("hover over a note lane, then Shift+digit assigns that key to it", true);
  for (const k of Object.keys(S.keymap)) if (S.keymap[k] === t && +k !== d) delete S.keymap[k];
  S.keymap[d] = t;
  lsSet("keymap:" + S.rigPath, S.keymap);
  updateNoteCounts();
  msg(`key ${d} records into ${t}`);
}
function tap(d) {
  const track = trackForKey(d);
  if (!track) return msg(`key ${d} has no lane (Shift+${d} over a lane assigns it)`, true);
  const lat = (parseFloat($("#rec-lat").value) || 0) / 1000;
  const t = clamp(snapT(curT() - lat), S.tStart, S.duration);
  if (!S.take) S.take = { t0: S.playStartT, before: deep(S.notes), tracks: new Set(), added: new Set() };
  const h = { time: +t.toFixed(4), vel: 1 };
  (S.notes.tracks[track] = S.notes.tracks[track] || []).push(h);
  S.take.tracks.add(track); S.take.added.add(h);
  sortTrack(track); buildEx(track);
  scheduleKind(`canvas[data-kind="notes"][data-track="${CSS.escape(track)}"]`);
  const k = document.querySelector(`.lane.notes[data-track="${CSS.escape(track)}"] .keys`);
  if (k) { k.classList.add("hot"); setTimeout(() => k.classList.remove("hot"), 120); }
  updateNoteCounts();
  postNotes(400);
}
function endTake() {
  const take = S.take;
  if (!take) return;
  S.take = null;
  let removed = 0;
  if ($("#rec-replace").classList.contains("on")) {
    const t1 = curT();
    for (const tr of take.tracks) {
      const l = S.notes.tracks[tr] || [];
      S.notes.tracks[tr] = l.filter((h) => {
        const drop = h.pattern == null && !take.added.has(h) && h.time >= take.t0 - 1e-6 && h.time <= t1 + 1e-6;
        if (drop) removed++;
        return !drop;
      });
    }
    buildAllEx();
  }
  pushUndo("notes", take.before);
  notesChangedViews();
  postNotes();
  msg(`take: ${take.added.size} hit${take.added.size === 1 ? "" : "s"} into ${[...take.tracks].join(", ")}${removed ? `, replaced ${removed}` : ""}`);
}
$("#rec-replace").onclick = (e) => { e.currentTarget.classList.toggle("on"); lsSet("replace", e.currentTarget.classList.contains("on")); };
$("#rec-lat").addEventListener("change", (e) => lsSet("latency", +e.target.value));
$("#rec-lat").addEventListener("keydown", (e) => { if (e.key === "Enter") e.target.blur(); e.stopPropagation(); });

// ------------------------------------------------------------------------------------------- live link
// Playhead link rule: while this page plays it drives the editor and ignores the editor's playhead; while paused it
// follows the editor (events at <= 30 Hz), except for 600 ms after it last moved the editor itself.
function applyLive(st) {
  if (!st) return;
  S.live = st;
  S.link = !!st.link_playhead;
  syncMute();
  if (S.link && S.editorAudio == null && !S.audioProbe && !POPOUT) {      // linked already (a reload): ask the editor once
    S.audioProbe = true;
    api("/api/playhead").then(editorAudio, () => {}).finally(() => { S.audioProbe = false; });
  }
  $("#live").classList.toggle("on", !!st.enabled);
  $("#link").classList.toggle("on", S.link);
  const seq = $("#live-seq");
  if (document.activeElement !== seq && st.sequence && !seq.value) seq.value = st.sequence;
  const info = $("#live-info");
  let text = "", err = false;
  if (st.state === "offline") { text = "editor offline"; err = true; }
  else if (st.state === "unavailable") { text = "live link not installed"; err = true; }
  else if (st.enabled && st.pending) text = "pushing…";
  else if (st.enabled && st.last) {
    const p = st.last.pushed, n = Array.isArray(p) ? p.length : p;
    text = `pushed ${n ?? 0}${st.last.skipped != null ? `, kept ${Array.isArray(st.last.skipped) ? st.last.skipped.length : st.last.skipped}` : ""} · ${fmt(+st.last.ms, 0)} ms · ${st.last.at}`;
  } else if (st.enabled) text = "live";
  info.textContent = text;
  info.classList.toggle("err", err);
  info.title = st.error || (st.last ? JSON.stringify(st.last) : "");
}
$("#live").onclick = async () => {
  const on = !S.live.enabled, seq = $("#live-seq").value.trim();
  try { applyLive(await api("/api/live", { enabled: on, sequence: seq })); msg(on ? `live: pushing changes into ${seq}` : "live off"); }
  catch (e) { msg("live: " + e.message, true); const i = $("#live-info"); i.textContent = e.message; i.classList.add("err"); }
};
$("#live-seq").addEventListener("change", async (e) => {
  lsSet("sequence", e.target.value.trim());
  try { applyLive(await api("/api/live", { sequence: e.target.value.trim() })); } catch (err) { msg("live: " + err.message, true); }
});
$("#live-seq").addEventListener("keydown", (e) => { if (e.key === "Enter") e.target.blur(); e.stopPropagation(); });
$("#link").onclick = async () => {
  try {
    applyLive(await api("/api/live", { link_playhead: !S.link }));
    if (S.link) pushPlayhead();
    msg(S.link ? "playhead linked: playing here drives the editor; paused, this page follows it" : "playhead unlinked");
  } catch (e) { msg("link: " + e.message, true); }
};
// Linked to a sequence that plays the music itself (an audio track): this page goes quiet while linked, so the song
// isn't heard twice. Unlinking, or a sequence without audio, gives the page its sound back.
function editorAudio(d) {
  if (!d || d.audio == null) return;
  if (S.live.sequence && d.sequence && d.sequence.split(".")[0] !== S.live.sequence.split(".")[0]) return;
  const was = !!S.editorAudio;
  S.editorAudio = !!d.audio;
  syncMute();
  if (S.link && S.editorAudio && !was) msg("the sequence plays the music: page audio muted while linked");
}
function syncMute() { audio.muted = !!(S.link && S.editorAudio); }
let phBusy = false, phNext = null;
function pushPlayhead() {
  if (!S.link || POPOUT) return;
  phNext = { frame: frameOf(curT()), playing: !audio.paused };
  S.lastPagePush = performance.now();
  if (phBusy) return;
  const send = async () => {
    const body = phNext; phNext = null; phBusy = true;
    try { editorAudio(await api("/api/playhead", body)); } catch (e) { msg("playhead: " + e.message, true); }
    phBusy = false;
    if (phNext) send();
  };
  send();
}
function onEditorPlayhead(d) {
  if (S.link) editorAudio(d);
  if (!S.link || playing() || d == null || d.frame == null) return;
  if (POPOUT && Remote.main) return;                                              // the main page drives the pop-out
  if (S.live.sequence && d.sequence && d.sequence !== S.live.sequence) return;    // another sequence is open: ignore it
  if (performance.now() - S.lastPagePush < 600) return;
  const t = tOfFrame(+d.frame);
  if (Math.abs(t - S.pausedT) < 0.5 / S.fps) return;
  seek(t, true);
  followTo(t);
}
$("#scan").onclick = async () => {
  const b = $("#scan"); b.disabled = true; b.textContent = "Scanning…";
  try {
    const d = await api("/api/scan", {});
    S.scene = d.scene; S.geo = null;
    sceneInfo(); drawSchem();
    // the server may have pointed the rig at the new scene.json (an unsaved rig edit): take its rig and dirty state
    await reloadState(`scanned ${Object.keys(d.scene.actors || {}).length} actors into ${d.path.split(/[\\/]/).pop()}`);
  } catch (e) { msg("scan: " + e.message, true); }
  b.disabled = false; b.textContent = "Scan";
};
$("#flash").onclick = () => {
  const all = [];
  for (const l of Object.values(S.flags || {})) for (const [a] of l) all.push(a);
  all.sort((a, b) => a - b);
  if (!all.length) return;
  const t = curT(), next = all.find((a) => a > t + 0.05) ?? all[0];
  seek(Math.max(0, next - 0.5));
  if (xOf(next) < 0 || xOf(next) > S.width) { S.view.t0 = next - (S.width * 0.2) / S.view.pps; clampView(); schedule(); }
};

// ------------------------------------------------------------------------------------------- pop-out window sync
// Main page -> pop-out: {type: "time", t, playing, rate} on play, pause and seek, every 250 ms while playing and in
// answer to each pop-out heartbeat (the pop-out extrapolates in between). Pop-out -> main page: "hello", "popout"
// heartbeats every 2 s (the main page folds its own preview away while a pop-out is open) and "cmd" toggle / seek: the
// pop-out's transport drives the main page, which owns the audio, so the song is never heard twice. Edits reach the
// pop-out through the server's event stream, like any other change. With no main page the pop-out plays silently.
const Pop = { bc: null, sentAt: 0, open: false, at: 0, here: false, scrubF: null };
try { Pop.bc = new BroadcastChannel("beatrig-preview"); } catch { /* no BroadcastChannel: no pop-out */ }
function bcPost(m) { if (Pop.bc) { try { Pop.bc.postMessage(m); } catch { /* closed */ } } }
function bcTime() {
  if (!Pop.bc || POPOUT || !Pop.open) return;
  Pop.sentAt = performance.now();
  bcPost({ type: "time", t: curT(), playing: !audio.paused, rate: audio.playbackRate || 1 });
}
// main page: fold the preview away while a pop-out shows it ("Show it here too" keeps both)
function setAway() {
  const away = Pop.open && !Pop.here;
  if (away === !!P3.away) return;
  P3.away = away;
  $("#schem-away").hidden = !away;
  schemView.classList.toggle("away", away);
  if (away) meterCv.hidden = true;
  else { P3.meterStale = true; P3.ovStale = true; if (P3.r) sizeView3(); drawSchem(); }
}
function initMainSync() {
  const btn = $("#popout");
  if (!Pop.bc) { btn.disabled = true; btn.title = "This browser has no BroadcastChannel, so a pop-out window can't follow the page"; return; }
  btn.onclick = () => {
    const w = window.open("/preview", "beatrig-preview", "popup,width=1100,height=760");
    if (!w) return msg("the browser blocked the pop-out window: allow pop-ups for this page", true);
    w.focus();
  };
  Pop.bc.onmessage = (e) => {
    const d = e.data || {};
    if (d.type === "popout") {
      if (d.open) { Pop.open = true; Pop.at = performance.now(); bcTime(); }
      else { Pop.open = false; Pop.here = false; }
      setAway();
    } else if (d.type === "hello") { Pop.open = true; Pop.at = performance.now(); bcTime(); setAway(); }
    else if (d.type === "cmd" && Pop.open) {
      if (d.cmd === "toggle") togglePlay();
      else if (d.cmd === "seek" && Number.isFinite(d.t)) { seek(d.t); followTo(curT()); }
    }
  };
  // a pop-out that vanished without saying so (crashed, killed) stops counting after 6 s
  setInterval(() => { if (Pop.open && performance.now() - Pop.at > 6000) { Pop.open = false; Pop.here = false; setAway(); } }, 2000);
  $("#schem-back").onclick = () => { Pop.here = true; setAway(); };
  window.addEventListener("pagehide", () => bcPost({ type: "bye" }));
}
// pop-out: follow the main page, or play on a silent clock of its own
function initPopout() {
  const info = $("#sync-info");
  info.hidden = false;
  const syncInfo = () => {
    const t = Remote.main ? "following the main page" : "no main page open: plays silently";
    if (info.textContent !== t) info.textContent = t;
    info.title = Remote.main ? "Play, pause and seek here drive the main page, which plays the audio" : "Open the main page for sound: this window never plays audio";
  };
  Pop.syncInfo = syncInfo;
  syncInfo();
  if (Pop.bc) {
    Pop.bc.onmessage = (e) => {
      const d = e.data || {};
      if (d.type === "time" && Number.isFinite(d.t)) {
        Remote.main = true; Remote.mainAt = performance.now(); Remote.rate = d.rate || 1;
        if (d.playing) {
          Remote.t0 = d.t; Remote.at = performance.now();
          if (!Remote.playing) { Remote.playing = true; setPlayGlyph(true); if (!raf) raf = requestAnimationFrame(tick); }
        } else {
          Remote.playing = false; setPlayGlyph(false);
          S.pausedT = clamp(d.t, S.tStart, S.duration);
          onTimeChanged();
        }
        syncInfo();
      } else if (d.type === "bye") { Remote.main = false; popStop(curT()); syncInfo(); }
    };
    const beat = () => {
      bcPost({ type: "popout", open: true });
      // the main page answers every heartbeat; silence for 6 s means it is gone
      if (Remote.main && performance.now() - Remote.mainAt > 6000) { Remote.main = false; popStop(curT()); syncInfo(); }
    };
    bcPost({ type: "hello" });
    beat();
    setInterval(beat, 2000);
    window.addEventListener("pagehide", () => bcPost({ type: "popout", open: false }));
  }
  const scrub = $("#pop-scrub");
  scrub.addEventListener("input", () => popSeek(S.tStart + (+scrub.value / 1000) * (S.duration - S.tStart)));
  popScrubShow();
}
function popStop(t) {
  Remote.playing = false;
  S.pausedT = clamp(t, S.tStart, S.duration);
  setPlayGlyph(false);
  onTimeChanged();
}
function popToggle() {
  if (Remote.main) return bcPost({ type: "cmd", cmd: "toggle" });
  if (Remote.playing) return popStop(curT());
  Remote.t0 = S.pausedT >= S.duration - 1e-3 ? S.tStart : S.pausedT;
  Remote.at = performance.now(); Remote.rate = 1; Remote.playing = true;
  setPlayGlyph(true);
  if (!raf) raf = requestAnimationFrame(tick);
}
function popSeek(t) {
  t = clamp(t, S.tStart, S.duration);
  if (Remote.main) bcPost({ type: "cmd", cmd: "seek", t });
  S.pausedT = t; Remote.t0 = t; Remote.at = performance.now();
  if (!Remote.playing) onTimeChanged();
}
function popScrubShow() {
  const scrub = $("#pop-scrub"), f = frameOf(curT());
  if (f === Pop.scrubF || (mouseDown && document.activeElement === scrub)) return;
  Pop.scrubF = f;
  scrub.value = Math.round(((curT() - S.tStart) / Math.max(1e-6, S.duration - S.tStart)) * 1000);
}
// the pop-out's keys: Space, V, Home, Left / Right (Shift: a beat); nothing that edits
function popKey(e) {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === " ") { e.preventDefault(); popToggle(); }
  else if (e.key.toLowerCase() === "v") cycleView3();
  else if (e.key === "Home") popSeek(S.tStart);
  else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
    e.preventDefault();
    const dir = e.key === "ArrowLeft" ? -1 : 1, t = curT();
    popSeek(e.shiftKey ? timeAt(Math.round(beatAt(t)) + dir) : tOfFrame(frameOf(t) + dir) + 1e-4);
  }
}

// ------------------------------------------------------------------------------------------- events (SSE)
function connectEvents() {
  const es = new EventSource("/api/events");
  es.addEventListener("rev", (e) => {
    const d = JSON.parse(e.data);
    renderActivity(d.activity || []);
    if (S.server && d.server && d.server !== S.server) return reloadState("the server restarted");
    S.maxSeenRev = Math.max(S.maxSeenRev || 0, d.rev);
    if (S.rev != null && d.rev > S.rev) reloadState();
  });
  es.addEventListener("live", (e) => applyLive(JSON.parse(e.data)));
  es.addEventListener("playhead", (e) => onEditorPlayhead(JSON.parse(e.data)));
  const conn = $("#conn");
  es.onopen = () => {
    conn.textContent = "connected"; conn.classList.remove("lost");
    if ($("#msg").textContent.startsWith("lost the server")) msg("reconnected");
  };
  es.onerror = () => {
    if (es.readyState === EventSource.OPEN) return;
    conn.textContent = "reconnecting"; conn.classList.add("lost");
    msg("lost the server; reconnecting", true);
  };
}
let mouseDown = false;
window.addEventListener("mousedown", () => { mouseDown = true; }, true);
window.addEventListener("mouseup", () => {
  mouseDown = false;
  if (S.pendingReload && !S.baking && !S.queued) { S.pendingReload = false; if (S.maxSeenRev > S.rev) reloadState(); }
}, true);
let reloading = false;
async function reloadState(why) {
  // our own edits come back with their rev in the response; anything newer is someone else (an agent)
  if (S.baking || S.queued || mouseDown || document.activeElement.matches("#param-list input, #graph-form *, #rename")) { S.pendingReload = true; return; }
  if (reloading) { S.pendingReload = true; return; }
  reloading = true;
  try {
    const d = await api("/api/state");
    if (S.baking || S.queued) { S.pendingReload = true; return; }
    if (d.rev === S.rev && d.server === S.server && !why) return;
    loadState(d, false);
    const last = (d.activity || [])[d.activity.length - 1];
    msg(why || (last ? `${last.source}: ${last.action}${last.detail ? " — " + last.detail : ""}` : "state reloaded"));
  } catch (err) {
    msg("lost the server: " + err.message, true);
  } finally { reloading = false; }
}
let activityKey = "";
function renderActivity(list) {
  const key = list.map((a) => a.rev).join();
  if (key === activityKey) return;
  activityKey = key;
  const ol = $("#agent-list");
  ol.textContent = "";
  const recent = list.slice(-5).reverse();
  if (!recent.length) { ol.innerHTML = '<li class="empty-row">No agent activity yet. Changes an agent makes over MCP show up here.</li>'; return; }
  for (const a of recent) {
    const li = document.createElement("li");
    if (a.rev > (S.seenRev || 0)) li.className = "fresh";
    li.innerHTML = '<span class="at"></span><span class="src"></span><span class="what"><b></b><span></span></span>';
    li.querySelector(".at").textContent = a.time;
    li.querySelector(".src").textContent = a.source;
    li.querySelector("b").textContent = a.action;
    li.querySelector(".what span").textContent = a.detail || "";
    li.title = `${a.time}  ${a.source}  ${a.action}  ${a.detail || ""}`;
    ol.append(li);
  }
  $("#agent-rev").textContent = `rev ${S.rev}`;
}

// ------------------------------------------------------------------------------------------- keyboard
$("#play").onclick = togglePlay;
$("#zoom").addEventListener("input", (e) => {
  const a = Math.log(fitPps()), b = Math.log(maxPps());
  const ph = xOf(curT());
  zoomTo(Math.exp(a + (b - a) * (+e.target.value / 1000)), ph >= 0 && ph <= S.width ? ph : S.width / 2);
});
$("#zoom-in").onclick = () => zoomTo(S.view.pps * 1.5, clamp(xOf(curT()), 0, S.width));
$("#zoom-out").onclick = () => zoomTo(S.view.pps / 1.5, clamp(xOf(curT()), 0, S.width));
$("#zoom-fit").onclick = zoomFit;
$("#grid").onchange = () => schedule();
$("#save").onclick = save;
$("#mk-pat").onclick = makePattern;
$("#dup-pat").onclick = copyPatterns;
$("#unlink-pat").onclick = unlinkPatterns;
$("#add-sec").onclick = () => addSection(snapT(curT()), true);
$("#undo").onclick = () => undoRedo(S.undo, S.redo, "undo");
$("#redo").onclick = () => undoRedo(S.redo, S.undo, "redo");
$("#help").onclick = () => { $("#help-panel").hidden = !$("#help-panel").hidden; };
for (const id of ["#snap", "#follow", "#grid", "#zoom", "#click-vol", "#pop-scrub"]) $(id).addEventListener("keydown", (e) => { if (e.key === " ") e.preventDefault(); });

document.addEventListener("keydown", (e) => {
  if (POPOUT) return popKey(e);
  const mod = e.ctrlKey || e.metaKey, key = e.key.toLowerCase();
  if (mod && key === "s") { e.preventDefault(); save(); return; }
  if (e.target instanceof Element && e.target.matches("input[type=number], input[type=text], textarea, select, #graph-form *")) return;
  if (mod && key === "z") { e.preventDefault(); if (e.shiftKey) undoRedo(S.redo, S.undo, "redo"); else undoRedo(S.undo, S.redo, "undo"); return; }
  if (mod && key === "y") { e.preventDefault(); undoRedo(S.redo, S.undo, "redo"); return; }
  if (mod && key === "d") { e.preventDefault(); copyPatterns(); return; }
  if (mod || e.altKey) return;
  const dm = /^(?:Digit|Numpad)([1-9])$/.exec(e.code);
  if (dm) {
    e.preventDefault();
    if (e.shiftKey) assignKey(+dm[1]);
    else if (!audio.paused) tap(+dm[1]);
    else msg("tap-record works while playing (Space)");
    return;
  }
  if (e.key === " ") { e.preventDefault(); togglePlay(); }
  else if (e.key === "Delete" || e.key === "Backspace") { e.preventDefault(); deleteSelected(); }
  else if (e.key === "Escape") { S.sel.clear(); selChanged(); closeForm(); $("#help-panel").hidden = true; }
  else if (e.key === "Home") { seek(S.tStart); S.view.t0 = S.tStart; clampView(); schedule(); }
  else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
    e.preventDefault();
    const dir = e.key === "ArrowLeft" ? -1 : 1;
    if (!nudgeSelected(dir, e.shiftKey)) { const t = curT(); seek(e.shiftKey ? timeAt(Math.round(beatAt(t)) + dir) : tOfFrame(frameOf(t) + dir) + 1e-4); followTo(curT()); }
  } else if (e.key === "ArrowUp" || e.key === "ArrowDown") { e.preventDefault(); velSelected(e.key === "ArrowUp" ? 0.05 : -0.05); }
  else if (key === "c") setClick(!Metro.on);
  else if (key === "m") addSection(snapT(curT()), false);
  else if (key === "p") makePattern();
  else if (key === "u") unlinkPatterns();
  else if (key === "f") { $("#follow").checked = !$("#follow").checked; msg(`follow ${$("#follow").checked ? "on" : "off"}`); }
  else if (key === "v") cycleView3();
  else if (e.key === "=" || e.key === "+") zoomTo(S.view.pps * 1.5, clamp(xOf(curT()), 0, S.width));
  else if (e.key === "-") zoomTo(S.view.pps / 1.5, clamp(xOf(curT()), 0, S.width));
  else if (e.key === "\\") zoomFit();
  else if (e.key === "?") $("#help-panel").hidden = !$("#help-panel").hidden;
});
window.addEventListener("beforeunload", (e) => { if (S.dirty && !POPOUT) { e.preventDefault(); e.returnValue = ""; } });
// Buttons and checkboxes give focus back after a click, so Space always means play/pause
// (a focused button would otherwise also activate on Space and toggle twice).
document.addEventListener("click", (e) => {
  const b = e.target.closest && e.target.closest("button, input[type=checkbox]");
  if (b && !b.closest("#graph-form")) b.blur();
});

function measure() {
  const cv = document.querySelector('canvas[data-kind="ruler"]');
  const w = Math.round(cv.getBoundingClientRect().width) || 600;
  gutterW = $(".lane-head").getBoundingClientRect().width;
  const changed = w !== S.width;
  S.width = w;
  return changed;
}
new ResizeObserver(() => {
  if (measure()) syncZoomSlider();
  S.geo = null;
  fitLaneHeights(); schedule(); drawSchem();
}).observe(timeline);

// ------------------------------------------------------------------------------------------- boot
const baseName = (p) => (p || "").split(/[\\/]/).pop();
// Apply a full /api/state snapshot. On a reload (agent change) the view, zoom and playhead are kept.
function loadState(d, first) {
  $("#boot-error").hidden = true;
  S.rig = d.rig; S.notes = d.notes; S.analysis = d.analysis || {}; S.scene = d.scene || null;
  S.lastGood = { rig: deep(d.rig), notes: deep(d.notes) };
  S.opParams = d.op_params || {};
  S.rev = d.rev; S.server = d.server;
  S.rigPath = d.rig_path;
  S.fps = (d.bake && d.bake.fps) || d.rig.fps || 30;
  S.offset = d.rig.audio_offset_frames || 0;
  S.tStart = +d.rig.audio_in_s || 0;
  const full = S.analysis.duration || S.tStart + (((d.bake && d.bake.frames) || 0) - S.offset) / S.fps || 1;
  S.duration = Math.max(S.tStart + 0.1, d.rig.audio_out_s != null ? Math.min(+d.rig.audio_out_s, full) : full);
  rebuildTempo();
  $("#engine").textContent = d.safety ? "flash check on" : "flash check off";
  const paths = $("#paths");
  paths.textContent = `${baseName(d.rig_path)}  ·  ${baseName(d.notes_path)}${d.scene ? "  ·  " + baseName(d.scene_path) : ""}`;
  paths.title = `${d.rig_path}\n${d.notes_path}\n${d.scene_path}`;
  $("#audio-name").textContent = baseName(d.rig.audio);
  if (first) {
    measure(); S.view.pps = fitPps(); S.view.t0 = S.tStart; S.pausedT = S.tStart;
    S.keymap = lsGet("keymap:" + d.rig_path, {}) || {};
  } else {
    S.undo.length = 0; S.redo.length = 0; updUndo();   // the files changed under us: old undo points would revert them
  }
  S.sel.clear();
  buildAllEx();
  buildNoteLanes();
  S.bake = null;                       // force the curve lanes to rebuild
  applyBake(d);
  buildParams();
  applyLive(d.live);
  sceneInfo();
  selChanged();
  notesChangedViews();
  clampView(); syncZoomSlider();
  onTimeChanged();
  renderActivity(d.activity || []);
  const pk = `${d.rig.audio}|${S.analysis.duration}|${S.analysis.sample_rate}`;
  if (pk !== S.peaksKey && !POPOUT) { S.peaksKey = pk; loadPeaks(); }
}
async function boot() {
  if (POPOUT) { document.body.classList.add("popout"); document.title = "beatrig preview"; }
  for (const k of ["ruler", "sections", "wave"]) canvases.add(document.querySelector(`canvas[data-kind="${k}"]`));
  attachSeek(document.querySelector('canvas[data-kind="ruler"]'), true);
  attachSeek(document.querySelector('canvas[data-kind="wave"]'), true);
  const t0 = performance.now(), first = api("/api/state");          // in flight while the 3D preview starts
  init3D();
  S.perf.glInitMs = performance.now() - t0;
  setView3(lsGet(POPOUT ? "view3:popout" : "view3", "orbit"));
  $("#click").classList.toggle("on", Metro.on);
  $("#rec-replace").classList.toggle("on", !!lsGet("replace", false));
  $("#rec-lat").value = lsGet("latency", 30);
  $("#live-seq").value = lsGet("sequence", "");
  if (!POPOUT) audio.src = "/api/audio";                            // the pop-out never plays sound
  try {
    const d = await first;
    S.perf.stateMs = performance.now() - t0;
    S.seenRev = d.rev;
    loadState(d, true);
    msg(`loaded ${Object.keys(d.notes.tracks || {}).length} note tracks, ${(d.bake.outputs || []).length} outputs`);
    if (POPOUT) initPopout(); else initMainSync();
  } catch (err) {
    msg("failed to load /api/state: " + err.message, true);
    showError(err.message);
    $("#boot-error-msg").textContent = err.message;
    $("#boot-error").hidden = false;
  }
  connectEvents();
}

// ------------------------------------------------------------------------------------------- perf probes
// S.perf holds the last 300 redraw / playback-frame / preview times (ms). beatrigBench() redraws every lane on
// screen n times at the current view and reports per-kind costs; the perf table in TASKS.md comes from these.
window.beatrigPerf = () => {
  const st = (a) => { if (!a.length) return null; const s = [...a].sort((x, y) => x - y); return { n: a.length, p50: +s[s.length >> 1].toFixed(2), p95: +s[Math.floor(s.length * 0.95)].toFixed(2), max: +s[s.length - 1].toFixed(2) }; };
  return { glInitMs: S.perf.glInitMs != null ? +S.perf.glInitMs.toFixed(1) : null, stateMs: S.perf.stateMs && +S.perf.stateMs.toFixed(1), dataReadyMs: S.perf.dataReady && +S.perf.dataReady.toFixed(0), firstPaintMs: S.perf.firstPaint && +S.perf.firstPaint.toFixed(0), flush: st(S.perf.flush), tick: st(S.perf.tick), schem: st(S.perf.schem), drawn: S.perf.drawn, skipped: S.perf.skipped };
};
window.beatrigBench = (n = 20) => {
  const vr = timeline.getBoundingClientRect(), kinds = {}, all = [];
  const vis = [...canvases].filter((c) => { if (!c.isConnected || c.offsetParent === null) return false; const r = c.getBoundingClientRect(); return !(r.bottom < vr.top || r.top > vr.bottom); });
  for (let i = 0; i < n; i++) {
    const t0 = performance.now();
    for (const c of vis) { const a = performance.now(); drawCanvas(c); const k = c.dataset.kind; (kinds[k] = kinds[k] || []).push(performance.now() - a); }
    const s0 = performance.now(); drawSchem(); (kinds.preview = kinds.preview || []).push(performance.now() - s0);
    all.push(performance.now() - t0);
  }
  const avg = (a) => +(a.reduce((x, y) => x + y, 0) / a.length).toFixed(2);
  const per = {}; for (const k in kinds) per[k] = { avg: avg(kinds[k]), max: +Math.max(...kinds[k]).toFixed(2), lanes: kinds[k].length / n };
  return { visible: vis.length, frameAvg: avg(all), frameMax: +Math.max(...all).toFixed(2), pps: +S.view.pps.toFixed(3), framesPerPx: +(S.fps / S.view.pps).toFixed(2), per };
};
boot();
