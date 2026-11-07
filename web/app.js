// beatrig preview page. Plain script, no build, no dependencies.
// One shared time axis (audio seconds) across the ruler, waveform, band, note and curve lanes.
// Canvases redraw only when something changes; the only animation loop runs while the audio plays.
"use strict";

const $ = (s) => document.querySelector(s);
const audio = $("#audio");
const timeline = $("#timeline");
const lanesEl = $("#lanes");
const playheadEl = $("#playhead");

const css = getComputedStyle(document.documentElement);
const C = {
  bg: "#111213", panel: "#161719", line: "#24262a", line2: "#303237",
  text: "#cdd0d5", dim: "#7b8089", faint: "#4b4f56", wave: "#6c717a",
  accent: css.getPropertyValue("--accent").trim() || "#e2a23a",
  note: "#a7abb2", light: [255, 214, 150],
};
const MAX_CANVAS_PX = 4096;
const BEATS_PER_BAR = 4; // DESIGN.md has no meter; 4/4 assumed for downbeats and bar numbers

const S = {
  rig: null, notes: null, bake: null, bands: [], analysis: null,
  dirty: false, bakeMs: null, engine: "",
  duration: 1, fps: 30, offset: 0,
  view: { t0: 0, pps: 50 },        // seconds at the left edge, pixels per second
  width: 600,                       // canvas area width in CSS px
  sel: new Set(),                   // selected notes, as "track:index"
  collapsed: new Set(), stacked: new Set(), seenOutputs: new Set(),
  baking: false, queued: null, rev: null, seenRev: 0,
  lastGood: { rig: null, notes: null },
};

// ------------------------------------------------------------------------------------------- utils
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const fmt = (v, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : "-");
const deep = (o) => JSON.parse(JSON.stringify(o));
function msg(text, isErr = false) {
  const m = $("#msg");
  m.textContent = text || "";
  m.classList.toggle("err", !!isErr);
}
const xOf = (t) => (t - S.view.t0) * S.view.pps;
const tOf = (x) => S.view.t0 + x / S.view.pps;
const frameOf = (t) => Math.floor(t * S.fps + 1e-6) + S.offset; // sample at frame start: never ahead of the audio
const tOfFrame = (f) => (f - S.offset) / S.fps;

// ------------------------------------------------------------------------------------------- tempo grid
// From notes.json: bpm_map [{time, bpm}] + grid_offset (time of beat 0). Piecewise-constant tempo.
function gridSegs() {
  const n = S.notes || {};
  const map = (n.bpm_map && n.bpm_map.length ? n.bpm_map : [{ time: 0, bpm: rigBpm() || 120 }])
    .slice().sort((a, b) => a.time - b.time);
  const g0 = n.grid_offset || 0;
  const bpmAt = (t) => { let b = map[0].bpm; for (const m of map) if (m.time <= t) b = m.bpm; return b; };
  const segs = [{ t: g0, b: 0, bpm: bpmAt(g0) }];
  for (const m of map) {
    if (m.time <= g0) continue;
    const p = segs[segs.length - 1];
    segs.push({ t: m.time, b: p.b + (m.time - p.t) * p.bpm / 60, bpm: m.bpm });
  }
  return segs;
}
function rigBpm() {
  const t = S.rig && S.rig.tempo;
  if (t && typeof t.bpm === "number") return t.bpm;
  return S.analysis && S.analysis.bpm;
}
let SEGS = [{ t: 0, b: 0, bpm: 120 }];
function beatAt(t) {
  let s = SEGS[0];
  for (const x of SEGS) if (x.t <= t) s = x;
  return s.b + (t - s.t) * s.bpm / 60;
}
function timeAt(b) {
  let s = SEGS[0];
  for (const x of SEGS) if (x.b <= b) s = x;
  return s.t + (b - s.b) * 60 / s.bpm;
}
function snapT(t) {
  if (!$("#snap").checked) return t;
  const g = parseFloat($("#grid").value);
  return timeAt(Math.round(beatAt(t) / g) * g);
}
const gridSize = () => parseFloat($("#grid").value);

// Visible grid lines: [{x, beat, level}] level 2 = bar, 1 = beat, 0 = subdivision.
function gridLines(w, sub) {
  const out = [];
  const b0 = Math.floor(beatAt(tOf(0))), b1 = Math.ceil(beatAt(tOf(w)));
  const pxPerBeat = S.view.pps * 60 / SEGS[0].bpm;
  const showBeats = pxPerBeat > 6;
  const step = sub && pxPerBeat * sub > 7 ? sub : 1;
  if (!showBeats) {
    // bars only, possibly every Nth bar
    let every = 1;
    while (pxPerBeat * BEATS_PER_BAR * every < 10) every *= 2;
    const start = Math.floor(b0 / (BEATS_PER_BAR * every)) * BEATS_PER_BAR * every;
    for (let b = start; b <= b1; b += BEATS_PER_BAR * every) out.push({ x: xOf(timeAt(b)), beat: b, level: 2 });
    return out;
  }
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
  for (const g of gridLines(w, sub)) {
    if (g.x < -1 || g.x > w + 1) continue;
    ctx.fillStyle = g.level === 2 ? `rgba(150,155,165,${0.30 * strong})` : g.level === 1 ? `rgba(150,155,165,${0.13 * strong})` : `rgba(150,155,165,${0.06 * strong})`;
    ctx.fillRect(Math.round(g.x), 0, 1, h);
  }
}

// ------------------------------------------------------------------------------------------- canvases
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
const dirtyCanvases = new Set();
function schedule(cv) {
  if (cv) dirtyCanvases.add(cv); else for (const c of canvases) dirtyCanvases.add(c);
  if (redrawPending) return;
  redrawPending = true;
  requestAnimationFrame(() => {
    redrawPending = false;
    for (const c of dirtyCanvases) if (c.isConnected && c.offsetParent !== null) drawCanvas(c);
    dirtyCanvases.clear();
    placePlayhead();
  });
}
function drawCanvas(cv) {
  const { ctx, w, h } = setupCanvas(cv);
  const k = cv.dataset.kind;
  if (k === "ruler") drawRuler(ctx, w, h);
  else if (k === "wave") drawWave(ctx, w, h);
  else if (k === "band") drawBand(ctx, w, h, cv.dataset.id);
  else if (k === "notes") drawNotes(ctx, w, h, cv.dataset.track);
  else if (k === "curve") drawCurve(ctx, w, h, cv.dataset.id);
}

// ------------------------------------------------------------------------------------------- ruler + waveform
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
  ctx.font = "11px Consolas, monospace";
  ctx.textBaseline = "top";
  for (let t = t0, i = 0; t <= tOf(w) && i < 2000; t += minor, i++) {
    const x = Math.round(xOf(t));
    const major = Math.abs(t / step - Math.round(t / step)) < 1e-6;
    ctx.fillStyle = major ? C.dim : C.faint;
    ctx.fillRect(x, major ? h - 9 : h - 4, 1, major ? 9 : 4);
    if (major && t >= -1e-9) { ctx.fillStyle = C.dim; ctx.fillText(tLabel(Math.max(0, t), step), x + 3, 3); }
  }
  // end of clip
  const xe = xOf(S.duration);
  if (xe < w) { ctx.fillStyle = "rgba(0,0,0,0.35)"; ctx.fillRect(xe, 0, w - xe, h); }
}
function drawWave(ctx, w, h) {
  ctx.fillStyle = C.bg; ctx.fillRect(0, 0, w, h);
  drawGrid(ctx, w, h, gridSize(), 1.2);
  // bar numbers
  ctx.font = "10px Consolas, monospace"; ctx.textBaseline = "top"; ctx.fillStyle = C.faint;
  let lastX = -1e9;
  for (const g of gridLines(w, 0)) if (g.level === 2 && g.x - lastX > 28 && g.x >= 0) {
    ctx.fillText(String(Math.round(g.beat / BEATS_PER_BAR) + 1), g.x + 3, 2); lastX = g.x;
  }
  const p = S.analysis && S.analysis.peaks;
  const mid = h / 2 + 4, amp = h / 2 - 8;
  if (p && p.data && p.data.length) {
    const d = p.data, n = d.length / 2, rate = p.rate;
    ctx.fillStyle = C.wave;
    for (let x = 0; x < w; x++) {
      const i0 = Math.max(0, Math.floor(tOf(x) * rate)), i1 = Math.min(n, Math.max(i0 + 1, Math.floor(tOf(x + 1) * rate)));
      if (i0 >= n || i1 <= 0) continue;
      let lo = 127, hi = -127;
      for (let i = i0; i < i1; i++) { if (d[2 * i] < lo) lo = d[2 * i]; if (d[2 * i + 1] > hi) hi = d[2 * i + 1]; }
      const y0 = mid - (hi / 127) * amp, y1 = mid - (lo / 127) * amp;
      ctx.fillRect(x, y0, 1, Math.max(1, y1 - y0));
    }
  } else {
    ctx.fillStyle = C.faint; ctx.font = "11px Segoe UI, sans-serif";
    ctx.fillText("no waveform (wav only)", 8, h / 2);
  }
  // detected beats from the analysis: short ticks along the bottom edge
  const beats = S.analysis && S.analysis.beats;
  if (beats && beats.length) {
    ctx.fillStyle = C.accent; ctx.globalAlpha = 0.7;
    for (const t of beats) { const x = xOf(t); if (x >= 0 && x <= w) ctx.fillRect(Math.round(x), h - 5, 1, 5); }
    ctx.globalAlpha = 1;
  }
  const xe = xOf(S.duration);
  if (xe < w) { ctx.fillStyle = "rgba(0,0,0,0.35)"; ctx.fillRect(xe, 0, w - xe, h); }
}

// Series helpers: a channel (or band) has values (maybe min/max pairs with stride) and maybe keys.
function seriesValueAtFrame(ch, f) {
  const k = ch.keys;
  if (k && k.length) {
    if (f <= k[0][0]) return k[0][1];
    if (f >= k[k.length - 1][0]) return k[k.length - 1][1];
    let lo = 0, hi = k.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (k[m][0] <= f) lo = m; else hi = m; }
    const a = k[lo], b = k[hi];
    return a[1] + (b[1] - a[1]) * (f - a[0]) / (b[0] - a[0]);
  }
  const v = ch.values || [];
  if (!v.length) return 0;
  if (ch.minmax) { const i = clamp(Math.floor(f / ch.stride), 0, v.length / 2 - 1); return v[2 * i + 1]; }
  return v[clamp(f, 0, v.length - 1)];
}
// Draw a series between y(lo..hi) in a box; aggregates to min/max per pixel column.
function drawSeries(ctx, ch, w, top, height, lo, hi, color, fill) {
  const span = hi - lo || 1;
  const yOf = (v) => top + height - clamp((v - lo) / span, -0.05, 1.05) * height;
  const pts = []; // [frame, value]
  const f0 = frameOf(tOf(0)) - 1, f1 = frameOf(tOf(w)) + 2;
  if (ch.keys && ch.keys.length) {
    const k = ch.keys;
    let a = 0, b = k.length - 1;
    while (b - a > 1) { const m = (a + b) >> 1; if (k[m][0] <= f0) a = m; else b = m; }
    for (let i = a; i < k.length; i++) { pts.push(k[i]); if (k[i][0] > f1) break; }
  } else if (ch.values) {
    const v = ch.values, st = ch.stride || 1;
    if (ch.minmax) {
      for (let i = Math.max(0, Math.floor(f0 / st)); i < v.length / 2 && i * st <= f1; i++) {
        pts.push([i * st, v[2 * i]]); pts.push([i * st + st / 2, v[2 * i + 1]]);
      }
    } else for (let f = Math.max(0, f0); f < v.length && f <= f1; f++) pts.push([f, v[f]]);
  }
  if (!pts.length) return;
  ctx.beginPath();
  let col = null, cmin = 0, cmax = 0, first = true;
  const xF = (f) => xOf(tOfFrame(f));
  for (const [f, v] of pts) {
    const x = xF(f), y = yOf(v), c = Math.floor(x);
    if (c === col) { cmin = Math.min(cmin, y); cmax = Math.max(cmax, y); ctx.lineTo(x, y); continue; }
    col = c; cmin = cmax = y;
    if (first) { ctx.moveTo(x, y); first = false; } else ctx.lineTo(x, y);
  }
  if (fill) {
    ctx.lineTo(xF(pts[pts.length - 1][0]), top + height); ctx.lineTo(xF(pts[0][0]), top + height); ctx.closePath();
    ctx.fillStyle = fill; ctx.fill();
  } else { ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.stroke(); }
}
function drawBand(ctx, w, h, id) {
  ctx.fillStyle = C.bg; ctx.fillRect(0, 0, w, h);
  drawGrid(ctx, w, h, 0, 0.6);
  const b = S.bands.find((x) => x.id === id);
  if (!b) return;
  let mx = 0; for (const v of b.values) if (v > mx) mx = v;
  drawSeries(ctx, b, w, 2, h - 3, 0, mx || 1, null, "rgba(150,155,165,0.45)");
}

// ------------------------------------------------------------------------------------------- note lanes
const noteKey = (track, i) => `${track}:${i}`;
function noteDur(n) {
  const b = beatAt(n.time);
  return Math.max(0.001, timeAt(b + (n.len_beats || 0.25)) - n.time);
}
function drawNotes(ctx, w, h, track) {
  ctx.fillStyle = C.bg; ctx.fillRect(0, 0, w, h);
  drawGrid(ctx, w, h, gridSize(), 1);
  const list = (S.notes.tracks || {})[track] || [];
  const pad = 4, H = h - pad * 2;
  for (let i = 0; i < list.length; i++) {
    const n = list[i];
    const x = xOf(n.time);
    const wpx = Math.max(3, noteDur(n) * S.view.pps);
    if (x + wpx < 0 || x > w) continue;
    const sel = S.sel.has(noteKey(track, i));
    const vel = clamp(n.vel == null ? 1 : n.vel, 0, 1);
    const vh = Math.max(2, vel * H);
    ctx.fillStyle = sel ? "rgba(226,162,58,0.22)" : "rgba(167,171,178,0.12)";
    ctx.fillRect(x, pad + H - vh, wpx, vh);
    ctx.fillStyle = sel ? C.accent : C.note;
    ctx.fillRect(Math.round(x), pad + H - vh, 2, vh);           // onset stem, height = velocity
    ctx.fillRect(x, pad + H - vh, wpx, 1);                       // velocity line
  }
}
function hitNote(track, x) {
  const list = (S.notes.tracks || {})[track] || [];
  let best = -1, bestD = 1e9;
  for (let i = 0; i < list.length; i++) {
    const nx = xOf(list[i].time), wpx = Math.max(6, noteDur(list[i]) * S.view.pps);
    if (x >= nx - 4 && x <= nx + wpx) {
      const d = Math.abs(x - nx);
      if (d < bestD) { bestD = d; best = i; }
    }
  }
  return best;
}
function sortTrack(track) {
  // keep selection attached to the moved notes across the re-sort
  const list = S.notes.tracks[track];
  const tagged = list.map((n, i) => ({ n, sel: S.sel.has(noteKey(track, i)) }));
  tagged.sort((a, b) => a.n.time - b.n.time);
  for (const k of [...S.sel]) if (k.startsWith(track + ":")) S.sel.delete(k);
  S.notes.tracks[track] = tagged.map((t, i) => { if (t.sel) S.sel.add(noteKey(track, i)); return t.n; });
}
function attachNoteLane(cv, track) {
  cv.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    const list = (S.notes.tracks[track] = S.notes.tracks[track] || []);
    let idx = hitNote(track, x);
    let added = false;
    if (idx < 0) {
      const t = Math.max(0, snapT(tOf(x)));
      const H = r.height - 8, vel = clamp(Math.round((1 - (y - 4) / H) * 100) / 100, 0.05, 1);
      list.push({ time: +t.toFixed(4), vel, len_beats: 0.5, shape: "punch" });
      idx = list.length - 1;
      added = true;
    }
    const key = noteKey(track, idx);
    if (e.shiftKey && !added) { if (S.sel.has(key)) S.sel.delete(key); else S.sel.add(key); }
    else if (!S.sel.has(key)) { S.sel.clear(); S.sel.add(key); }
    if (!added && e.shiftKey) { scheduleNotes(); return; }
    const sel = [...S.sel].filter((k) => k.startsWith(track + ":")).map((k) => +k.split(":")[1]);
    const start = sel.map((i) => ({ i, time: list[i].time, vel: list[i].vel == null ? 1 : list[i].vel }));
    const anchor = list[idx].time;
    const x0 = e.clientX, y0 = e.clientY;
    let axis = null, changed = added;
    const H = r.height - 8;
    const move = (ev) => {
      const dx = ev.clientX - x0, dy = ev.clientY - y0;
      if (!axis) { if (Math.abs(dx) < 3 && Math.abs(dy) < 3) return; axis = Math.abs(dx) >= Math.abs(dy) ? "x" : "y"; }
      if (axis === "x") {
        const target = snapT(anchor + dx / S.view.pps);
        const dt = target - anchor;
        for (const s of start) list[s.i].time = +Math.max(0, s.time + dt).toFixed(4);
      } else {
        for (const s of start) list[s.i].vel = +clamp(s.vel - dy / H, 0.01, 1).toFixed(3);
        msg(`velocity ${list[idx].vel.toFixed(2)}`);
      }
      changed = true;
      scheduleNotes();
    };
    const up = () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      if (changed) { sortTrack(track); postNotes(); }
      scheduleNotes();
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    scheduleNotes();
  });
}
function scheduleNotes() { document.querySelectorAll('canvas[data-kind="notes"]').forEach((c) => schedule(c)); }
function deleteSelected() {
  if (!S.sel.size) return;
  const byTrack = {};
  for (const k of S.sel) { const [t, i] = k.split(":"); (byTrack[t] = byTrack[t] || new Set()).add(+i); }
  for (const t in byTrack) S.notes.tracks[t] = S.notes.tracks[t].filter((_, i) => !byTrack[t].has(i));
  const n = S.sel.size;
  S.sel.clear();
  msg(`removed ${n} note${n > 1 ? "s" : ""}`);
  buildNoteLanes();
  postNotes();
}

// ------------------------------------------------------------------------------------------- curve lanes
function outputById(id) { return S.bake && (S.bake.outputs || []).find((o) => o.id === id); }
function targetSummary(t) {
  if (!t) return "";
  if (t.kind === "light") return `light ×${(t.actors || []).length} ${t.property || ""}`;
  if (t.kind === "mpc") return `mpc ${t.param || ""}`;
  if (t.kind === "material") return `material ${t.param || ""}`;
  return `${t.kind || ""} ${t.param || t.property || ""}`;
}
// Curve lanes share whatever vertical space the timeline has left, up to a readable maximum.
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
      const cap = S.stacked.has(o.id) && o.channels.length > 1 ? o.channels.length * 36 : 240;
      cv.style.height = Math.floor(Math.max(base, Math.min(cap, base + extra))) + "px";
    }
  }
  for (const cv of cvs) schedule(cv);
}
function curveHeight(o) {
  const n = o.channels.length;
  if (S.stacked.has(o.id) && n > 1) return clamp(n * 13, 48, 260);
  return 78;
}
function drawCurve(ctx, w, h, id) {
  ctx.fillStyle = C.bg; ctx.fillRect(0, 0, w, h);
  drawGrid(ctx, w, h, 0, 0.7);
  const o = outputById(id);
  if (!o) return;
  const lo = o.min ?? 0, hi = o.max ?? 1;
  const n = o.channels.length;
  if (S.stacked.has(id) && n > 1) {
    const rowH = h / n;
    for (let i = 0; i < n; i++) {
      const top = i * rowH;
      ctx.fillStyle = C.line; ctx.fillRect(0, Math.round(top + rowH) - 1, w, 1);
      drawSeries(ctx, o.channels[i], w, top + 1.5, rowH - 3, lo, hi, null, "rgba(226,162,58,0.55)");
    }
  } else {
    ctx.globalAlpha = n > 1 ? 0.75 : 1;
    for (let i = 0; i < n; i++) drawSeries(ctx, o.channels[i], w, 4, h - 8, lo, hi, C.accent);
    ctx.globalAlpha = 1;
  }
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
    const nch = o.channels.length;
    head.innerHTML = `<div class="row"><button class="chev" title="Collapse"><svg viewBox="0 0 10 10" width="10" height="10"><path d="M1 3h8L5 8z"/></svg></button><span class="name"></span></div>
      <small class="dim tgt"></small>
      <div class="row"><small class="dim rng"></small>${nch > 1 ? '<button class="tog" title="Overlay or one row per channel">Rows</button>' : ""}</div>`;
    head.querySelector(".name").textContent = o.id;
    head.querySelector(".tgt").textContent = targetSummary(o.target);
    head.querySelector(".rng").textContent = `${nch} ch  ${o.min}–${o.max}`;
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
    const head = document.createElement("div");
    head.className = "lane-head";
    head.innerHTML = `<span class="name"></span><small class="dim cnt"></small>`;
    head.querySelector(".name").textContent = t;
    head.querySelector(".cnt").textContent = `${S.notes.tracks[t].length} notes`;
    const cv = document.createElement("canvas");
    cv.dataset.kind = "notes"; cv.dataset.track = t;
    attachNoteLane(cv, t);
    lane.append(head, cv);
    host.append(lane);
    canvases.add(cv);
  }
  if (!tracks.length) host.innerHTML = '<div class="group-head dim">notes.json has no tracks</div>';
  pruneCanvases();
  schedule();
}
function buildBandLanes() {
  const host = $("#band-lanes");
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
function pruneCanvases() { for (const c of [...canvases]) if (!c.isConnected) canvases.delete(c); }
function updateNoteCounts() {
  document.querySelectorAll(".lane.notes").forEach((lane) => {
    const t = lane.querySelector("canvas").dataset.track;
    lane.querySelector(".cnt").textContent = `${(S.notes.tracks[t] || []).length} notes`;
  });
}

// ------------------------------------------------------------------------------------------- schematic
const schem = $("#schem-canvas");
function schemLayout() {
  const outs = (S.bake && S.bake.outputs) || [];
  let h = 8;
  for (const o of outs) h += o.target && o.target.kind === "light" ? 150 : 26;
  return Math.max(60, h + 4);
}
function drawSchem() {
  const t = audio.currentTime || 0;
  const f = frameOf(t);
  $("#schem-frame").textContent = `frame ${f}`;
  const want = schemLayout() + "px";
  if (schem.style.height !== want) schem.style.height = want;
  const { ctx, w } = setupCanvas(schem);
  ctx.fillStyle = C.bg; ctx.fillRect(0, 0, w, schem.height);
  ctx.textBaseline = "middle";
  let y = 8;
  for (const o of (S.bake && S.bake.outputs) || []) {
    const lo = o.min ?? 0, hi = o.max ?? 1, span = hi - lo || 1;
    const vals = o.channels.map((ch) => seriesValueAtFrame(ch, f));
    if (o.target && o.target.kind === "light") {
      const cx = w / 2, cy = y + 74, R = 52;
      const n = Math.max(vals.length, vals.length === 1 ? (o.target.actors || [1]).length : 0);
      ctx.font = "11px Segoe UI, sans-serif"; ctx.fillStyle = C.dim; ctx.textAlign = "left";
      ctx.fillText(o.id, 10, y + 6);
      ctx.textAlign = "right"; ctx.fillStyle = C.faint;
      ctx.fillText(o.target.property || "", w - 10, y + 6);
      ctx.strokeStyle = C.line; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();
      for (let i = 0; i < n; i++) {
        const v = vals.length === 1 ? vals[0] : vals[i];
        const b = clamp((v - lo) / span, 0, 1);
        const a = -Math.PI / 2 + (i / n) * Math.PI * 2;
        const x = cx + Math.cos(a) * R, yy = cy + Math.sin(a) * R;
        const [r, g, bl] = C.light;
        const base = 30;
        ctx.fillStyle = `rgb(${Math.round(base + (r - base) * b)},${Math.round(base + (g - base) * b)},${Math.round(base + 2 + (bl - base) * b)})`;
        ctx.beginPath(); ctx.arc(x, yy, 8, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = C.line2; ctx.stroke();
      }
      const mean = vals.reduce((s, v) => s + v, 0) / (vals.length || 1);
      ctx.textAlign = "center"; ctx.fillStyle = C.dim; ctx.font = "11px Consolas, monospace";
      ctx.fillText(fmtVal(mean), cx, cy);
      y += 150;
    } else {
      const label = (o.target && (o.target.param || o.target.property)) || o.id;
      ctx.font = "11px Segoe UI, sans-serif"; ctx.textAlign = "left"; ctx.fillStyle = C.dim;
      ctx.fillText(label, 10, y + 10);
      const bx = 84, bw = w - bx - 64, v = vals[0] ?? 0;
      const b = clamp((v - lo) / span, 0, 1);
      ctx.fillStyle = C.line; ctx.fillRect(bx, y + 6, bw, 8);
      ctx.fillStyle = C.accent; ctx.fillRect(bx, y + 6, Math.round(bw * b), 8);
      ctx.textAlign = "right"; ctx.fillStyle = C.text; ctx.font = "11px Consolas, monospace";
      ctx.fillText(fmtVal(v), w - 10, y + 10);
      y += 26;
    }
  }
  ctx.textAlign = "left";
}
function fmtVal(v) {
  const a = Math.abs(v);
  return a >= 1000 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : v.toFixed(3);
}

// ------------------------------------------------------------------------------------------- params
const SKIP_KEYS = new Set(["id", "op", "in", "track", "signal", "target", "layer", "fn", "shape", "order", "preset", "direction"]);
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
  if (k === "damping" || k === "falloff" || k === "level" || k === "threshold" || k === "sustain" || k === "strength" || k === "amount" || k === "phase" || k === "phase_step") return [0, 1, 0.001];
  if (k === "key_tolerance") return [0, 0.05, 0.0005];
  if (k === "min" || k === "max") { const m = Math.max(1, Math.abs(v) * 3); return [v < 0 ? -m : 0, m, m / 1000]; }
  const m = Math.max(1, Math.abs(v) * 2);
  return [v < 0 ? -m : 0, m, m / 1000];
}
// Frequencies get a log slider; everything else linear.
const isLog = (key) => key === "lo" || key === "hi";
function toSlider(key, v, r) { return isLog(key) ? (Math.log(clamp(v, r[0], r[1]) / r[0]) / Math.log(r[1] / r[0])) * 1000 : v; }
function fromSlider(key, s, r) { return isLog(key) ? Math.round(r[0] * Math.pow(r[1] / r[0], s / 1000)) : s; }

function paramEntries() {
  const groups = [];
  const rig = S.rig;
  const g0 = { title: "rig", sub: "", rows: [] };
  if (typeof rig.audio_offset_frames === "number") g0.rows.push({ path: ["audio_offset_frames"], key: "audio_offset_frames" });
  if (rig.tempo && typeof rig.tempo.bpm === "number") g0.rows.push({ path: ["tempo", "bpm"], key: "bpm" });
  if (typeof rig.seed === "number") g0.rows.push({ path: ["seed"], key: "seed" });
  if (rig.bake && typeof rig.bake.key_tolerance === "number") g0.rows.push({ path: ["bake", "key_tolerance"], key: "key_tolerance" });
  (rig.nodes || []).forEach((n, ni) => {
    const g = { title: n.id, sub: n.op, rows: [] };
    for (const [k, v] of Object.entries(n)) if (!SKIP_KEYS.has(k) && typeof v === "number") g.rows.push({ path: ["nodes", ni, k], key: k });
    // params the rig leaves at the op's default (engine registry): shown dimmed, written into the node when changed
    const defs = (S.opParams || {})[n.op] || {};
    for (const [k, v] of Object.entries(defs)) if (!(k in n) && !SKIP_KEYS.has(k)) g.rows.push({ path: ["nodes", ni, k], key: k, def: v });
    if (g.rows.length) groups.push(g);
  });
  (rig.outputs || []).forEach((o, oi) => {
    const g = { title: o.id, sub: "output → " + targetSummary(o.target), rows: [] };
    for (const [k, v] of Object.entries(o)) if (!SKIP_KEYS.has(k) && typeof v === "number") g.rows.push({ path: ["outputs", oi, k], key: k });
    if (g.rows.length) groups.push(g);
  });
  if (g0.rows.length) groups.push(g0);
  return groups;
}
const getPath = (o, p) => p.reduce((a, k) => a[k], o);
function setPath(o, p, v) { const last = p[p.length - 1]; getPath(o, p.slice(0, -1))[last] = v; }

function buildParams() {
  const host = $("#param-list");
  host.textContent = "";
  for (const g of paramEntries()) {
    const box = document.createElement("div");
    box.className = "pgroup";
    const head = document.createElement("div");
    head.className = "pgroup-head";
    head.innerHTML = "<b></b><small class=dim></small>";
    head.querySelector("b").textContent = g.title;
    head.querySelector("small").textContent = g.sub;
    box.append(head);
    for (const row of g.rows) box.append(paramRow(row));
    host.append(box);
  }
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
    postRig(rig, { el, path, show, def });
  };
  sl.addEventListener("input", () => { num.value = +(+fromSlider(key, +sl.value, r)).toPrecision(6); });
  sl.addEventListener("change", () => commit(+fromSlider(key, +sl.value, r)));
  num.addEventListener("change", () => commit(parseFloat(num.value)));
  num.addEventListener("keydown", (e) => { if (e.key === "Enter") num.blur(); e.stopPropagation(); });
  el._show = show;
  return el;
}

// ------------------------------------------------------------------------------------------- server I/O
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
  const prevIds = ((S.bake && S.bake.outputs) || []).map((o) => o.id + ":" + o.channels.length).join();
  S.bake = data.bake;
  S.bands = data.bands || [];
  S.bakeMs = data.bake_ms;
  if (data.rev != null) S.rev = data.rev;
  S.offset = (S.rig && S.rig.audio_offset_frames) || 0;
  SEGS = gridSegs();                       // keep the grid in step with tempo edits
  $("#bpm").textContent = fmt(SEGS[0].bpm, 2);
  setDirty(!!data.dirty);
  setBakeInfo(`bake ${fmt(S.bakeMs, 0)} ms`);
  const warns = data.warnings || [];
  $("#bake-info").title = warns.length ? "Warnings:\n" + warns.join("\n") : "";
  if (warns.length) $("#bake-info").textContent += `  · ${warns.length} warning${warns.length > 1 ? "s" : ""}`;
  const ids = (S.bake.outputs || []).map((o) => o.id + ":" + o.channels.length).join();
  if (ids !== prevIds) buildCurveLanes();
  buildBandLanes();
  schedule();
  drawSchem();
}
// One request at a time; if edits pile up while baking, only the latest is sent next.
async function runQueue() {
  if (S.baking || !S.queued) return;
  const job = S.queued; S.queued = null;
  S.baking = true;
  setBakeInfo("baking…");
  try {
    const data = await api(job.path, job.body);
    if (job.path === "/api/rig") S.lastGood.rig = deep(job.body); else S.lastGood.notes = deep(job.body);
    applyBake(data);
    clearError();
    if (job.ctx && job.ctx.el) job.ctx.el.classList.remove("bad");
    msg(job.path === "/api/rig" ? "rig re-baked" : "notes re-baked");
  } catch (err) {
    // server kept its last good state; mirror that here
    if (job.path === "/api/rig") {
      S.rig = deep(S.lastGood.rig);
      if (job.ctx) { job.ctx.show(valueOr(job.ctx.path, job.ctx.def)); job.ctx.el.classList.add("bad"); }
    } else {
      S.notes = deep(S.lastGood.notes); S.sel.clear(); buildNoteLanes();
    }
    showError(err.message + (job.ctx ? `\n(${job.ctx.path.join(".")} reverted)` : " (notes reverted)"));
    setBakeInfo(`bake ${fmt(S.bakeMs, 0)} ms`);
  } finally {
    S.baking = false;
    runQueue();
  }
}
function postRig(rig, ctx) {
  S.rig = rig;
  S.queued = { path: "/api/rig", body: rig, ctx };
  runQueue();
}
function postNotes() {
  updateNoteCounts();
  S.queued = { path: "/api/notes", body: deep(S.notes), ctx: null };
  runQueue();
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

// ------------------------------------------------------------------------------------------- view + transport
function clampView() {
  const vis = S.width / S.view.pps;
  S.view.t0 = clamp(S.view.t0, 0, Math.max(0, S.duration - vis * 0.5));
}
const fitPps = () => S.width / Math.max(0.1, S.duration);
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
function placePlayhead() {
  const t = audio.currentTime || 0;
  const x = xOf(t);
  const gutter = $(".lane-head").getBoundingClientRect().width;
  playheadEl.style.transform = `translateX(${Math.round(gutter + x)}px)`;
  playheadEl.style.display = x < 0 || x > S.width ? "none" : "block";
}
function updateReadouts() {
  const t = audio.currentTime || 0;
  $("#t-sec").textContent = t.toFixed(3);
  $("#t-frame").textContent = frameOf(t);
  const b = beatAt(t);
  if (b >= 0) {
    const bar = Math.floor(b / BEATS_PER_BAR) + 1, beat = Math.floor(b % BEATS_PER_BAR) + 1, six = Math.floor((b % 1) * 4) + 1;
    $("#t-beat").textContent = `${bar}.${beat}.${six}`;
  } else $("#t-beat").textContent = "0.0.0";
}
function onTimeChanged() {
  updateReadouts();
  placePlayhead();
  drawSchem();
}
let raf = 0;
function tick() {
  raf = 0;
  if (audio.paused) return;
  const t = audio.currentTime;
  if ($("#follow").checked) {
    const x = xOf(t);
    if (x > S.width * 0.92 || x < 0) { S.view.t0 = t - (S.width * 0.08) / S.view.pps; clampView(); schedule(); }
  }
  onTimeChanged();
  raf = requestAnimationFrame(tick);
}
function togglePlay() { if (audio.paused) audio.play().catch((e) => msg("audio: " + e.message, true)); else audio.pause(); }
audio.addEventListener("play", () => {
  $("#play").classList.add("on");
  $("#play-glyph").setAttribute("d", "M4 2.5h3v11H4zM9 2.5h3v11H9z");
  if (!raf) raf = requestAnimationFrame(tick);
});
audio.addEventListener("pause", () => {
  $("#play").classList.remove("on");
  $("#play-glyph").setAttribute("d", "M4 2.5v11l9-5.5z");
  onTimeChanged();
});
audio.addEventListener("seeked", () => { if (audio.paused) onTimeChanged(); });
audio.addEventListener("error", () => msg("audio failed to load from /api/audio", true));
function seek(t) {
  audio.currentTime = clamp(t, 0, S.duration);
  if (audio.paused) onTimeChanged();
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

$("#play").onclick = togglePlay;
$("#zoom").addEventListener("input", (e) => {
  const a = Math.log(fitPps()), b = Math.log(maxPps());
  const ph = xOf(audio.currentTime || 0);
  zoomTo(Math.exp(a + (b - a) * (+e.target.value / 1000)), ph >= 0 && ph <= S.width ? ph : S.width / 2);
});
$("#zoom-in").onclick = () => zoomTo(S.view.pps * 1.5, clamp(xOf(audio.currentTime || 0), 0, S.width));
$("#zoom-out").onclick = () => zoomTo(S.view.pps / 1.5, clamp(xOf(audio.currentTime || 0), 0, S.width));
$("#zoom-fit").onclick = () => { S.view.pps = fitPps(); S.view.t0 = 0; syncZoomSlider(); schedule(); };
$("#grid").onchange = () => schedule();
$("#save").onclick = save;
for (const id of ["#snap", "#follow", "#grid", "#zoom"]) $(id).addEventListener("keydown", (e) => { if (e.key === " ") e.preventDefault(); });

document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") { e.preventDefault(); save(); return; }
  if (e.target.matches("input[type=number], input[type=text], textarea, select")) return;
  if (e.key === " ") { e.preventDefault(); togglePlay(); }
  else if (e.key === "Delete" || e.key === "Backspace") { e.preventDefault(); deleteSelected(); }
  else if (e.key === "Escape") { S.sel.clear(); scheduleNotes(); }
  else if (e.key === "Home") { seek(0); S.view.t0 = 0; clampView(); schedule(); }
});
window.addEventListener("beforeunload", (e) => { if (S.dirty) { e.preventDefault(); e.returnValue = ""; } });

function measure() {
  const cv = document.querySelector('canvas[data-kind="ruler"]');
  const w = Math.round(cv.getBoundingClientRect().width) || 600;
  const changed = w !== S.width;
  const ratio = S.width ? w / S.width : 1;
  S.width = w;
  return { changed, ratio };
}
new ResizeObserver(() => {
  const { changed } = measure();
  if (changed) { syncZoomSlider(); }
  fitLaneHeights(); schedule(); drawSchem();
}).observe(timeline);

// ------------------------------------------------------------------------------------------- boot
const baseName = (p) => (p || "").split(/[\\/]/).pop();
// Apply a full /api/state snapshot. On a reload (agent change) the view, zoom and playhead are kept.
function loadState(d, first) {
  S.rig = d.rig; S.notes = d.notes; S.analysis = d.analysis || {};
  S.lastGood = { rig: deep(d.rig), notes: deep(d.notes) };
  S.engine = d.engine;
  S.opParams = d.op_params || {};
  S.rev = d.rev;
  S.fps = (d.bake && d.bake.fps) || d.rig.fps || 30;
  S.offset = d.rig.audio_offset_frames || 0;
  S.duration = S.analysis.duration || ((d.bake && d.bake.frames) || 0) / S.fps || 1;
  SEGS = gridSegs();
  $("#fps").textContent = S.fps;
  $("#engine").textContent = `${d.engine} engine`;
  const paths = $("#paths");
  paths.textContent = `${baseName(d.rig_path)}  ·  ${baseName(d.notes_path)}`;
  paths.title = `${d.rig_path}
${d.notes_path}`;
  $("#audio-name").textContent = baseName(d.rig.audio);
  if (first) { measure(); S.view.pps = fitPps(); S.view.t0 = 0; }
  S.sel.clear();
  S.bake = null;                       // force the curve lanes to rebuild
  buildNoteLanes();
  applyBake(d);
  buildParams();
  clampView(); syncZoomSlider();
  onTimeChanged();
  renderActivity(d.activity || []);
}
async function boot() {
  canvases.add(document.querySelector('canvas[data-kind="ruler"]'));
  canvases.add(document.querySelector('canvas[data-kind="wave"]'));
  attachSeek(document.querySelector('canvas[data-kind="ruler"]'), true);
  attachSeek(document.querySelector('canvas[data-kind="wave"]'), true);
  try {
    const d = await api("/api/state");
    S.seenRev = d.rev;
    loadState(d, true);
    msg(`loaded ${Object.keys(d.notes.tracks || {}).length} note tracks, ${(d.bake.outputs || []).length} outputs`);
  } catch (err) {
    msg("failed to load /api/state: " + err.message, true);
    showError(err.message);
  }
  setInterval(poll, 1000);
}

// ------------------------------------------------------------------------------------------- agent activity
// Poll /api/state?since=<rev> once a second while the tab is visible. Our own edits return the new rev in
// their response, so only outside changes (an agent via the MCP server) trigger a reload.
let mouseDown = false;
window.addEventListener("mousedown", () => { mouseDown = true; }, true);
window.addEventListener("mouseup", () => { mouseDown = false; }, true);
let polling = false;
async function poll() {
  if (polling || document.visibilityState !== "visible" || S.rev == null) return;
  if (S.baking || S.queued || mouseDown || document.activeElement.matches("#param-list input")) return;
  polling = true;
  try {
    const d = await api(`/api/state?since=${S.rev}`);
    if (d.unchanged) { renderActivity(d.activity || []); return; }
    if (S.baking || S.queued) return;  // an edit started meanwhile; next tick
    loadState(d, false);
    const last = (d.activity || [])[d.activity.length - 1];
    msg(last ? `${last.source}: ${last.action}${last.detail ? " — " + last.detail : ""}` : "state reloaded");
  } catch (err) {
    msg("lost the server: " + err.message, true);
  } finally { polling = false; }
}
let activityKey = "";
function renderActivity(list) {
  const key = list.map((a) => a.rev).join();
  if (key === activityKey) return;
  activityKey = key;
  const ol = $("#agent-list");
  ol.textContent = "";
  const recent = list.slice(-5).reverse();
  if (!recent.length) { ol.innerHTML = '<li class="dim">No agent activity yet</li>'; return; }
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
boot();
// Buttons and checkboxes give focus back after a click, so Space always means play/pause
// (a focused button would otherwise also activate on Space and toggle twice).
document.addEventListener("click", (e) => {
  const b = e.target.closest && e.target.closest("button, input[type=checkbox]");
  if (b) b.blur();
});
