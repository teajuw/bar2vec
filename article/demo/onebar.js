// "Follow one bar": draws one bar of a song going through bar2vec, from scripts/bar_story_data.py's output.
// oneBar(data) wires up the elements with ids obStage, obPlay, obSeek, obSteps, obN, obText and obNote,
// and returns {start, stop}. Drawing is a pure function of the time t, so any moment can be jumped to.
//
// The middle of the animation is inside the audio model (MATPAC++), and shows what its code does:
//   the bar's spectrogram is cut into 16x16 patches (5 pitch rows by 9 moments for this bar),
//   each patch becomes 768 numbers, 12 layers update them (every patch reads every other patch),
//   the 5 patches of each moment are joined end to end, and the moments are averaged: 3,840 numbers.
// The tiles are the model's real values for this bar at each layer.
function oneBar(D){
const B = D.bar;
const bytes = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
const peaks = bytes(D.peaks), wave = bytes(B.wave), spec = bytes(B.spec), emb = bytes(B.emb), layers = bytes(B.layers);
const N = D.bars, K = B.index, DUR = D.duration;
const TW = 32, TH = B.patchDims / TW;  // a patch's numbers are drawn as a tile this many cells wide and tall
const PR = B.pitchRows, PM = B.moments, PD = B.patchDims, NL = layers.length / (PR * PM * PD) - 1;  // 5, 9, 768, 12
const clock = t => Math.floor(t / 60) + ":" + String(Math.floor(t % 60)).padStart(2, "0");
const near = B.sim.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(1, 9).map(p => p[1]);
const inChorus = near.filter(i => D.sections.some(s => s.name === "chorus" && D.edges[i] >= s.start - 0.05 && D.edges[i] < s.end)).length;
const choruses = D.sections.filter(s => s.name === "chorus").length;
const words = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"];
const FILL = {intro:"#dbd8cd",verse:"#8fc7e8",prechorus:"#b9a6ea",chorus:"#f2a65a",postchorus:"#f3c98b",bridge:"#7fcfa8",outro:"#b4b1a5",instrumental:"#e79ab0"};
const SHORT = {prechorus:"pre",postchorus:"post",instrumental:"inst"};
const secs = (B.end - B.start).toFixed(1), dims = D.dims.toLocaleString();

/* The steps: start time (s), short name, caption. */
const STEPS = [
  [0, "Sound", `This is ${D.title} as it arrives: sound and nothing else. A beat tracker finds ${N} bar lines. Follow bar ${K + 1}, ${secs} seconds starting at ${clock(Math.round(B.start))}.`],
  [5, "Spectrogram", "The bar is redrawn as a spectrogram: time runs left to right, pitch runs bottom to top, and loud is dark. This is the picture the audio model works from."],
  [10, "Patches", `Now inside the frozen audio model, MATPAC++. It cuts the picture into ${PR * PM} small patches: ${words[PR]} pitch rows by ${words[PM]} moments, each about a sixth of a second.`],
  [15, "Numbers", `Each patch is turned into a list of ${PD} numbers, with a note of where it sits in the picture. Each tile is one patch's list.`],
  [20, "Layers", `Then ${words[NL]} layers. In each one, every patch reads every other patch and updates its numbers; the tiles show the real values after each layer, with the average patch taken away so the differences show. This is the attention that language models use, and it is frozen: it has never seen a labelled song section.`],
  [27, "Embedding", `The ${words[PR]} patches of each moment are joined end to end, and the ${words[PM]} moments are averaged. What is left is the bar's embedding: one list of ${dims} numbers. Two bars that sound alike get similar lists.`],
  [33, "Colour", `Nobody can read ${dims} numbers, so for the picture they are squeezed to the three directions that vary most across the song, shown as red, green and blue. That keeps about ${Math.round(100 * D.kept)}% of the variation.`],
  [39, "Every bar", `The same happens to all ${N} bars. Bars that sound alike come out a similar colour.`],
  [44, "Alike", `The eight bars most like bar ${K + 1}, judged on all ${dims} numbers, are marked. ${inChorus === 8 ? "All eight" : words[inChorus][0].toUpperCase() + words[inChorus].slice(1) + " of the eight"} sit in the ${words[choruses]} choruses. Nobody told it where the chorus is.`],
  [50, "Sections", "Comparing every bar with every other bar is how bar2vec cuts the song into sections. A small classifier then names them."],
];
const END = 56;
const LAYER0 = 20.5, LAYER_S = 0.5;  // when the layers start, and how long each one shows

document.getElementById("obNote").textContent = "Everything drawn is computed from the song's audio. The spectrogram and the tiles are the audio model's own values for this bar. Until the last layer, each tile shows a patch's numbers minus the average patch at that layer. In the layers step the lines show which patches a patch reads (all of them), not how strongly. The colour is only for looking: bar2vec compares bars on the full list of numbers.";

const cv = document.getElementById("obStage"), cx = cv.getContext("2d");
let W = 0, H = 0, wide = true, C = {}, specCv = null, embCv = null, atlas = [], rawLast = null, built = "";
const clamp = x => Math.max(0, Math.min(1, x));
const ease = x => { x = clamp(x); return x * x * (3 - 2 * x); };
const pr = (t, a, b) => ease((t - a) / (b - a));
const mix = (a, b, p) => a + (b - a) * p;
const mixR = (a, b, p) => ({x: mix(a.x, b.x, p), y: mix(a.y, b.y, p), w: mix(a.w, b.w, p), h: mix(a.h, b.h, p)});
const hex = h => { h = h.trim(); if (h.length === 4) h = "#" + [1,2,3].map(i => h[i] + h[i]).join(""); return [1,3,5].map(i => parseInt(h.slice(i, i + 2), 16)); };
const FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif", MONO = "ui-monospace,Menlo,Consolas,monospace";

function tokens(){
  const cs = getComputedStyle(document.documentElement);
  // the site's tokens when the page has them, these values when it is opened on its own
  const DEF = {bg:"#faf9f5",surface:"#f1efe7",line:"#d6d2c6",ink:"#141413","ink-2":"#5c5b56","ink-3":"#9a9891",spec:"#0c447c",neg:"#2f6fb3",pos:"#d2622a",accent:"#c2613f"};
  for (const k in DEF){ const v = cs.getPropertyValue("--" + k).trim(); C[k] = /^#[0-9a-f]{3}([0-9a-f]{3})?$/i.test(v) ? v : DEF[k]; }
  built = "";
}

/* Pictures made once per theme and tile shape: the spectrogram, one sheet of tiles per layer, the embedding. */
function pictures(){
  const key = [C.bg, C.spec, C.neg, C.pos, C.surface].join();
  if (key === built) return; built = key;
  const bg = hex(C.bg), sp = hex(C.spec), ng = hex(C.neg), ps = hex(C.pos), sf = hex(C.surface);
  const canvas = (w, h) => { const c = document.createElement("canvas"); c.width = w; c.height = h; return c; };
  const cell = (im, i, byte) => { const v = (byte - 127.5) / 127.5, to = v < 0 ? ng : ps, a = Math.pow(Math.abs(v), 0.7);
    for (let c = 0; c < 3; c++) im.data[4 * i + c] = mix(sf[c], to[c], a); im.data[4 * i + 3] = 255; };
  specCv = canvas(B.specCols, B.specRows);
  let g = specCv.getContext("2d"), im = g.createImageData(B.specCols, B.specRows);
  for (let i = 0; i < spec.length; i++){ const v = spec[i] / 255; for (let c = 0; c < 3; c++) im.data[4 * i + c] = mix(bg[c], sp[c], v); im.data[4 * i + 3] = 255; }
  g.putImageData(im, 0, 0);
  // A sheet per layer: every patch's numbers as a TW x TH tile, laid out like the patches (high pitch at the top).
  // In the early layers the patches share a large common part that hides their differences, so these sheets show
  // each patch after the average patch of that layer is taken away. rawLast is the last layer as it is.
  const sw = PM * TW, sh = PR * TH, np = PR * PM;
  const sheet = (l, value) => { const c = canvas(sw, sh), g = c.getContext("2d"), im = g.createImageData(sw, sh);
    for (let f = 0; f < PR; f++) for (let m = 0; m < PM; m++){
      const pi = f * PM + m, x0 = m * TW, y0 = (PR - 1 - f) * TH;
      for (let k = 0; k < PD; k++) cell(im, (y0 + Math.floor(k / TW)) * sw + x0 + k % TW, value(pi, k));
    }
    g.putImageData(im, 0, 0); return c; };
  atlas = [];
  for (let l = 0; l <= NL; l++){
    const at = (pi, k) => layers[(l * np + pi) * PD + k], rel = new Float32Array(np * PD);
    for (let k = 0; k < PD; k++){ let s = 0; for (let pi = 0; pi < np; pi++) s += at(pi, k); s /= np;
      for (let pi = 0; pi < np; pi++) rel[pi * PD + k] = at(pi, k) - s; }
    const mags = Float32Array.from(rel, Math.abs).sort(), top = mags[Math.floor(mags.length * 0.99)] || 1;
    atlas.push(sheet(l, (pi, k) => 127.5 + 127.5 * Math.max(-1, Math.min(1, rel[pi * PD + k] / top))));
    if (l === NL) rawLast = sheet(l, at);
  }
  // the bar's embedding: the same shape as one moment's joined patches
  embCv = canvas(TW, sh); g = embCv.getContext("2d"); im = g.createImageData(TW, sh);
  for (let f = 0; f < PR; f++) for (let k = 0; k < PD; k++) cell(im, ((PR - 1 - f) * TH + Math.floor(k / TW)) * TW + k % TW, emb[f * PD + k]);
  g.putImageData(im, 0, 0);
}

function size(){
  const w = cv.clientWidth; if (!w) return;
  wide = w >= 600; W = w; H = Math.round(wide ? w * 0.62 : w * 1.18);
  const d = Math.min(window.devicePixelRatio || 1, 2.5);
  cv.style.height = H + "px"; cv.width = Math.round(W * d); cv.height = Math.round(H * d);
  cx.setTransform(d, 0, 0, d, 0, 0);
}

function fit(r, aspect, maxW){ // the largest box of a given width/height ratio centred in r
  let w = Math.min(r.w, maxW || r.w), h = w / aspect; if (h > r.h){ h = r.h; w = h * aspect; }
  return {x: r.x + (r.w - w) / 2, y: r.y + (r.h - h) / 2, w, h};
}
function layout(){
  const L = {}, fs = wide ? 13 : 12, pad = wide ? 14 : 8;
  L.fs = fs;
  L.wave = {x: 0, y: 22, w: W, h: wide ? H * 0.09 : 40};
  const secH = wide ? 34 : 30, stripH = wide ? 34 : 30;
  L.sec = {x: 0, y: H - 20 - secH, w: W, h: secH};
  L.strip = {x: 0, y: L.sec.y - 8 - stripH, w: W, h: stripH};
  const top = L.wave.y + L.wave.h + 28, bot = L.strip.y - 28;
  L.frame = {x: 0.75, y: top, w: W - 1.5, h: bot - top};                           // the audio model's outline
  L.inner = {x: pad, y: top + 24, w: W - 2 * pad, h: bot - top - 24 - 26};         // where things are drawn
  L.labelY = bot - 9;
  L.spec = fit(L.inner, B.specCols / B.specRows, wide ? W * 0.5 : W);
  const gap = wide ? 5 : 3, ta = TW / TH;                                          // the patches, pulled apart
  const gw = Math.min(L.inner.w, (L.inner.h - (PR - 1) * gap) / PR * ta * PM + (PM - 1) * gap);
  const tw = (gw - (PM - 1) * gap) / PM, th = tw / ta, gh = PR * th + (PR - 1) * gap;
  L.grid = {x: L.inner.x + (L.inner.w - gw) / 2, y: L.inner.y + (L.inner.h - gh) / 2, w: gw, h: gh, tw, th, gap};
  L.emb = fit(L.inner, TW / (PR * TH));                                            // the one list left at the end
  const ex = wide ? L.inner.x + L.inner.w * 0.12 : L.inner.x + 6;
  L.embLeft = {x: ex, y: L.emb.y, w: L.emb.w, h: L.emb.h};
  const rx = ex + L.emb.w + (wide ? 40 : 14);
  L.rgb = {x: rx, y: L.inner.y, w: Math.min(L.inner.x + L.inner.w - rx, 420), h: L.inner.h};
  return L;
}
const bx = t => W * t / DUR;
const slot = (L, i) => ({x: bx(D.edges[i]), y: L.strip.y, w: Math.max(bx(D.edges[i + 1]) - bx(D.edges[i]) - 0.6, 0.6), h: L.strip.h});
const rgb = i => `rgb(${D.rgb[i][0]},${D.rgb[i][1]},${D.rgb[i][2]})`;

function text(s, x, y, o = {}){
  cx.font = `${o.weight || 400} ${o.size || 13}px ${o.mono ? MONO : FONT}`;
  cx.textAlign = o.align || "left"; cx.textBaseline = o.base || "alphabetic"; cx.fillStyle = o.color || C["ink-2"];
  cx.fillText(s, x, y);
}
function rr(r, rad){ cx.beginPath(); cx.roundRect(r.x, r.y, r.w, r.h, rad); }
function img(src, sx, sy, sw, sh, r){ cx.imageSmoothingEnabled = r.w < sw; cx.drawImage(src, sx, sy, sw, sh, r.x, r.y, r.w, r.h); }

function draw(t){
  const L = layout(), fs = L.fs;
  pictures();
  cx.clearRect(0, 0, W, H);

  /* the song's sound, with the bar marked */
  text(`${D.title}, ${clock(DUR)} of sound`, 0, 13, {size: fs});
  const wv = L.wave, mid = wv.y + wv.h / 2, n = peaks.length, step = wv.w / n;
  cx.fillStyle = C["ink-3"];
  for (let i = 0; i < n; i++){ const h = Math.max(peaks[i] / 255 * wv.h / 2, 0.5); cx.fillRect(i * step, mid - h, Math.max(step - 1, 0.8), 2 * h); }
  const b0 = bx(B.start), b1 = bx(B.end), mk = pr(t, 1.2, 2.2);
  if (mk > 0){
    cx.save(); cx.globalAlpha = mk; cx.fillStyle = C.accent; cx.fillRect(b0 - 0.5, wv.y - 3, Math.max(b1 - b0, 2) + 1, wv.h + 6);
    const right = b0 < W * 0.7;
    text(`bar ${K + 1} of ${N}`, right ? b1 + 6 : b0 - 6, wv.y + wv.h + 15, {size: fs, color: C.accent, weight: 600, align: right ? "left" : "right"});
    cx.restore();
  }

  /* the audio model's outline: everything from the patches to the embedding happens inside it */
  const inside = pr(t, 10, 10.9) * (1 - pr(t, 32, 32.9));
  if (inside > 0){
    cx.save(); cx.globalAlpha = inside;
    cx.fillStyle = C.surface; cx.globalAlpha = inside * 0.5; rr(L.frame, 6); cx.fill();
    cx.globalAlpha = inside; cx.strokeStyle = C.line; cx.lineWidth = 1.5; rr(L.frame, 6); cx.stroke();
    cx.font = `600 ${fs}px ${FONT}`; const a = "inside the frozen audio model", wa = cx.measureText(a).width;
    text(a, L.frame.x + 10, L.frame.y + 17, {size: fs, weight: 600, color: C.ink});
    text("MATPAC++", L.frame.x + 10 + wa + 9, L.frame.y + 17, {size: 11, mono: true});
    cx.restore();
  }
  let label = "";

  /* the bar: sound, then spectrogram, then patches */
  const grow = pr(t, 2.4, 4.2), S = L.spec, G = L.grid, apart = pr(t, 11.4, 13);
  if (grow > 0 && apart <= 0){
    const r = mixR({x: b0, y: wv.y, w: b1 - b0, h: wv.h}, S, grow), toSpec = pr(t, 5.2, 7), gl = pr(t, 10.2, 11.2);
    if (toSpec < 1){
      cx.fillStyle = C.surface; rr(r, 3); cx.fill();
      cx.save(); cx.globalAlpha = 1 - toSpec; cx.fillStyle = C.ink;
      const m = wave.length, st = r.w / m, cy = r.y + r.h / 2;
      for (let i = 0; i < m; i++){ const h = Math.max(wave[i] / 255 * r.h * 0.44, 0.5); cx.fillRect(r.x + i * st, cy - h, Math.max(st - 0.8, 0.6), 2 * h); }
      cx.restore();
    }
    if (toSpec > 0){ cx.save(); cx.globalAlpha = toSpec; img(specCv, 0, 0, B.specCols, B.specRows, r); cx.restore(); }
    if (gl > 0){
      cx.save(); cx.globalAlpha = gl; cx.strokeStyle = C.bg; cx.lineWidth = 1.5; cx.beginPath();
      for (let i = 1; i < PM; i++){ cx.moveTo(r.x + r.w * i / PM, r.y); cx.lineTo(r.x + r.w * i / PM, r.y + r.h); }
      for (let j = 1; j < PR; j++){ cx.moveTo(r.x, r.y + r.h * j / PR); cx.lineTo(r.x + r.w, r.y + r.h * j / PR); }
      cx.stroke(); cx.restore();
    }
    if (grow >= 1) label = toSpec < 0.5 ? `bar ${K + 1}: ${secs} seconds of sound` : gl < 0.5 ? "the same bar as a spectrogram: time across, pitch up" : `cut into ${PR * PM} patches`;
  }

  /* inside the model: patches, their numbers, the layers, joining and averaging */
  const join = pr(t, 27.4, 28.8), avg = pr(t, 29.6, 31.6), shift = pr(t, 33, 34.4), fly = pr(t, 39, 40.6);
  if (apart > 0 && avg < 1){
    const lf = (t - LAYER0) / LAYER_S, l = t < LAYER0 ? 0 : Math.min(NL, 1 + Math.floor(lf)), blend = t < LAYER0 ? 1 : clamp((lf - Math.floor(lf)) / 0.35);
    const rects = [], asIs = pr(t, 26.5, 27.2);
    for (let r = 0; r < PR; r++) for (let m = 0; m < PM; m++){
      const a = {x: S.x + m * S.w / PM, y: S.y + r * S.h / PR, w: S.w / PM, h: S.h / PR};
      const b = {x: G.x + m * (G.tw + G.gap), y: G.y + r * (G.th + G.gap), w: G.tw, h: G.th};
      const c = {x: b.x, y: G.y + (G.h - PR * G.th) / 2 + r * G.th, w: G.tw, h: G.th};
      rects.push(mixR(mixR(a, b, apart), c, join));
    }
    if (avg <= 0){
      rects.forEach((q, i) => {
        const r = Math.floor(i / PM), m = i % PM, d = (m + r) / (PM + PR) * 1.5, num = pr(t, 15.3 + d, 16.3 + d);
        if (num < 1) img(specCv, m * 16, r * 16, 16, 16, q);
        if (num > 0){
          cx.save(); cx.globalAlpha = num;
          if (l > 0 && blend < 1 && l <= NL) img(atlas[l - 1], m * TW, r * TH, TW, TH, q);
          cx.globalAlpha = num * (l > 0 ? blend : 1); img(atlas[l], m * TW, r * TH, TW, TH, q);
          if (asIs > 0){ cx.globalAlpha = asIs; img(rawLast, m * TW, r * TH, TW, TH, q); }
          cx.restore();
        }
      });
      // one patch per layer is picked out, with a line to every patch it reads
      if (t >= LAYER0 && lf < NL){
        const focal = (l * 7 + 3) % (PR * PM), f = rects[focal], a = Math.sin(Math.PI * clamp(lf - Math.floor(lf)));
        cx.save(); cx.globalAlpha = 0.5 * a; cx.strokeStyle = C.ink; cx.lineWidth = 0.6; cx.beginPath();
        rects.forEach((q, i) => { if (i !== focal){ cx.moveTo(f.x + f.w / 2, f.y + f.h / 2); cx.lineTo(q.x + q.w / 2, q.y + q.h / 2); } });
        cx.stroke(); cx.globalAlpha = a; cx.lineWidth = 1.5; cx.strokeRect(f.x - 1, f.y - 1, f.w + 2, f.h + 2); cx.restore();
      }
    } else {
      // each moment's joined patches slide together and are averaged into one list
      for (let m = 0; m < PM; m++){
        const c0 = rects[m], col = {x: c0.x, y: c0.y, w: G.tw, h: PR * G.th}, q = mixR(col, L.emb, avg);
        cx.save(); cx.globalAlpha = mix(1, 0.3, avg); img(rawLast, m * TW, 0, TW, PR * TH, q); cx.restore();
      }
      const done = pr(t, 31, 31.6);
      if (done > 0){ cx.save(); cx.globalAlpha = done; img(embCv, 0, 0, TW, PR * TH, mixR(L.emb, L.emb, 1)); cx.restore(); }
    }
    const numAll = pr(t, 15.3, 17.8);
    label = avg > 0 ? `the ${PM} moments averaged: one list of ${dims} numbers`
      : join > 0 ? `the ${PR} patches of each moment joined: ${PM} lists of ${dims}`
      : t >= LAYER0 ? (lf < NL ? `layer ${l} of ${NL}: every patch reads every other patch` : `after ${NL} layers: each patch's ${PD} numbers`)
      : numAll > 0.5 ? `each patch is now ${PD} numbers` : `${PR * PM} patches: ${PR} pitch rows by ${PM} moments`;
  }

  /* the embedding, squeezed to three numbers and one colour */
  if (avg >= 1){
    const E = mixR(L.emb, L.embLeft, shift), dim = 1 - 0.5 * fly, R = L.rgb;
    cx.save(); cx.globalAlpha = dim; img(embCv, 0, 0, TW, PR * TH, E); cx.restore();
    label = `bar ${K + 1}'s embedding: ${dims} numbers`;
    const bars = pr(t, 34.2, 35.2), sw = pr(t, 36.4, 37.6);
    if (bars > 0){
      cx.save(); cx.globalAlpha = bars * dim;
      const names = ["red", "green", "blue"], cols = ["#d8433b", "#2f9e5b", "#2f6fd6"];
      const rowH = Math.min(R.h / 3, 34), labW = 44, valW = 36, swW = Math.min(R.h, wide ? 84 : 64);
      const x0 = R.x + labW, trackW = Math.max(R.w - labW - valW - swW - 14, 20), y0 = R.y + (R.h - 3 * rowH) / 2;
      for (let k = 0; k < 3; k++){
        const y = y0 + k * rowH + rowH * 0.2, h = rowH * 0.6, v = B.scaled[k] * pr(t, 34.4 + k * 0.3, 35.8 + k * 0.3);
        text(names[k], x0 - 8, y + h / 2, {size: fs, align: "right", base: "middle"});
        cx.fillStyle = C.surface; cx.fillRect(x0, y, trackW, h);
        cx.fillStyle = cols[k]; cx.fillRect(x0, y, trackW * v, h);
        text(v.toFixed(2), x0 + trackW + 6, y + h / 2, {size: 12, base: "middle", mono: true, color: C.ink});
      }
      cx.restore();
      const home = {x: x0 + trackW + valW + 14, y: R.y + (R.h - swW) / 2, w: swW, h: swW};
      if (fly > 0){ cx.save(); cx.globalAlpha = 0.5 * fly; cx.strokeStyle = C.line; cx.lineWidth = 1; cx.setLineDash([3, 3]); rr(home, 4); cx.stroke(); cx.restore(); }
      if (sw > 0 && fly < 1){ const q = mixR(home, slot(L, K), fly); cx.save(); cx.globalAlpha = sw; cx.fillStyle = rgb(K); rr(q, mix(4, 0, fly)); cx.fill(); cx.restore(); }
      label = sw < 0.5 ? "its three directions that vary most across the song" : "one colour for the bar";
    }
  }
  if (label){ cx.save(); cx.globalAlpha = avg >= 1 ? 1 - 0.5 * fly : 1; text(label, W / 2, L.labelY, {size: fs, align: "center"}); cx.restore(); }

  /* the strip: one colour per bar */
  if (fly > 0){
    const alike = pr(t, 44, 45.2) * (1 - pr(t, 49.6, 50.6)), st = L.strip;
    cx.save(); cx.globalAlpha = fly; text(t < 41 ? `bar ${K + 1}'s colour, in its place` : `all ${N} bars, one colour each`, 0, st.y - 9, {size: fs}); cx.restore();
    const wipe = (t - 40.6) / 2.8;
    for (let i = 0; i < N; i++){
      const r = slot(L, i), a = i === K ? (fly >= 1 ? 1 : 0) : clamp((wipe - i / N) * 8);
      if (a <= 0) continue;
      cx.globalAlpha = a * (i === K || near.includes(i) ? 1 : 1 - 0.72 * alike);
      cx.fillStyle = rgb(i); cx.fillRect(r.x, r.y, r.w, r.h);
    }
    cx.globalAlpha = 1;
    if (alike > 0){
      cx.save(); cx.globalAlpha = alike; cx.fillStyle = C.ink; cx.strokeStyle = C.ink; cx.lineWidth = 1.5;
      for (const i of near.concat([K])){ const r = slot(L, i), m = r.x + r.w / 2;
        if (i === K){ cx.strokeRect(r.x - 1, r.y - 1, r.w + 2, r.h + 2); continue; }
        const y = r.y + r.h; cx.beginPath(); cx.moveTo(m - 4, y + 10); cx.lineTo(m + 4, y + 10); cx.lineTo(m, y + 4); cx.closePath(); cx.fill(); }
      cx.restore();
    }
  }

  /* the sections */
  const sIn = pr(t, 50.4, 52);
  if (sIn > 0){
    const sc = L.sec; cx.save(); cx.globalAlpha = sIn;
    for (const s of D.sections){
      const a = bx(s.start), b = bx(s.end), r = {x: a + 1, y: sc.y, w: Math.max(b - a - 2, 1), h: sc.h};
      cx.fillStyle = FILL[s.name] || "#ccc"; rr(r, 3); cx.fill();
      if (s.unsure){ cx.save(); rr(r, 3); cx.clip(); cx.strokeStyle = "rgba(255,255,255,.6)"; cx.lineWidth = 3; cx.beginPath();
        for (let x = r.x - r.h; x < r.x + r.w; x += 8){ cx.moveTo(x, r.y + r.h); cx.lineTo(x + r.h, r.y); } cx.stroke(); cx.restore(); }
      const lab = (wide ? s.name : (SHORT[s.name] || s.name)) + (s.unsure ? "?" : "");
      cx.font = `500 ${fs - 1}px ${FONT}`;
      if (cx.measureText(lab).width < r.w - 6) text(lab, r.x + r.w / 2, r.y + r.h / 2 + 0.5, {size: fs - 1, weight: 500, align: "center", base: "middle", color: "#1c1c1a"});
      if (s.start > 0){ cx.strokeStyle = C.accent; cx.lineWidth = 1.25; cx.beginPath(); cx.moveTo(a, L.strip.y - 2); cx.lineTo(a, sc.y + sc.h); cx.stroke(); }
    }
    text("0:00", 0, H - 5, {size: 11, color: C["ink-3"]}); text(clock(DUR), W, H - 5, {size: 11, color: C["ink-3"], align: "right"});
    cx.restore();
  }
}

/* transport */
const playBtn = document.getElementById("obPlay"), seek = document.getElementById("obSeek"), stepsEl = document.getElementById("obSteps");
const capn = document.getElementById("obN"), capt = document.getElementById("obText");
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
let t = 0, playing = false, last = 0, cur = -1;
const stepEnd = i => (i + 1 < STEPS.length ? STEPS[i + 1][0] : END) - 0.01;
STEPS.forEach((s, i) => { const li = document.createElement("li"), b = document.createElement("button");
  b.type = "button"; b.textContent = `${i + 1} ${s[1]}`; b.onclick = () => { t = reduced ? stepEnd(i) : s[0]; if (!reduced) setPlaying(true); paint(); };
  li.appendChild(b); stepsEl.appendChild(li); });
function paint(){
  draw(t);
  let i = 0; while (i + 1 < STEPS.length && t >= STEPS[i + 1][0]) i++;
  if (i !== cur){ cur = i; capn.textContent = `${i + 1} / ${STEPS.length}`; capt.textContent = STEPS[i][2];
    [...stepsEl.querySelectorAll("button")].forEach((b, j) => j === i ? b.setAttribute("aria-current", "step") : b.removeAttribute("aria-current")); }
  if (document.activeElement !== seek) seek.value = Math.round(1000 * t / END);
  playBtn.textContent = playing ? "Pause" : t >= END - 0.02 ? "Replay" : "Play";
}
function tick(now){
  if (!playing) return;
  t = Math.min(END, t + Math.min((now - last) / 1000, 0.1)); last = now;
  if (t >= END) playing = false;
  paint(); if (playing) requestAnimationFrame(tick);
}
function setPlaying(p){ if (p === playing) return; playing = p; if (p){ last = performance.now(); requestAnimationFrame(tick); } }
playBtn.onclick = () => { if (!playing && t >= END - 0.02) t = 0; setPlaying(!playing); paint(); };
seek.oninput = () => { setPlaying(false); t = END * seek.value / 1000; paint(); };
function refresh(){ tokens(); size(); paint(); }
new ResizeObserver(() => { size(); paint(); }).observe(cv);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", refresh);
new MutationObserver(refresh).observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});
if (reduced) t = END;
refresh();
return {start(){ if (reduced) return; t = 0; setPlaying(true); paint(); }, stop(){ setPlaying(false); paint(); }, end: END};
}
