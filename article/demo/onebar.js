// "Follow one bar": draws one bar of a song going through bar2vec, from scripts/bar_story_data.py's output.
// oneBar(data) wires up the elements with ids obStage, obPlay, obSeek, obSteps, obN, obText and obNote,
// and returns {start, stop}. Drawing is a pure function of the time t, so any moment can be jumped to.
function oneBar(D){
const B = D.bar;
const bytes = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
const peaks = bytes(D.peaks), wave = bytes(B.wave), spec = bytes(B.spec), emb = bytes(B.emb);
const N = D.bars, K = B.index, DUR = D.duration;
const clock = t => Math.floor(t / 60) + ":" + String(Math.floor(t % 60)).padStart(2, "0");
const near = B.sim.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(1, 9).map(p => p[1]);
const inChorus = near.filter(i => D.sections.some(s => s.name === "chorus" && D.edges[i] >= s.start - 0.05 && D.edges[i] < s.end)).length;
const choruses = D.sections.filter(s => s.name === "chorus").length;
const words = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight"];
const FILL = {intro:"#dbd8cd",verse:"#8fc7e8",prechorus:"#b9a6ea",chorus:"#f2a65a",postchorus:"#f3c98b",bridge:"#7fcfa8",outro:"#b4b1a5",instrumental:"#e79ab0"};
const SHORT = {prechorus:"pre",postchorus:"post",instrumental:"inst"};
const secs = (B.end - B.start).toFixed(1);

/* The steps: start time (s), short name, caption. */
const STEPS = [
  [0, "Sound", `This is ${D.title} as it arrives: sound and nothing else. A beat tracker finds ${N} bar lines. Follow bar ${K + 1}, ${secs} seconds starting at ${clock(Math.round(B.start))}.`],
  [5, "Spectrogram", "The bar is redrawn as a spectrogram: time runs left to right, pitch runs bottom to top, and loud is dark. It is then split into small patches."],
  [11, "Audio model", "A frozen audio model reads the patches. It has never seen a labelled song section, so it has no idea what a chorus is."],
  [15, "Embedding", `Out comes the bar's embedding: a list of ${D.dims.toLocaleString()} numbers. Each square is one of them. Two bars that sound alike get similar lists.`],
  [20, "Colour", `Nobody can read ${D.dims.toLocaleString()} numbers, so for the picture they are squeezed to the three directions that vary most across the song, shown as red, green and blue. That keeps about ${Math.round(100 * D.kept)}% of the variation.`],
  [26, "Every bar", `The same happens to all ${N} bars. Bars that sound alike come out a similar colour.`],
  [31, "Alike", `The eight bars most like bar ${K + 1}, judged on all ${D.dims.toLocaleString()} numbers, are marked. ${inChorus === 8 ? "All eight" : words[inChorus][0].toUpperCase() + words[inChorus].slice(1) + " of the eight"} sit in the ${words[choruses]} choruses. Nobody told it where the chorus is.`],
  [37, "Sections", "Comparing every bar with every other bar is how bar2vec cuts the song into sections. A small classifier then names them."],
];
const END = 43;

document.getElementById("obNote").textContent = `Everything drawn is computed from the song's audio. The spectrogram shown is ours (80 pitch bands); the audio model makes its own from each bar. The colour is only for looking: bar2vec compares bars on the full list of numbers.`;

const cv = document.getElementById("obStage"), cx = cv.getContext("2d");
let W = 0, H = 0, wide = true, C = {}, specCv = null, embCv = null;
const clamp = x => Math.max(0, Math.min(1, x));
const ease = x => { x = clamp(x); return x * x * (3 - 2 * x); };
const pr = (t, a, b) => ease((t - a) / (b - a));
const mix = (a, b, p) => a + (b - a) * p;
const hex = h => { h = h.trim(); if (h.length === 4) h = "#" + [1,2,3].map(i => h[i] + h[i]).join(""); return [1,3,5].map(i => parseInt(h.slice(i, i + 2), 16)); };

function tokens(){
  const cs = getComputedStyle(document.documentElement);
  // the site's tokens when the page has them, these values when it is opened on its own
  const DEF = {bg:"#faf9f5",surface:"#f1efe7",line:"#d6d2c6",ink:"#141413","ink-2":"#5c5b56","ink-3":"#9a9891",spec:"#0c447c",neg:"#2f6fb3",pos:"#d2622a",accent:"#c2613f"};
  for (const k in DEF){ const v = cs.getPropertyValue("--" + k).trim(); C[k] = /^#[0-9a-f]{3}([0-9a-f]{3})?$/i.test(v) ? v : DEF[k]; }
  const bg = hex(C.bg), sp = hex(C.spec), ng = hex(C.neg), ps = hex(C.pos), sf = hex(C.surface);
  // the bar's spectrogram, background to the spectrogram colour
  specCv = document.createElement("canvas"); specCv.width = B.specCols; specCv.height = B.specRows;
  let g = specCv.getContext("2d"), im = g.createImageData(B.specCols, B.specRows);
  for (let i = 0; i < spec.length; i++){ const v = spec[i] / 255; for (let c = 0; c < 3; c++) im.data[4*i+c] = mix(bg[c], sp[c], v); im.data[4*i+3] = 255; }
  g.putImageData(im, 0, 0);
  // the embedding: 96 columns by 40 rows, one cell per number; 128 is zero
  embCv = document.createElement("canvas"); embCv.width = 96; embCv.height = 40;
  g = embCv.getContext("2d"); im = g.createImageData(96, 40);
  for (let i = 0; i < 3840; i++){ const v = (emb[i] - 127.5) / 127.5, to = v < 0 ? ng : ps, a = Math.pow(Math.abs(v), 0.7);
    for (let c = 0; c < 3; c++) im.data[4*i+c] = mix(sf[c], to[c], a); im.data[4*i+3] = 255; }
  g.putImageData(im, 0, 0);
}

function size(){
  const w = cv.clientWidth; if (!w) return;
  wide = w >= 600; W = w; H = Math.round(wide ? w * 0.5 : w * 1.5);
  const d = Math.min(window.devicePixelRatio || 1, 2.5);
  cv.style.height = H + "px"; cv.width = Math.round(W * d); cv.height = Math.round(H * d);
  cx.setTransform(d, 0, 0, d, 0, 0);
}

function fit(r, aspect){ // the largest box of a given width/height ratio centred in r
  let w = r.w, h = w / aspect; if (h > r.h){ h = r.h; w = h * aspect; }
  return {x: r.x + (r.w - w) / 2, y: r.y + (r.h - h) / 2, w, h};
}
function layout(){
  const L = {}, fs = wide ? 13 : 12;
  L.fs = fs;
  L.wave = {x: 0, y: 22, w: W, h: wide ? H * 0.1 : 44};
  const secH = wide ? 34 : 30, stripH = wide ? 34 : 30;
  L.sec = {x: 0, y: H - 20 - secH, w: W, h: secH};
  L.strip = {x: 0, y: L.sec.y - 8 - stripH, w: W, h: stripH};
  const top = L.wave.y + L.wave.h + (wide ? 34 : 28), bot = L.strip.y - 26, sh = bot - top;
  if (wide){
    L.lens = {x: 0, y: top, w: W * 0.34, h: sh};
    L.model = {x: W * 0.41, y: top + sh * 0.3, w: W * 0.17, h: sh * 0.4};
    L.out = {x: W * 0.64, y: top, w: W * 0.36, h: sh};
  } else {
    const a = sh * 0.36, m = sh * 0.11, gap = (sh - 2 * a - m) / 2;
    L.lens = {x: 0, y: top, w: W, h: a};
    L.model = {x: W * 0.2, y: top + a + gap, w: W * 0.6, h: m};
    L.out = {x: 0, y: top + a + gap + m + gap, w: W, h: a};
  }
  const inset = r => ({x: r.x, y: r.y + 4, w: r.w, h: r.h - 26});
  L.spec = fit(inset(L.lens), wide ? 1.8 : 2.4);
  L.grid = fit(inset(L.out), 96 / 40);
  return L;
}
const bx = t => W * t / DUR;
const slot = (L, i) => ({x: bx(D.edges[i]), y: L.strip.y, w: Math.max(bx(D.edges[i + 1]) - bx(D.edges[i]) - 0.6, 0.6), h: L.strip.h});
const rgb = i => `rgb(${D.rgb[i][0]},${D.rgb[i][1]},${D.rgb[i][2]})`;

function text(s, x, y, o = {}){
  cx.font = `${o.weight || 400} ${o.size || 13}px ${o.mono ? "ui-monospace,Menlo,Consolas,monospace" : "-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif"}`;
  cx.textAlign = o.align || "left"; cx.textBaseline = o.base || "alphabetic"; cx.fillStyle = o.color || C["ink-2"];
  cx.fillText(s, x, y);
}
function arrow(x1, y1, x2, y2, a){
  if (a <= 0) return;
  cx.save(); cx.globalAlpha *= a; cx.strokeStyle = C["ink-3"]; cx.lineWidth = 1.25; cx.beginPath();
  cx.moveTo(x1, y1); cx.lineTo(x2, y2);
  const an = Math.atan2(y2 - y1, x2 - x1), h = 5;
  cx.moveTo(x2 - h * Math.cos(an - 0.6), y2 - h * Math.sin(an - 0.6)); cx.lineTo(x2, y2); cx.lineTo(x2 - h * Math.cos(an + 0.6), y2 - h * Math.sin(an + 0.6));
  cx.stroke(); cx.restore();
}
function rr(r, rad){ cx.beginPath(); cx.roundRect(r.x, r.y, r.w, r.h, rad); }

function draw(t){
  const L = layout(), fs = L.fs;
  cx.clearRect(0, 0, W, H);
  cx.imageSmoothingEnabled = false;

  /* the song's sound, with the bar marked */
  text(`${D.title}, ${clock(DUR)} of sound`, 0, 13, {size: fs});
  const wv = L.wave, mid = wv.y + wv.h / 2, n = peaks.length, step = wv.w / n;
  cx.fillStyle = C["ink-3"];
  for (let i = 0; i < n; i++){ const h = Math.max(peaks[i] / 255 * wv.h / 2, 0.5); cx.fillRect(i * step, mid - h, Math.max(step - 1, 0.8), 2 * h); }
  const b0 = bx(B.start), b1 = bx(B.end), mk = pr(t, 1.2, 2.2);
  if (mk > 0){
    cx.save(); cx.globalAlpha = mk; cx.fillStyle = C.accent; cx.fillRect(b0 - 0.5, wv.y - 3, Math.max(b1 - b0, 2) + 1, wv.h + 6);
    const lab = `bar ${K + 1} of ${N}`, right = b0 < W * 0.7;
    text(lab, right ? b1 + 6 : b0 - 6, wv.y + wv.h + 15, {size: fs, color: C.accent, weight: 600, align: right ? "left" : "right"});
    cx.restore();
  }

  /* station 1: the bar, as sound then as a spectrogram */
  const grow = pr(t, 2.4, 4.2), S = L.spec;
  if (grow > 0){
    const r = {x: mix(b0, S.x, grow), y: mix(wv.y, S.y, grow), w: mix(b1 - b0, S.w, grow), h: mix(wv.h, S.h, grow)};
    const toSpec = pr(t, 5.2, 7), gone = pr(t, 11.2, 13.4), dim = 1 - 0.6 * gone;
    cx.save(); cx.globalAlpha = dim;
    if (toSpec < 1){ cx.fillStyle = C.surface; rr(r, 3); cx.fill(); }
    if (toSpec < 1){
      cx.save(); cx.globalAlpha = dim * (1 - toSpec); cx.fillStyle = C.ink;
      const m = wave.length, st = r.w / m, cy = r.y + r.h / 2;
      for (let i = 0; i < m; i++){ const h = Math.max(wave[i] / 255 * r.h * 0.44, 0.5); cx.fillRect(r.x + i * st, cy - h, Math.max(st - 0.8, 0.6), 2 * h); }
      cx.restore();
    }
    if (toSpec > 0){ cx.save(); cx.globalAlpha = dim * toSpec; cx.drawImage(specCv, r.x, r.y, r.w, r.h); cx.restore(); }
    const gl = pr(t, 8.2, 9.6);
    if (gl > 0){
      cx.save(); cx.globalAlpha = dim * gl * 0.9; cx.strokeStyle = C.bg; cx.lineWidth = 1; cx.beginPath();
      const pc = B.specCols / 16, prw = B.specRows / 16;
      for (let i = 1; i < pc; i++){ cx.moveTo(r.x + r.w * i / pc, r.y); cx.lineTo(r.x + r.w * i / pc, r.y + r.h); }
      for (let j = 1; j < prw; j++){ cx.moveTo(r.x, r.y + r.h * j / prw); cx.lineTo(r.x + r.w, r.y + r.h * j / prw); }
      cx.stroke(); cx.restore();
    }
    cx.restore();
    if (grow >= 1){
      const lab = toSpec < 0.5 ? `bar ${K + 1}: ${secs} seconds of sound` : gl < 0.5 ? "the same bar as a spectrogram" : `split into ${(B.specCols / 16) * (B.specRows / 16)} patches`;
      text(lab, S.x + S.w / 2, S.y + S.h + 17, {size: fs, align: "center"});
      if (toSpec > 0.5 && wide){ cx.save(); cx.globalAlpha = toSpec;
        text("time", S.x + S.w, S.y - 5, {size: 11, align: "right", color: C["ink-3"]}); cx.restore(); }
    }
  }

  /* station 2: the frozen audio model */
  const M = L.model, mIn = pr(t, 10.4, 11.4);
  if (mIn > 0){
    const busy = t > 11.4 && t < 16.2 ? 0.5 + 0.5 * Math.sin(t * 7) : 0;
    cx.save(); cx.globalAlpha = mIn;
    cx.fillStyle = C.surface; rr(M, 5); cx.fill();
    cx.strokeStyle = busy ? C.accent : C.line; cx.lineWidth = 1 + busy; rr(M, 5); cx.stroke();
    if (wide){
      text("frozen", M.x + M.w / 2, M.y + M.h / 2 - 12, {size: fs, align: "center", color: C.ink, weight: 600});
      text("audio model", M.x + M.w / 2, M.y + M.h / 2 + 5, {size: fs, align: "center", color: C.ink, weight: 600});
      text("never told what", M.x + M.w / 2, M.y + M.h / 2 + 24, {size: 11, align: "center"});
      text("a chorus is", M.x + M.w / 2, M.y + M.h / 2 + 38, {size: 11, align: "center"});
    } else text("frozen audio model", M.x + M.w / 2, M.y + M.h / 2, {size: fs, align: "center", base: "middle", color: C.ink, weight: 600});
    cx.restore();
    if (wide){ arrow(S.x + S.w + 8, M.y + M.h / 2, M.x - 8, M.y + M.h / 2, mIn); }
    else arrow(W / 2, S.y + S.h + 24, W / 2, M.y - 5, mIn);
  }
  /* the patches travel into the model */
  if (t > 11.2 && t < 13.6){
    const pc = B.specCols / 16, prw = B.specRows / 16, tx = M.x + M.w / 2, ty = M.y + M.h / 2;
    for (let j = 0; j < prw; j++) for (let i = 0; i < pc; i++){
      const d = (i * prw + j) / (pc * prw) * 1.3, p = pr(t, 11.2 + d, 12.2 + d); if (p <= 0 || p >= 1) continue;
      const w0 = S.w / pc, h0 = S.h / prw, sc = mix(1, 0.3, p);
      const x = mix(S.x + i * w0, tx - w0 * sc / 2, p), y = mix(S.y + j * h0, ty - h0 * sc / 2, p);
      cx.save(); cx.globalAlpha = 1 - p * p * 0.8; cx.drawImage(specCv, i * 16, j * 16, 16, 16, x, y, w0 * sc, h0 * sc);
      cx.strokeStyle = C.accent; cx.lineWidth = 1; cx.strokeRect(x, y, w0 * sc, h0 * sc); cx.restore();
    }
  }

  /* station 3: the embedding, then three numbers, then one colour */
  const G = L.grid, fill = clamp((t - 13.6) / 3), squeeze = pr(t, 20, 21.6), sw = pr(t, 22.6, 24), fly = pr(t, 26, 27.6);
  if (t > 13.2){
    if (wide) arrow(M.x + M.w + 8, M.y + M.h / 2, L.out.x + (L.out.w - G.w) / 2 - 8, M.y + M.h / 2, pr(t, 13.2, 13.8));
    else arrow(W / 2, M.y + M.h + 5, W / 2, G.y - 8, pr(t, 13.2, 13.8));
  }
  if (fill > 0 && squeeze < 1){
    cx.save(); cx.globalAlpha = 1 - squeeze;
    const rows = Math.ceil(fill * 40), g = {x: G.x, y: mix(G.y, G.y + G.h * 0.35, squeeze), w: G.w, h: mix(G.h, G.h * 0.3, squeeze)};
    cx.drawImage(embCv, 0, 0, 96, rows, g.x, g.y, g.w, g.h * rows / 40);
    cx.restore();
    if (squeeze < 0.3){
      const shown = Math.min(D.dims, Math.round(fill * D.dims));
      text(`${shown.toLocaleString()} numbers`, G.x + G.w / 2, G.y + G.h + 17, {size: fs, align: "center"});
      if (fill >= 1 && wide) text("[" + B.first.slice(0, wide ? 4 : 5).map(v => v.toFixed(3)).join(", ") + ", …]", G.x + G.w / 2, G.y - 8, {size: 11, align: "center", mono: true, color: C["ink-3"]});
    }
  }
  if (squeeze > 0){
    cx.save(); cx.globalAlpha = squeeze * (1 - 0.5 * fly);
    const names = ["red", "green", "blue"], cols = ["#d8433b", "#2f9e5b", "#2f6fd6"], O = L.out;
    const R = wide ? G : {x: W * 0.04, y: G.y, w: W * 0.92, h: G.h};
    const rowH = Math.min(R.h / 3, 34), labW = 46, valW = 38, swW = Math.min(R.h, wide ? 84 : 76);
    const x0 = R.x + labW, trackW = R.w - labW - valW - swW - 18, y0 = R.y + (R.h - 3 * rowH) / 2;
    for (let k = 0; k < 3; k++){
      const y = y0 + k * rowH + rowH * 0.2, h = rowH * 0.6, v = B.scaled[k] * pr(t, 20.6 + k * 0.3, 22 + k * 0.3);
      text(names[k], x0 - 8, y + h / 2, {size: fs, align: "right", base: "middle"});
      cx.fillStyle = C.surface; cx.fillRect(x0, y, trackW, h);
      cx.fillStyle = cols[k]; cx.fillRect(x0, y, trackW * v, h);
      text(v.toFixed(2), x0 + trackW + 6, y + h / 2, {size: 12, base: "middle", mono: true, color: C.ink});
    }
    cx.restore();
    const home = {x: R.x + R.w - swW, y: R.y + (R.h - swW) / 2, w: swW, h: swW}, to = slot(L, K);
    if (sw > 0){
      const r = {x: mix(home.x, to.x, fly), y: mix(home.y, to.y, fly), w: mix(home.w, to.w, fly), h: mix(home.h, to.h, fly)};
      if (fly > 0){ cx.save(); cx.globalAlpha = 0.5 * fly; cx.strokeStyle = C.line; cx.lineWidth = 1; cx.setLineDash([3, 3]); rr(home, 4); cx.stroke(); cx.restore(); }
      if (fly < 1){ cx.save(); cx.globalAlpha = sw; cx.fillStyle = rgb(K); rr(r, mix(4, 0, fly)); cx.fill(); cx.restore(); }
    }
    const lab = sw < 0.5 ? "the three directions that vary most" : "one colour for the bar";
    cx.save(); cx.globalAlpha = squeeze * (1 - 0.5 * fly); text(lab, G.x + G.w / 2, G.y + G.h + 17, {size: fs, align: "center"}); cx.restore();
  }

  /* the strip: one colour per bar */
  if (fly > 0){
    const alike = pr(t, 31, 32.2) * (1 - pr(t, 36.6, 37.6)), st = L.strip;
    cx.save(); cx.globalAlpha = fly; text(t < 28 ? `bar ${K + 1}'s colour, in its place` : `all ${N} bars, one colour each`, 0, st.y - 9, {size: fs}); cx.restore();
    const wipe = (t - 27.6) / 2.8;
    for (let i = 0; i < N; i++){
      const r = slot(L, i); let a = i === K ? (fly >= 1 ? 1 : 0) : clamp((wipe - i / N) * 8);
      if (a <= 0) continue;
      const isNear = i === K || near.includes(i);
      cx.globalAlpha = a * (isNear ? 1 : 1 - 0.72 * alike);
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
  const sIn = pr(t, 37.4, 39);
  if (sIn > 0){
    const sc = L.sec; cx.save(); cx.globalAlpha = sIn;
    for (const s of D.sections){
      const a = bx(s.start), b = bx(s.end), r = {x: a + 1, y: sc.y, w: Math.max(b - a - 2, 1), h: sc.h};
      cx.fillStyle = FILL[s.name] || "#ccc"; rr(r, 3); cx.fill();
      if (s.unsure){ cx.save(); rr(r, 3); cx.clip(); cx.strokeStyle = "rgba(255,255,255,.6)"; cx.lineWidth = 3; cx.beginPath();
        for (let x = r.x - r.h; x < r.x + r.w; x += 8){ cx.moveTo(x, r.y + r.h); cx.lineTo(x + r.h, r.y); } cx.stroke(); cx.restore(); }
      const lab = (wide ? s.name : (SHORT[s.name] || s.name)) + (s.unsure ? "?" : "");
      cx.font = `500 ${fs - 1}px -apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif`;
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
let t = 0, playing = false, last = 0, cur = -1;
STEPS.forEach((s, i) => { const li = document.createElement("li"), b = document.createElement("button");
  b.type = "button";  b.textContent = `${i + 1} ${s[1]}`; b.onclick = () => { t = reduced ? stepEnd(i) : s[0]; if (!reduced) setPlaying(true); paint(); };
  li.appendChild(b); stepsEl.appendChild(li); });
const stepEnd = i => (i + 1 < STEPS.length ? STEPS[i + 1][0] : END) - 0.01;
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
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
if (document.fonts && document.fonts.ready) document.fonts.ready.then(paint);
if (reduced) t = END;
refresh();
return {start(){ if (reduced) return; t = 0; setPlaying(true); paint(); }, stop(){ setPlaying(false); paint(); }};
}
