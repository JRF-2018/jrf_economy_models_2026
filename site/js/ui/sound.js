// sound.js: 効果音(Web Audio で、その場で合成する。音のファイルは使わない)。
// ブラウザは、人が一度押すまで音を出させないので、初めの押し(どこでも)で鳴らせるようにする(unlock)。
// 海の音(低い波の音)が後ろで鳴り、嵐の年は風が強くなる。

const NOTE = n => 440 * Math.pow(2, (n - 69) / 12);       // MIDI の番号 → 周波数

export class Sound {
  constructor() {
    this.ctx = null;
    this.on = true;
    this.vol = 0.7;
    try {
      const s = localStorage.getItem('im-sandbox-sound');
      if (s) { const o = JSON.parse(s); this.on = !!o.on; this.vol = +o.vol; }
    } catch (e) { /* 使えない窓 */ }
    this.listeners = [];
  }
  save() { try { localStorage.setItem('im-sandbox-sound', JSON.stringify({ on: this.on, vol: this.vol })); } catch (e) { /* 使えない窓 */ } }
  get ready() { return !!this.ctx && this.ctx.state === 'running'; }
  unlock() {
    try {
      if (!this.ctx) {
        const AC = window.AudioContext || window.webkitAudioContext;
        if (!AC) return;
        this.ctx = new AC();
        this.master = this.ctx.createGain();
        this.master.gain.value = this.on ? this.vol : 0;
        const comp = this.ctx.createDynamicsCompressor();
        this.master.connect(comp); comp.connect(this.ctx.destination);
        // 白い雑音(2 秒)を使い回す
        const len = this.ctx.sampleRate * 2, buf = this.ctx.createBuffer(1, len, this.ctx.sampleRate), d = buf.getChannelData(0);
        let seed = 12345;
        for (let i = 0; i < len; i++) { seed = (seed * 16807) % 2147483647; d[i] = seed / 1073741823.5 - 1; }
        this.noiseBuf = buf;
        this._ambient();
      }
      if (this.ctx.state !== 'running') this.ctx.resume();
    } catch (e) { this.ctx = null; }
    for (const f of this.listeners) f();
  }
  setOn(on) { this.on = on; if (this.master) this.master.gain.setTargetAtTime(on ? this.vol : 0, this.ctx.currentTime, 0.05); this.save(); }
  setVol(v) { this.vol = v; if (this.master && this.on) this.master.gain.setTargetAtTime(v, this.ctx.currentTime, 0.05); this.save(); }

  // ------------------------------------------------ 部品
  _noise(t0, dur, { type = 'lowpass', f0 = 1000, f1 = null, q = 0.7, gain = 0.5, attack = 0.005 } = {}) {
    const c = this.ctx, src = c.createBufferSource(), flt = c.createBiquadFilter(), g = c.createGain();
    src.buffer = this.noiseBuf; src.loop = true;
    flt.type = type; flt.frequency.setValueAtTime(f0, t0); flt.Q.value = q;
    if (f1) flt.frequency.exponentialRampToValueAtTime(f1, t0 + dur);
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.exponentialRampToValueAtTime(gain, t0 + attack);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    src.connect(flt); flt.connect(g); g.connect(this.master);
    src.start(t0, Math.random() * 1.5); src.stop(t0 + dur + 0.05);
  }
  _tone(t0, dur, freq, { type = 'sine', gain = 0.3, attack = 0.01, f1 = null, lp = null, detune = 0 } = {}) {
    const c = this.ctx, o = c.createOscillator(), g = c.createGain();
    o.type = type; o.frequency.setValueAtTime(freq, t0); o.detune.value = detune;
    if (f1) o.frequency.exponentialRampToValueAtTime(f1, t0 + dur);
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.exponentialRampToValueAtTime(gain, t0 + attack);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    let last = o;
    if (lp) { const f = c.createBiquadFilter(); f.type = 'lowpass'; f.frequency.value = lp; o.connect(f); last = f; }
    last.connect(g); g.connect(this.master);
    o.start(t0); o.stop(t0 + dur + 0.05);
  }
  _bell(t0, freq, gain = 0.25, dur = 2.2) {
    for (const [r, a] of [[1, 1], [2.76, 0.45], [5.4, 0.25], [8.93, 0.12]]) this._tone(t0, dur / Math.sqrt(r), freq * r, { gain: gain * a, attack: 0.003 });
  }
  _drum(t0, gain = 0.8) {
    this._tone(t0, 0.5, 110, { f1: 45, gain, attack: 0.002 });
    this._noise(t0, 0.18, { type: 'bandpass', f0: 1800, q: 0.8, gain: gain * 0.5 });
  }
  _ambient() {
    // 海: 低く流した雑音を、ゆっくり波打たせる
    const c = this.ctx, src = c.createBufferSource(), flt = c.createBiquadFilter(), g = c.createGain(), lfo = c.createOscillator(), lg = c.createGain();
    src.buffer = this.noiseBuf; src.loop = true;
    flt.type = 'lowpass'; flt.frequency.value = 420;
    g.gain.value = 0.05;
    lfo.frequency.value = 0.09; lg.gain.value = 0.035;
    lfo.connect(lg); lg.connect(g.gain);
    src.connect(flt); flt.connect(g); g.connect(this.master);
    src.start(); lfo.start();
    this.sea = { flt, g };
    // 風(嵐の年に強くなる)
    const w = c.createBufferSource(), wf = c.createBiquadFilter(), wg = c.createGain();
    w.buffer = this.noiseBuf; w.loop = true;
    wf.type = 'bandpass'; wf.frequency.value = 600; wf.Q.value = 0.6;
    wg.gain.value = 0.0;
    w.connect(wf); wf.connect(wg); wg.connect(this.master);
    w.start();
    this.wind = { wf, wg };
  }
  storm(level) {
    if (!this.ready || !this.wind) return;
    const t = this.ctx.currentTime;
    this.wind.wg.gain.setTargetAtTime(0.18 * level, t, 0.4);
    this.wind.wf.frequency.setTargetAtTime(500 + 500 * level + 200 * Math.random(), t, 0.6);
  }

  // ------------------------------------------------ 効果音
  play(name, opt = {}) {
    if (!this.ready || !this.on) return;
    const c = this.ctx, t = c.currentTime + 0.02;
    switch (name) {
      case 'disaster': {      // 雷鳴と地響き
        this._noise(t, 3.2, { f0: 2400, f1: 80, gain: 0.9, attack: 0.01 });
        this._noise(t + 0.25, 2.6, { f0: 300, f1: 60, gain: 0.7, attack: 0.3 });
        this._tone(t, 3.0, 48, { f1: 30, gain: 0.5, attack: 0.2 });
        this._noise(t + 0.9, 1.6, { type: 'highpass', f0: 3000, gain: 0.12, attack: 0.4 });   // 雨
        break;
      }
      case 'local': {         // 局地の災害: 短い雷と雨
        this._noise(t, 1.6, { f0: 1800, f1: 120, gain: 0.6 });
        this._noise(t + 0.2, 2.0, { type: 'highpass', f0: 2500, gain: 0.15, attack: 0.3 });
        break;
      }
      case 'collapse': {      // 建物が崩れる: 割れる音と鈍い響き
        for (let k = 0; k < 6; k++) this._noise(t + k * 0.07 + Math.random() * 0.05, 0.25, { type: 'bandpass', f0: 1200 + 1500 * Math.random(), q: 2, gain: 0.35 });
        this._tone(t + 0.05, 1.2, 70, { f1: 35, gain: 0.6, attack: 0.01 });
        this._noise(t + 0.3, 1.8, { f0: 500, f1: 120, gain: 0.3, attack: 0.2 });
        break;
      }
      case 'hike': {          // 利上げ: 下がる二つの鐘
        this._bell(t, NOTE(76), 0.22); this._bell(t + 0.35, NOTE(69), 0.22);
        break;
      }
      case 'euphoria': {      // 熱狂: きらきらと上がる分散和音
        [72, 76, 79, 84, 88, 91].forEach((n, k) => this._tone(t + k * 0.07, 0.9, NOTE(n), { type: 'triangle', gain: 0.12 }));
        this._noise(t, 1.0, { type: 'highpass', f0: 6000, gain: 0.05, attack: 0.3 });
        break;
      }
      case 'revolution': {    // 革命: 太鼓の連打と、金管の短調の和音
        for (let k = 0; k < 8; k++) this._drum(t + k * 0.11, 0.35 + 0.08 * k);
        this._drum(t + 1.0, 1.0);
        for (const n of [48, 55, 60, 63, 67]) this._tone(t + 1.0, 2.6, NOTE(n), { type: 'sawtooth', gain: 0.07, attack: 0.15, lp: 1800, detune: (Math.random() - 0.5) * 12 });
        this._noise(t + 1.0, 2.5, { type: 'bandpass', f0: 700, q: 0.4, gain: 0.12, attack: 0.5 });   // 群衆のどよめき
        break;
      }
      case 'restoration': {   // 体制がもどる: 下がる和音
        for (const n of [55, 60, 64]) this._tone(t, 1.8, NOTE(n + 12), { type: 'sawtooth', f1: NOTE(n + 5), gain: 0.06, lp: 1400, attack: 0.1 });
        this._drum(t, 0.6);
        break;
      }
      case 'election': {      // 選挙: 鐘を三つ
        [79, 76, 72].forEach((n, k) => this._bell(t + k * 0.28, NOTE(n), 0.18));
        break;
      }
      case 'workshop': {      // 工房が開く: 澄んだ響き
        [84, 91, 96].forEach((n, k) => this._tone(t + k * 0.09, 1.2, NOTE(n), { gain: 0.1 }));
        break;
      }
      case 'upgrade': {       // 格上げ: ファンファーレ
        [[60, 0], [64, 0.14], [67, 0.28], [72, 0.42]].forEach(([n, d]) => this._tone(t + d, 0.35, NOTE(n), { type: 'square', gain: 0.07, lp: 2600 }));
        for (const n of [60, 64, 67, 72, 76]) this._tone(t + 0.6, 2.2, NOTE(n), { type: 'sawtooth', gain: 0.05, attack: 0.05, lp: 3000 });
        this._bell(t + 0.6, NOTE(84), 0.12);
        break;
      }
      case 'gone': {          // 工房が消える
        this._tone(t, 0.6, NOTE(67), { type: 'triangle', gain: 0.12 }); this._tone(t + 0.3, 0.9, NOTE(63), { type: 'triangle', gain: 0.12 });
        break;
      }
      case 'breakups': {      // 一家離散: 低い一打
        this._tone(t, 0.9, 82, { f1: 55, gain: 0.35 }); this._tone(t + 0.12, 0.8, NOTE(56), { type: 'triangle', gain: 0.06 });
        break;
      }
      case 'alarm': {         // 財政の行き詰まり、踏み倒し: 警報
        for (let k = 0; k < 4; k++) { this._tone(t + k * 0.36, 0.17, 880, { type: 'sawtooth', gain: 0.06, lp: 2000 }); this._tone(t + k * 0.36 + 0.18, 0.17, 660, { type: 'sawtooth', gain: 0.06, lp: 2000 }); }
        break;
      }
      case 'shortage': {      // 品不足、価格の上限: 低い警告
        this._tone(t, 0.3, 220, { type: 'square', gain: 0.06, lp: 900 }); this._tone(t + 0.35, 0.45, 196, { type: 'square', gain: 0.06, lp: 900 });
        break;
      }
      case 'generation': {    // 代替わり: オルゴールの旋律
        [72, 76, 79, 84, 83, 79, 76, 81].forEach((n, k) => this._tone(t + k * 0.22, 0.9, NOTE(n), { type: 'triangle', gain: 0.1 }));
        break;
      }
      case 'child': {         // 子が生まれる: 高い二つの響き
        this._bell(t, NOTE(88), 0.12, 1.4); this._bell(t + 0.18, NOTE(93), 0.12, 1.6);
        break;
      }
      case 'dice': {          // サイコロ: 転がる音
        for (let k = 0; k < 9; k++) this._noise(t + k * 0.055 * (1 + k * 0.08), 0.05, { type: 'bandpass', f0: 2500 + 1500 * Math.random(), q: 4, gain: 0.4 });
        this._tone(t + 0.62, 0.15, 1400, { type: 'triangle', gain: 0.08 });
        break;
      }
      case 'fork': {          // 新しい試し: 風切り
        this._noise(t, 0.7, { type: 'bandpass', f0: 300, f1: 3000, q: 1.5, gain: 0.25, attack: 0.15 });
        break;
      }
      case 'undo': {
        this._noise(t, 0.5, { type: 'bandpass', f0: 2500, f1: 300, q: 1.5, gain: 0.2, attack: 0.1 });
        break;
      }
      case 'click': {
        this._tone(t, 0.06, 1800, { type: 'triangle', gain: 0.05, attack: 0.002 });
        break;
      }
      case 'tick': {          // 年が進む(×1 と ×4 だけ)
        this._tone(t, 0.08, 1200, { type: 'sine', gain: 0.025, attack: 0.002 });
        break;
      }
    }
  }
}
