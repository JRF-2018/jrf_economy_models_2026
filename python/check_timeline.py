# 時間のつまみの関門: 鍵の年から計算し直した枝が、初めから通しで計算した数と完全に同じであることを、quickjs で確かめる。
#   python python/check_timeline.py
import json, os, sys, time
import quickjs
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ctx = quickjs.Context()
ctx.set_memory_limit(1 << 30); ctx.set_time_limit(-1)
for f in ('model/rng.js', 'model/solver.js', 'model/im_core.js', 'model/im1.js', 'model/im3.js', 'engine.js'):
    ctx.eval(open(os.path.join(ROOT, 'site', 'js', f), encoding='utf-8').read())
t0 = time.time()
out = json.loads(ctx.eval('''JSON.stringify((function () {
  const frames = new Map();
  const E = new IM.Engine(m => { if (m.type === 'frame') { if (!frames.has(m.branch)) frames.set(m.branch, []); frames.get(m.branch)[m.frame.t] = m.frame; } });
  const setup = IM.defaultSetup(); setup.Ns = 60;
  const b0 = E.reset(setup);
  E.compute(b0, 35);
  E.command(b0, 35, [{type: 'param', region: 0, data: {r_cb: 0.04}}]);          // 終わりで足す(枝は分けない)
  E.compute(b0, 47);
  // 年 23 に戻って、別の試し(富裕税とサイコロ)。枝の年 0..22 は、親の数を写す
  const b1 = E.command(b0, 23, [{type: 'param', region: 2, data: {wealth_tax: 0.01}}, {type: 'dice', data: {seed: 12345}}]);
  frames.set(b1, frames.get(b0).slice(0, 23));
  E.compute(b1, 52);
  // 年 41 に戻って、災害と能力の半減
  const b2 = E.command(b1, 41, [{type: 'disaster', region: 1, data: {size: 0.3, scope: 'common'}}, {type: 'capital_loss', region: 0, data: {share: 0.5}}]);
  frames.set(b2, frames.get(b1).slice(0, 41));
  E.compute(b2, 60);
  // 古い枝 b0 の先を、もう一度伸ばす(今の世界は b2 にいるので、鍵の年から計算し直す)
  E.compute(b0, 55);
  const res = {};
  for (const b of [b0, b1, b2]) res['branch ' + b] = E.verify(b, frames.get(b));
  // セーブ → ロード
  const rec = E.recipe(b2);
  const json = JSON.stringify(rec);
  const loaded = [];
  const E2 = new IM.Engine(m => { if (m.type === 'frame') loaded[m.frame.t] = m.frame; });
  E2.load(JSON.parse(json));
  let diff = 0, n = 0;
  for (let t = 0; t < rec.end; t++) loaded[t].regions.forEach((r, i) => { for (const k in r) if (r[k] !== frames.get(b2)[t].regions[i][k] && !(Number.isNaN(r[k]) && Number.isNaN(frames.get(b2)[t].regions[i][k]))) { n++; diff = Math.max(diff, Math.abs(r[k] - frames.get(b2)[t].regions[i][k])); } });
  res['save/load'] = {years: rec.end, mismatches: n, maxAbsDiff: diff, bytes: json.length};
  // 一国だけの世界(危機時の生産)と、プレイヤーの選び方の命令
  const fs = new Map();
  const E3 = new IM.Engine(m => { if (m.type === 'frame') { if (!fs.has(m.branch)) fs.set(m.branch, []); fs.get(m.branch)[m.frame.t] = m.frame; } });
  const one = IM.defaultSetup(); one.regions = ['A']; one.Ns = 60; one.warmup = 20;
  one.common = Object.assign({}, one.common, { crisis_track: true, crisis_rule: 'last', g_stock: 0.05 }); one.region_kw = { A: { player: true, vote: true, vote_noise: true } };
  const c0 = E3.reset(one);
  E3.compute(c0, 30);
  E3.command(c0, 30, [{type: 'param', region: 0, data: {cap_N: 2.2}}, {type: 'act', region: 0, data: {spend: 0.7, lux: 0.1, work: 0.8, vote: 'hi', campaign: 1}}]);
  E3.compute(c0, 45);
  const c1 = E3.command(c0, 33, [{type: 'act', region: 0, data: {child: true}}, {type: 'dice', data: {seed: 7}}]);
  fs.set(c1, fs.get(c0).slice(0, 33));
  E3.compute(c1, 50);
  res['single country, branch 0'] = E3.verify(c0, fs.get(c0));
  res['single country, branch 1'] = E3.verify(c1, fs.get(c1));
  res.single_capped = {capped_years: fs.get(c0).filter(f => f.regions[0].capped > 0).length, pl_gen: fs.get(c1)[49].regions[0].pl_gen};
  // 枝が本当に違う道を通ったか(違わなければ、確かめにならない)
  const E35 = (b) => frames.get(b)[34].regions[2].E, E50 = (b) => frames.get(b)[45].regions[2].E;
  res.differ = {b0_vs_b1_t45: E50(b0) - E50(b1), b0_vs_b1_t34: E35(b0) - E35(b1)};
  return res;
})())'''))
for k, v in out.items():
    print(k, v)
ok = all(v['mismatches'] == 0 for k, v in out.items() if k not in ('differ', 'single_capped')) and out['single_capped']['capped_years'] > 0 and out['differ']['b0_vs_b1_t45'] != 0 and out['differ']['b0_vs_b1_t34'] != 0
print(f'{time.time() - t0:.1f}s', 'OK' if ok else 'FAIL')
sys.exit(0 if ok else 1)
