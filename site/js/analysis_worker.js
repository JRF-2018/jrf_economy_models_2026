// analysis_worker.js: 分析の画面の仕事(平行世界、政策の地図)を回す Worker。状態から始めて年を進め、系列を返す。
/* global importScripts, IM */
importScripts('model/rng.js', 'model/solver.js', 'model/im_core.js', 'model/im1.js', 'model/im3.js', 'engine.js');
onmessage = ev => {
  const m = ev.data;
  try {
    const r = IM.runJob(m.setup, m.state, m.cmds, m.years, m.keys);
    postMessage({ type: 'job', id: m.id, result: r });
  } catch (err) {
    postMessage({ type: 'job', id: m.id, error: String(err && err.stack || err) });
  }
};
