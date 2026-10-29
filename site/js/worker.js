// worker.js: 模型を Web Worker で回す(画面を止めない)。命令の受け方は engine.js の IM.handle。
/* global importScripts, IM */
importScripts('model/rng.js', 'model/solver.js', 'model/im_core.js', 'model/im1.js', 'model/im3.js', 'engine.js');
const E = new IM.Engine(m => postMessage(m));
onmessage = ev => IM.handle(E, ev.data, m => postMessage(m));
