// scene.js: 三つの国の島の 3D の情景(three.js)。海、島、三つの分野の建物、家々(標本の記入)、銀行と政府の塔、
// 島の間の流れ(財・資本・銀行の間の貸し)、出来事の演出、昼と夜、ブルーム。
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { t, fmt, pct } from '../i18n.js';

export const ACCENT = ['#2dd4bf', '#f59e0b', '#a78bfa'];
const REGIME_COLOR = new THREE.Color('#ff3b3b');
const SOOT = new THREE.Color('#2a2730');                   // 局地の災害で打たれた家           // 段 B7: 革命のあとの体制の旗
const ISLAND_POS = [new THREE.Vector3(-17.5, 0, 9), new THREE.Vector3(17.5, 0, 9), new THREE.Vector3(0, 0, -19)];
const TOP_Y = 1.6;
const HOUSE_R0 = 4.8, HOUSE_R1 = 10.2;
const K0 = [1250, 400, 1100];

function radialTexture(inner = 'rgba(255,255,255,1)', outer = 'rgba(255,255,255,0)') {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  const gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  gr.addColorStop(0, inner);
  gr.addColorStop(0.35, inner.replace(/[\d.]+\)$/, '0.6)'));
  gr.addColorStop(1, outer);
  g.fillStyle = gr;
  g.fillRect(0, 0, 64, 64);
  const tx = new THREE.CanvasTexture(c);
  tx.colorSpace = THREE.SRGBColorSpace;
  return tx;
}

// 島の輪郭の揺らぎ(角度ごと。決まった形)
function shoreNoise(theta, seed) {
  return 1 + 0.055 * Math.sin(3 * theta + seed) + 0.035 * Math.sin(7 * theta + 2.1 * seed) + 0.02 * Math.sin(13 * theta + 0.7 * seed);
}
function wobbleCylinder(geo, seed, keepCenter = true) {
  const p = geo.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), z = p.getZ(i);
    const r = Math.hypot(x, z);
    if (keepCenter && r < 1e-6) continue;
    const th = Math.atan2(z, x);
    const f = shoreNoise(th, seed);
    p.setX(i, x * f);
    p.setZ(i, z * f);
  }
  geo.computeVertexNormals();
  return geo;
}

const WATER_VS = `
uniform float uTime;
varying vec3 vPos;
varying vec3 vNormal;
float h(vec2 p) {
  return 0.25 * sin(p.x * 0.18 + uTime * 0.9) + 0.18 * sin(p.y * 0.23 - uTime * 0.7)
       + 0.08 * sin((p.x + p.y) * 0.5 + uTime * 1.7) + 0.05 * sin((p.x - 1.3 * p.y) * 0.9 + uTime * 2.3);
}
void main() {
  vec3 p = position;
  vec2 q = (modelMatrix * vec4(p, 1.0)).xz;
  float e = 0.5;
  float y = h(q);
  vec3 n = normalize(vec3(-(h(q + vec2(e, 0.0)) - y) / e, 1.0, -(h(q + vec2(0.0, e)) - y) / e));
  vec4 wp = modelMatrix * vec4(p.x, p.y, p.z, 1.0);
  wp.y += y;
  vPos = wp.xyz;
  vNormal = n;
  gl_Position = projectionMatrix * viewMatrix * wp;
}`;
const WATER_FS = `
uniform vec3 uSun;
uniform vec3 uDeep;
uniform vec3 uShallow;
uniform vec3 uSky;
uniform float uDay;
varying vec3 vPos;
varying vec3 vNormal;
void main() {
  vec3 V = normalize(cameraPosition - vPos);
  vec3 N = normalize(vNormal);
  float fres = pow(1.0 - max(dot(N, V), 0.0), 3.0);
  float dist = length(vPos.xz);
  vec3 base = mix(uShallow, uDeep, smoothstep(10.0, 90.0, dist));
  vec3 col = mix(base, uSky, 0.55 * fres);
  vec3 H = normalize(normalize(uSun) + V);
  float spec = pow(max(dot(N, H), 0.0), 120.0) * (0.25 + 1.6 * uDay);
  col += vec3(1.0, 0.92, 0.75) * spec;
  float fog = smoothstep(80.0, 260.0, length(vPos - cameraPosition));
  col = mix(col, uSky * 0.8, fog);
  gl_FragColor = vec4(col, 1.0);
}`;

export class Scene {
  constructor(container, n = 3, Ns = 400) {
    this.container = container;
    this.n = n;
    this.Ns = Ns;
    this.quality = 'high';
    this.selected = 0;
    this.pickCbs = [];
    this.clock = new THREE.Clock();
    this.phase = 0.25;
    this.storms = [];
    this.shake = 0;

    const w = container.clientWidth, h = container.clientHeight;
    const R = this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
    R.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    R.setSize(w, h);
    R.toneMapping = THREE.ACESFilmicToneMapping;
    R.toneMappingExposure = 1.05;
    R.outputColorSpace = THREE.SRGBColorSpace;
    R.shadowMap.enabled = true;
    R.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(R.domElement);
    R.domElement.classList.add('scene-canvas');

    this.labels = new CSS2DRenderer();
    this.labels.setSize(w, h);
    this.labels.domElement.className = 'scene-labels';
    container.appendChild(this.labels.domElement);

    const S = this.scene = new THREE.Scene();
    S.background = new THREE.Color('#0b1430');
    S.fog = new THREE.FogExp2('#0b1430', 0.0042);
    const cam = this.camera = new THREE.PerspectiveCamera(42, w / h, 0.5, 900);
    cam.position.set(0, 56, 80);
    const ctl = this.controls = new OrbitControls(cam, this.labels.domElement);
    ctl.target.set(0, 0, -2);
    ctl.enableDamping = true;
    ctl.dampingFactor = 0.07;
    ctl.maxPolarAngle = 1.38;
    ctl.minDistance = 14;
    ctl.maxDistance = 160;
    ctl.autoRotate = false;

    // 光
    this.hemi = new THREE.HemisphereLight('#bcd7ff', '#1d2a3a', 0.55);
    S.add(this.hemi);
    const sun = this.sun = new THREE.DirectionalLight('#fff1d6', 2.2);
    sun.position.set(40, 60, 20);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const sc = sun.shadow.camera;
    sc.left = -45; sc.right = 45; sc.top = 45; sc.bottom = -45; sc.near = 10; sc.far = 200;
    sun.shadow.bias = -0.0005;
    S.add(sun);
    S.add(sun.target);
    this.moon = new THREE.DirectionalLight('#7aa2ff', 0.0);
    this.moon.position.set(-40, 50, -30);
    S.add(this.moon);

    this._buildSky();
    this._buildWater();
    this.islands = [];
    for (let i = 0; i < n; i++) this.islands.push(this._buildIsland(i));
    this._buildFlows();

    // 後処理
    this.composer = new EffectComposer(R);
    this.composer.addPass(new RenderPass(S, cam));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(w, h), 0.75, 0.55, 0.82);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    let down = null;
    this.labels.domElement.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; });
    this.labels.domElement.addEventListener('pointerup', e => {
      if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
      const r = this.renderer.domElement.getBoundingClientRect();
      this.pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      this.raycaster.setFromCamera(this.pointer, this.camera);
      const hit = this.raycaster.intersectObjects(this.islands.map(o => o.pick), false)[0];
      if (hit) for (const f of this.pickCbs) f(hit.object.userData.region);
    });
    // 島の上のものの説明(マウスを載せると出る)
    this.tip = document.createElement('div');
    this.tip.className = 'scene-tip';
    container.appendChild(this.tip);
    this.raycaster.params.Line.threshold = 0.6;
    let lastMove = 0;
    this.labels.domElement.addEventListener('pointermove', e => {
      const now = performance.now();
      if (now - lastMove < 40) return;
      lastMove = now;
      this.hover(e.clientX, e.clientY);
    });
    this.labels.domElement.addEventListener('pointerleave', () => { this.tip.classList.remove('show'); });
    window.addEventListener('resize', () => this.resize());
    this.resize();
    this._loop = this._loop.bind(this);
    requestAnimationFrame(this._loop);
  }

  onPick(f) { this.pickCbs.push(f); }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.labels.setSize(w, h);
    this.composer.setSize(w, h);
    this.bloom.setSize(w, h);
  }

  setQuality(q) {
    this.quality = q;
    this.renderer.setPixelRatio(q === 'high' ? Math.min(window.devicePixelRatio || 1, 2) : 1);   // 軽い画質: 画素の比を 1 に
    this.resize();
    this.renderer.shadowMap.enabled = q === 'high';
    this.sun.castShadow = q === 'high';
    this.scene.traverse(o => { if (o.material) o.material.needsUpdate = true; });
  }

  // ------------------------------------------------ 空と海
  _buildSky() {
    const g = new THREE.BufferGeometry();
    const N = 1500, pos = new Float32Array(N * 3);
    let seed = 7;
    const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
    for (let i = 0; i < N; i++) {
      const u = rnd() * 2 - 1, th = rnd() * Math.PI * 2, r = 420;
      const s = Math.sqrt(1 - u * u);
      pos.set([r * s * Math.cos(th), Math.abs(r * u) + 20, r * s * Math.sin(th)], i * 3);
    }
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    this.stars = new THREE.Points(g, new THREE.PointsMaterial({ size: 1.6, sizeAttenuation: false, color: '#dfe8ff', transparent: true, opacity: 0.8, depthWrite: false, fog: false }));
    this.scene.add(this.stars);
  }
  _buildWater() {
    const geo = new THREE.PlaneGeometry(700, 700, 220, 220);
    geo.rotateX(-Math.PI / 2);
    this.waterMat = new THREE.ShaderMaterial({
      uniforms: { uTime: { value: 0 }, uSun: { value: new THREE.Vector3(1, 1, 0) }, uDeep: { value: new THREE.Color('#06213f') },
                  uShallow: { value: new THREE.Color('#0f6e8c') }, uSky: { value: new THREE.Color('#9cc6ff') }, uDay: { value: 1 } },
      vertexShader: WATER_VS, fragmentShader: WATER_FS,
    });
    const water = new THREE.Mesh(geo, this.waterMat);
    water.position.y = -0.4;
    this.scene.add(water);
  }

  // ------------------------------------------------ 島
  _buildIsland(i) {
    const S = this.scene;
    const grp = new THREE.Group();
    grp.position.copy(ISLAND_POS[i]);
    S.add(grp);
    const accent = new THREE.Color(ACCENT[i]);
    const seed = 1.7 + i * 2.3;
    // 砂浜と崖
    const sand = new THREE.Mesh(wobbleCylinder(new THREE.CylinderGeometry(12.6, 14.2, 2.6, 96, 1), seed),
                                new THREE.MeshStandardMaterial({ color: '#d9c39a', roughness: 0.95 }));
    sand.position.y = -0.2;
    sand.receiveShadow = true;
    grp.add(sand);
    const grassCol = new THREE.Color('#3f8f4f').lerp(accent, 0.12);
    const top = new THREE.Mesh(wobbleCylinder(new THREE.CylinderGeometry(11.0, 12.0, 0.9, 96, 1), seed),
                               new THREE.MeshStandardMaterial({ color: grassCol, roughness: 0.9 }));
    top.position.y = TOP_Y - 0.45;
    top.receiveShadow = true;
    top.userData.region = i;
    grp.add(top);
    // 選んだ国の輪
    const ring = new THREE.Mesh(new THREE.TorusGeometry(14.8, 0.12, 8, 160),
                                new THREE.MeshBasicMaterial({ color: accent, transparent: true, opacity: 0.0 }));
    ring.rotation.x = Math.PI / 2;
    ring.position.y = 0.3;
    grp.add(ring);
    // 地価の輝き(熱狂は金、崩れは赤)
    const glow = new THREE.Mesh(new THREE.TorusGeometry(13.6, 0.5, 12, 160),
                                new THREE.MeshBasicMaterial({ color: '#ffcc55', transparent: true, opacity: 0.0, blending: THREE.AdditiveBlending, depthWrite: false }));
    glow.rotation.x = Math.PI / 2;
    glow.position.y = 0.5;
    grp.add(glow);

    // 銀行(神殿)と政府の塔
    const bank = new THREE.Group();
    const stone = new THREE.MeshStandardMaterial({ color: '#e9e4d8', roughness: 0.55, metalness: 0.05 });
    const base = new THREE.Mesh(new THREE.BoxGeometry(2.6, 0.35, 1.9), stone);
    base.position.y = 0.17;
    bank.add(base);
    for (let k = 0; k < 6; k++) {
      const col = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.13, 1.3, 12), stone);
      col.position.set(-1.05 + k * 0.42, 1.0, 0.7);
      col.castShadow = true;
      bank.add(col);
      const col2 = col.clone(); col2.position.z = -0.7; bank.add(col2);
    }
    const roof = new THREE.Mesh(new THREE.ConeGeometry(1.9, 0.75, 4), stone);
    roof.rotation.y = Math.PI / 4;
    roof.scale.set(1.0, 1.0, 0.72);
    roof.position.y = 2.05;
    roof.castShadow = true;
    bank.add(roof);
    const bankGlow = new THREE.PointLight(accent, 0, 9);
    bankGlow.position.y = 1.2;
    bank.add(bankGlow);
    bank.position.set(-1.8, TOP_Y, 0.4);
    grp.add(bank);
    const gov = new THREE.Group();
    const govMat = new THREE.MeshStandardMaterial({ color: '#cfd8e6', roughness: 0.35, metalness: 0.4, emissive: new THREE.Color('#ff2a2a'), emissiveIntensity: 0 });
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.55, 0.8, 4.4, 20), govMat);
    shaft.position.y = 2.2;
    shaft.castShadow = true;
    gov.add(shaft);
    const dome = new THREE.Mesh(new THREE.SphereGeometry(0.75, 24, 12, 0, Math.PI * 2, 0, Math.PI / 2),
                                new THREE.MeshStandardMaterial({ color: accent, metalness: 0.7, roughness: 0.25, emissive: accent, emissiveIntensity: 0.25 }));
    dome.position.y = 4.4;
    gov.add(dome);
    const flagPole = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 1.4, 6), new THREE.MeshStandardMaterial({ color: '#ddd' }));
    flagPole.position.y = 5.6;
    gov.add(flagPole);
    const flag = new THREE.Mesh(new THREE.PlaneGeometry(0.8, 0.45), new THREE.MeshStandardMaterial({ color: accent, side: THREE.DoubleSide, emissive: accent, emissiveIntensity: 0.4 }));
    flag.position.set(0.42, 6.05, 0);
    gov.add(flag);
    gov.position.set(1.9, TOP_Y, -0.4);
    grp.add(gov);

    // 三つの分野の建物
    const fieldGeo = new THREE.BoxGeometry(1.25, 0.08, 0.9);
    const farm = new THREE.InstancedMesh(fieldGeo, new THREE.MeshStandardMaterial({ roughness: 0.95 }), 14);
    const siloGeo = new THREE.CylinderGeometry(0.28, 0.28, 1, 12).translate(0, 0.5, 0);
    const silo = new THREE.InstancedMesh(siloGeo, new THREE.MeshStandardMaterial({ color: '#c9b79c', roughness: 0.6 }), 14);
    const towerGeo = new THREE.CylinderGeometry(0.32, 0.4, 1, 8).translate(0, 0.5, 0);
    const towerMat = new THREE.MeshStandardMaterial({ color: '#8fd3ff', roughness: 0.1, metalness: 0.8, emissive: new THREE.Color('#ff7ad9'), emissiveIntensity: 0.6 });
    const tower = new THREE.InstancedMesh(towerGeo, towerMat, 12);
    const facGeo = new THREE.BoxGeometry(1.0, 1, 0.8).translate(0, 0.5, 0);
    const fac = new THREE.InstancedMesh(facGeo, new THREE.MeshStandardMaterial({ color: '#8a8f99', roughness: 0.7, metalness: 0.3 }), 12);
    const chimGeo = new THREE.CylinderGeometry(0.1, 0.14, 1, 8).translate(0, 0.5, 0);
    const chim = new THREE.InstancedMesh(chimGeo, new THREE.MeshStandardMaterial({ color: '#5b5f66' }), 12);
    for (const m of [farm, silo, tower, fac, chim]) { m.castShadow = true; m.receiveShadow = true; grp.add(m); }
    const slots = (a0, a1, k, r0, r1) => {
      const out = [];
      for (let q = 0; q < k; q++) {
        const a = a0 + (a1 - a0) * ((q * 0.618) % 1);
        const r = r0 + (r1 - r0) * ((q * 0.381 + 0.2) % 1);
        out.push([r * Math.cos(a), r * Math.sin(a), a]);
      }
      return out;
    };
    const sectors = {
      farm, silo, tower, fac, chim, towerMat,
      farmSlots: slots(Math.PI * 0.55, Math.PI * 1.2, 14, 2.6, 4.3),
      towerSlots: slots(Math.PI * 1.25, Math.PI * 1.85, 12, 2.6, 4.2),
      facSlots: slots(-Math.PI * 0.1, Math.PI * 0.5, 12, 2.7, 4.3),
    };
    // 煙(原料の工場)
    const smokeN = 160;
    const sg = new THREE.BufferGeometry();
    sg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(smokeN * 3), 3));
    const smoke = new THREE.Points(sg, new THREE.PointsMaterial({ size: 1.1, map: radialTexture('rgba(190,190,200,0.55)', 'rgba(190,190,200,0)'),
                                                                  transparent: true, depthWrite: false, opacity: 0.7 }));
    smoke.userData.seeds = Float32Array.from({ length: smokeN }, (_, k) => (k * 0.6180339) % 1);
    grp.add(smoke);

    // 家々
    const Ns = this.Ns + 1;                 // プレイヤーの記入の分を足す
    const houseGeo = new THREE.BoxGeometry(0.44, 1, 0.44).translate(0, 0.5, 0);
    const houses = new THREE.InstancedMesh(houseGeo, new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.15 }), Ns);
    const roofGeo = new THREE.ConeGeometry(0.38, 0.32, 4).rotateY(Math.PI / 4).translate(0, 0.16, 0);
    const roofs = new THREE.InstancedMesh(roofGeo, new THREE.MeshStandardMaterial({ color: '#7b3b2a', roughness: 0.8 }), Ns);
    houses.castShadow = true; houses.receiveShadow = true; roofs.castShadow = true;
    grp.add(houses); grp.add(roofs);
    const hpos = [];
    const golden = Math.PI * (3 - Math.sqrt(5));
    for (let k = 0; k < Ns; k++) {
      const r = Math.sqrt(HOUSE_R0 * HOUSE_R0 + (HOUSE_R1 * HOUSE_R1 - HOUSE_R0 * HOUSE_R0) * (k + 0.5) / Ns);
      const a = k * golden;
      const f = shoreNoise(a, seed);
      hpos.push([r * Math.cos(a) * Math.min(f, 1.0), r * Math.sin(a) * Math.min(f, 1.0), a]);
    }
    const col = new THREE.Color('#4f8cff');
    for (let k = 0; k < Ns; k++) { houses.setColorAt(k, col); roofs.setColorAt(k, new THREE.Color('#7b3b2a')); }
    const H = { hPrev: new Float32Array(Ns).fill(0.4), hNext: new Float32Array(Ns).fill(0.4), c: new Float32Array(Ns), r: new Float32Array(Ns) };

    // 嵐(災害)
    const rainN = 700;
    const rg = new THREE.BufferGeometry();
    const rp = new Float32Array(rainN * 3);
    for (let k = 0; k < rainN; k++) { const a = Math.random() * Math.PI * 2, rr = Math.sqrt(Math.random()) * 12; rp.set([rr * Math.cos(a), Math.random() * 14, rr * Math.sin(a)], 3 * k); }
    rg.setAttribute('position', new THREE.BufferAttribute(rp, 3));
    const rain = new THREE.Points(rg, new THREE.PointsMaterial({ color: '#a9c7ff', size: 0.16, transparent: true, opacity: 0.0, depthWrite: false }));
    grp.add(rain);
    const cloudMat = new THREE.MeshStandardMaterial({ color: '#2a2f3a', roughness: 1, transparent: true, opacity: 0.0, depthWrite: false });
    const cloud = new THREE.Group();
    for (let k = 0; k < 9; k++) {
      const s = new THREE.Mesh(new THREE.SphereGeometry(2.6 + (k % 3), 16, 12), cloudMat);
      const a = k * 0.7;
      s.position.set(5.5 * Math.cos(a) * (k % 2 ? 1 : 0.5), 14 + (k % 3) * 0.6, 5.5 * Math.sin(a) * (k % 2 ? 1 : 0.5));
      s.scale.y = 0.55;
      cloud.add(s);
    }
    grp.add(cloud);
    const flash = new THREE.PointLight('#cfe0ff', 0, 40);
    flash.position.y = 12;
    grp.add(flash);

    // 守りの堤(災害に備えて積む資本)
    const wall = new THREE.Mesh(wobbleCylinder(new THREE.CylinderGeometry(12.9, 12.9, 1, 96, 1, true), seed),
                                new THREE.MeshStandardMaterial({ color: '#9fb8d6', roughness: 0.5, metalness: 0.3, transparent: true, opacity: 0.55, side: THREE.DoubleSide }));
    wall.position.y = 0.2;
    wall.scale.y = 0.001;
    grp.add(wall);
    // 壊れた建物の土埃
    const dg = new THREE.BufferGeometry();
    dg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(90 * 3), 3));
    const dust = new THREE.Points(dg, new THREE.PointsMaterial({ size: 1.4, map: radialTexture('rgba(170,150,120,0.6)', 'rgba(170,150,120,0)'), transparent: true, depthWrite: false, opacity: 0 }));
    dust.userData.seeds = Float32Array.from({ length: 90 }, (_, k) => (k * 0.6180339 + 0.21) % 1);
    grp.add(dust);
    // 段 B7: 新しい財の工房(光る丸屋根。数は工房の割合に比例)
    const domeGeo = new THREE.SphereGeometry(0.42, 16, 10, 0, Math.PI * 2, 0, Math.PI / 2);
    const domes = new THREE.InstancedMesh(domeGeo, new THREE.MeshStandardMaterial({ color: '#7ff6ff', emissive: new THREE.Color('#33e0ff'), emissiveIntensity: 1.2, roughness: 0.2, metalness: 0.3 }), 30);
    domes.count = 0;
    grp.add(domes);
    const domeSlots = [];
    for (let k = 0; k < 30; k++) { const a = Math.PI * (0.62 + 1.25 * ((k * 0.618) % 1)), r = 4.35 + 0.25 * (k % 2); domeSlots.push([r * Math.cos(a), r * Math.sin(a)]); }
    // プレイヤーの家の光の柱
    const beacon = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 14, 12, 1, true).translate(0, 7, 0),
                                  new THREE.MeshBasicMaterial({ color: '#ffe08a', transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending, depthWrite: false }));
    beacon.visible = false;
    // 光の柱は細いので、マウスの当たりは太めの見えない筒で(色を書かない)
    const beaconHit = new THREE.Mesh(new THREE.CylinderGeometry(0.7, 0.7, 14, 8, 1, true).translate(0, 7, 0), new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: false }));
    beacon.add(beaconHit);
    grp.add(beacon);
    // 名札
    const el = document.createElement('div');
    el.className = 'island-label';
    el.style.setProperty('--accent', ACCENT[i]);
    el.innerHTML = `<div class="il-name"></div><div class="il-cur"></div><div class="il-stats"></div>`;
    el.addEventListener('click', () => { for (const f of this.pickCbs) f(i); });
    const label = new CSS2DObject(el);
    label.position.set(0, 11.5, 0);
    grp.add(label);

    const pick = top;
    const accentCol = accent.clone();
    return { grp, top, ring, glow, bank, bankGlow, gov, govMat, dome, flag, sectors, smoke, houses, roofs, hpos, H, rain, cloud, cloudMat, flash, el, pick, label,
             dust, ruin: 0, dustAmt: 0, lastR: null, accent: accentCol, wall, wallTarget: 0, beacon, beaconIdx: null, election: 0, domes, domeSlots, regime: 0, revFlash: 0, upFlash: 0,
             storm: 0, crack: 0, glowAmt: 0, glowCol: new THREE.Color('#ffcc55'), facCount: 0, chimneyTops: [] };
  }

  // ------------------------------------------------ 島の間の流れ
  _buildFlows() {
    const S = this.scene;
    this.arcs = [];
    const tex = radialTexture();
    const mk = (a, b, lift, side, color, size, N) => {
      const pa = ISLAND_POS[a].clone().setY(3.2), pb = ISLAND_POS[b].clone().setY(3.2);
      const mid = pa.clone().add(pb).multiplyScalar(0.5);
      const dir = pb.clone().sub(pa).normalize();
      const perp = new THREE.Vector3(-dir.z, 0, dir.x).multiplyScalar(side);
      mid.add(perp).setY(lift);
      const curve = new THREE.QuadraticBezierCurve3(pa, mid, pb);
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(N * 3), 3));
      const pts = new THREE.Points(g, new THREE.PointsMaterial({ color, size, map: tex, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
      pts.frustumCulled = false;
      S.add(pts);
      const lineG = new THREE.BufferGeometry().setFromPoints(curve.getPoints(60));
      const line = new THREE.Line(lineG, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.0, blending: THREE.AdditiveBlending, depthWrite: false }));
      S.add(line);
      const u = Float32Array.from({ length: N }, (_, k) => k / N);
      return { a, b, curve, pts, line, u, N, amt: 0, target: 0, speed: 0.12 };
    };
    for (let a = 0; a < this.n; a++) for (let b = 0; b < this.n; b++) {
      if (a === b) continue;
      this.arcs.push(Object.assign(mk(a, b, 13 + 2 * ((a + b) % 2), 2.2, '#e8f4ff', 0.75, 90), { kind: 'trade' }));
      this.arcs.push(Object.assign(mk(a, b, 19, -3.4, '#ffc94d', 1.0, 60), { kind: 'capital' }));
    }
    // 銀行の間の貸し(同じ通貨の国どうし): 低い紫の帯
    this.ibArcs = [];
    for (let a = 0; a < this.n; a++) for (let b = a + 1; b < this.n; b++) {
      const pa = ISLAND_POS[a].clone().setY(1.0), pb = ISLAND_POS[b].clone().setY(1.0);
      const mid = pa.clone().add(pb).multiplyScalar(0.5).setY(4.5);
      const curve = new THREE.QuadraticBezierCurve3(pa, mid, pb);
      const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, 64, 0.22, 8, false),
                                  new THREE.MeshBasicMaterial({ color: '#b26bff', transparent: true, opacity: 0.0, blending: THREE.AdditiveBlending, depthWrite: false }));
      S.add(tube);
      this.ibArcs.push({ a, b, tube, amt: 0, target: 0 });
    }
  }

  // ------------------------------------------------ 島の上のものの説明
  pickables() {
    if (this._picks) return this._picks;
    const out = [];
    const tag = (o, kind, region, extra) => { o.traverse(c => { c.userData.tip = Object.assign({ kind, region }, extra || {}); }); out.push(o); };
    this.islands.forEach((I, i) => {
      tag(I.houses, 'house', i); tag(I.roofs, 'house', i);
      tag(I.sectors.farm, 'farm', i); tag(I.sectors.silo, 'farm', i);
      tag(I.sectors.tower, 'tower', i);
      tag(I.sectors.fac, 'factory', i); tag(I.sectors.chim, 'factory', i);
      tag(I.bank, 'bank', i); tag(I.gov, 'gov', i);
      tag(I.wall, 'wall', i); tag(I.beacon, 'beacon', i); tag(I.domes, 'dome', i);
      tag(I.cloud, 'storm', i); tag(I.glow, 'glow', i); tag(I.top, 'island', i);
    });
    for (const A of this.arcs) tag(A.line, A.kind, A.a, { b: A.b, arc: A });
    for (const B of this.ibArcs) tag(B.tube, 'ib', B.a, { b: B.b, arc: B });
    this._picks = out;
    return out;
  }
  hover(cx, cy) {
    const r = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set(((cx - r.left) / r.width) * 2 - 1, -((cy - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObjects(this.pickables(), true);
    const shown = o => {
      for (let q = o; q; q = q.parent) if (!q.visible) return false;
      const m = o.material;
      if (m && m.transparent && m.opacity < 0.05) return false;
      if (o.isInstancedMesh && o.count === 0) return false;
      if (o === undefined) return false;
      return true;
    };
    const hit = hits.find(h => h.object.userData.tip && shown(h.object) && (h.object.userData.tip.region < (this.count || this.n)));
    if (!hit || !this.frame) { this.tip.classList.remove('show'); return; }
    const html = this.tipText(hit.object.userData.tip, hit.instanceId);
    if (!html) { this.tip.classList.remove('show'); return; }
    this.tip.innerHTML = html;
    const cr = this.container.getBoundingClientRect();
    const x = Math.min(cx - cr.left + 14, cr.width - 290), y = Math.max(cy - cr.top - 10, 8);
    this.tip.style.left = x + 'px';
    this.tip.style.top = y + 'px';
    this.tip.classList.add('show');
  }
  tipText(tp, inst) {
    const f = this.frame, i = tp.region, r = f.regions[i];
    if (!r) return '';
    const nm = (this.names || ['A', 'B', 'C'])[i];
    const head = (title) => `<b>${title}</b> <small>${nm}</small>`;
    const line = (k, v) => `<div><span>${t(k)}</span><em>${v}</em></div>`;
    switch (tp.kind) {
      case 'house': {
        const hs = f.houses[i];
        if (inst === undefined || !hs || inst >= hs.h.length) return '';
        const me = f.pl && f.pl[i] === inst;
        return head(me ? t('tip_house_you') : t('tip_house')) + `<p>${t('tip_house_p')}</p>`
          + line('tip_nw', fmt(hs.h[inst], 2) + '×') + line('tip_asset', pct(hs.c[inst], 0)) + line('tip_risk', pct(Math.min(hs.r[inst], 9.99), 0));
      }
      case 'farm': return head(t('tip_farm')) + `<p>${t('tip_farm_p')}</p>` + line('tip_K', fmt(r.KN, 0)) + line('tip_u', pct(r.uN, 0)) + line('tip_price', fmt(r.PN, 2))
        + (r.capped > 0 ? `<div class="warn">${t('tip_capped')}</div>` : '') + (r.short > 0.005 ? `<div class="warn">${t('tip_short', { p: pct(r.short, 0) })}</div>` : '');
      case 'tower': return head(t('tip_tower')) + `<p>${t('tip_tower_p')}</p>` + line('tip_K', fmt(r.KL, 0)) + line('tip_u', pct(r.uL, 0));
      case 'factory': return head(t('tip_factory')) + `<p>${t('tip_factory_p')}</p>` + line('tip_K', fmt(r.KI, 0)) + line('tip_u', pct(r.uI, 0));
      case 'bank': return head(t('tip_bank')) + `<p>${t('tip_bank_p')}</p>` + line('p_r_cb', pct(r.r_policy, 2)) + line('tip_r_loan', pct(r.r_loan, 2)) + line('tip_loans', fmt(r.loans, 0));
      case 'gov': return head(t('tip_gov')) + `<p>${t('tip_gov_p')}</p>` + line('ind_tau', pct(r.tau, 1)) + line('ind_b', fmt(r.b, 2) + '×')
        + line('ind_tau_k', pct(r.tau_k < 0 ? r.tau : r.tau_k, 0)) + (r.regime ? `<div class="warn">${t('tip_regime')}</div>` : '') + (r.crisis > 0 ? `<div class="warn">${t('tip_crisis')}</div>` : '');
      case 'wall': return head(t('tip_wall')) + `<p>${t('tip_wall_p')}</p>` + line('tip_prot', fmt(r.prot, 0));
      case 'beacon': return head(t('tip_beacon')) + `<p>${t('tip_beacon_p')}</p>` + line('tip_nw', fmt(r.pl_nw_rel, 2) + '×') + line('life_gen', fmt(r.pl_gen, 0));
      case 'dome': return head(t('tip_dome')) + `<p>${t('tip_dome_p')}</p>` + line('il_workshop', pct(r.up_s || 0, 0));
      case 'storm': return r.dis_size > 0 ? head(t('tip_storm')) + `<p>${t('tip_storm_p')}</p>` + line('tip_dis', pct(r.dis_size, 0)) : '';
      case 'glow': return head(t('tip_glow')) + `<p>${t('tip_glow_p')}</p>` + line('ind_pT', fmt(r.pT, 0));
      case 'island': return head(t('tip_island')) + `<p>${t('tip_island_p')}</p>` + line('ind_E', fmt(r.E, 0)) + line('ind_real_wage', fmt(r.real_wage, 2)) + line('ind_Y', fmt(r.Y, 0));
      case 'trade': case 'capital': {
        const a = tp.region, b = tp.b, names = this.names || ['A', 'B', 'C'];
        const v = tp.kind === 'trade' ? f.flows[b][a] : (f.dest[a] === b ? f.KF[a] : 0);
        return `<b>${t(tp.kind === 'trade' ? 'tip_trade' : 'tip_capital')}</b> <small>${names[a]} → ${names[b]}</small><p>${t(tp.kind === 'trade' ? 'tip_trade_p' : 'tip_capital_p')}</p>`
          + line('tip_amount', fmt(v, 0));
      }
      case 'ib': {
        const names = this.names || ['A', 'B', 'C'];
        return `<b>${t('tip_ib')}</b> <small>${names[tp.region]} — ${names[tp.b]}</small><p>${t('tip_ib_p')}</p>` + line('tip_ib_v', fmt(f.regions[tp.region].ib || 0, 0));
      }
    }
    return '';
  }

  // ------------------------------------------------ 年の数を情景に当てる
  // frame: その年の記録(engine.js の makeFrame)。prev: 前の年。info: {names, curLabel(i), dur}
  setFrame(frame, prev, info = {}) {
    if (!frame) return;
    this.frame = frame;
    if (info.names) this.names = info.names;
    const n = this.n;
    for (let i = 0; i < Math.min(n, frame.regions.length); i++) {
      const I = this.islands[i], r = frame.regions[i], hs = frame.houses[i];
      // 家
      const H = I.H;
      H.hPrev.set(this._curHeights(I));
      const m = Math.min(hs.h.length, H.hNext.length);
      I.houses.count = m; I.roofs.count = m;
      for (let k = 0; k < m; k++) {
        const v = hs.h[k];
        H.hNext[k] = v <= 0 ? 0.18 : Math.min(0.35 + 1.25 * Math.log10(1 + 6 * v), 7.5);
        H.c[k] = hs.c[k];
        H.r[k] = hs.r[k];
      }
      I.hBlend = 0;
      I.hSettled = false;
      this._houseColors(I, 0);
      // 分野の建物: 数は能力に、高さと明るさは稼働率に。能力が急に減った年(災害・能力の半減)は、建物が壊れて、土埃が立つ
      if (prev) {
        const pr = prev.regions[i];
        const drop = Math.max(1 - r.KN / pr.KN, 1 - r.KL / pr.KL, 1 - r.KI / pr.KI);
        if (drop > 0.05) { I.ruin = Math.max(I.ruin || 0, Math.min(1, drop * 3)); I.dustAmt = 1; }
      }
      // 局地の災害: 打たれた世帯の家だけが煤ける(その年だけ)
      I.H.hit = r.dis_size > 0 && hs.hit ? hs.hit : null;
      I.lastR = r;
      this._sectors(I, r);
      // 出来事
      if (r.dis_size > 0) { I.storm = 1.0; this.shake = Math.max(this.shake, 0.6 * Math.min(r.dis_size / 0.3, 1.5)); }
      const crisis = r.crisis > 0 || r.defaulted > 0;
      I.crack = r.defaulted > 0 ? 1.4 : crisis ? 0.8 : Math.max(0, Math.min((r.b - 0.8) / 0.6, 0.6));
      if (prev) {
        const pr = prev.regions[i];
        const g = Math.log(r.pT / pr.pT);
        I.glowAmt = Math.min(Math.abs(g) * 6, 1.0);
        I.glowCol.set(g >= 0 ? '#ffcc55' : '#ff4d4d');
      }
      I.bankGlow.intensity = 6 * Math.min(Math.max(r.r_policy || 0, 0) / 0.06, 2);
      // 守り(災害の守りの資本)は、島を囲む堤の高さ
      I.wallTarget = Math.min((r.prot || 0) / 200, 1.6);
      // 段 B7: 体制(旗の色)、革命の年の閃光と揺れ、工房の丸屋根
      I.regime = r.regime || 0;
      if (r.revolution > 0) { I.revFlash = 1.0; I.election = 1.0; this.shake = Math.max(this.shake, 0.5); }
      if (r.upgraded > 0) I.upFlash = 1.0;
      const nd = r.up_s > 0 ? Math.max(1, Math.min(30, Math.round(30 * r.up_s))) : 0;
      if (nd !== I.domes.count) {
        const m4 = new THREE.Matrix4();
        for (let k = 0; k < nd; k++) { const [dx, dz] = I.domeSlots[k]; m4.makeTranslation(dx, TOP_Y, dz); I.domes.setMatrixAt(k, m4); }
        I.domes.count = nd;
        I.domes.instanceMatrix.needsUpdate = true;
        I.domes.boundingSphere = null; I.domes.boundingBox = null;
      }
      // 選挙の年は旗が光る
      if (prev && r.vote_hi !== undefined && (r.vote_hi !== prev.regions[i].vote_hi || r.vote_lo !== prev.regions[i].vote_lo)) I.election = 1.0;
      // プレイヤーの家の光の柱
      const pl = frame.pl ? frame.pl[i] : null;
      I.beacon.visible = pl !== null && pl !== undefined;
      if (I.beacon.visible) { I.beaconIdx = pl; }
      // 名札
      const names = info.names || ['A', 'B', 'C'];
      I.el.querySelector('.il-name').textContent = names[i];
      I.el.querySelector('.il-cur').textContent = info.curLabel ? info.curLabel(i) : '';
      I.el.querySelector('.il-stats').innerHTML = info.stats ? info.stats(i) : '';
      I.el.classList.toggle('sel', i === this.selected);
    }
    // 流れ: 財(産地 r から買い手 s へ)、資本(出し手から行き先へ)
    let maxT = 1e-9;
    for (let s = 0; s < frame.regions.length; s++) for (let r = 0; r < frame.regions.length; r++) if (s !== r) maxT = Math.max(maxT, frame.flows[s][r]);
    const Ysum = frame.regions.reduce((a, r) => a + r.Y, 0) / frame.regions.length;
    const nr = frame.regions.length;
    for (const A of this.arcs) {
      if (A.a >= nr || A.b >= nr) { A.target = 0; continue; }
      if (A.kind === 'trade') {
        const v = frame.flows[A.b][A.a];                 // 買い手 b が、産地 a に払う(財は a から b へ)
        A.target = Math.min(v / (0.18 * Ysum), 1.0);
        A.speed = 0.08 + 0.12 * A.target;
      } else {
        const o = A.a, d = frame.dest[o];
        const kf = frame.KF[o];
        A.target = (d === A.b && kf > 0) ? Math.min(kf / (0.01 * Ysum), 1.0) : 0;
        if (d === A.a && frame.KF[A.b] < 0 && frame.dest[A.b] === A.a) A.target = Math.min(-frame.KF[A.b] / (0.01 * Ysum), 1.0);
        A.speed = 0.15 + 0.2 * A.target;
      }
    }
    for (const B of this.ibArcs) {
      if (B.a >= nr || B.b >= nr) { B.target = 0; continue; }
      const same = frame.cur[B.a] === frame.cur[B.b];
      const ra = frame.regions[B.a], rb = frame.regions[B.b];
      const v = same ? Math.max(Math.abs(ra.ib || 0) / ra.Y, Math.abs(rb.ib || 0) / rb.Y) : 0;
      B.target = Math.min(v * 2.5, 0.85);
    }
    this.yearDur = info.dur || 1.0;
  }

  _curHeights(I) {
    const H = I.H, b = Math.min(I.hBlend || 0, 1), out = new Float32Array(H.hNext.length);
    const e = b * b * (3 - 2 * b);
    for (let k = 0; k < out.length; k++) out[k] = H.hPrev[k] + (H.hNext[k] - H.hPrev[k]) * e;
    return out;
  }
  _houseColors(I, time) {
    const H = I.H, blue = new THREE.Color('#3f7dff'), gold = new THREE.Color('#ffc23d'), red = new THREE.Color('#ff2d3d'), neg = new THREE.Color('#7a3d4a');
    const c = new THREE.Color();
    const blink = 0.5 + 0.5 * Math.sin(time * 9);
    for (let k = 0; k < H.c.length; k++) {
      if (I.H.hNext[k] <= 0.18) c.copy(neg);
      else c.copy(blue).lerp(gold, Math.min(Math.max(H.c[k], 0), 1));
      if (H.r[k] > 0.8) c.lerp(red, 0.35 + 0.6 * blink * Math.min((H.r[k] - 0.8) / 0.2, 1));
      if (H.hit && H.hit[k] && I.storm > 0.02) c.lerp(SOOT, 0.75 * Math.min(I.storm * 1.5, 1));
      I.houses.setColorAt(k, c);
    }
    I.houses.instanceColor.needsUpdate = true;
  }
  _sectors(I, r) {
    const S = I.sectors, m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), up = new THREE.Vector3(0, 1, 0);
    const nF = Math.max(1, Math.min(14, Math.round(8 * r.KN / K0[0])));
    const nT = Math.max(1, Math.min(12, Math.round(6 * r.KL / K0[1])));
    const nI = Math.max(1, Math.min(12, Math.round(6 * r.KI / K0[2])));
    const green = new THREE.Color('#5fae4f'), wheat = new THREE.Color('#e3c55a');
    const c = new THREE.Color();
    const short = Math.min(Math.max((r.short || 0) * 8, r.capped ? 0.6 : 0), 1);    // 品不足は、畑が赤く
    const red = new THREE.Color('#ff3b3b'), ash = new THREE.Color('#3b3a40');
    // 壊れた建物(災害・能力の半減のあと): 一部が傾き、沈み、暗くなる。能力が戻るにつれて建ち直る(ruin は年とともに減る)
    const ruin = I.ruin || 0;
    const broken = k => ((k * 0.618034 + 0.13) % 1) < ruin;
    const tilt = new THREE.Quaternion(), ax = new THREE.Vector3(1, 0, 0.4).normalize();
    const wreck = (k, pos, qq, scl) => {
      if (!broken(k)) return m4.compose(pos, qq, scl);
      tilt.setFromAxisAngle(ax, 0.55 * ruin * (k % 2 ? 1 : -1));
      return m4.compose(pos.clone().setY(pos.y - 0.25 * ruin), qq.clone().multiply(tilt), scl.clone().setY(scl.y * (1 - 0.65 * ruin)));
    };
    S.farmSlots.forEach(([x, z, a], k) => {
      const on = k < nF;
      q.setFromAxisAngle(up, -a);
      m4.compose(new THREE.Vector3(x, TOP_Y + 0.04, z), q, new THREE.Vector3(on ? 1 : 0.001, 1, on ? 1 : 0.001));
      S.farm.setMatrixAt(k, m4);
      c.copy(green).lerp(wheat, Math.min(Math.max(r.uN, 0), 1.2) * 0.8).lerp(red, short);
      if (broken(k)) c.lerp(ash, 0.7 * ruin);
      S.farm.setColorAt(k, c);
      const hs = on && k % 2 === 0 ? 0.8 + 0.9 * Math.min(r.uN, 1.3) : 0.001;
      wreck(k, new THREE.Vector3(x + 0.5, TOP_Y, z + 0.4), q, new THREE.Vector3(1, hs, 1));
      S.silo.setMatrixAt(k, m4);
    });
    S.towerSlots.forEach(([x, z], k) => {
      const on = k < nT;
      const h = on ? 1.8 + 3.2 * Math.min(r.uL, 1.3) * (0.7 + 0.3 * ((k * 0.37) % 1)) : 0.001;
      wreck(k, new THREE.Vector3(x, TOP_Y, z), q.identity(), new THREE.Vector3(on ? 1 : 0.001, h, on ? 1 : 0.001));
      S.tower.setMatrixAt(k, m4);
    });
    S.towerMat.emissiveIntensity = (0.3 + 1.6 * Math.min(Math.max(r.uL, 0), 1.3)) * (1 - 0.6 * ruin);
    I.chimneyTops = [];
    S.facSlots.forEach(([x, z, a], k) => {
      const on = k < nI;
      q.setFromAxisAngle(up, -a);
      const h = on ? 0.8 + 0.5 * Math.min(r.uI, 1.3) : 0.001;
      wreck(k, new THREE.Vector3(x, TOP_Y, z), q, new THREE.Vector3(on ? 1 : 0.001, h, on ? 1 : 0.001));
      S.fac.setMatrixAt(k, m4);
      const ch = on ? 1.6 + 0.6 * Math.min(r.uI, 1.3) : 0.001;
      wreck(k, new THREE.Vector3(x + 0.3, TOP_Y, z - 0.25), q, new THREE.Vector3(1, ch, 1));
      S.chim.setMatrixAt(k, m4);
      if (on) I.chimneyTops.push([x + 0.3, TOP_Y + ch, z - 0.25]);
    });
    I.smokeRate = Math.min(Math.max(r.uI, 0), 1.4);
    for (const m of [S.farm, S.silo, S.tower, S.fac, S.chim]) { m.instanceMatrix.needsUpdate = true; if (m.instanceColor) m.instanceColor.needsUpdate = true; m.boundingSphere = null; m.boundingBox = null; }
    // ↑ 包む球を捨てる(InstancedMesh は、包む球を一度計算すると覚えている。置き場所を変えたら捨てないと、マウスの当たりを外す)
  }

  // 島の数(一国だけの世界では、ほかの島と流れを隠す)
  setCount(m) {
    this.islands.forEach((I, i) => { I.grp.visible = i < m; I.label.visible = i < m; I.el.style.display = i < m ? '' : 'none'; });
    if (m === 1 && this.count !== 1) this.focus(0);
    else if (m > 1 && this.count === 1) this.focus(null);
    for (const A of this.arcs) { const v = A.a < m && A.b < m; A.pts.visible = v; A.line.visible = v; }
    for (const B of this.ibArcs) B.tube.visible = B.a < m && B.b < m;
    this.count = m;
  }
  select(i) {
    this.selected = i;
    this.islands.forEach((I, k) => I.el.classList.toggle('sel', k === i));
  }
  focus(i) {
    const p = i === null || i === undefined ? new THREE.Vector3(0, 0, -2) : ISLAND_POS[i];
    const from = this.controls.target.clone(), camFrom = this.camera.position.clone();
    const off = i === null || i === undefined ? new THREE.Vector3(0, 56, 80) : new THREE.Vector3(0, 26, 34);
    const camTo = p.clone().add(off);
    const g = window.gsap;
    if (g) {
      const o = { k: 0 };
      g.to(o, { k: 1, duration: 1.2, ease: 'power2.inOut', onUpdate: () => {
        this.controls.target.lerpVectors(from, p, o.k);
        this.camera.position.lerpVectors(camFrom, camTo, o.k);
      } });
    } else {
      this.controls.target.copy(p);
      this.camera.position.copy(camTo);
    }
  }
  // 一人の暮らし: プレイヤーの家に寄る
  focusHouse(i, k) {
    const I = this.islands[i];
    if (!I || k === null || k === undefined || !I.hpos[k]) return this.focus(i);
    const [x, z] = I.hpos[k];
    const p = ISLAND_POS[i].clone().add(new THREE.Vector3(x, TOP_Y + 1, z));
    const out = new THREE.Vector3(x, 0, z).normalize();
    const camTo = p.clone().add(out.multiplyScalar(15)).add(new THREE.Vector3(0, 13, 0));
    const from = this.controls.target.clone(), camFrom = this.camera.position.clone();
    const g = window.gsap;
    if (g) {
      const o = { k: 0 };
      g.to(o, { k: 1, duration: 1.4, ease: 'power2.inOut', onUpdate: () => {
        this.controls.target.lerpVectors(from, p, o.k);
        this.camera.position.lerpVectors(camFrom, camTo, o.k);
      } });
    } else { this.controls.target.copy(p); this.camera.position.copy(camTo); }
  }
  setPhase(ph) { this.phase = ph; }

  // ------------------------------------------------ 毎こま
  _loop() {
    requestAnimationFrame(this._loop);
    const dt = Math.min(this.clock.getDelta(), 0.1);
    const time = this.clock.elapsedTime;
    // 昼と夜
    const ang = (this.phase - 0.25) * Math.PI * 2;
    const sy = Math.sin(ang + Math.PI / 2), sx = Math.cos(ang + Math.PI / 2);
    const day = Math.max(0, Math.min(1, sy * 1.6 + 0.35));
    this.sun.position.set(70 * sx, 20 + 70 * Math.max(sy, 0.05), 30);
    this.sun.intensity = 0.25 + 2.3 * day;
    this.moon.intensity = 0.55 * (1 - day);
    this.hemi.intensity = 0.25 + 0.5 * day;
    const night = new THREE.Color('#070b1d'), dusk = new THREE.Color('#e9876a'), noon = new THREE.Color('#8fc3f2');
    const sky = day > 0.5 ? dusk.clone().lerp(noon, (day - 0.5) * 2) : night.clone().lerp(dusk, day * 2);
    this.scene.background.copy(sky);
    this.scene.fog.color.copy(sky);
    this.stars.material.opacity = 0.9 * (1 - day);
    this.waterMat.uniforms.uTime.value = time;
    this.waterMat.uniforms.uSun.value.copy(this.sun.position).normalize();
    this.waterMat.uniforms.uSky.value.copy(sky);
    this.waterMat.uniforms.uDay.value = day;
    this.bloom.strength = this.quality === 'high' ? 0.55 + 0.5 * (1 - day) : 0;
    this.bloom.enabled = this.quality === 'high';

    const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), v = new THREE.Vector3(), sc = new THREE.Vector3();
    for (let i = 0; i < (this.count || this.n); i++) {
      const I = this.islands[i];
      // 家の高さを、年の初めの 4 割で新しい値へ
      I.hBlend = Math.min((I.hBlend || 0) + dt / Math.max(0.4 * (this.yearDur || 1), 0.05), 1);
      const hh = this._curHeights(I);
      const settled = I.hBlend >= 1 && I.hSettled;      // 高さが新しい値に着いたら、家の行列は書き直さない(軽くする)
      I.hSettled = I.hBlend >= 1;
      for (let k = 0; k < (settled ? 0 : hh.length); k++) {
        const [x, z, a] = I.hpos[k];
        q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), -a);
        v.set(x, TOP_Y, z); sc.set(1, hh[k], 1);
        m4.compose(v, q, sc);
        I.houses.setMatrixAt(k, m4);
        v.set(x, TOP_Y + hh[k], z); sc.set(1, 1, 1);
        m4.compose(v, q, sc);
        I.roofs.setMatrixAt(k, m4);
      }
      if (!settled) {
        I.houses.instanceMatrix.needsUpdate = true; I.roofs.instanceMatrix.needsUpdate = true;
        I.houses.boundingSphere = null; I.roofs.boundingSphere = null;
      }
      if (I.H.r.some(x => x > 0.8) || (I.H.hit && I.storm > 0.01)) this._houseColors(I, time);
      if ((I.ruin || 0) > 0.003 && I.lastR) {
        I.ruin = Math.max(0, I.ruin - dt / (Math.max(this.yearDur || 1, 0.6) * 4));    // 四年ほどで建ち直る
        this._sectors(I, I.lastR);
      }
      I.dustAmt = Math.max(0, (I.dustAmt || 0) - dt / Math.max(this.yearDur || 1, 0.6));
      I.dust.material.opacity = 0.75 * I.dustAmt;
      if (I.dustAmt > 0.01) {
        const dp = I.dust.geometry.attributes.position, sd = I.dust.userData.seeds, life = 1 - I.dustAmt;
        for (let k = 0; k < dp.count; k++) { const a = sd[k] * 40, rr = 2.2 + 2.3 * ((sd[k] * 7.3) % 1) + life * 1.5; dp.setXYZ(k, rr * Math.cos(a), TOP_Y + 0.3 + life * (1.5 + 2.5 * ((sd[k] * 3.1) % 1)), rr * Math.sin(a)); }
        dp.needsUpdate = true;
      }
      // 堤、選挙の旗、プレイヤーの光の柱
      I.wall.scale.y += ((I.wallTarget || 0.001) * 2.2 - I.wall.scale.y) * Math.min(dt * 2, 1);
      I.wall.visible = I.wall.scale.y > 0.01;
      I.election = Math.max(0, I.election - dt * 0.6);
      I.flag.material.emissiveIntensity = 0.4 + 2.5 * I.election * (0.5 + 0.5 * Math.sin(time * 10));
      const fc = I.regime ? REGIME_COLOR : I.accent;
      I.flag.material.color.copy(fc); I.flag.material.emissive.copy(fc);
      I.revFlash = Math.max(0, I.revFlash - dt * 0.5);
      I.upFlash = Math.max(0, I.upFlash - dt * 0.4);
      I.domes.material.emissiveIntensity = 1.0 + 3.0 * I.upFlash * (0.5 + 0.5 * Math.sin(time * 8));
      if (I.revFlash > 0.01) { I.flash.color.set('#ff5a3c'); I.flash.intensity = Math.max(I.flash.intensity, 300 * I.revFlash * (Math.random() < 0.3 ? 1 : 0.3)); }
      else if (I.storm <= 0.01) I.flash.color.set('#cfe0ff');
      I.flag.scale.setScalar(1 + 0.8 * I.election);
      if (I.beacon.visible && I.beaconIdx !== null && I.hpos[I.beaconIdx]) {
        const [bx, bz] = I.hpos[I.beaconIdx];
        I.beacon.position.set(bx, TOP_Y + hh[I.beaconIdx], bz);
        I.beacon.material.opacity = 0.35 + 0.25 * Math.sin(time * 3);
      }
      // 選んだ国の輪
      I.ring.material.opacity = i === this.selected ? 0.55 + 0.35 * Math.sin(time * 2.4) : 0.0;
      // 地価の輝き
      I.glow.material.color.copy(I.glowCol);
      I.glow.material.opacity += (I.glowAmt * 0.55 - I.glow.material.opacity) * Math.min(dt * 2, 1);
      // 政府の塔のひび
      I.govMat.emissiveIntensity = I.crack > 0 ? I.crack * (0.5 + 0.5 * Math.sin(time * (6 + 6 * I.crack))) : 0;
      // 嵐
      I.storm = Math.max(0, I.storm - dt / Math.max(this.yearDur || 1, 0.6) * 0.8);
      I.cloudMat.opacity = Math.min(I.storm * 1.2, 0.85);
      I.rain.material.opacity = Math.min(I.storm * 1.4, 0.9);
      const rp = I.rain.geometry.attributes.position;
      if (I.storm > 0.01) {
        for (let k = 0; k < rp.count; k++) { let y = rp.getY(k) - dt * 22; if (y < 0) y += 14; rp.setY(k, y); }
        rp.needsUpdate = true;
        I.flash.intensity = Math.random() < 0.04 * I.storm ? 400 * I.storm : I.flash.intensity * 0.8;
      } else I.flash.intensity = 0;
      // 煙
      const spos = I.smoke.geometry.attributes.position, seeds = I.smoke.userData.seeds;
      const tops = I.chimneyTops;
      for (let k = 0; k < spos.count; k++) {
        if (!tops.length) { spos.setXYZ(k, 0, -50, 0); continue; }
        const life = (seeds[k] + time * 0.12 * (0.5 + (I.smokeRate || 0))) % 1;
        const tp = tops[k % tops.length];
        spos.setXYZ(k, tp[0] + life * 1.6 + Math.sin(k + time) * 0.15 * life, tp[1] + life * 4.5, tp[2] + life * 0.8);
      }
      spos.needsUpdate = true;
      I.smoke.material.opacity = 0.15 + 0.55 * Math.min(I.smokeRate || 0, 1);
    }
    // 流れの粒
    for (const A of this.arcs) {
      A.amt += (A.target - A.amt) * Math.min(dt * 1.5, 1);
      const nvis = Math.round(A.N * A.amt);
      A.pts.geometry.setDrawRange(0, nvis);
      A.line.material.opacity = 0.05 + 0.25 * A.amt;
      if (nvis === 0) { A.line.material.opacity = 0; continue; }
      const pos = A.pts.geometry.attributes.position;
      for (let k = 0; k < nvis; k++) {
        A.u[k] = (A.u[k] + dt * A.speed) % 1;
        const p = A.curve.getPoint(A.u[k]);
        pos.setXYZ(k, p.x, p.y, p.z);
      }
      pos.needsUpdate = true;
    }
    for (const B of this.ibArcs) {
      B.amt += (B.target - B.amt) * Math.min(dt * 1.5, 1);
      B.tube.material.opacity = B.amt * (0.6 + 0.4 * Math.sin(time * 3));
    }
    // 揺れ
    if (this.shake > 0.001) {
      this.shake *= Math.pow(0.04, dt);
      this.camera.position.x += (Math.random() - 0.5) * this.shake;
      this.camera.position.y += (Math.random() - 0.5) * this.shake;
    }
    this.controls.update();
    if (this.quality === 'high') this.composer.render();
    else this.renderer.render(this.scene, this.camera);
    this.labels.render(this.scene, this.camera);
  }
}
