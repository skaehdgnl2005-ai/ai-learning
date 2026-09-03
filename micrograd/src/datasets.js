// datasets.js — 2D 분류용 장난감 데이터셋 4종.
// 모든 좌표는 대략 [-2, 2] 안에 들어오도록 맞춰 두었다 (시각화 범위와 동일).
(function (global) {
  'use strict';

  const makeRng = global.makeRng || require('./rng.js').makeRng;

  const KINDS = [
    { id: 'moons', name: '초승달 (moons)' },
    { id: 'circles', name: '동심원 (circles)' },
    { id: 'xor', name: 'XOR' },
    { id: 'spiral', name: '나선 (spiral)' },
  ];

  function moons(n, noise, rng) {
    const pts = [];
    for (let i = 0; i < n; i++) {
      const inner = i % 2 === 1;
      const t = Math.PI * rng();
      let x, y;
      if (inner) {
        x = 1 - Math.cos(t);
        y = 0.5 - Math.sin(t);
      } else {
        x = Math.cos(t);
        y = Math.sin(t);
      }
      // 중심 이동 후 확대
      x = (x - 0.5) * 1.2;
      y = (y - 0.25) * 1.2;
      pts.push({
        x: x + rng.gauss() * noise,
        y: y + rng.gauss() * noise,
        label: inner ? -1 : 1,
      });
    }
    return pts;
  }

  function circles(n, noise, rng) {
    const pts = [];
    for (let i = 0; i < n; i++) {
      const inner = i % 2 === 1;
      const t = 2 * Math.PI * rng();
      const r = inner ? 0.55 : 1.45;
      pts.push({
        x: r * Math.cos(t) + rng.gauss() * noise,
        y: r * Math.sin(t) + rng.gauss() * noise,
        label: inner ? -1 : 1,
      });
    }
    return pts;
  }

  function xor(n, noise, rng) {
    const pts = [];
    for (let i = 0; i < n; i++) {
      const cx = i % 2 === 0 ? 1 : -1;
      const cy = i % 4 < 2 ? 1 : -1;
      const x = cx + rng.gauss() * (0.28 + noise);
      const y = cy + rng.gauss() * (0.28 + noise);
      pts.push({ x, y, label: cx * cy > 0 ? 1 : -1 });
    }
    return pts;
  }

  function spiral(n, noise, rng) {
    const pts = [];
    for (let i = 0; i < n; i++) {
      const arm = i % 2 === 0 ? 1 : -1;
      const t = (i / n) * 3.2 + 0.25;       // 반지름
      const theta = t * 2.2 + (arm === 1 ? 0 : Math.PI);
      const r = t * 0.55;
      pts.push({
        x: r * Math.cos(theta) + rng.gauss() * noise,
        y: r * Math.sin(theta) + rng.gauss() * noise,
        label: arm,
      });
    }
    return pts;
  }

  const GENERATORS = { moons, circles, xor, spiral };

  // makeDataset('moons', { n: 100, noise: 0.1, seed: 1 })
  function makeDataset(kind, { n = 100, noise = 0.1, seed = 1 } = {}) {
    const gen = GENERATORS[kind];
    if (!gen) throw new Error(`알 수 없는 데이터셋: ${kind}`);
    const rng = makeRng(seed);
    return { kind, points: gen(n, noise, rng) };
  }

  Object.assign(global, { makeDataset, DATASET_KINDS: KINDS });
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { makeDataset, DATASET_KINDS: KINDS };
  }
})(typeof globalThis !== 'undefined' ? globalThis : this);
