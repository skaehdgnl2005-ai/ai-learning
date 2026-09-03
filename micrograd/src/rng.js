// rng.js — 시드를 넣으면 항상 같은 수열이 나오는 난수기.
// 같은 시드로 같은 데이터셋·같은 초기 가중치를 재현하기 위해 쓴다.
(function (global) {
  'use strict';

  // mulberry32: 32비트 시드 하나로 도는 작고 품질 좋은 PRNG.
  function makeRng(seed) {
    let a = (seed >>> 0) || 1;
    const rng = function () {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    // [-1, 1) 균등분포
    rng.uniform = () => rng() * 2 - 1;
    // 표준정규분포 (Box-Muller)
    rng.gauss = () => {
      let u = 0;
      while (u === 0) u = rng();
      const v = rng();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
    return rng;
  }

  global.makeRng = makeRng;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { makeRng };
  }
})(typeof globalThis !== 'undefined' ? globalThis : this);
