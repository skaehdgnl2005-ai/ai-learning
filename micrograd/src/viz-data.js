// viz-data.js — 데이터 점 + 결정 경계 캔버스.
(function (global) {
  'use strict';

  const POS = '#4b9fff', NEG = '#ff6b6b', ACCENT = '#ffc857';

  function fitCanvas(canvas) {
    const dpr = global.devicePixelRatio || 1;
    const w = canvas.clientWidth || canvas.width;
    // 논리 높이는 처음 한 번만 읽어 둔다.
    // canvas.height 에 값을 넣으면 height 속성 자체가 바뀌므로,
    // 속성을 매번 다시 읽으면 dpr 이 1이 아닌 화면에서 h 가 dpr 배씩 커진다.
    if (!canvas.dataset.logicalHeight) {
      canvas.dataset.logicalHeight =
        String(parseInt(canvas.getAttribute('height'), 10) || canvas.clientHeight || 360);
    }
    const h = parseInt(canvas.dataset.logicalHeight, 10);
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      canvas.style.height = h + 'px';
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  class DataViz {
    constructor(canvas) {
      this.canvas = canvas;
      this.range = 2.4;
      this.res = 44;
      // 격자를 작은 캔버스에 그린 뒤 확대해서 부드러운 배경을 만든다.
      this.off = document.createElement('canvas');
      this.off.width = this.res;
      this.off.height = this.res;
    }

    // opts.field: 미리 계산한 격자 (없으면 결정 경계를 새로 계산)
    // opts.heatmap: true 면 활성 히트맵 색으로 칠한다
    draw(trainer, opts = {}) {
      const { ctx, w, h } = fitCanvas(this.canvas);
      const size = Math.min(w, h);
      const ox = (w - size) / 2, oy = (h - size) / 2;
      const range = this.range;
      const sx = (x) => ox + ((x + range) / (2 * range)) * size;
      const sy = (y) => oy + ((range - y) / (2 * range)) * size;

      ctx.clearRect(0, 0, w, h);

      const field = opts.field || trainer.decisionGrid(this.res, range);
      this._paintField(field, opts.heatmap === true);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(this.off, ox, oy, size, size);

      // 축
      ctx.strokeStyle = 'rgba(255,255,255,0.10)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(sx(-range), sy(0)); ctx.lineTo(sx(range), sy(0));
      ctx.moveTo(sx(0), sy(-range)); ctx.lineTo(sx(0), sy(range));
      ctx.stroke();

      // 점
      const pts = trainer.data.points;
      const wrong = new Set(trainer.misclassified);
      const showWrong = trainer.step > 0 || trainer.phase !== 'forward';
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];
        const x = sx(p.x), y = sy(p.y);
        ctx.beginPath();
        ctx.arc(x, y, 4.2, 0, Math.PI * 2);
        ctx.fillStyle = p.label === 1 ? POS : NEG;
        ctx.fill();
        if (showWrong && wrong.has(i)) {
          ctx.strokeStyle = ACCENT;
          ctx.lineWidth = 2.2;
          ctx.stroke();
          ctx.beginPath();
          ctx.arc(x, y, 7.5, 0, Math.PI * 2);
          ctx.strokeStyle = 'rgba(255,200,87,0.55)';
          ctx.lineWidth = 1;
          ctx.stroke();
        } else {
          ctx.strokeStyle = 'rgba(10,14,22,0.85)';
          ctx.lineWidth = 1.4;
          ctx.stroke();
        }
      }
    }

    _paintField(field, heatmap) {
      const { res, grid } = field;
      if (this.off.width !== res) { this.off.width = res; this.off.height = res; }
      const octx = this.off.getContext('2d');
      const img = octx.createImageData(res, res);
      let max = 1e-9;
      for (let i = 0; i < grid.length; i++) max = Math.max(max, Math.abs(grid[i]));

      for (let i = 0; i < grid.length; i++) {
        const v = grid[i] / max;               // -1 .. 1
        const m = Math.min(1, Math.abs(v));
        let r, g, b;
        if (heatmap) {
          // 활성 히트맵: 0 = 어두움, 큰 값 = 노랑
          r = 26 + 229 * m; g = 32 + 168 * m; b = 44 - 9 * m;
        } else if (v >= 0) {
          r = 20 + 40 * m; g = 34 + 60 * m; b = 52 + 110 * m;   // 파랑 계열 = +1 예측
        } else {
          r = 52 + 110 * m; g = 26 + 24 * m; b = 34 + 26 * m;   // 빨강 계열 = -1 예측
        }
        const o = i * 4;
        img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
      }
      octx.putImageData(img, 0, 0);
    }
  }

  global.DataViz = DataViz;
})(typeof globalThis !== 'undefined' ? globalThis : this);
