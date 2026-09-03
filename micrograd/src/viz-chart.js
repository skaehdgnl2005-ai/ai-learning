// viz-chart.js — 손실(로그 스케일)과 정답률 두 줄 그래프.
(function (global) {
  'use strict';

  const LOSS = '#ffc857', ACC = '#4ade80', GRID = 'rgba(255,255,255,0.07)';

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

  class ChartViz {
    constructor(canvas) {
      this.canvas = canvas;
    }

    draw(trainer) {
      const { ctx, w, h } = fitCanvas(this.canvas);
      ctx.clearRect(0, 0, w, h);
      const padL = 46, padR = 46, padT = 14, padB = 26;
      const iw = w - padL - padR, ih = h - padT - padB;

      const hist = trainer.history;
      const n = Math.max(hist.length, 2);

      // 손실은 로그 스케일. 발산해도 보이도록 위쪽을 넉넉히 잡는다.
      const finite = hist.map((d) => d.loss).filter((v) => Number.isFinite(v) && v > 0);
      const lo = finite.length ? Math.max(1e-4, Math.min.apply(null, finite) * 0.7) : 1e-2;
      const hi = finite.length ? Math.max.apply(null, finite) * 1.4 : 10;
      const logLo = Math.log10(lo), logHi = Math.log10(Math.max(hi, lo * 10));

      const X = (i) => padL + (iw * i) / (n - 1);
      const Yloss = (v) => {
        if (!Number.isFinite(v) || v <= 0) return padT;   // 발산 = 천장에 붙는다
        const t = (Math.log10(v) - logLo) / (logHi - logLo);
        return padT + ih * (1 - Math.max(0, Math.min(1, t)));
      };
      const Yacc = (v) => padT + ih * (1 - v);

      ctx.font = '10px system-ui, sans-serif';
      ctx.lineWidth = 1;
      for (let k = 0; k <= 4; k++) {
        const y = padT + (ih * k) / 4;
        ctx.strokeStyle = GRID;
        ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + iw, y); ctx.stroke();
        const lossVal = Math.pow(10, logHi - ((logHi - logLo) * k) / 4);
        ctx.fillStyle = LOSS; ctx.textAlign = 'right';
        ctx.fillText(fmt(lossVal), padL - 6, y + 3);
        ctx.fillStyle = ACC; ctx.textAlign = 'left';
        ctx.fillText(Math.round((1 - k / 4) * 100) + '%', padL + iw + 6, y + 3);
      }

      if (hist.length === 0) {
        ctx.fillStyle = '#8f9bb3';
        ctx.textAlign = 'center';
        ctx.fillText('학습을 시작하면 곡선이 그려집니다', w / 2, h / 2);
        return;
      }

      ctx.beginPath();
      hist.forEach((d, i) => (i ? ctx.lineTo(X(i), Yacc(d.acc)) : ctx.moveTo(X(i), Yacc(d.acc))));
      ctx.strokeStyle = ACC; ctx.lineWidth = 1.8; ctx.stroke();

      ctx.beginPath();
      hist.forEach((d, i) => (i ? ctx.lineTo(X(i), Yloss(d.loss)) : ctx.moveTo(X(i), Yloss(d.loss))));
      ctx.strokeStyle = LOSS; ctx.lineWidth = 1.8; ctx.stroke();

      const last = hist.length - 1;
      [[Yloss(hist[last].loss), LOSS], [Yacc(hist[last].acc), ACC]].forEach(([y, c]) => {
        ctx.beginPath(); ctx.arc(X(last), y, 3, 0, Math.PI * 2);
        ctx.fillStyle = c; ctx.fill();
      });

      ctx.fillStyle = '#8f9bb3'; ctx.textAlign = 'center';
      ctx.fillText('스텝 ' + hist[last].step, padL + iw / 2, h - 7);
    }
  }

  function fmt(v) {
    if (v >= 1e4) return v.toExponential(0);
    if (v >= 100) return v.toFixed(0);
    if (v >= 1) return v.toFixed(1);
    if (v >= 0.01) return v.toFixed(2);
    return v.toExponential(0);
  }

  global.ChartViz = ChartViz;
})(typeof globalThis !== 'undefined' ? globalThis : this);
