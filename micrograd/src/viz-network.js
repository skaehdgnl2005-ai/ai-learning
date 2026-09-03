// viz-network.js — 네트워크 다이어그램.
// 엣지 굵기 = |w|, 색 = 부호(파랑 +, 빨강 −). 역전파/갱신 단계에서는 grad 오버레이.
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

  class NetworkViz {
    constructor(canvas) {
      this.canvas = canvas;
      this.nodes = [];      // {col, idx, x, y, r, layerIndex, neuronIndex}
      this.edges = [];      // {from, to, layerIndex, neuronIndex, inputIndex, paramIndex}
      this.hover = null;
      this.model = null;
    }

    // 모델 구조가 바뀔 때만 다시 계산한다.
    layout(model, w, h) {
      const sizes = model.sizes;                 // [2, 16, 16, 1]
      const padX = 46, padY = 22;
      const cols = sizes.length;
      const nodes = [];
      const colX = (c) => padX + ((w - 2 * padX) * c) / (cols - 1);

      for (let c = 0; c < cols; c++) {
        const n = sizes[c];
        const usable = h - 2 * padY;
        const gap = n > 1 ? usable / (n - 1) : 0;
        const r = Math.max(3, Math.min(9, usable / (n * 2.6)));
        for (let i = 0; i < n; i++) {
          const y = n > 1 ? padY + gap * i : h / 2;
          nodes.push({ col: c, idx: i, x: colX(c), y, r, layerIndex: c - 1, neuronIndex: i });
        }
      }

      // 파라미터 배열에서의 위치 (Δw / grad 조회용)
      const edges = [];
      let base = 0;
      for (let li = 0; li < model.layers.length; li++) {
        const layer = model.layers[li];
        for (let ni = 0; ni < layer.neurons.length; ni++) {
          const neuron = layer.neurons[ni];
          const target = nodes.find((nd) => nd.col === li + 1 && nd.idx === ni);
          for (let i = 0; i < neuron.w.length; i++) {
            edges.push({
              from: nodes.find((nd) => nd.col === li && nd.idx === i),
              to: target,
              layerIndex: li, neuronIndex: ni, inputIndex: i,
              paramIndex: base + i,
            });
          }
          target.paramIndex = base + neuron.w.length;   // bias 의 위치
          base += neuron.w.length + 1;
        }
      }

      this.nodes = nodes;
      this.edges = edges;
      this.model = model;
      this._w = w;
      this._h = h;
      this._sizeKey = sizes.join('-');
    }

    draw(trainer) {
      const { ctx, w, h } = fitCanvas(this.canvas);
      const model = trainer.model;
      if (this._sizeKey !== model.sizes.join('-') || this._w !== w || this._h !== h || this.model !== model) {
        this.layout(model, w, h);
      }

      ctx.clearRect(0, 0, w, h);

      // 정규화 기준: 이번 프레임의 최대 |w|, 최대 |grad|
      let maxW = 1e-9, maxG = 1e-9;
      for (const e of this.edges) {
        const p = model.layers[e.layerIndex].neurons[e.neuronIndex].w[e.inputIndex];
        if (Math.abs(p.data) > maxW) maxW = Math.abs(p.data);
        if (Math.abs(p.grad) > maxG) maxG = Math.abs(p.grad);
      }
      // backward 를 막 끝낸 상태(다음 단계가 update)에서 grad 를 겹쳐 보여준다.
      const gradPhase = trainer.phase === 'update';

      // 엣지
      for (const e of this.edges) {
        const p = model.layers[e.layerIndex].neurons[e.neuronIndex].w[e.inputIndex];
        const mag = Math.abs(p.data) / maxW;
        const isHover = this.hover && this.hover.type === 'edge' && this.hover.edge === e;
        ctx.beginPath();
        ctx.moveTo(e.from.x, e.from.y);
        ctx.lineTo(e.to.x, e.to.y);
        ctx.strokeStyle = p.data >= 0 ? POS : NEG;
        ctx.globalAlpha = isHover ? 1 : 0.14 + 0.6 * mag;
        ctx.lineWidth = isHover ? 3.5 : 0.35 + 3.4 * mag * mag;
        ctx.stroke();

        if (gradPhase && maxG > 1e-12) {
          const gm = Math.abs(p.grad) / maxG;
          if (gm > 0.03) {
            ctx.beginPath();
            ctx.moveTo(e.from.x, e.from.y);
            ctx.lineTo(e.to.x, e.to.y);
            ctx.strokeStyle = ACCENT;
            ctx.globalAlpha = 0.10 + 0.75 * gm;
            ctx.lineWidth = 0.3 + 2.6 * gm;
            ctx.stroke();
          }
        }
      }
      ctx.globalAlpha = 1;

      // 노드
      for (const nd of this.nodes) {
        const isHover = this.hover && this.hover.type === 'node' && this.hover.node === nd;
        ctx.beginPath();
        ctx.arc(nd.x, nd.y, nd.r, 0, Math.PI * 2);
        ctx.fillStyle = nd.col === 0 ? '#243049' : '#1e2637';
        ctx.fill();
        ctx.lineWidth = isHover ? 2.5 : 1.2;
        ctx.strokeStyle = isHover ? ACCENT : '#3c4761';
        ctx.stroke();
      }

      // 열 라벨
      ctx.fillStyle = '#8f9bb3';
      ctx.font = '11px system-ui, sans-serif';
      ctx.textAlign = 'center';
      const cols = model.sizes.length;
      for (let c = 0; c < cols; c++) {
        const x = this.nodes.find((n) => n.col === c).x;
        const label = c === 0 ? '입력 (x, y)' : c === cols - 1 ? '출력' : `은닉 ${c} (${model.sizes[c]})`;
        ctx.fillText(label, x, h - 4);
      }
    }

    // 마우스 좌표(캔버스 기준)에서 가장 가까운 노드/엣지를 찾는다.
    hitTest(mx, my) {
      for (const nd of this.nodes) {
        const d = Math.hypot(mx - nd.x, my - nd.y);
        if (d <= nd.r + 4) return { type: 'node', node: nd };
      }
      let best = null, bestD = 4.5;
      for (const e of this.edges) {
        const d = distToSegment(mx, my, e.from.x, e.from.y, e.to.x, e.to.y);
        if (d < bestD) { bestD = d; best = e; }
      }
      return best ? { type: 'edge', edge: best } : null;
    }

    setHover(hit) {
      this.hover = hit;
    }
  }

  function distToSegment(px, py, x1, y1, x2, y2) {
    const dx = x2 - x1, dy = y2 - y1;
    const len2 = dx * dx + dy * dy;
    let t = len2 === 0 ? 0 : ((px - x1) * dx + (py - y1) * dy) / len2;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
  }

  global.NetworkViz = NetworkViz;
})(typeof globalThis !== 'undefined' ? globalThis : this);
