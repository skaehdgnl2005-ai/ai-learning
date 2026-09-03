// app.js — UI 배선. 컨트롤 → Trainer → 시각화 3개.
(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);

  const el = {
    dataset: $('c-dataset'), n: $('c-n'), noise: $('c-noise'), seed: $('c-seed'),
    hidden: $('c-hidden'), act: $('c-act'), lr: $('c-lr'), speed: $('c-speed'),
    vN: $('v-n'), vNoise: $('v-noise'), vLr: $('v-lr'), vSpeed: $('v-speed'), vParams: $('v-params'),
    play: $('b-play'), step: $('b-step'), sub: $('b-sub'), reset: $('b-reset'),
    statStep: $('stat-step'), statLoss: $('stat-loss'), statAcc: $('stat-acc'),
    badge: $('phase-badge'), dataSub: $('data-sub'), netSub: $('net-sub'),
    tooltip: $('tooltip'),
    cvNet: $('cv-net'), cvData: $('cv-data'), cvChart: $('cv-chart'),
  };

  const netViz = new NetworkViz(el.cvNet);
  const dataViz = new DataViz(el.cvData);
  const chartViz = new ChartViz(el.cvChart);

  let trainer = null;
  let playing = false;
  let stepping = false;      // "1 스텝" 애니메이션 진행 중
  let lastPhase = null;      // 방금 끝낸 부분 스텝
  let acc = 0;               // 재생 루프의 시간 누적
  let lastT = 0;

  function readOpts() {
    const hidden = parseInt(el.hidden.value, 10);
    return {
      dataset: el.dataset.value,
      n: parseInt(el.n.value, 10),
      noise: parseFloat(el.noise.value),
      seed: parseInt(el.seed.value, 10) || 1,
      hidden: [hidden, hidden],
      activation: el.act.value,
      lr: parseFloat(el.lr.value),
    };
  }

  function rebuild() {
    stop();
    trainer = new Trainer(readOpts());
    lastPhase = null;
    netViz.layout(trainer.model, el.cvNet.clientWidth || 800, 360);
    syncLabels();
    render();
  }

  function syncLabels() {
    el.vN.textContent = el.n.value;
    el.vNoise.textContent = parseFloat(el.noise.value).toFixed(2);
    el.vLr.textContent = parseFloat(el.lr.value).toFixed(2);
    el.vSpeed.textContent = el.speed.value + ' 스텝/초';
    el.vParams.textContent = trainer ? trainer.model.parameters().length : '—';
  }

  // --- 렌더 -------------------------------------------------------------

  function render() {
    netViz.draw(trainer);

    // 은닉 노드에 마우스를 올리면 그 뉴런의 활성 히트맵을 보여준다.
    const hv = netViz.hover;
    if (hv && hv.type === 'node' && hv.node.col > 0 && hv.node.col < trainer.model.sizes.length - 1) {
      const field = trainer.activationGrid(hv.node.layerIndex, hv.node.neuronIndex, dataViz.res, dataViz.range);
      dataViz.draw(trainer, { field, heatmap: true });
      el.dataSub.textContent = `은닉 ${hv.node.col} · 뉴런 ${hv.node.idx + 1} 의 활성값`;
    } else {
      dataViz.draw(trainer);
      el.dataSub.textContent = '배경색 = 모델의 예측 부호';
    }

    chartViz.draw(trainer);

    el.statStep.textContent = trainer.step;
    el.statLoss.textContent = fmtLoss(trainer.lossValue, trainer.step);
    el.statLoss.classList.toggle('diverged', trainer.diverged || trainer.lossValue > 1e3);
    el.statAcc.textContent = trainer.step || trainer.phase !== 'forward'
      ? (trainer.acc * 100).toFixed(0) + '%' : '—';

    if (lastPhase) {
      el.badge.textContent = PHASE_LABELS[lastPhase] + ' 완료';
      el.badge.classList.add('active');
    } else {
      el.badge.textContent = trainer.step ? `스텝 ${trainer.step} 완료` : '준비됨';
      el.badge.classList.toggle('active', false);
    }

    el.netSub.textContent = trainer.phase === 'update'
      ? '노란색 = 역전파로 구한 grad (굵기 = |grad|)'
      : '굵기 = |w|, 색 = 부호 · 마우스를 올리면 값이 보입니다';
  }

  // --- 실행 -------------------------------------------------------------

  function substepOnce() {
    lastPhase = trainer.substep();
    render();
  }

  // 4단계를 순서대로 보여준다.
  function oneStepAnimated() {
    if (stepping || playing) return;
    stepping = true;
    setButtons();
    const tick = () => {
      substepOnce();
      if (trainer.phase === 'forward') {
        stepping = false;
        lastPhase = null;
        render();
        setButtons();
        return;
      }
      setTimeout(tick, 320);
    };
    tick();
  }

  function loop(t) {
    if (!playing) return;
    if (!lastT) lastT = t;
    const dt = (t - lastT) / 1000;
    lastT = t;
    acc += dt * parseFloat(el.speed.value);
    let budget = 0;
    const start = performance.now();
    while (acc >= 1 && budget < 8 && performance.now() - start < 30) {
      trainer.stepFull();
      acc -= 1;
      budget++;
    }
    if (acc > 4) acc = 4;   // 밀린 스텝이 무한히 쌓이지 않게
    if (budget > 0) {       // 바뀐 게 있을 때만 다시 그린다
      lastPhase = null;
      render();
    }
    requestAnimationFrame(loop);
  }

  function play() {
    if (playing) return;
    // 부분 스텝 중이었다면 스텝 경계까지 맞춰 놓는다.
    while (trainer.phase !== 'forward') trainer.substep();
    playing = true;
    lastT = 0;
    acc = 0;
    setButtons();
    requestAnimationFrame(loop);
  }

  function stop() {
    playing = false;
    stepping = false;
    setButtons();
  }

  function setButtons() {
    if (!el.play) return;
    el.play.textContent = playing ? '⏸ 일시정지' : '▶ 재생';
    el.step.disabled = playing || stepping;
    el.sub.disabled = playing || stepping;
  }

  // --- 이벤트 -----------------------------------------------------------

  ['dataset', 'n', 'noise', 'seed', 'hidden', 'act'].forEach((k) => {
    el[k].addEventListener('input', rebuild);
  });
  el.lr.addEventListener('input', () => {
    trainer.opts.lr = parseFloat(el.lr.value);
    syncLabels();
  });
  el.speed.addEventListener('input', syncLabels);

  el.play.addEventListener('click', () => (playing ? stop() : play()));
  el.step.addEventListener('click', oneStepAnimated);
  el.sub.addEventListener('click', () => {
    if (playing || stepping) return;
    substepOnce();
  });
  el.reset.addEventListener('click', rebuild);

  // 네트워크 hover → 툴팁 + 활성 히트맵
  el.cvNet.addEventListener('mousemove', (e) => {
    const r = el.cvNet.getBoundingClientRect();
    const hit = netViz.hitTest(e.clientX - r.left, e.clientY - r.top);
    const changed = describeHover(hit) !== describeHover(netViz.hover);
    netViz.setHover(hit);
    if (hit) {
      el.tooltip.style.display = 'block';
      el.tooltip.style.left = e.clientX + 14 + 'px';
      el.tooltip.style.top = e.clientY + 14 + 'px';
      el.tooltip.textContent = tooltipText(hit);
    } else {
      el.tooltip.style.display = 'none';
    }
    if (!playing && changed) render();
  });
  el.cvNet.addEventListener('mouseleave', () => {
    netViz.setHover(null);
    el.tooltip.style.display = 'none';
    if (!playing) render();
  });

  function describeHover(hit) {
    if (!hit) return '';
    return hit.type === 'node'
      ? `n${hit.node.col}-${hit.node.idx}`
      : `e${hit.edge.layerIndex}-${hit.edge.neuronIndex}-${hit.edge.inputIndex}`;
  }

  function tooltipText(hit) {
    const model = trainer.model;
    if (hit.type === 'edge') {
      const e = hit.edge;
      const p = model.layers[e.layerIndex].neurons[e.neuronIndex].w[e.inputIndex];
      const d = trainer.deltas ? trainer.deltas[e.paramIndex] : null;
      const src = e.layerIndex === 0 ? `입력 ${e.inputIndex === 0 ? 'x' : 'y'}` : `은닉${e.layerIndex}·${e.inputIndex + 1}`;
      const dst = e.layerIndex === model.layers.length - 1 ? '출력' : `은닉${e.layerIndex + 1}·${e.neuronIndex + 1}`;
      return `${src} → ${dst}\nw    = ${fmt(p.data)}\ngrad = ${fmt(p.grad)}` +
        (d != null ? `\nΔw   = ${fmt(d)}` : '');
    }
    const nd = hit.node;
    if (nd.col === 0) return `입력 ${nd.idx === 0 ? 'x' : 'y'}`;
    const neuron = model.layers[nd.layerIndex].neurons[nd.neuronIndex];
    const d = trainer.deltas && nd.paramIndex != null ? trainer.deltas[nd.paramIndex] : null;
    const name = nd.col === model.sizes.length - 1 ? '출력 뉴런' : `은닉${nd.col} · 뉴런 ${nd.idx + 1}`;
    return `${name}\nb    = ${fmt(neuron.b.data)}\ngrad = ${fmt(neuron.b.grad)}` +
      (d != null ? `\nΔb   = ${fmt(d)}` : '');
  }

  // 헤더의 큰 손실 숫자. 발산하면 지수 표기로 짧게 보여준다.
  function fmtLoss(v, step) {
    if (!Number.isFinite(v)) return step ? '발산' : '—';
    if (v >= 1e4) return v.toExponential(1);
    if (v >= 100) return v.toFixed(1);
    return v.toFixed(4);
  }

  function fmt(v) {
    if (!Number.isFinite(v)) return String(v);
    if (v !== 0 && Math.abs(v) < 1e-3) return v.toExponential(2);
    return v.toFixed(4);
  }

  let resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (!playing) render(); }, 120);
  });

  // 키보드: 스페이스 = 재생/정지, S = 1스텝, D = 부분 스텝
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.code === 'Space') { e.preventDefault(); playing ? stop() : play(); }
    else if (e.key === 's' || e.key === 'S') oneStepAnimated();
    else if (e.key === 'd' || e.key === 'D') { if (!playing && !stepping) substepOnce(); }
  });

  rebuild();
})();
