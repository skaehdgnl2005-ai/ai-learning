// trainer.js — 학습 루프를 4개의 부분 스텝으로 쪼갠 상태기계.
//   forward -> loss -> backward -> update
// 각 부분 스텝이 끝난 상태를 시각화가 그대로 그린다.
(function (global) {
  'use strict';

  const Value = global.Value || require('./engine.js').Value;
  const MLP = global.MLP || require('./nn.js').MLP;
  const makeDataset = global.makeDataset || require('./datasets.js').makeDataset;

  const PHASES = ['forward', 'loss', 'backward', 'update'];
  const PHASE_LABELS = {
    forward: '1. 순전파 (forward)',
    loss: '2. 손실 (loss)',
    backward: '3. 역전파 (backward)',
    update: '4. 가중치 갱신 (update)',
  };

  class Trainer {
    constructor(opts = {}) {
      this.opts = {
        dataset: 'moons',
        n: 100,
        noise: 0.1,
        seed: 1,
        hidden: [16, 16],
        activation: 'relu',
        lr: 0.1,
        alpha: 1e-4,          // L2 정규화 세기 (강의 데모와 동일)
        ...opts,
      };
      this.reset();
    }

    reset(patch = {}) {
      Object.assign(this.opts, patch);
      const o = this.opts;
      this.data = makeDataset(o.dataset, { n: o.n, noise: o.noise, seed: o.seed });
      this.model = new MLP(2, [...o.hidden, 1], {
        seed: o.seed + 1000,
        activation: o.activation,
      });
      this.step = 0;
      this.phase = 'forward';
      this.history = [];
      this.scores = [];          // 이번 스텝의 예측값 (숫자)
      this.misclassified = [];   // 오분류된 점의 인덱스
      this.acc = 0;
      this.loss = null;          // Value
      this.lossValue = NaN;
      this.dataLoss = NaN;
      this.regLoss = NaN;
      this.deltas = null;        // 직전 update 의 Δw (파라미터 순서)
      this.diverged = false;
      return this;
    }

    get phaseLabel() {
      return PHASE_LABELS[this.phase];
    }

    // --- 부분 스텝 -------------------------------------------------------

    _forward() {
      const pts = this.data.points;
      this._inputs = pts.map((p) => [new Value(p.x), new Value(p.y)]);
      this._scoreValues = this._inputs.map((xi) => this.model.call(xi));
      this.scores = this._scoreValues.map((s) => s.data);
      this.misclassified = [];
      let correct = 0;
      for (let i = 0; i < pts.length; i++) {
        const ok = (this.scores[i] > 0 ? 1 : -1) === pts[i].label;
        if (ok) correct++;
        else this.misclassified.push(i);
      }
      this.acc = correct / pts.length;
    }

    _loss() {
      const pts = this.data.points;
      // SVM max-margin hinge loss: mean(relu(1 - y * score))
      let total = new Value(0);
      for (let i = 0; i < pts.length; i++) {
        total = total.add(this._scoreValues[i].mul(-pts[i].label).add(1).relu());
      }
      const dataLoss = total.mul(1 / pts.length);

      // L2 정규화
      let sq = new Value(0);
      for (const p of this.model.parameters()) sq = sq.add(p.mul(p));
      const regLoss = sq.mul(this.opts.alpha);

      this.loss = dataLoss.add(regLoss);
      this.lossValue = this.loss.data;
      this.dataLoss = dataLoss.data;
      this.regLoss = regLoss.data;
      if (!Number.isFinite(this.lossValue)) this.diverged = true;
    }

    _backward() {
      this.model.zeroGrad();
      this.loss.backward();
    }

    _update() {
      const lr = this.opts.lr;
      const params = this.model.parameters();
      this.deltas = new Float64Array(params.length);
      for (let i = 0; i < params.length; i++) {
        const d = -lr * params[i].grad;
        this.deltas[i] = d;
        params[i].data += d;
        if (!Number.isFinite(params[i].data)) this.diverged = true;
      }
      this.step++;
      this.history.push({ step: this.step, loss: this.lossValue, acc: this.acc });
    }

    // 다음 부분 스텝 하나를 실행하고, 방금 끝낸 단계 이름을 돌려준다.
    substep() {
      const done = this.phase;
      switch (this.phase) {
        case 'forward': this._forward(); break;
        case 'loss': this._loss(); break;
        case 'backward': this._backward(); break;
        case 'update': this._update(); break;
      }
      this.phase = PHASES[(PHASES.indexOf(this.phase) + 1) % PHASES.length];
      return done;
    }

    // 4단계를 한 번에 실행하고 히스토리의 마지막 항목을 돌려준다.
    stepFull() {
      do {
        this.substep();
      } while (this.phase !== 'forward');
      return this.history[this.history.length - 1];
    }

    // --- 시각화 보조 -----------------------------------------------------

    // 결정 경계용 격자. predictFast 로 그래프 없이 계산한다.
    decisionGrid(res = 40, range = 2.4) {
      const grid = new Float64Array(res * res);
      for (let j = 0; j < res; j++) {
        const y = range - (2 * range * j) / (res - 1);
        for (let i = 0; i < res; i++) {
          const x = -range + (2 * range * i) / (res - 1);
          grid[j * res + i] = this.model.predictFast(x, y);
        }
      }
      return { res, range, grid };
    }

    // 은닉 뉴런 하나의 활성값 격자 (노드 hover 히트맵용).
    activationGrid(layerIndex, neuronIndex, res = 40, range = 2.4) {
      const grid = new Float64Array(res * res);
      for (let j = 0; j < res; j++) {
        const y = range - (2 * range * j) / (res - 1);
        for (let i = 0; i < res; i++) {
          const x = -range + (2 * range * i) / (res - 1);
          grid[j * res + i] = this.model.activationAt(layerIndex, neuronIndex, [x, y]);
        }
      }
      return { res, range, grid };
    }
  }

  Object.assign(global, { Trainer, PHASES, PHASE_LABELS });
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Trainer, PHASES, PHASE_LABELS };
  }
})(typeof globalThis !== 'undefined' ? globalThis : this);
