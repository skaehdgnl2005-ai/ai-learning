// nn.js — micrograd 의 nn.py 를 자바스크립트로 1:1 재구현.
// engine.js (Value) 와 rng.js (makeRng) 가 먼저 로드되어 있어야 한다.
(function (global) {
  'use strict';

  const Value = global.Value || require('./engine.js').Value;
  const makeRng = global.makeRng || require('./rng.js').makeRng;

  class Module {
    zeroGrad() {
      for (const p of this.parameters()) p.grad = 0;
    }
    parameters() {
      return [];
    }
  }

  class Neuron extends Module {
    // nonlin=false 면 활성화 없이 선형 출력 (강의의 마지막 레이어).
    constructor(nin, { rng, seed = 1337, nonlin = true, activation = 'relu' } = {}) {
      super();
      const r = rng || makeRng(seed);
      this.w = Array.from({ length: nin }, () => new Value(r.uniform()));
      this.b = new Value(0);
      this.nonlin = nonlin;
      this.activation = activation;
    }

    // x: Value 배열 -> Value
    call(x) {
      let act = this.b;
      for (let i = 0; i < this.w.length; i++) {
        act = act.add(this.w[i].mul(x[i]));
      }
      if (!this.nonlin) return act;
      return this.activation === 'tanh' ? act.tanh() : act.relu();
    }

    parameters() {
      return [...this.w, this.b];
    }
  }

  class Layer extends Module {
    constructor(nin, nout, opts = {}) {
      super();
      const rng = opts.rng || makeRng(opts.seed != null ? opts.seed : 1337);
      this.neurons = Array.from(
        { length: nout },
        () => new Neuron(nin, { ...opts, rng })
      );
    }

    call(x) {
      return this.neurons.map((n) => n.call(x));
    }

    parameters() {
      return this.neurons.flatMap((n) => n.parameters());
    }
  }

  class MLP extends Module {
    // new MLP(2, [16, 16, 1]) => 2-16-16-1, 마지막 레이어만 linear
    constructor(nin, nouts, opts = {}) {
      super();
      const rng = opts.rng || makeRng(opts.seed != null ? opts.seed : 1337);
      this.activation = opts.activation || 'relu';
      this.sizes = [nin, ...nouts];
      this.layers = nouts.map((nout, i) =>
        new Layer(this.sizes[i], nout, {
          rng,
          nonlin: i !== nouts.length - 1,
          activation: this.activation,
        })
      );
    }

    // Value 그래프를 만드는 정식 forward. 출력이 1개면 Value 하나를 돌려준다.
    call(x) {
      let out = x;
      for (const layer of this.layers) out = layer.call(out);
      return out.length === 1 ? out[0] : out;
    }

    // 그래프 없이 숫자만으로 계산하는 빠른 forward.
    // 결정 경계용 40x40 격자를 매 스텝 그려야 하는데, Value 로 하면
    // 노드가 100만 개 생겨 느리다. 수학은 call() 과 완전히 동일하다.
    predictFast(...inputs) {
      let act = inputs.length === 1 && Array.isArray(inputs[0]) ? inputs[0].slice() : inputs;
      for (let li = 0; li < this.layers.length; li++) {
        const layer = this.layers[li];
        const next = new Array(layer.neurons.length);
        for (let ni = 0; ni < layer.neurons.length; ni++) {
          const n = layer.neurons[ni];
          let s = n.b.data;
          for (let i = 0; i < n.w.length; i++) s += n.w[i].data * act[i];
          if (n.nonlin) s = this.activation === 'tanh' ? Math.tanh(s) : (s < 0 ? 0 : s);
          next[ni] = s;
        }
        act = next;
      }
      return act.length === 1 ? act[0] : act;
    }

    // 은닉 뉴런 하나의 활성값 (노드 hover 시 히트맵용).
    activationAt(layerIndex, neuronIndex, inputs) {
      let act = inputs.slice();
      for (let li = 0; li <= layerIndex; li++) {
        const layer = this.layers[li];
        const next = new Array(layer.neurons.length);
        for (let ni = 0; ni < layer.neurons.length; ni++) {
          const n = layer.neurons[ni];
          let s = n.b.data;
          for (let i = 0; i < n.w.length; i++) s += n.w[i].data * act[i];
          if (n.nonlin) s = this.activation === 'tanh' ? Math.tanh(s) : (s < 0 ? 0 : s);
          next[ni] = s;
        }
        act = next;
      }
      return act[neuronIndex];
    }

    parameters() {
      return this.layers.flatMap((l) => l.parameters());
    }
  }

  Object.assign(global, { Module, Neuron, Layer, MLP });
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Module, Neuron, Layer, MLP };
  }
})(typeof globalThis !== 'undefined' ? globalThis : this);
