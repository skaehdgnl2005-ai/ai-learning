'use strict';
const test = require('node:test');
const assert = require('node:assert');

require('../src/engine.js');
require('../src/nn.js');
require('../src/datasets.js');
require('../src/trainer.js');
const { Value, MLP, Neuron, Layer, makeDataset, Trainer } = globalThis;

test('MLP 구조와 파라미터 개수 (2, [16,16,1]) = 337', () => {
  const mlp = new MLP(2, [16, 16, 1], { seed: 1 });
  // (2*16+16) + (16*16+16) + (16*1+1) = 48 + 272 + 17 = 337
  assert.strictEqual(mlp.parameters().length, 337);
  assert.strictEqual(mlp.layers.length, 3);
  assert.strictEqual(mlp.layers[0].neurons.length, 16);
  // 마지막 레이어만 linear
  assert.strictEqual(mlp.layers[0].neurons[0].nonlin, true);
  assert.strictEqual(mlp.layers[2].neurons[0].nonlin, false);
});

test('predictFast 는 Value forward 와 같은 값을 낸다 (relu / tanh 모두)', () => {
  for (const activation of ['relu', 'tanh']) {
    const mlp = new MLP(2, [8, 8, 1], { seed: 7, activation });
    for (const [x, y] of [[0.3, -1.2], [-2.0, 0.5], [0, 0], [1.5, 1.5]]) {
      const slow = mlp.call([new Value(x), new Value(y)]);
      const fast = mlp.predictFast(x, y);
      assert.ok(
        Math.abs(slow.data - fast) < 1e-12,
        `${activation} (${x},${y}): slow=${slow.data} fast=${fast}`
      );
    }
  }
});

test('같은 시드는 같은 초기 가중치를 준다', () => {
  const a = new MLP(2, [4, 1], { seed: 42 }).parameters().map((p) => p.data);
  const b = new MLP(2, [4, 1], { seed: 42 }).parameters().map((p) => p.data);
  const c = new MLP(2, [4, 1], { seed: 43 }).parameters().map((p) => p.data);
  assert.deepStrictEqual(a, b);
  assert.notDeepStrictEqual(a, c);
});

test('zeroGrad 후 모든 grad 가 0', () => {
  const mlp = new MLP(2, [4, 4, 1], { seed: 3 });
  const out = mlp.call([new Value(0.5), new Value(-0.5)]);
  out.backward();
  assert.ok(mlp.parameters().some((p) => p.grad !== 0), 'backward 후엔 grad 가 있어야 한다');
  mlp.zeroGrad();
  assert.ok(mlp.parameters().every((p) => p.grad === 0));
});

test('Neuron / Layer 도 개별적으로 동작한다', () => {
  const n = new Neuron(3, { seed: 1 });
  assert.strictEqual(n.parameters().length, 4);
  const l = new Layer(3, 5, { seed: 1 });
  assert.strictEqual(l.parameters().length, 20);
  assert.strictEqual(l.call([new Value(1), new Value(2), new Value(3)]).length, 5);
});

test('데이터셋 4종이 요청한 개수와 ±1 라벨을 준다', () => {
  for (const kind of ['moons', 'circles', 'xor', 'spiral']) {
    const ds = makeDataset(kind, { n: 60, noise: 0.1, seed: 5 });
    assert.strictEqual(ds.points.length, 60, kind);
    assert.ok(ds.points.every((p) => p.label === 1 || p.label === -1), kind);
    assert.ok(ds.points.every((p) => Number.isFinite(p.x) && Number.isFinite(p.y)), kind);
    // 두 클래스가 모두 존재해야 한다
    assert.ok(ds.points.some((p) => p.label === 1) && ds.points.some((p) => p.label === -1), kind);
  }
});

test('같은 시드의 데이터셋은 동일하다', () => {
  const a = makeDataset('moons', { n: 30, noise: 0.1, seed: 9 }).points;
  const b = makeDataset('moons', { n: 30, noise: 0.1, seed: 9 }).points;
  assert.deepStrictEqual(a, b);
});

test('moons 100점을 100스텝 학습하면 loss 가 줄고 accuracy >= 0.9', () => {
  const trainer = new Trainer({
    dataset: 'moons', n: 100, noise: 0.1, seed: 1,
    hidden: [16, 16], activation: 'relu', lr: 0.1,
  });
  const first = trainer.stepFull();
  for (let i = 0; i < 99; i++) trainer.stepFull();
  const last = trainer.history[trainer.history.length - 1];

  assert.ok(last.loss < first.loss, `loss ${first.loss} -> ${last.loss}`);
  assert.ok(last.acc >= 0.9, `accuracy = ${last.acc}`);
  assert.strictEqual(trainer.history.length, 100);
});

test('부분 스텝 4개를 순서대로 돌리면 stepFull 과 같은 결과', () => {
  const opts = { dataset: 'xor', n: 40, noise: 0.1, seed: 2, hidden: [8, 8], lr: 0.2 };
  const a = new Trainer(opts);
  const b = new Trainer(opts);

  for (let i = 0; i < 5; i++) {
    a.stepFull();
    for (let k = 0; k < 4; k++) b.substep();
  }
  assert.deepStrictEqual(
    b.model.parameters().map((p) => p.data),
    a.model.parameters().map((p) => p.data)
  );
  assert.deepStrictEqual(b.history, a.history);
});

test('reset 하면 step 과 히스토리가 초기화된다', () => {
  const trainer = new Trainer({ dataset: 'circles', n: 40, seed: 4, hidden: [8, 8] });
  trainer.stepFull();
  trainer.reset();
  assert.strictEqual(trainer.step, 0);
  assert.strictEqual(trainer.history.length, 0);
  assert.strictEqual(trainer.phase, 'forward');
});
