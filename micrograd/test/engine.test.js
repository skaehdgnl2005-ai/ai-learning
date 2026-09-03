'use strict';
const test = require('node:test');
const assert = require('node:assert');

require('../src/engine.js');
const { Value } = globalThis;

// 표현식 f(a, b, c, ...) 를 받아, 각 leaf 에 대한 grad 를 중심차분과 비교한다.
function checkGrads(expr, xs, tol = 1e-6) {
  const vals = xs.map((x) => new Value(x));
  const out = expr(vals);
  out.backward();
  const analytic = vals.map((v) => v.grad);

  const h = 1e-5;
  const numeric = xs.map((_, i) => {
    const plus = xs.slice();
    const minus = xs.slice();
    plus[i] += h;
    minus[i] -= h;
    const fp = expr(plus.map((x) => new Value(x))).data;
    const fm = expr(minus.map((x) => new Value(x))).data;
    return (fp - fm) / (2 * h);
  });

  for (let i = 0; i < xs.length; i++) {
    assert.ok(
      Math.abs(analytic[i] - numeric[i]) < tol,
      `leaf ${i}: analytic=${analytic[i]} numeric=${numeric[i]}`
    );
  }
}

test('강의 sanity check: 스칼라 그래프의 forward/backward 값', () => {
  const x = new Value(-4.0);
  const z = x.mul(2).add(2).add(x);          // z = 2x + 2 + x
  const q = z.relu().add(z.mul(x));          // q = relu(z) + z*x
  const h = z.mul(z).relu();                 // h = relu(z*z)
  const y = h.add(q).add(q.mul(x));          // y = h + q + q*x
  y.backward();

  assert.ok(Math.abs(y.data - (-20.0)) < 1e-9, `y.data=${y.data}`);
  assert.ok(Math.abs(x.grad - 46.0) < 1e-9, `x.grad=${x.grad}`);
});

test('add / mul / sub / neg 의 grad', () => {
  checkGrads(([a, b, c]) => a.mul(b).sub(c).add(a.neg()), [1.3, -2.7, 0.4]);
});

test('pow 의 grad', () => {
  checkGrads(([a, b]) => a.pow(3).add(b.pow(2)).mul(a), [1.7, -0.6]);
});

test('div 의 grad', () => {
  checkGrads(([a, b]) => a.div(b).add(b.div(2)), [2.5, 1.9]);
});

test('tanh 의 grad', () => {
  checkGrads(([a, b]) => a.mul(b).tanh().add(a.tanh()), [0.8, -1.4]);
});

test('relu 의 grad', () => {
  checkGrads(([a, b]) => a.mul(2).relu().add(b.relu()).add(a.sub(b).relu()), [0.9, -1.2]);
});

test('exp 의 grad', () => {
  checkGrads(([a, b]) => a.exp().add(b.mul(0.5).exp()).mul(a), [0.4, -0.9]);
});

test('숫자 인자는 자동으로 Value 로 감싼다', () => {
  const a = new Value(3);
  const out = a.add(2).mul(4);
  assert.strictEqual(out.data, 20);
  out.backward();
  assert.strictEqual(a.grad, 4);
});

test('같은 노드를 여러 번 쓰면 grad 가 누적된다', () => {
  const a = new Value(3);
  const b = a.add(a); // b = 2a
  b.backward();
  assert.strictEqual(a.grad, 2);
});

test('메타데이터: _prev / _op / label', () => {
  const a = new Value(1, [], '', 'a');
  const b = new Value(2, [], '', 'b');
  const c = a.add(b);
  assert.strictEqual(c._op, '+');
  assert.deepStrictEqual([...c._prev], [a, b]);
  assert.strictEqual(a.label, 'a');
});
