// engine.js — micrograd 의 engine.py 를 자바스크립트로 1:1 재구현.
//
// 파이썬과 달리 JS 에는 연산자 오버로딩이 없으므로 메서드 체인을 쓴다.
//   python:  d = a * b + c
//   js:      const d = a.mul(b).add(c)
// 숫자를 넘기면 자동으로 Value 로 감싸므로 a.mul(2) 처럼 써도 된다.
(function (global) {
  'use strict';

  class Value {
    constructor(data, children = [], op = '', label = '') {
      this.data = data;
      this.grad = 0;
      this._backward = () => {};
      this._prev = new Set(children);
      this._op = op;
      this.label = label;
    }

    static wrap(other) {
      return other instanceof Value ? other : new Value(other);
    }

    add(other) {
      other = Value.wrap(other);
      const out = new Value(this.data + other.data, [this, other], '+');
      out._backward = () => {
        this.grad += out.grad;
        other.grad += out.grad;
      };
      return out;
    }

    mul(other) {
      other = Value.wrap(other);
      const out = new Value(this.data * other.data, [this, other], '*');
      out._backward = () => {
        this.grad += other.data * out.grad;
        other.grad += this.data * out.grad;
      };
      return out;
    }

    // 지수는 상수만 지원한다 (강의 engine.py 와 동일).
    pow(k) {
      if (typeof k !== 'number') {
        throw new TypeError('pow() 의 지수는 숫자여야 합니다');
      }
      const out = new Value(Math.pow(this.data, k), [this], `**${k}`);
      out._backward = () => {
        this.grad += k * Math.pow(this.data, k - 1) * out.grad;
      };
      return out;
    }

    neg() {
      return this.mul(-1);
    }

    sub(other) {
      return this.add(Value.wrap(other).neg());
    }

    div(other) {
      return this.mul(Value.wrap(other).pow(-1));
    }

    exp() {
      const e = Math.exp(this.data);
      const out = new Value(e, [this], 'exp');
      out._backward = () => {
        this.grad += e * out.grad;
      };
      return out;
    }

    tanh() {
      const t = Math.tanh(this.data);
      const out = new Value(t, [this], 'tanh');
      out._backward = () => {
        this.grad += (1 - t * t) * out.grad;
      };
      return out;
    }

    relu() {
      const out = new Value(this.data < 0 ? 0 : this.data, [this], 'ReLU');
      out._backward = () => {
        this.grad += (out.data > 0 ? 1 : 0) * out.grad;
      };
      return out;
    }

    // 위상 정렬 후 역순으로 _backward 를 호출한다.
    backward() {
      const topo = [];
      const visited = new Set();
      // 깊은 그래프에서 스택이 넘치지 않도록 재귀 대신 명시적 스택을 쓴다.
      const stack = [[this, false]];
      while (stack.length) {
        const [node, expanded] = stack.pop();
        if (expanded) {
          topo.push(node);
          continue;
        }
        if (visited.has(node)) continue;
        visited.add(node);
        stack.push([node, true]);
        for (const child of node._prev) {
          if (!visited.has(child)) stack.push([child, false]);
        }
      }

      this.grad = 1;
      for (let i = topo.length - 1; i >= 0; i--) {
        topo[i]._backward();
      }
      return topo;
    }

    toString() {
      return `Value(data=${this.data}, grad=${this.grad})`;
    }
  }

  global.Value = Value;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Value };
  }
})(typeof globalThis !== 'undefined' ? globalThis : this);
