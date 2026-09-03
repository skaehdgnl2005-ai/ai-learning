# 신경망 학습 과정 시각화 (micrograd-js)

Karpathy 의 [micrograd](https://github.com/karpathy/micrograd) 를 자바스크립트로 1:1 재구현하고,
**결과값에 따라 각 가중치가 조절되며 정답률이 올라가는 과정**을 브라우저에서 눈으로 보는 도구입니다.

강의 마지막 데모와 같은 과제입니다: 2D 점 분류, `MLP(2, [16, 16, 1])`, 파라미터 337개,
SVM max-margin hinge loss + L2 정규화(alpha = 1e-4).

## 여는 법

서버가 필요 없습니다. `index.html` 을 브라우저에서 그냥 열면 됩니다.

```
start index.html          # Windows
```

테스트:

```
npm test                  # = node --test test/engine.test.js test/nn.test.js
```

## 화면 설명

### 상단
현재 **스텝 / 손실(loss) / 정답률**. 왼쪽 배지는 방금 끝난 부분 스텝을 보여줍니다.

### 네트워크
2-16-16-1 노드를 열로 배치합니다.

- 선 굵기 = `|w|`, 색 = 부호 (파랑 +, 빨강 −)
- **역전파 단계**에서는 노란색 오버레이로 `grad` 를 겹쳐 그립니다 (굵기 = `|grad|`)
- 엣지나 노드에 마우스를 올리면 `w`(또는 `b`), `grad`, 직전 스텝의 `Δw` 툴팁이 뜹니다
- 은닉 노드에 마우스를 올리면 그 뉴런의 **활성값 히트맵**이 아래 데이터 화면에 나타납니다
  (ReLU 뉴런이 만드는 반평면 경계를 직접 볼 수 있습니다)

### 데이터와 결정 경계
- 점 색 = 라벨(+1 파랑 / −1 빨강)
- 배경색 = `predictFast` 로 계산한 모델의 예측 부호
- **오분류된 점**은 노란 테두리로 강조됩니다

### 학습 곡선
손실(로그 스케일, 노랑)과 정답률(초록). 학습률을 크게 잡으면 손실이 위로 튀는 것이 그대로 보입니다.

## 조작

| 버튼 | 하는 일 | 단축키 |
| --- | --- | --- |
| **재생 / 일시정지** | 단계 표시 없이 스텝을 반복 (속도 슬라이더) | `Space` |
| **1 스텝** | forward → loss → backward → update 4단계를 0.32초 간격으로 차례로 보여줌 | `S` |
| **부분 스텝** | 다음 부분 스텝 하나만 실행 | `D` |
| **초기화** | 데이터·가중치를 시드에서 다시 생성 | — |

컨트롤: 데이터셋(초승달 / 동심원 / XOR / 나선), 점 개수(50~300), 노이즈, 시드,
은닉층 크기(4·8·16 두 층), 활성화(ReLU / tanh), 학습률(0.01~5), 재생 속도.

## 직접 해볼 것

- **moons, lr 0.5** → 3스텝 만에 정답률 90%, 100스텝이면 100%. 결정 경계가 초승달을 따라 휩니다.
- **나선, 은닉 16 vs 은닉 4** → 16은 60스텝 근처에서 95%를 넘지만, 4는 300스텝을 돌려도 70%대에 갇힙니다. 모델 용량 차이가 눈에 보입니다.
- **학습률 5** → 손실이 위로 폭발하고 정답률이 50%(찍기)로 떨어집니다.
- **"부분 스텝"을 계속 누르기** → 역전파 단계에서 grad 가 출력층 쪽에서 크고 입력층으로 갈수록 작아지는 것, update 단계에서 굵은 grad 를 가진 선이 실제로 굵기가 바뀌는 것을 확인할 수 있습니다.

## 학습 노트

이 도구를 만들면서 정리한 개념 노트가 [NOTES.md](NOTES.md) 에 있습니다.
score의 의미, 역전파가 흐르는 순서, grad가 정확히 무엇의 기울기인지,
손실 함수의 연속성, 극소 vs 최소, 활성화 함수의 진짜 역할 —
전부 이 코드를 직접 돌려서 얻은 숫자로 확인한 내용입니다.

## 강의 Python 코드 ↔ 이 프로젝트의 JS

자바스크립트에는 연산자 오버로딩이 없어서 **메서드 체인**을 씁니다. 그 외에는 이름과 구조가 같습니다.

### `engine.py` → [src/engine.js](src/engine.js)

| micrograd (Python) | 이 프로젝트 (JS) |
| --- | --- |
| `Value(2.0, label='a')` | `new Value(2.0, [], '', 'a')` |
| `a + b`, `a + 2` | `a.add(b)`, `a.add(2)` |
| `a * b` | `a.mul(b)` |
| `a - b`, `-a` | `a.sub(b)`, `a.neg()` |
| `a / b` | `a.div(b)` |
| `a ** 3` | `a.pow(3)` |
| `a.tanh()`, `a.relu()`, `a.exp()` | 같음 |
| `a._prev`, `a._op`, `a.grad` | 같음 |
| `a._backward` | 같음 (클로저) |
| `loss.backward()` | 같음 |

`backward()` 는 강의와 똑같이 위상 정렬 후 역순으로 `_backward` 를 호출합니다.
다만 재귀 대신 명시적 스택을 써서 깊은 그래프에서도 스택이 넘치지 않습니다.

### `nn.py` → [src/nn.js](src/nn.js)

| micrograd (Python) | 이 프로젝트 (JS) |
| --- | --- |
| `Neuron(nin)` / `Layer(nin, nout)` / `MLP(nin, nouts)` | 같음 (`new MLP(2, [16, 16, 1])`) |
| `n(x)` (`__call__`) | `n.call(x)` |
| `.parameters()` | 같음 |
| `.zero_grad()` | `.zeroGrad()` |
| 마지막 레이어만 `nonlin=False` | 같음 (`activation` 옵션으로 relu/tanh 선택) |
| 초기값 `uniform(-1, 1)` | 같음 (시드 있는 난수기 [src/rng.js](src/rng.js)) |
| — | **`mlp.predictFast(x, y)`** (아래 참고) |

### `predictFast` 를 추가한 이유

결정 경계를 그리려면 44×44 격자를 **매 스텝** 계산해야 합니다.
`Value` 그래프로 하면 격자 한 장에 노드가 100만 개 넘게 생겨 느립니다.
`predictFast` 는 파라미터의 `.data` 만 읽어 순수 숫자로 같은 수식을 계산합니다.
수학은 `call()` 과 완전히 동일하며, [test/nn.test.js](test/nn.test.js) 에서 두 결과가 일치함을 검증합니다.

### 학습 루프 → [src/trainer.js](src/trainer.js)

강의의 `loss()` + 학습 루프와 같습니다.

```python
# 강의
losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(yb, scores)]
data_loss = sum(losses) * (1.0/len(losses))
reg_loss = alpha * sum((p*p for p in model.parameters()))
total_loss = data_loss + reg_loss
model.zero_grad(); total_loss.backward()
for p in model.parameters(): p.data -= learning_rate * p.grad
```

이 프로젝트는 같은 계산을 **4개의 부분 스텝**으로 쪼개 각각을 그립니다.

| 부분 스텝 | 하는 일 | 화면에 보이는 것 |
| --- | --- | --- |
| `forward` | 모든 점 예측, 정답률 계산 | 오분류 점 강조 |
| `loss` | hinge loss + L2 | 손실값 표시 |
| `backward` | `zeroGrad()` → `loss.backward()` | 엣지 grad 오버레이 |
| `update` | `p.data -= lr * p.grad` | 가중치 굵기 변화, 곡선 갱신 |

`trainer.stepFull()` 은 이 4개를 한 번에 돌립니다.
[test/nn.test.js](test/nn.test.js) 에서 부분 스텝 4번과 `stepFull()` 한 번의 결과가 같음을 검증합니다.

## 파일 구조

```
index.html          레이아웃 + 컨트롤 + 캔버스 3개
style.css
src/
  rng.js            시드 있는 난수기 (재현성)
  engine.js         Value 클래스           ← engine.py
  nn.js             Neuron / Layer / MLP   ← nn.py
  datasets.js       moons / circles / xor / spiral
  trainer.js        4단계 학습 상태기계
  viz-network.js    네트워크 다이어그램
  viz-data.js       데이터 점 + 결정 경계
  viz-chart.js      loss / accuracy 그래프
  app.js            UI 배선
test/
  engine.test.js    수치 미분과 grad 비교
  nn.test.js        구조 / predictFast 일치 / 학습 수렴
```

ES 모듈이 아니라 평범한 `<script>` 태그로 로드하므로 `file://` 로 그냥 열립니다.
각 파일은 `globalThis` 에 내보내면서 동시에 `module.exports` 도 채워서, Node 테스트가 같은 코드를 그대로 씁니다.

## 성능

100개 점 × 스텝당 약 6만 개 `Value` 노드 → 스텝당 30ms 안팎.
재생은 초당 1~60 스텝까지 설정할 수 있고, 실제로는 초당 30 스텝 근처가 상한입니다.
결정 경계는 `predictFast` 로 그리므로 여기에 거의 비용이 들지 않습니다.
