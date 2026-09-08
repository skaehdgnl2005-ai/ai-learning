# AI 학습 도구 모음

AI 가 어떻게 동작하는지 직접 눈으로 보면서 이해하려고 만드는 시각화 도구 모음입니다.

> Zero-dependency browser visualizations of how AI actually works. First tool: a scalar autograd engine (micrograd) reimplemented in JS, training a neural net step by step.

**▶ https://skaehdgnl2005-ai.github.io/ai-learning/**

빌드 과정도, 서버도, 설치할 패키지도 없습니다. 전부 순수 HTML + 자바스크립트라
저장소를 내려받아 `index.html` 을 브라우저에서 그냥 열어도 똑같이 동작합니다.

## 도구 목록

| 도구 | 내용 | 링크 |
|---|---|---|
| [micrograd](micrograd/) | 신경망 학습 과정 시각화. 가중치가 조절되며 정답률이 오르는 과정을 스텝 단위로 관찰 | [열기](https://skaehdgnl2005-ai.github.io/ai-learning/micrograd/) |

## 새 도구 추가하는 법

1. 저장소 루트에 폴더를 하나 만들고 그 안에 `index.html` 을 둡니다.
2. 루트 [`index.html`](index.html) 의 카드 목록에 링크를 추가합니다.
3. 위 표에 한 줄 추가합니다.
4. push 하면 몇 분 안에 자동으로 배포됩니다.

폴더 안에서는 상대 경로만 쓰면 됩니다. 별도 설정은 없습니다.

## 어떻게 검증했나

- 자동미분 엔진(`micrograd/src/engine.js`)과 신경망 층(`nn.js`)은 Node 내장 테스트 러너로 검증합니다. 현재 20개 테스트 통과.
- 시드 고정 RNG(`rng.js`)로 데모가 매번 같은 궤적을 재현합니다.
- 시각화 레이어(`viz-*.js`)는 엔진과 완전히 분리되어 있어 엔진만 단독 테스트할 수 있습니다.

```
cd micrograd && npm test     # node --test → 20 pass
```

## 설계 원칙

- 무빌드 · 무서버 · 무의존. 배포는 GitHub Pages에 push하는 것이 전부입니다.
- 엔진과 시각화의 경계를 지킵니다. 새 도구는 이 경계를 그대로 따릅니다.
- 관련 노트: [micrograd 강의 정리 (한국어)](https://github.com/skaehdgnl2005-ai/ai-experiments/tree/main/micrograd-lecture-notes-ko)
