# Day 3: 딥러닝과 신경망 — 뉴런에서 GPT까지

> 이전 학습에서 머신러닝의 세 패러다임(지도·비지도·강화학습)을 다뤘습니다. 오늘은 그 세 패러다임 모두의 내부 엔진으로 자리잡은 **딥러닝(Deep Learning, DL)** 이 어떤 구조로 작동하며, 1958년의 단일 퍼셉트론에서 2024년의 AlphaFold 3·GPT까지 어떻게 진화했는지를 학습합니다.

## 개요

**딥러닝** 은 여러 층의 **인공 신경망(Artificial Neural Network, ANN)** 을 통해 데이터로부터 복잡한 비선형 함수를 학습하는 기술입니다. 핵심 강점은 "특징(feature)을 사람이 설계하지 않고 데이터로부터 자동 학습" 한다는 점입니다. 전통적 QSAR는 logP·TPSA·분자량 같은 기술자를 사람이 골라 넣었지만, 딥러닝은 분자 그래프·SMILES 토큰·단백질 서열을 원시 형태 그대로 받아 과제에 맞는 표현을 내부에서 구성합니다. AlphaFold가 단백질 구조를, ChatGPT가 자연어를, AlphaFold 3가 단백질-리간드 복합체를 동일한 원리로 풀어내는 이유입니다. 오늘의 목표는 단일 뉴런부터 Transformer까지 주요 구조를 원리 수준에서 이해하고, 각 아키텍처가 신약개발의 어떤 문제에 대응하는지 지도를 그리는 것입니다.

## 핵심 개념

### 인공 뉴런 — 하나의 세포에서 시작된 이야기

딥러닝의 최소 단위는 **인공 뉴런(artificial neuron)** 또는 **퍼셉트론(perceptron)** 입니다. 입력 x₁, x₂, …, xₙ에 각각 가중치 w₁, w₂, …, wₙ을 곱해 더하고, 편향 b를 보탠 뒤, **활성화 함수(activation function)** σ를 통과시켜 출력을 만듭니다.

$$y = \sigma(w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b)$$

이 구조는 1958년 Rosenblatt가 제안한 퍼셉트론과 동일한 형태입니다. 약학 전공자에게 친숙한 **로지스틱 회귀** — BBB 투과를 logP·TPSA·분자량으로 예측하는 모델 — 는 시그모이드(sigmoid)를 활성화 함수로 쓰는 **단일층 신경망** 과 수학적으로 같습니다. 로지스틱 QSAR가 곧 가장 얕은 딥러닝임을 기억하면, 층을 쌓는 확장이 자연스럽게 이해됩니다.

### 층을 쌓는 이유 — 보편 근사와 표현 학습

뉴런 하나는 선형 결합 + 단순 비선형에 불과하지만, 뉴런을 옆으로 여러 개 두고(한 층) 그 출력을 다음 층의 입력으로 넘기면 표현력이 급증합니다. 이 구조가 **다층 퍼셉트론(Multi-Layer Perceptron, MLP)** 이며, 1989년 Cybenko·Hornik의 **보편 근사 정리(Universal Approximation Theorem)** 는 "충분히 많은 뉴런의 2층 MLP가 임의 연속 함수를 원하는 정밀도로 근사한다"고 말합니다. 다만 이론상 얕고 넓은 망도 가능하나, 실무적으로 **깊은 망이 훨씬 적은 파라미터로 동일 함수를 표현** 합니다. 분자 활성 예측에서 얕은 층은 원자 연결 패턴을, 중간 층은 작용기(functional group)를, 깊은 층은 약리단(pharmacophore) 수준의 모티프를 형성합니다. 이는 약학 전공자가 SAR를 직관할 때 원자 → 작용기 → 약리단으로 해석하는 과정과 정확히 대응합니다.

### 활성화 함수 — 비선형성의 원천

층을 아무리 쌓아도 활성화 함수가 선형이면 전체는 하나의 선형 변환에 불과합니다. 따라서 비선형 활성화 함수가 딥러닝의 표현력의 근본입니다.

| 함수 | 형태 | 특징 | 실무 사용처 |
|------|------|------|------------|
| Sigmoid | 1/(1+e^−x) | 0~1 출력, 기울기 소실 문제 | 이진 분류 출력층 (hERG 차단 여부 등) |
| Tanh | (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ) | −1~1 출력 | 초기 RNN |
| ReLU | max(0, x) | 계산 간단, 기울기 소실 완화 | 현대 CNN·MLP의 기본값 |
| GELU | x·Φ(x) | 부드러운 ReLU | Transformer (BERT·GPT·MolFormer) |

ReLU의 등장(2011)이 기울기 소실(vanishing gradient) 문제를 상당 부분 해결하며 깊은 망 학습의 실용화를 이끌었습니다.

### 주요 아키텍처의 진화

딥러닝의 역사는 "어떤 데이터 구조에 어떤 귀납 편향을 넣을 것인가"의 진화입니다.

| 아키텍처 | 등장 | 핵심 아이디어 | 신약개발 대응 |
|---------|------|-----------|-------------|
| **MLP** | 1986 (역전파) | 전결합 층 쌓기 | 기술자 기반 QSAR |
| **CNN** (Convolutional NN) | 1998/2012 | 국소 패턴 + 파라미터 공유 | 세포 이미지, 의료 영상 |
| **RNN·LSTM** | 1997 | 시퀀스 순서 기억 | 초기 SMILES 생성 모델 |
| **GNN** (Graph Neural Network) | 2017 | 이웃 원자 메시지 전달 | 분자 그래프 물성 예측 |
| **Transformer** | 2017 | 자기 주의(self-attention) | AlphaFold, GPT, MolFormer |
| **Diffusion** | 2020 | 노이즈 역추적 생성 | AlphaFold 3, 3D 분자 생성 |

Transformer의 자기 주의(self-attention)는 "임의의 두 원소가 직접 상호작용하는 가중치를 데이터에서 학습" 하여 이전 순차 모델의 한계를 깼습니다. GPT의 장문 맥락, AlphaFold의 잔기-잔기 상호작용, MolFormer의 분자 토큰 의미가 모두 같은 구조에서 나오는 이유입니다.

## 작동 원리와 아키텍처

### 학습 루프의 네 단계

어떤 딥러닝 모델이든 학습은 다음 4단계가 반복됩니다.

```
1. Forward pass  : 입력 → 층 통과 → 예측값 ŷ
2. Loss 계산     : L = f(ŷ, y)  (회귀면 MSE, 분류면 교차엔트로피)
3. Backward pass : ∂L/∂w 계산 (역전파, chain rule)
4. 파라미터 갱신 : w ← w − η · ∂L/∂w  (경사하강법)
```

**역전파(backpropagation)** 는 1986년 Rumelhart 등이 체계화한 알고리즘으로, 출력층의 오차를 미분의 연쇄법칙으로 입력층까지 되밀어 각 가중치의 책임을 배분합니다. 옵티마이저로는 현재 Adam·AdamW가 가장 널리 쓰이며, 학습률(learning rate) η가 수렴의 핵심 하이퍼파라미터입니다.

### 바이브코딩 관점: 신약개발 딥러닝 시스템의 구성

```
시스템 구성 (예: 분자 물성 예측 딥러닝 서비스):
1. 데이터 모듈  : ChEMBL/PubChem에서 (SMILES, 활성값) 수집
2. 전처리       : SMILES 정규화 → 분자 그래프 or 토큰 시퀀스
3. 인코더       : GNN 또는 Transformer로 분자 임베딩 추출(256~768차원)
4. 예측 헤드    : MLP 2층 → pIC50(회귀) 또는 확률(분류)
5. 학습 루프    : Adam + 조기 종료(early stopping) + scaffold split
6. 서빙         : FastAPI + Docker, 예측 시 불확실성(MC Dropout) 반환
```

| 설계 결정 | 선택지 | 권장 | 이유 |
|----------|--------|------|------|
| 인코더 | MLP / CNN / GNN / Transformer | GNN or Transformer | 분자 구조 귀납 편향 내장 |
| 깊이 | 2~20+ 층 | 3~6 층부터 시작 | 데이터 규모와 과적합 균형 |
| 활성화 | ReLU / GELU | GELU (Transformer), ReLU (MLP·GNN) | 아키텍처 관행 |
| 정규화 | BatchNorm / LayerNorm | LayerNorm (Transformer), BatchNorm (CNN) | 안정성 |
| 평가 | Random split / Scaffold split | **Scaffold split** | 실제 신규 화합물 일반화 반영 |

Scaffold split은 약학 전공자가 반드시 주장해야 할 설계 결정입니다. 무작위 분할은 동일 scaffold의 유사 분자가 train/test에 섞여 성능을 과대평가합니다. 약물 탐색은 본질적으로 "처음 보는 scaffold로의 외삽"이기 때문에, 평가 분할이 임상 현실과 괴리되면 모델 선정 자체가 왜곡됩니다.

## 신약개발 적용

딥러닝이 실제 신약개발 파이프라인에 어떻게 스며들었는지 네 가지 사례로 확인합니다.

- **AlphaFold 3 (Google DeepMind · Isomorphic Labs, 2024, *Nature*)**: 2024년 5월 공개된 AF3는 Transformer 계열 **Pairformer** 와 **확산 모델(diffusion)** 을 결합하여 단백질-리간드·핵산·이온 복합체 구조를 단일 네트워크로 예측합니다. 단백질-리간드 상호작용에서 기존 도킹 대비 정확도가 약 50% 이상 향상된 것으로 보고되었고, Isomorphic Labs는 2024년 Novartis·Eli Lilly와 총 30억 달러 규모 파트너십을 체결했습니다.
- **MolFormer-XL (IBM, 2022)**: 약 11억 개 SMILES를 마스크 언어 모델링으로 사전학습한 Transformer. 다운스트림 11개 물성 과제에서 그래프 기반 모델을 다수 상회했습니다. Day 2의 자기지도학습과 딥러닝 아키텍처의 결합 사례입니다.
- **Insilico Medicine Pharma.AI**: GNN·Transformer·강화학습 생성 모델을 한 플랫폼에 통합해, 특발성 폐섬유증 치료제 INS018_055를 타겟 발굴부터 임상 Phase II까지 약 30개월에 도달시켰습니다.
- **Chai-1 (Chai Discovery, 2024)**: AF3급 복합체 구조 예측 모델을 연구용으로 공개해 접근성을 넓혔습니다 (상업 이용 조건은 버전별 확인 필요).

기존 구조 기반 신약설계 대비 딥러닝 기반 접근의 차이는 다음과 같습니다.

| 항목 | 고전적 도킹/QSAR | 딥러닝 기반 |
|------|---------------|-----------|
| 특징 설계 | 사람 수작업 | 모델이 자동 학습 |
| 단백질-리간드 복합체 예측 | 물리 기반 도킹 (제한적 정확도) | AF3 수준의 end-to-end 예측 |
| 신규 scaffold 일반화 | 약함 | SSL+딥러닝으로 크게 개선 |
| 계산 비용 | 상대적으로 저렴 | GPU 필요, 점차 경량화 진행 |

## 창업 관점

Phase 1 단계에서는 간략히만 짚습니다. 약학 전공자 관점에서 딥러닝은 그 자체로 제품이라기보다 **모든 제품의 엔진** 입니다. 창업 기회의 차별화는 "어떤 아키텍처를 쓰느냐"가 아니라 "어떤 신약개발 문제에 어떤 아키텍처를 어떻게 약학 지식으로 조합하느냐"에서 발생합니다. 구체적 BM·시장 분석은 Phase 5에서 다룹니다.

## 오늘의 과제

1. **아키텍처-문제 매핑 (40분)**: 신약개발 문제 5개 — ① 세포 이미지 기반 표현형 스크리닝, ② 분자 물성(logP) 예측, ③ 단백질 구조 예측, ④ 분자 생성, ⑤ SMILES 기반 독성 분류 — 각각에 가장 적합한 딥러닝 아키텍처(MLP / CNN / GNN / Transformer / Diffusion)를 배정하고 선택 이유를 1-2줄로 정리합니다. A4 1쪽.
2. **AlphaFold 3 논문/블로그 리서치 (50분)**: Abramson *et al.* (2024) "Accurate structure prediction of biomolecular interactions with AlphaFold 3" *Nature* 논문과 Isomorphic Labs 블로그를 참고해, ① Pairformer와 확산 모델이 결합된 방식, ② 기존 AF2 대비 추가된 예측 범위(리간드·핵산·이온), ③ 약학적 한계(동적 유연성, 알로스테릭 예측) 3가지를 각 3줄씩 요약합니다.
3. **내 관심 모달리티 선정 (30분)**: 관심 있는 신약개발 모달리티(예: 항체, 경구 저분자, RNA 치료제) 1개를 고르고, 그 모달리티에 가장 적합할 딥러닝 아키텍처를 선정한 뒤, 왜 그 선택이 약학적 타당성을 갖는지 반 페이지로 주장합니다. 이 메모는 Phase 2에서 심화 학습 시 재사용합니다.

## 참고 자료

- Abramson, J. *et al.* (2024). "Accurate structure prediction of biomolecular interactions with AlphaFold 3." *Nature*, 630, 493-500. — 단백질 복합체 예측의 새로운 표준.
- Vaswani, A. *et al.* (2017). "Attention is all you need." *NeurIPS*. — Transformer 원전. 오늘날 분자·단백질·언어 모델의 공통 뿌리.
- Rumelhart, D. E., Hinton, G. E., Williams, R. J. (1986). "Learning representations by back-propagating errors." *Nature*, 323, 533-536. — 역전파 알고리즘 체계화.
- Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press (deeplearningbook.org에 무료 공개). — 딥러닝의 고전 교과서.
- Isomorphic Labs 공식 블로그 (isomorphiclabs.com) — AF3 기반 합리적 약물 설계 사례.
