# Day 4: AI 모델의 학습 원리 — 데이터, 손실함수, 최적화

> 이전 학습에서 인공 뉴런부터 Transformer·확산 모델까지 주요 딥러닝 아키텍처를 살폈습니다. 오늘은 그 모든 아키텍처가 공통으로 거치는 **학습(training)** 의 내부를 — 데이터를 어떻게 나눌 것인가, 손실 함수를 어떻게 정의할 것인가, 어떤 최적화 알고리즘으로 파라미터를 업데이트할 것인가 — 원리 수준에서 학습합니다.

## 개요

딥러닝 모델의 성능은 아키텍처보다 **데이터 분할 전략, 손실 함수 설계, 최적화·정규화 절차** 의 합에서 결정되는 경우가 더 많습니다. 같은 GNN을 써도 무작위 분할은 과대평가된 R²를, scaffold 분할은 임상 현실에 가까운 R²를 산출합니다. 같은 분류기를 써도 가중치 없는 교차엔트로피는 hERG처럼 양성이 5%인 데이터에서 무력하지만, 클래스 가중치를 약학적 위해도 기준으로 조정하면 임상적 가치가 살아납니다. 오늘 학습 목표는 학습 절차의 5대 구성요소(데이터 분할 → 손실 함수 → 옵티마이저 → 정규화 → 평가)를 이해하고, 약학 전공자가 각 지점에서 어떤 의사결정에 기여할 수 있는지 정리하는 것입니다.

## 핵심 개념

### 데이터 분할 — 평가 신뢰도의 출발

학습은 본질적으로 **일반화(generalization)**, 즉 보지 못한 데이터에서의 성능을 만드는 작업입니다. 따라서 데이터를 **훈련(train) / 검증(validation) / 시험(test)** 으로 분할합니다. 훈련셋은 파라미터 학습에, 검증셋은 하이퍼파라미터 선택과 조기 종료에, 시험셋은 단 한 번의 최종 평가에만 사용합니다. 시험셋을 모델 선택에 재사용하는 순간 평가의 독립성이 무너집니다.

분자 데이터에서는 **분할 방식 자체가 결과를 좌우** 합니다.

| 분할 방식 | 작동 원리 | 위험·강점 |
|----------|---------|---------|
| Random split | 무작위로 80/10/10 | 동일 scaffold 분자가 train·test에 섞여 성능 과대평가 |
| Scaffold split | Bemis-Murcko scaffold가 train과 test에 겹치지 않게 | 신규 화학공간으로의 외삽력을 측정 |
| Time split | 합성·등록 시점 기준 절단 | 실제 신약개발의 시간적 일반화를 모사 |
| Cluster split | Tanimoto 유사도 군집 단위로 분리 | 유사 화학군 누수 차단 |

약학 전공자가 첫 번째로 주장해야 할 설계 결정은 거의 항상 **scaffold 또는 time split** 입니다. 신약 탐색은 정의상 "처음 보는 scaffold로의 외삽"이며, 무작위 분할은 임상 현실과 괴리됩니다.

### 손실 함수 — 모델에게 무엇을 잘하라고 말할 것인가

**손실 함수(loss function)** 는 예측이 얼마나 틀렸는지를 수치화하는 함수입니다. 손실 함수의 선택이 곧 모델이 최적화하는 목적입니다.

| 과제 | 손실 함수 | 해석 |
|------|---------|------|
| 회귀 (logP, pIC50) | MSE = mean((ŷ−y)²) | 큰 오차에 민감, 정규분포 가정 |
| 회귀 (이상치 강건) | MAE, Huber | 이상치 영향 완화 |
| 이진 분류 (hERG) | Binary cross-entropy | 확률 출력의 로그우도 |
| 다중 분류 | Categorical cross-entropy | 소프트맥스 + 로그우도 |
| 불균형 분류 | Focal loss, weighted CE | 소수 클래스 가중 |
| 다중 과제 | Σ wᵢ·Lᵢ | 과제 간 가중치 설계가 핵심 |

약학 전공자의 차별화는 **가중치 wᵢ를 설계하는 단계** 에서 발휘됩니다. ADMET 다중 과제에서 hERG·간독성처럼 임상적 위해도가 큰 항목의 가중치를 logP보다 높게 두는 결정은 통계만으로는 도출되지 않습니다. 이 결정에는 임상시험 단계별 실패 비용과 규제 기준에 대한 이해가 필요합니다.

손실 함수는 **약학 전공자에게 익숙한 비선형 회귀 적합** 과 본질적으로 같은 일을 합니다. PK 데이터에 1-구획 모델 C(t) = (D/V)·exp(−kt)를 적합할 때 MSE를 최소화해 V·k를 추정하는 절차는, 신경망이 가중치 w를 추정하는 절차와 수학적으로 동일합니다. 모델 형태만 더 복잡해진 것입니다.

### 최적화 — 손실을 줄이는 방향으로 파라미터 갱신

**경사하강법(gradient descent)** 은 손실에 대한 파라미터의 그래디언트 ∂L/∂w 반대 방향으로 w를 조금씩 옮깁니다. 핵심 변형은 다음 셋입니다.

- **SGD(Stochastic Gradient Descent)**: 미니배치 단위 그래디언트. 단순하지만 학습률 튜닝 민감.
- **Momentum/Nesterov**: 과거 그래디언트의 지수가중평균을 더해 진동 감쇠.
- **Adam / AdamW**: 그래디언트의 1차·2차 모멘트를 함께 추적해 파라미터별로 학습률을 적응. 현대 딥러닝의 사실상 표준. AdamW는 weight decay를 분리해 일반화를 개선.

**학습률(learning rate, η)** 은 한 걸음 크기로, 가장 영향력이 큰 단일 하이퍼파라미터입니다. 너무 크면 발산, 너무 작으면 수렴이 느리거나 국소 최소에 갇힙니다. **학습률 스케줄러(scheduler)** — warmup 후 cosine decay, ReduceLROnPlateau 등 — 가 대형 모델 학습의 안정성을 좌우합니다.

### 정규화 — 과적합과의 싸움

훈련 손실은 줄어드는데 검증 손실이 다시 올라가는 현상이 **과적합(overfitting)** 입니다. 정규화 기법은 모델이 훈련 데이터를 외우는 대신 일반 패턴을 학습하도록 강제합니다.

| 기법 | 작동 | 활용 |
|------|------|------|
| L2 정규화 / weight decay | 가중치 크기에 페널티 | 모든 모델 기본값 |
| Dropout | 학습 중 일부 뉴런 무작위 0 처리 | MLP·Transformer 표준 |
| Early stopping | 검증 손실 악화 시 학습 중단 | 거의 모든 학습 |
| Data augmentation | SMILES enumeration, 회전 등 | 데이터가 적을 때 |
| Batch/Layer Norm | 활성값 분포 안정화 | 깊은 망의 수렴성 |

데이터가 1만 건 이하인 약리 데이터에서는 정규화 강도가 모델 성능을 크게 좌우합니다. 또한 **MC Dropout** 은 추론 시 dropout을 켠 채 여러 번 예측을 수행해 **불확실성 추정** 으로 활용됩니다. ADMET처럼 "모델이 자신없는 분자에 대해 wet-lab 검증을 우선 배치"하는 활용 사례에서 본질적인 가치를 제공합니다.

## 작동 원리와 아키텍처

### 학습 파이프라인의 표준 흐름

```
[데이터 준비]
  raw 데이터 → 전처리(정규화·중복 제거) → scaffold split → DataLoader

[학습 루프]
  for epoch in range(E):
      for batch in train_loader:
          ŷ = model(x)                  # forward
          L = loss_fn(ŷ, y)             # 손실
          L.backward()                  # 역전파
          optimizer.step()              # AdamW 업데이트
          optimizer.zero_grad()
      val_loss = evaluate(val_loader)
      scheduler.step(val_loss)
      if early_stopping(val_loss): break

[최종 평가]
  best_model.load() → test_loader → 단 1회 측정
```

### 신약개발 학습 시스템의 설계 결정

| 단계 | 결정 항목 | Phase 1 권장 출발점 | 약학 관점의 개입 |
|------|---------|------------------|----------------|
| 분할 | random / scaffold / time | **scaffold** | 임상 현실 모사 |
| 손실 | MSE / MAE / CE / focal | 데이터 분포 따라 선택 | 위해도 기반 가중치 |
| 옵티마이저 | SGD / AdamW | **AdamW** | 거의 모든 경우 안전 |
| 학습률 | 1e-2 ~ 1e-5 | 1e-3에서 시작 | 데이터 규모 고려 |
| 정규화 | L2 + Dropout + Early stop | 셋 다 적용 | 임상적 신뢰도 확보 |
| 평가 | 단일 지표 / 다중 지표 | RMSE + R² + MAE 동시 | 임상 활용 형태에 맞춰 |

바이브코딩 관점에서는 위 표가 그대로 AI에게 전달할 사양서가 됩니다. "데이터는 scaffold split, 옵티마이저 AdamW(lr=1e-3), early stopping 인내 10 epoch, MC Dropout으로 불확실성 추정 포함" 같은 명세가 PyTorch Lightning·HuggingFace Trainer 등의 코드로 곧장 변환됩니다.

## 신약개발 적용

학습 절차의 차이가 실제 결과를 가른 사례는 풍부합니다.

- **MoleculeNet 벤치마크 (Wu *et al.*, 2018, *Chem. Sci.*)**: 동일 데이터셋에서 random split과 scaffold split의 평가 결과가 자주 0.1~0.2 AUC가량 차이 나며, scaffold split이 모델의 실제 신규 화합물 일반화 능력을 훨씬 보수적으로 측정한다는 점이 표준화되었습니다. 이후 **Therapeutics Data Commons (TDC, 2021)** 가 표준 분할·평가 프로토콜을 통일해 모델 비교의 재현성을 끌어올렸습니다.
- **Insilico Medicine Pharma.AI**: 분자 생성 단계에서 활성·합성 가능성·ADMET을 가중 합산한 다목적 손실을 강화학습 보상으로 통합해, 특발성 폐섬유증 후보 INS018_055를 타겟 발굴에서 임상 Phase II까지 약 30개월 만에 도달시켰습니다.
- **AlphaFold 2 / 3 (DeepMind, 2021/2024)**: 학습 단계에서 self-distillation·재활용 학습(recycling) 등 비표준적인 학습 절차가 성능에 결정적으로 기여했음이 후속 분석들에서 보고되었습니다. 아키텍처 못지않게 학습 레시피가 핵심임을 보여주는 대표 사례입니다.

기존 통계 기반 QSAR와의 학습 절차 비교는 다음과 같습니다.

| 항목 | 고전 QSAR | 딥러닝 기반 |
|------|----------|-----------|
| 데이터 분할 | random·k-fold가 흔함 | scaffold·time split이 표준화 진행 |
| 손실 | OLS / MLE | 다양한 손실 + 다목적 가중 |
| 정규화 | Ridge·Lasso 정도 | dropout·weight decay·data augmentation 조합 |
| 불확실성 | 신뢰구간(분포 가정) | MC Dropout·앙상블·Conformal Prediction |

## 창업 관점

Phase 1 단계에서는 간략히만 짚습니다. 학습 절차는 모델의 외피가 아니라 신뢰성의 근간입니다. 약학 전공자는 "scaffold split으로 평가했고, 가중치는 임상 위해도 기반이며, 불확실성은 MC Dropout으로 보고한다"고 설명할 수 있는 능력만으로도 빅파마·규제 기관과의 신뢰 자본을 확보합니다. 구체적 비즈니스 모델은 Phase 5에서 다룹니다.

## 오늘의 과제

1. **분할 전략 비교 정리 (40분)**: 관심 있는 ADMET·활성 데이터셋 1개(예: BBBP, hERG, BACE)를 골라, ① 데이터 규모, ② 양/음 비율, ③ scaffold split과 random split이 동일 모델에서 어떤 성능 차이를 보이는지를 MoleculeNet 또는 TDC 리더보드에서 찾아 표로 정리합니다. A4 1쪽.
2. **손실 함수 가중치 설계 (50분)**: 가상의 ADMET 다중 과제 모델(hERG, 간독성, logP, 용해도, CYP3A4 억제)을 가정하고, 각 과제의 손실 가중치를 약학적 위해도 기준으로 배정한 뒤 그 근거를 1줄씩 적습니다. 이어 가중치를 균등하게 둘 때와의 차이가 임상적으로 어떤 의미를 갖는지 반 페이지로 논증합니다.
3. **불확실성과 의사결정 메모 (30분)**: 분자 1,000개의 활성 예측에서 모델이 자신있는 상위 50개와 자신없는 하위 50개 중 wet-lab 검증을 어디에 우선 배치할지를, 탐색·활용(exploration·exploitation) 균형 관점에서 반 페이지로 주장합니다. 이 메모는 Phase 2의 활성 학습(active learning) 학습 시 재사용합니다.

## 참고 자료

- Wu, Z. *et al.* (2018). "MoleculeNet: a benchmark for molecular machine learning." *Chemical Science*, 9, 513-530. — 분자 ML 벤치마크와 분할 방식 표준화의 출발점.
- Huang, K. *et al.* (2021). "Therapeutics Data Commons: Machine learning datasets and tasks for drug discovery and development." *NeurIPS Datasets and Benchmarks*. — 표준화된 분할·평가 프로토콜 제공.
- Loshchilov, I., Hutter, F. (2019). "Decoupled weight decay regularization." *ICLR*. — AdamW 원전.
- Gal, Y., Ghahramani, Z. (2016). "Dropout as a Bayesian approximation: Representing model uncertainty in deep learning." *ICML*. — MC Dropout으로 불확실성을 추정하는 이론적 근거.
- Therapeutics Data Commons (tdcommons.ai) — 분할 코드와 리더보드를 즉시 활용 가능한 오픈 플랫폼.
