# Day 2: 피드포워드 신경망 구축

## 학습 목표
- `nn.Module`로 신경망을 정의한다
- 활성 함수(ReLU, Sigmoid)와 레이어의 역할을 이해한다
- 분자 물성 예측용 피드포워드 네트워크를 구축한다

## 이론 (1시간)

### 신경망의 구조
```
입력(기술자) → [선형 변환 + 활성 함수] × N → 출력(예측값)

분자 기술자(5개) → Dense(64) → ReLU → Dense(32) → ReLU → Dense(1) → logS 예측
```

### nn.Module로 신경망 정의

```python
import torch
import torch.nn as nn

class MoleculeNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

model = MoleculeNet(n_features=5)
print(model)
print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
```

### 활성 함수의 역할

| 활성 함수 | 수식 | 용도 |
|-----------|------|------|
| ReLU | max(0, x) | 은닉층 기본, 빠르고 안정적 |
| Sigmoid | 1/(1+e⁻ˣ) | 이진 분류 출력 (0~1 확률) |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | 출력이 -1~1 필요할 때 |
| LeakyReLU | max(0.01x, x) | ReLU의 "죽은 뉴런" 문제 해결 |

> **생화학 비유**: 활성 함수는 효소의 활성화 에너지 장벽과 비슷합니다. 입력(기질)이 특정 수준을 넘어야 출력(반응)이 발생합니다. ReLU는 임계값(0) 아래를 차단하는 on/off 스위치입니다.

### 손실 함수와 옵티마이저

```python
# 회귀: MSE Loss
criterion = nn.MSELoss()

# 분류: Binary Cross-Entropy
criterion_cls = nn.BCEWithLogitsLoss()

# 옵티마이저: Adam (가장 범용적)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

### Week 2 sklearn 모델 vs 신경망 비교

```python
# sklearn: 한 줄로 학습
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor().fit(X_train, y_train)

# PyTorch: 학습 과정을 직접 제어
model = MoleculeNet(5)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_t = torch.FloatTensor(X_train)
y_t = torch.FloatTensor(y_train)

for epoch in range(100):
    y_pred = model(X_t)
    loss = criterion(y_pred, y_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**딥러닝의 장점**: 아키텍처를 자유롭게 설계 가능 (GNN, Transformer 등). Week 4에서 활용합니다.

## 실습 (1.5시간)

### Exercise 12: 피드포워드 신경망

파일: `exercises/phase1/week3/ex12_feedforward_nn.py`

**과제:**
1. `nn.Module`로 MoleculeNet 클래스 정의 (입력→64→32→1)
2. 분자 기술자 데이터로 용해도 예측 학습
3. 학습 곡선 (epoch vs loss) 그리기
4. sklearn RandomForest와 성능 비교 (RMSE, R²)

## 참고 자료
- PyTorch: nn.Module 튜토리얼
- "Deep Learning for Molecular Design" 리뷰
