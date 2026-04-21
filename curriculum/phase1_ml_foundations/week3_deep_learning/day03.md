# Day 3: 학습 루프와 최적화

## 학습 목표
- PyTorch 학습 루프(Training Loop)의 전체 구조를 마스터한다
- 옵티마이저(SGD, Adam)와 학습률 스케줄링을 이해한다
- 배치 처리(mini-batch)와 DataLoader를 사용한다
- Early Stopping으로 과적합을 방지한다

## 이론 (1시간)

### 완전한 학습 루프

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 데이터 준비
dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
loader = DataLoader(dataset, batch_size=32, shuffle=True)

val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
val_loader = DataLoader(val_dataset, batch_size=64)

model = MoleculeNet(n_features=5)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
for epoch in range(200):
    # --- 학습 ---
    model.train()
    train_loss = 0
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(X_batch)
    train_loss /= len(dataset)

    # --- 검증 ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            val_loss += criterion(y_pred, y_batch).item() * len(X_batch)
    val_loss /= len(val_dataset)

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
```

### model.train() vs model.eval()

| 모드 | Dropout | BatchNorm | 용도 |
|------|---------|-----------|------|
| `model.train()` | 활성 | 이동 통계 업데이트 | 학습 시 |
| `model.eval()` | 비활성 | 고정 통계 사용 | 검증/추론 시 |

> 반드시 검증/테스트 전에 `model.eval()`과 `torch.no_grad()`를 사용하세요.

### 옵티마이저 비교

```python
# SGD: 기본, 모멘텀 추가 권장
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam: 가장 범용적, 대부분의 경우 첫 선택
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# AdamW: weight decay(L2 정규화)가 올바르게 적용
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

### 학습률 스케줄링

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

for epoch in range(200):
    # ... 학습 ...
    scheduler.step(val_loss)  # val_loss가 개선 안 되면 lr 절반
```

### Early Stopping

```python
best_val_loss = float('inf')
patience = 20
counter = 0

for epoch in range(500):
    # ... 학습 & 검증 ...

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# 최적 모델 복원
model.load_state_dict(torch.load('best_model.pt'))
```

### 재현성 보장

```python
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)
```

## 실습 (1.5시간)

### Exercise 13: 완전한 학습 루프

파일: `exercises/phase1/week3/ex13_training_loop.py`

**과제:**
1. DataLoader로 배치 처리 구현 (batch_size=32)
2. train/validation 분리 + 양쪽 loss 기록
3. 학습 곡선 시각화 (train_loss vs val_loss)
4. Early Stopping 구현 (patience=15)
5. ReduceLROnPlateau 스케줄러 적용

## 참고 자료
- PyTorch: DataLoader 튜토리얼
- "Practical Deep Learning" (학습률 찾기 전략)
