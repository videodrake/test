# Day 1: PyTorch 텐서와 자동미분

## 학습 목표
- PyTorch 텐서(Tensor)의 생성과 연산을 익힌다
- 자동미분(Autograd)의 원리를 이해한다
- NumPy와 PyTorch 간 데이터 변환을 수행한다

## 이론 (1시간)

### 왜 PyTorch인가?
scikit-learn은 전통적 ML에 강하지만, **딥러닝**에는 한계가 있습니다:
- 사용자 정의 아키텍처 (GNN, Transformer 등) 불가
- GPU 가속 미지원
- 자동미분 없음

PyTorch는 이 모든 것을 제공하면서도 **Python 친화적**입니다.

### 텐서 기초

```python
import torch
import numpy as np

# 텐서 생성
x = torch.tensor([1.0, 2.0, 3.0])          # 리스트에서
x = torch.zeros(3, 4)                        # 0으로 채운 3x4
x = torch.randn(3, 4)                        # 정규분포 랜덤
x = torch.from_numpy(np.array([1, 2, 3]))    # NumPy에서 변환

# NumPy ↔ PyTorch (메모리 공유!)
arr = x.numpy()           # Tensor → NumPy
t = torch.from_numpy(arr) # NumPy → Tensor

# 디바이스 (GPU 가속)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)
```

### 텐서 연산 — NumPy와 거의 동일

```python
# 기본 연산
a = torch.randn(3, 4)
b = torch.randn(3, 4)
c = a + b          # 원소별 덧셈
d = a @ b.T        # 행렬 곱 (matmul)
e = a * b          # 원소별 곱 (Hadamard)

# 분자 데이터 예시: 물성 정규화
properties = torch.tensor([
    [180.16, 1.19, 63.60],   # 아스피린
    [151.16, 0.46, 49.33],   # 아세트아미노펜
    [206.28, 3.18, 49.33],   # 이부프로펜
])
mean = properties.mean(dim=0)
std = properties.std(dim=0)
normalized = (properties - mean) / std
```

### 자동미분 (Autograd) — 딥러닝의 핵심

딥러닝은 **경사 하강법**으로 학습합니다. 이를 위해 손실 함수의 **기울기(gradient)**가 필요합니다.

```python
# requires_grad=True → 이 텐서에 대한 기울기를 추적
x = torch.tensor([2.0, 3.0], requires_grad=True)

# 순전파: 계산 수행
y = x ** 2 + 3 * x + 1  # y = x² + 3x + 1

# 역전파: 기울기 계산
loss = y.sum()
loss.backward()

# x의 기울기: dy/dx = 2x + 3
print(x.grad)  # tensor([7., 9.])  → 2*2+3=7, 2*3+3=9
```

### 자동미분의 실제 활용

```python
# 간단한 선형 회귀 from scratch
torch.manual_seed(42)

# 분자 기술자 → 용해도 예측 (간단 버전)
X = torch.randn(100, 5)      # 100개 분자, 5개 기술자
y_true = X @ torch.tensor([0.5, -1.2, 0.3, 0.8, -0.5]) + 0.1  # 실제 관계

# 학습 가능한 파라미터
w = torch.randn(5, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr = 0.01
for epoch in range(100):
    # 순전파
    y_pred = X @ w + b
    loss = ((y_pred - y_true) ** 2).mean()  # MSE

    # 역전파
    loss.backward()

    # 파라미터 업데이트 (경사 하강법)
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

print(f"학습된 가중치: {w.detach().numpy().round(2)}")
```

## 실습 (1.5시간)

### Exercise 11: PyTorch 텐서와 자동미분

파일: `exercises/phase1/week3/ex11_pytorch_basics.py`

**과제:**
1. 다양한 방식으로 텐서 생성 (zeros, randn, from_numpy)
2. 분자 물성 텐서의 정규화 (mean, std)
3. 자동미분으로 간단한 함수의 기울기 계산
4. 경사 하강법으로 선형 회귀 직접 구현

## 참고 자료
- PyTorch 공식 튜토리얼: Tensors
- PyTorch Autograd 문서
