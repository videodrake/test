"""
Exercise 11: PyTorch 기초 - 텐서, 자동미분, 선형 회귀

목표:
  - PyTorch 텐서 생성 및 기본 연산
  - 분자 기술자 정규화 (StandardScaler 구현)
  - autograd를 이용한 자동 미분 이해
  - 순수 PyTorch로 선형 회귀 구현 (logS 예측)

배경:
  PyTorch는 딥러닝 기반 신약개발의 핵심 프레임워크입니다.
  텐서 연산과 자동미분(autograd)을 이해하면 분자 속성 예측
  모델의 내부 작동 원리를 파악할 수 있습니다.
  이 실습에서는 분자 기술자로부터 용해도(logS)를 예측하는
  선형 회귀를 PyTorch로 직접 구현합니다.

사용 라이브러리:
  torch, numpy
"""

import numpy as np
import torch


def generate_molecular_data(n_samples: int = 200) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    분자 기술자와 logS(용해도) 데이터를 생성합니다.

    Parameters
    ----------
    n_samples : int
        생성할 분자 수

    Returns
    -------
    X : np.ndarray, shape (n_samples, 4)
        분자 기술자 [MW, logP, HBD, TPSA]
    y : np.ndarray, shape (n_samples,)
        logS 값 (log mol/L)
    feature_names : list[str]
        기술자 이름
    """
    np.random.seed(42)
    feature_names = ["MW", "logP", "HBD", "TPSA"]

    MW = np.random.uniform(150, 500, n_samples)
    logP = np.random.normal(2.5, 1.5, n_samples)
    HBD = np.random.randint(0, 5, n_samples).astype(float)
    TPSA = np.random.uniform(20, 160, n_samples)

    logS = (
        0.8
        - 0.008 * MW
        - 0.6 * logP
        + 0.15 * HBD
        + 0.004 * TPSA
        + np.random.normal(0, 0.3, n_samples)
    )

    X = np.column_stack([MW, logP, HBD, TPSA])
    return X, logS, feature_names


# =============================================================================
# TODO 1: 텐서 생성 및 기본 연산
# =============================================================================
def create_molecular_tensors(
    X: np.ndarray, y: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    NumPy 배열을 PyTorch 텐서로 변환합니다.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        분자 기술자 행렬
    y : np.ndarray, shape (n_samples,)
        타깃 값 (logS)

    Returns
    -------
    X_tensor : torch.Tensor, dtype=float32, shape (n_samples, n_features)
    y_tensor : torch.Tensor, dtype=float32, shape (n_samples, 1)
        y는 (n_samples, 1) shape으로 reshape 해야 합니다.

    힌트:
      - torch.tensor(X, dtype=torch.float32) 또는 torch.from_numpy(X).float()
      - y_tensor = y_tensor.reshape(-1, 1)
    """
    # TODO: NumPy 배열을 float32 텐서로 변환하고, y를 (n, 1)로 reshape하세요
    pass


# =============================================================================
# TODO 2: 텐서 정규화 (StandardScaler)
# =============================================================================
def normalize_tensor(
    X: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    텐서를 Z-score 정규화합니다 (각 특성별 평균=0, 표준편차=1).

    분자 기술자는 스케일이 다릅니다:
      MW: 150~500, logP: -2~6, HBD: 0~4, TPSA: 20~160
    정규화를 통해 모든 기술자가 동등하게 학습에 기여하도록 합니다.

    Parameters
    ----------
    X : torch.Tensor, shape (n_samples, n_features)

    Returns
    -------
    X_norm : torch.Tensor
        정규화된 텐서
    mean : torch.Tensor, shape (n_features,)
        각 특성의 평균
    std : torch.Tensor, shape (n_features,)
        각 특성의 표준편차

    힌트:
      - mean = X.mean(dim=0)
      - std = X.std(dim=0)
      - X_norm = (X - mean) / std
    """
    # TODO: Z-score 정규화를 구현하세요
    pass


# =============================================================================
# TODO 3: autograd 체험 - 간단한 미분
# =============================================================================
def compute_gradient_example() -> tuple[float, float]:
    """
    autograd를 사용하여 간단한 함수의 기울기를 계산합니다.

    함수: f(x) = 3x^2 + 2x + 1
    미분: f'(x) = 6x + 2

    x = 2.0에서의 기울기를 autograd로 계산하고,
    해석적 해와 비교합니다.

    Returns
    -------
    autograd_result : float
        autograd로 계산한 f'(2.0)
    analytical_result : float
        해석적으로 계산한 f'(2.0) = 6*2 + 2 = 14.0

    힌트:
      - x = torch.tensor(2.0, requires_grad=True)
      - f = 3 * x**2 + 2 * x + 1
      - f.backward()
      - autograd_result = x.grad.item()
    """
    # TODO: autograd로 기울기를 계산하고 해석적 해와 함께 반환하세요
    pass


# =============================================================================
# TODO 4: 순수 PyTorch 선형 회귀
# =============================================================================
def train_linear_regression(
    X: torch.Tensor,
    y: torch.Tensor,
    lr: float = 0.01,
    n_epochs: int = 200,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """
    순수 PyTorch (autograd)로 선형 회귀를 학습합니다.
    nn.Module을 사용하지 않고, 직접 가중치와 편향을 정의합니다.

    모델: y_pred = X @ W + b
    손실: MSE = mean((y_pred - y)^2)

    Parameters
    ----------
    X : torch.Tensor, shape (n_samples, n_features)
        정규화된 기술자 텐서
    y : torch.Tensor, shape (n_samples, 1)
        타깃 텐서
    lr : float
        학습률
    n_epochs : int
        에폭 수

    Returns
    -------
    W : torch.Tensor, shape (n_features, 1)
        학습된 가중치
    b : torch.Tensor, shape (1,)
        학습된 편향
    losses : list[float]
        에폭별 손실 기록

    힌트:
      - torch.manual_seed(42)
      - W = torch.randn(n_features, 1, requires_grad=True)
      - b = torch.zeros(1, requires_grad=True)
      - 각 에폭에서:
        1. y_pred = X @ W + b
        2. loss = ((y_pred - y) ** 2).mean()
        3. loss.backward()
        4. with torch.no_grad():
             W -= lr * W.grad
             b -= lr * b.grad
        5. W.grad.zero_()
           b.grad.zero_()
    """
    # TODO: 순수 PyTorch 선형 회귀를 구현하세요
    pass


# =============================================================================
# TODO 5: 예측 및 평가
# =============================================================================
def evaluate_predictions(
    X: torch.Tensor, y: torch.Tensor, W: torch.Tensor, b: torch.Tensor
) -> dict[str, float]:
    """
    학습된 선형 모델의 예측 성능을 평가합니다.

    Parameters
    ----------
    X : torch.Tensor
        기술자 텐서
    y : torch.Tensor
        실제 logS 값
    W : torch.Tensor
        학습된 가중치
    b : torch.Tensor
        학습된 편향

    Returns
    -------
    metrics : dict[str, float]
        {"mse": ..., "mae": ..., "r2": ...}

    힌트:
      - with torch.no_grad():
          y_pred = X @ W + b
      - MSE = ((y_pred - y) ** 2).mean().item()
      - MAE = (y_pred - y).abs().mean().item()
      - R² = 1 - SS_res / SS_tot
        SS_res = ((y - y_pred) ** 2).sum()
        SS_tot = ((y - y.mean()) ** 2).sum()
    """
    # TODO: MSE, MAE, R²를 계산하세요
    pass


# =============================================================================
# 메인 실행 및 자가 검증
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("Exercise 11: PyTorch 기초 - 텐서, 자동미분, 선형 회귀")
    print("=" * 60)

    # 1. 데이터 생성
    X_np, y_np, feature_names = generate_molecular_data(200)
    print(f"\n[데이터] 분자 수: {X_np.shape[0]}, 기술자: {feature_names}")

    # 2. 텐서 변환
    result = create_molecular_tensors(X_np, y_np)
    assert result is not None, "create_molecular_tensors()가 None을 반환했습니다."
    X_t, y_t = result
    assert X_t.dtype == torch.float32, f"X 텐서 dtype이 float32이어야 하지만 {X_t.dtype}입니다."
    assert y_t.shape == (200, 1), f"y 텐서 shape이 (200,1)이어야 하지만 {y_t.shape}입니다."
    print(f"\n[텐서 변환] X: {X_t.shape}, y: {y_t.shape}, dtype: {X_t.dtype}")

    # 3. 정규화
    norm_result = normalize_tensor(X_t)
    assert norm_result is not None, "normalize_tensor()가 None을 반환했습니다."
    X_norm, mean, std = norm_result
    assert X_norm.shape == X_t.shape
    assert torch.allclose(X_norm.mean(dim=0), torch.zeros(4), atol=1e-5), \
        "정규화 후 평균이 0이 아닙니다."
    assert torch.allclose(X_norm.std(dim=0), torch.ones(4), atol=0.1), \
        "정규화 후 표준편차가 1이 아닙니다."
    print(f"[정규화] 평균: {X_norm.mean(dim=0).tolist()}")
    print(f"         표준편차: {X_norm.std(dim=0).tolist()}")

    # 4. autograd 체험
    grad_result = compute_gradient_example()
    assert grad_result is not None, "compute_gradient_example()이 None을 반환했습니다."
    auto_grad, anal_grad = grad_result
    assert abs(auto_grad - anal_grad) < 1e-4, \
        f"autograd ({auto_grad})와 해석적 해 ({anal_grad})가 일치하지 않습니다."
    print(f"\n[Autograd] f'(2.0) = {auto_grad:.4f} (해석적: {anal_grad:.4f})")

    # 5. 선형 회귀 학습
    lr_result = train_linear_regression(X_norm, y_t, lr=0.01, n_epochs=200)
    assert lr_result is not None, "train_linear_regression()이 None을 반환했습니다."
    W, b, losses = lr_result
    assert W.shape == (4, 1), f"가중치 shape이 (4,1)이어야 하지만 {W.shape}입니다."
    assert len(losses) == 200, f"에폭별 손실이 200개여야 하지만 {len(losses)}개입니다."
    assert losses[-1] < losses[0], "학습이 진행되면서 손실이 감소해야 합니다."
    print(f"\n[선형 회귀 학습]")
    print(f"  초기 손실: {losses[0]:.4f}")
    print(f"  최종 손실: {losses[-1]:.4f}")
    print(f"  가중치: {W.detach().flatten().tolist()}")
    print(f"  편향: {b.item():.4f}")

    # 6. 평가
    metrics = evaluate_predictions(X_norm, y_t, W, b)
    assert metrics is not None, "evaluate_predictions()이 None을 반환했습니다."
    assert "mse" in metrics and "mae" in metrics and "r2" in metrics
    assert metrics["r2"] > 0.5, f"R²가 0.5 이상이어야 하지만 {metrics['r2']:.4f}입니다."
    print(f"\n[평가]")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f} log mol/L")
    print(f"  R²:  {metrics['r2']:.4f}")

    print("\n" + "=" * 60)
    print("모든 검증을 통과했습니다!")
    print("=" * 60)
