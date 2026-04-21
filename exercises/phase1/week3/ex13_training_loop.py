"""
Exercise 13: 학습 루프 심화 - DataLoader, 검증, Early Stopping

목표:
  - torch.utils.data.DataLoader를 이용한 미니배치 학습
  - 학습/검증 분할 및 에폭별 성능 모니터링
  - 손실 곡선(loss curve) 기록 및 분석
  - Early stopping 구현
  - 학습률 스케줄러(LR Scheduler) 적용

배경:
  실제 신약개발 프로젝트에서 모델 학습은 수천~수만 분자로 이루어집니다.
  효율적인 미니배치 학습과 과적합 방지를 위한 early stopping은
  안정적인 QSAR/QSPR 모델 개발에 필수적입니다.
  검증 손실을 모니터링하여 모델이 새로운 분자에 대해 잘 일반화하는
  시점을 찾는 것이 핵심입니다.

사용 라이브러리:
  torch, torch.nn, torch.utils.data, numpy
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def generate_bioactivity_data(n_samples: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """
    분자 기술자와 생물학적 활성(pIC50) 데이터를 생성합니다.

    Parameters
    ----------
    n_samples : int
        분자 수

    Returns
    -------
    X : np.ndarray, shape (n_samples, 10)
        10개 분자 기술자
    y : np.ndarray, shape (n_samples,)
        pIC50 값
    """
    np.random.seed(42)

    X = np.random.randn(n_samples, 10).astype(np.float32)

    # 비선형 관계 포함
    true_weights = np.array([0.5, -0.3, 0.8, -0.2, 0.6, -0.4, 0.3, -0.1, 0.7, -0.5])
    y = X @ true_weights
    y += 0.3 * X[:, 0] * X[:, 2]  # 상호작용 항
    y += 0.2 * X[:, 1] ** 2       # 비선형 항
    y += np.random.normal(0, 0.3, n_samples).astype(np.float32)
    y = y + 6.0  # pIC50 범위로 이동

    return X, y.astype(np.float32)


class BioactivityMLP(nn.Module):
    """생물학적 활성 예측 MLP 모델 (사전 정의)."""

    def __init__(self, n_features: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# TODO 1: DataLoader 생성
# =============================================================================
def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader]:
    """
    학습/검증 DataLoader를 생성합니다.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        학습 데이터
    X_val, y_val : np.ndarray
        검증 데이터
    batch_size : int
        미니배치 크기

    Returns
    -------
    train_loader : DataLoader
        shuffle=True인 학습용 DataLoader
    val_loader : DataLoader
        shuffle=False인 검증용 DataLoader

    힌트:
      - 텐서 변환: torch.tensor(X_train, dtype=torch.float32)
      - y를 (n, 1)로 reshape
      - TensorDataset(X_tensor, y_tensor)
      - DataLoader(dataset, batch_size=batch_size, shuffle=True/False)
    """
    # TODO: DataLoader를 생성하세요
    pass


# =============================================================================
# TODO 2: 1 에폭 학습
# =============================================================================
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """
    1 에폭 동안 모델을 학습하고 평균 손실을 반환합니다.

    Parameters
    ----------
    model : nn.Module
    train_loader : DataLoader
    criterion : 손실 함수 (예: nn.MSELoss)
    optimizer : 옵티마이저

    Returns
    -------
    avg_loss : float
        에폭 평균 손실

    힌트:
      - model.train()
      - total_loss = 0.0
      - for X_batch, y_batch in train_loader:
          optimizer.zero_grad()
          y_pred = model(X_batch)
          loss = criterion(y_pred, y_batch)
          loss.backward()
          optimizer.step()
          total_loss += loss.item() * len(X_batch)
      - avg_loss = total_loss / len(train_loader.dataset)
    """
    # TODO: 1 에폭 학습을 구현하세요
    pass


# =============================================================================
# TODO 3: 검증
# =============================================================================
def validate(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module
) -> float:
    """
    검증 데이터에 대한 평균 손실을 계산합니다.

    Parameters
    ----------
    model : nn.Module
    val_loader : DataLoader
    criterion : 손실 함수

    Returns
    -------
    avg_loss : float
        검증 평균 손실

    힌트:
      - model.eval()
      - with torch.no_grad():
          for X_batch, y_batch in val_loader:
              ...
    """
    # TODO: 검증 루프를 구현하세요
    pass


# =============================================================================
# TODO 4: Early Stopping이 포함된 전체 학습 루프
# =============================================================================
def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 200,
    lr: float = 0.001,
    patience: int = 15,
) -> dict:
    """
    Early stopping과 LR scheduler가 포함된 전체 학습 루프입니다.

    Parameters
    ----------
    model : nn.Module
    train_loader, val_loader : DataLoader
    n_epochs : int
        최대 에폭 수
    lr : float
        초기 학습률
    patience : int
        검증 손실 미개선 허용 에폭 수

    Returns
    -------
    history : dict
        {
            "train_losses": list[float],  -- 에폭별 학습 손실
            "val_losses": list[float],    -- 에폭별 검증 손실
            "best_epoch": int,            -- 최적 에폭 (0-indexed)
            "stopped_epoch": int,         -- 조기 종료된 에폭
        }

    힌트:
      - criterion = nn.MSELoss()
      - optimizer = torch.optim.Adam(model.parameters(), lr=lr)
      - scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5)
      - Early stopping 로직:
        best_val_loss = float('inf')
        wait = 0
        for epoch in range(n_epochs):
            train_loss = train_one_epoch(...)
            val_loss = validate(...)
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break  # 조기 종료
    """
    # TODO: Early stopping + LR scheduler 학습 루프를 구현하세요
    pass


# =============================================================================
# 메인 실행 및 자가 검증
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("Exercise 13: 학습 루프 - DataLoader, 검증, Early Stopping")
    print("=" * 60)

    # 1. 데이터 준비
    X, y = generate_bioactivity_data(1000)
    n_train = 700
    n_val = 150
    X_train, X_val, X_test = X[:n_train], X[n_train:n_train+n_val], X[n_train+n_val:]
    y_train, y_val, y_test = y[:n_train], y[n_train:n_train+n_val], y[n_train+n_val:]
    print(f"\n[데이터] 학습: {len(X_train)}, 검증: {len(X_val)}, 테스트: {len(X_test)}")

    # 2. DataLoader 생성
    loaders = create_dataloaders(X_train, y_train, X_val, y_val, batch_size=32)
    assert loaders is not None, "create_dataloaders()가 None을 반환했습니다."
    train_loader, val_loader = loaders

    # DataLoader 검증
    batch_X, batch_y = next(iter(train_loader))
    assert batch_X.shape[0] == 32, f"배치 크기가 32이어야 하지만 {batch_X.shape[0]}입니다."
    assert batch_X.shape[1] == 10
    assert batch_y.shape == (32, 1)
    print(f"[DataLoader] 배치 shape: X={batch_X.shape}, y={batch_y.shape}")

    # 3. 단일 에폭 학습/검증 테스트
    model = BioactivityMLP(n_features=10)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
    assert train_loss is not None, "train_one_epoch()이 None을 반환했습니다."
    assert train_loss > 0, f"학습 손실이 양수여야 합니다: {train_loss}"
    print(f"\n[단일 에폭] 학습 손실: {train_loss:.4f}")

    val_loss = validate(model, val_loader, criterion)
    assert val_loss is not None, "validate()가 None을 반환했습니다."
    assert val_loss > 0
    print(f"             검증 손실: {val_loss:.4f}")

    # 4. Early Stopping 전체 학습
    model2 = BioactivityMLP(n_features=10)
    history = train_with_early_stopping(
        model2, train_loader, val_loader,
        n_epochs=200, lr=0.001, patience=15
    )
    assert history is not None, "train_with_early_stopping()이 None을 반환했습니다."
    assert "train_losses" in history and "val_losses" in history
    assert "best_epoch" in history and "stopped_epoch" in history
    assert len(history["train_losses"]) == len(history["val_losses"])
    assert history["stopped_epoch"] <= 200

    print(f"\n[Early Stopping 학습]")
    print(f"  총 에폭: {history['stopped_epoch']}/{200}")
    print(f"  최적 에폭: {history['best_epoch']}")
    print(f"  최종 학습 손실: {history['train_losses'][-1]:.4f}")
    print(f"  최종 검증 손실: {history['val_losses'][-1]:.4f}")
    print(f"  최적 검증 손실: {min(history['val_losses']):.4f}")

    # 손실 곡선 추세 확인
    early_val = np.mean(history["val_losses"][:5])
    late_val = np.mean(history["val_losses"][-5:])
    assert late_val < early_val, "검증 손실이 학습 초기보다 감소해야 합니다."

    print("\n" + "=" * 60)
    print("모든 검증을 통과했습니다!")
    print("=" * 60)
