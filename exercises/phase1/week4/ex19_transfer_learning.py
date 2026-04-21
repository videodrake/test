"""
Exercise 19: 전이학습 - 사전학습 + 파인튜닝으로 소규모 데이터 극복

목표:
  - 대규모 데이터로 분자 표현 모델을 사전학습(pretrain)
  - 소규모 타겟 데이터로 파인튜닝(finetune)
  - Scratch vs Finetune vs Frozen backbone 성능 비교
  - 전이학습이 신약개발에서 효과적인 이유 이해

배경:
  신약개발에서 특정 타겟 단백질에 대한 활성 데이터는 수백 개 수준으로
  매우 부족합니다. 반면 ChEMBL 등 공공 데이터베이스에는 수백만 개의
  분자-활성 데이터가 있습니다.
  전이학습은 대규모 일반 데이터로 학습한 분자 표현 지식을
  소규모 타겟 데이터에 전이하여, 적은 데이터로도 높은 예측 성능을
  달성하는 핵심 전략입니다.

사용 라이브러리:
  torch, torch.nn, numpy
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy


def generate_pretrain_data(n_samples: int = 5000, n_features: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """
    대규모 사전학습 데이터를 생성합니다.
    (ChEMBL의 다양한 타겟에 대한 일반적인 분자 활성 시뮬레이션)

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # 복잡한 비선형 관계
    true_w = np.random.randn(n_features) * 0.5
    y = X @ true_w
    y += 0.3 * np.sin(X[:, 0] * 3) * X[:, 1]
    y += 0.2 * X[:, 2] ** 2
    y += np.random.normal(0, 0.3, n_samples)
    y = y.astype(np.float32)

    return X, y


def generate_finetune_data(
    n_samples: int = 100, n_features: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """
    소규모 파인튜닝 데이터를 생성합니다.
    (특정 키나아제에 대한 소규모 활성 데이터 시뮬레이션)

    사전학습 데이터와 유사한 패턴을 가지지만, 타겟 특이적 차이가 있습니다.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    """
    np.random.seed(123)
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # 사전학습 데이터와 부분적으로 유사한 관계 + 타겟 특이적 패턴
    true_w = np.random.randn(n_features) * 0.5
    np.random.seed(42)
    pretrain_w = np.random.randn(n_features) * 0.5
    # 50%는 공유 패턴, 50%는 타겟 특이적
    mixed_w = 0.5 * pretrain_w + 0.5 * true_w

    np.random.seed(123)
    y = X @ mixed_w
    y += 0.3 * np.sin(X[:, 0] * 3) * X[:, 1]  # 공유 패턴
    y += 0.4 * X[:, 3] * X[:, 4]               # 타겟 특이적 패턴
    y += np.random.normal(0, 0.3, n_samples)
    y = y.astype(np.float32)

    return X, y


class MolecularEncoder(nn.Module):
    """분자 표현 학습 인코더 (backbone)."""

    def __init__(self, n_features: int = 50, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class PredictionHead(nn.Module):
    """예측 헤드 (task-specific)."""

    def __init__(self, in_dim: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class FullModel(nn.Module):
    """인코더 + 헤드 전체 모델."""

    def __init__(self, encoder: MolecularEncoder, head: PredictionHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.head(features)


# =============================================================================
# TODO 1: 사전학습 함수
# =============================================================================
def pretrain_encoder(
    X: np.ndarray,
    y: np.ndarray,
    n_features: int = 50,
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
) -> FullModel:
    """
    대규모 데이터로 인코더(backbone)를 사전학습합니다.

    Parameters
    ----------
    X, y : np.ndarray
        사전학습 데이터
    n_features : int
        입력 차원
    n_epochs : int
    batch_size : int
    lr : float

    Returns
    -------
    pretrained_model : FullModel
        사전학습된 모델 (encoder + head)

    힌트:
      - encoder = MolecularEncoder(n_features)
      - head = PredictionHead()
      - model = FullModel(encoder, head)
      - 일반적인 학습 루프 (MSELoss + Adam)
    """
    # TODO: 사전학습을 구현하세요
    pass


# =============================================================================
# TODO 2: 처음부터 학습 (Scratch)
# =============================================================================
def train_from_scratch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_features: int = 50,
    n_epochs: int = 100,
    lr: float = 0.001,
) -> dict[str, float]:
    """
    사전학습 없이 소규모 데이터로만 학습합니다.

    Returns
    -------
    metrics : dict
        {"mae": float, "r2": float, "final_loss": float}

    힌트:
      - 새로운 MolecularEncoder + PredictionHead 생성
      - FullModel로 결합
      - 학습 후 테스트 데이터로 평가
    """
    # TODO: 처음부터 학습을 구현하세요
    pass


# =============================================================================
# TODO 3: 파인튜닝 (전체 모델)
# =============================================================================
def finetune_full(
    pretrained_model: FullModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_epochs: int = 100,
    lr: float = 0.0005,
) -> dict[str, float]:
    """
    사전학습된 모델의 전체를 파인튜닝합니다.

    Parameters
    ----------
    pretrained_model : FullModel
        사전학습된 모델
    X_train, y_train : np.ndarray
        파인튜닝 학습 데이터 (소규모)
    X_test, y_test : np.ndarray
        테스트 데이터
    n_epochs : int
    lr : float
        파인튜닝 학습률 (사전학습보다 작게)

    Returns
    -------
    metrics : dict
        {"mae": float, "r2": float, "final_loss": float}

    힌트:
      - model = copy.deepcopy(pretrained_model)  # 원본 보존
      - 새로운 PredictionHead로 교체: model.head = PredictionHead()
      - 작은 학습률로 전체 모델 학습
    """
    # TODO: 전체 파인튜닝을 구현하세요
    pass


# =============================================================================
# TODO 4: 동결 backbone + 헤드만 학습
# =============================================================================
def finetune_frozen_backbone(
    pretrained_model: FullModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_epochs: int = 100,
    lr: float = 0.001,
) -> dict[str, float]:
    """
    backbone(encoder)을 동결하고 head만 학습합니다.

    Parameters
    ----------
    pretrained_model : FullModel
    X_train, y_train, X_test, y_test : np.ndarray
    n_epochs : int
    lr : float

    Returns
    -------
    metrics : dict
        {"mae": float, "r2": float, "final_loss": float}

    힌트:
      - model = copy.deepcopy(pretrained_model)
      - model.head = PredictionHead()  # 새 헤드
      - encoder 파라미터 동결:
        for param in model.encoder.parameters():
            param.requires_grad = False
      - optimizer는 head 파라미터만:
        optimizer = Adam(model.head.parameters(), lr=lr)
    """
    # TODO: 동결 backbone 파인튜닝을 구현하세요
    pass


# =============================================================================
# 유틸리티: 평가 함수
# =============================================================================
def compute_metrics(
    model: nn.Module, X: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    """모델의 MAE와 R²를 계산합니다."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        y_pred = model(X_t).numpy().flatten()

    mae = np.mean(np.abs(y - y_pred))
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {"mae": float(mae), "r2": float(r2)}


# =============================================================================
# 메인 실행 및 자가 검증
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("Exercise 19: 전이학습 - Pretrain + Finetune")
    print("=" * 60)

    # 1. 데이터 생성
    X_pretrain, y_pretrain = generate_pretrain_data(5000)
    X_finetune, y_finetune = generate_finetune_data(100)

    # 파인튜닝 데이터를 train/test로 분할
    n_train = 70
    X_ft_train, X_ft_test = X_finetune[:n_train], X_finetune[n_train:]
    y_ft_train, y_ft_test = y_finetune[:n_train], y_finetune[n_train:]

    print(f"\n[데이터]")
    print(f"  사전학습: {X_pretrain.shape[0]}개 (대규모 ChEMBL 시뮬레이션)")
    print(f"  파인튜닝 학습: {X_ft_train.shape[0]}개 (소규모 타겟 데이터)")
    print(f"  파인튜닝 테스트: {X_ft_test.shape[0]}개")

    # 2. 사전학습
    print(f"\n--- 사전학습 ---")
    pretrained = pretrain_encoder(X_pretrain, y_pretrain, n_epochs=50, lr=0.001)
    assert pretrained is not None, "pretrain_encoder()가 None을 반환했습니다."
    assert isinstance(pretrained, FullModel)
    pretrain_metrics = compute_metrics(pretrained, X_pretrain[:500], y_pretrain[:500])
    print(f"  사전학습 데이터 MAE: {pretrain_metrics['mae']:.4f}")
    print(f"  사전학습 데이터 R²: {pretrain_metrics['r2']:.4f}")

    # 3. Scratch 학습
    print(f"\n--- Scratch (처음부터 학습) ---")
    torch.manual_seed(42)
    scratch_metrics = train_from_scratch(
        X_ft_train, y_ft_train, X_ft_test, y_ft_test, n_epochs=100
    )
    assert scratch_metrics is not None, "train_from_scratch()가 None을 반환했습니다."
    assert "mae" in scratch_metrics and "r2" in scratch_metrics
    print(f"  테스트 MAE: {scratch_metrics['mae']:.4f}")
    print(f"  테스트 R²: {scratch_metrics['r2']:.4f}")

    # 4. Full Finetune
    print(f"\n--- Full Finetune (전체 파인튜닝) ---")
    torch.manual_seed(42)
    finetune_metrics = finetune_full(
        pretrained, X_ft_train, y_ft_train, X_ft_test, y_ft_test,
        n_epochs=100, lr=0.0005
    )
    assert finetune_metrics is not None, "finetune_full()이 None을 반환했습니다."
    print(f"  테스트 MAE: {finetune_metrics['mae']:.4f}")
    print(f"  테스트 R²: {finetune_metrics['r2']:.4f}")

    # 5. Frozen Backbone
    print(f"\n--- Frozen Backbone (동결 인코더 + 헤드만 학습) ---")
    torch.manual_seed(42)
    frozen_metrics = finetune_frozen_backbone(
        pretrained, X_ft_train, y_ft_train, X_ft_test, y_ft_test,
        n_epochs=100, lr=0.001
    )
    assert frozen_metrics is not None, "finetune_frozen_backbone()이 None을 반환했습니다."
    print(f"  테스트 MAE: {frozen_metrics['mae']:.4f}")
    print(f"  테스트 R²: {frozen_metrics['r2']:.4f}")

    # 6. 종합 비교
    print(f"\n{'='*60}")
    print(f"{'전략':25s} {'MAE':>8s} {'R²':>8s}")
    print(f"{'-'*60}")
    print(f"{'Scratch (70개로 학습)':25s} {scratch_metrics['mae']:8.4f} {scratch_metrics['r2']:8.4f}")
    print(f"{'Full Finetune':25s} {finetune_metrics['mae']:8.4f} {finetune_metrics['r2']:8.4f}")
    print(f"{'Frozen Backbone':25s} {frozen_metrics['mae']:8.4f} {frozen_metrics['r2']:8.4f}")
    print(f"{'='*60}")

    print("\n" + "=" * 60)
    print("모든 검증을 통과했습니다!")
    print("=" * 60)
