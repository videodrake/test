"""
Exercise 14: 정규화 기법 비교 - Dropout, Weight Decay, 학습 곡선

목표:
  - 정규화 없는 모델 vs Dropout+Weight Decay 모델 비교
  - 학습 곡선(learning curve) 분석으로 과적합 진단
  - 정규화가 QSAR 모델의 일반화에 미치는 영향 이해

배경:
  신약개발의 QSAR 모델링에서 학습 데이터는 종종 수백 개 수준으로
  제한적입니다. 이런 상황에서 복잡한 신경망은 학습 데이터에 과적합되어
  새로운 화합물에 대한 예측 성능이 떨어집니다.
  Dropout과 Weight Decay는 과적합을 방지하는 대표적인 정규화 기법입니다.

사용 라이브러리:
  torch, torch.nn, numpy
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def generate_overfit_data(
    n_train: int = 150, n_val: int = 200, n_features: int = 50
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    과적합이 발생하기 쉬운 데이터를 생성합니다.
    (적은 학습 샘플, 높은 차원)

    실제 신약개발에서 high-throughput screening 데이터가 부족하고
    분자 기술자 차원이 높은 상황을 시뮬레이션합니다.

    Returns
    -------
    X_train, y_train, X_val, y_val : np.ndarray
    """
    np.random.seed(42)

    # 실제 유효한 특성은 10개뿐, 나머지 40개는 노이즈
    true_dim = 10
    true_weights = np.zeros(n_features)
    true_weights[:true_dim] = np.random.randn(true_dim)

    X_train = np.random.randn(n_train, n_features).astype(np.float32)
    y_train = (X_train @ true_weights + np.random.normal(0, 0.5, n_train)).astype(np.float32)

    X_val = np.random.randn(n_val, n_features).astype(np.float32)
    y_val = (X_val @ true_weights + np.random.normal(0, 0.5, n_val)).astype(np.float32)

    return X_train, y_train, X_val, y_val


# =============================================================================
# TODO 1: 정규화 없는 모델 정의
# =============================================================================
class UnregularizedModel(nn.Module):
    """
    정규화가 없는 큰 MLP 모델.
    과적합을 유발하기 위해 파라미터가 많은 구조를 사용합니다.

    아키텍처:
      Input(50) -> Linear(256) -> ReLU -> Linear(128) -> ReLU
      -> Linear(64) -> ReLU -> Linear(1)

    힌트:
      - nn.Sequential로 레이어를 정의하세요
    """

    def __init__(self, n_features: int = 50):
        super().__init__()
        # TODO: 정규화 없는 모델을 정의하세요
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: 순전파
        pass


# =============================================================================
# TODO 2: Dropout + Weight Decay 모델 정의
# =============================================================================
class RegularizedModel(nn.Module):
    """
    Dropout이 포함된 MLP 모델.
    (Weight Decay는 optimizer에서 적용)

    아키텍처:
      Input(50) -> Linear(256) -> ReLU -> Dropout(0.3)
      -> Linear(128) -> ReLU -> Dropout(0.3)
      -> Linear(64) -> ReLU -> Dropout(0.2)
      -> Linear(1)

    힌트:
      - nn.Dropout(p=0.3)을 ReLU 뒤에 추가
    """

    def __init__(self, n_features: int = 50, dropout_rate: float = 0.3):
        super().__init__()
        # TODO: Dropout이 포함된 모델을 정의하세요
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: 순전파
        pass


# =============================================================================
# TODO 3: 학습 곡선 생성 함수
# =============================================================================
def train_and_record(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_epochs: int = 150,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
) -> dict[str, list[float]]:
    """
    모델을 학습하면서 에폭별 학습/검증 손실을 기록합니다.

    Parameters
    ----------
    model : nn.Module
    X_train, y_train : np.ndarray
        학습 데이터
    X_val, y_val : np.ndarray
        검증 데이터
    n_epochs : int
        에폭 수
    lr : float
        학습률
    weight_decay : float
        L2 정규화 계수 (0이면 정규화 없음)
    batch_size : int
        배치 크기

    Returns
    -------
    curves : dict
        {"train_losses": [...], "val_losses": [...]}

    힌트:
      - criterion = nn.MSELoss()
      - optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                      weight_decay=weight_decay)
      - 각 에폭:
        1. model.train()으로 학습 모드 (Dropout 활성화)
        2. 미니배치 학습
        3. model.eval()으로 평가 모드 (Dropout 비활성화)
        4. torch.no_grad()에서 전체 학습/검증 손실 계산
    """
    # TODO: 학습 곡선을 기록하는 학습 루프를 구현하세요
    pass


# =============================================================================
# TODO 4: 과적합 분석
# =============================================================================
def analyze_overfitting(
    unreg_curves: dict[str, list[float]],
    reg_curves: dict[str, list[float]],
) -> dict[str, float]:
    """
    두 모델의 학습 곡선을 분석하여 과적합 지표를 계산합니다.

    Parameters
    ----------
    unreg_curves : dict
        정규화 없는 모델의 학습 곡선
    reg_curves : dict
        정규화된 모델의 학습 곡선

    Returns
    -------
    analysis : dict
        {
            "unreg_train_final": float,  -- 미정규화 최종 학습 손실
            "unreg_val_final": float,    -- 미정규화 최종 검증 손실
            "unreg_gap": float,          -- 미정규화 train-val 격차
            "reg_train_final": float,    -- 정규화 최종 학습 손실
            "reg_val_final": float,      -- 정규화 최종 검증 손실
            "reg_gap": float,            -- 정규화 train-val 격차
        }

    힌트:
      - gap = val_final - train_final (과적합일수록 gap이 큼)
      - 마지막 10 에폭의 평균을 사용하면 더 안정적
    """
    # TODO: 과적합 분석 지표를 계산하세요
    pass


# =============================================================================
# 메인 실행 및 자가 검증
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("Exercise 14: 정규화 기법 비교 (Dropout + Weight Decay)")
    print("=" * 60)

    # 1. 데이터 생성
    X_train, y_train, X_val, y_val = generate_overfit_data(
        n_train=150, n_val=200, n_features=50
    )
    print(f"\n[데이터] 학습: {X_train.shape[0]}개 (과적합 유도)")
    print(f"         검증: {X_val.shape[0]}개")
    print(f"         기술자: {X_train.shape[1]}개 (유효: 10개, 노이즈: 40개)")

    # 2. 정규화 없는 모델
    unreg_model = UnregularizedModel(n_features=50)
    assert unreg_model is not None
    n_params_unreg = sum(p.numel() for p in unreg_model.parameters())
    print(f"\n[미정규화 모델] 파라미터: {n_params_unreg:,}")

    unreg_curves = train_and_record(
        unreg_model, X_train, y_train, X_val, y_val,
        n_epochs=150, lr=0.001, weight_decay=0.0
    )
    assert unreg_curves is not None, "train_and_record()가 None을 반환했습니다."
    assert "train_losses" in unreg_curves and "val_losses" in unreg_curves
    assert len(unreg_curves["train_losses"]) == 150

    # 3. 정규화된 모델
    torch.manual_seed(42)
    reg_model = RegularizedModel(n_features=50, dropout_rate=0.3)
    assert reg_model is not None
    n_params_reg = sum(p.numel() for p in reg_model.parameters())
    print(f"[정규화 모델]   파라미터: {n_params_reg:,}")

    reg_curves = train_and_record(
        reg_model, X_train, y_train, X_val, y_val,
        n_epochs=150, lr=0.001, weight_decay=1e-4
    )
    assert reg_curves is not None

    # 4. 과적합 분석
    analysis = analyze_overfitting(unreg_curves, reg_curves)
    assert analysis is not None, "analyze_overfitting()이 None을 반환했습니다."
    assert "unreg_gap" in analysis and "reg_gap" in analysis

    print(f"\n[학습 곡선 분석]")
    print(f"  {'':20s} {'학습 손실':>12s} {'검증 손실':>12s} {'격차':>12s}")
    print(f"  {'미정규화':20s} {analysis['unreg_train_final']:12.4f} "
          f"{analysis['unreg_val_final']:12.4f} {analysis['unreg_gap']:12.4f}")
    print(f"  {'정규화(DO+WD)':20s} {analysis['reg_train_final']:12.4f} "
          f"{analysis['reg_val_final']:12.4f} {analysis['reg_gap']:12.4f}")

    # 정규화 모델의 격차가 더 작아야 함 (과적합 감소)
    assert analysis["reg_gap"] < analysis["unreg_gap"] * 2, \
        "정규화 모델의 train-val 격차가 기대보다 큽니다."
    print(f"\n  정규화로 과적합 격차 감소 확인!")

    print("\n" + "=" * 60)
    print("모든 검증을 통과했습니다!")
    print("=" * 60)
