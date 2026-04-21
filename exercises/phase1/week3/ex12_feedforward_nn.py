"""
Exercise 12: Feedforward 신경망 - nn.Module로 QSAR 모델 구축

목표:
  - torch.nn.Module을 상속하여 MLP(Multi-Layer Perceptron) 모델 정의
  - MoleculeNet 스타일의 분자 기술자 데이터로 회귀 모델 학습
  - sklearn Random Forest와 딥러닝 모델 성능 비교
  - 모델 아키텍처와 하이퍼파라미터가 성능에 미치는 영향 이해

배경:
  신약개발에서 QSAR 모델은 분자 기술자로부터 약물 활성(pIC50)을
  예측합니다. 이 실습에서는 기존 sklearn RF 모델과 동일한 데이터로
  PyTorch 신경망을 학습하여 딥러닝의 장단점을 직접 비교합니다.

사용 라이브러리:
  torch, torch.nn, numpy, sklearn (RandomForestRegressor, train_test_split,
  StandardScaler)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error


def generate_qsar_data(n_samples: int = 500) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    분자 기술자와 pIC50 데이터를 생성합니다.

    8개의 분자 기술자를 사용하여 비선형 관계를 포함합니다.

    Parameters
    ----------
    n_samples : int
        분자 수

    Returns
    -------
    X : np.ndarray, shape (n_samples, 8)
        분자 기술자 행렬
    y : np.ndarray, shape (n_samples,)
        pIC50 값 (높을수록 활성 강함)
    feature_names : list[str]
        기술자 이름
    """
    np.random.seed(42)
    feature_names = ["MW", "logP", "HBD", "HBA", "TPSA", "RotBonds", "AromaticRings", "HeavyAtoms"]

    MW = np.random.uniform(200, 550, n_samples)
    logP = np.random.normal(3.0, 1.5, n_samples)
    HBD = np.random.randint(0, 5, n_samples).astype(float)
    HBA = np.random.randint(1, 10, n_samples).astype(float)
    TPSA = np.random.uniform(30, 150, n_samples)
    RotBonds = np.random.randint(0, 12, n_samples).astype(float)
    AromaticRings = np.random.randint(0, 5, n_samples).astype(float)
    HeavyAtoms = np.random.randint(15, 40, n_samples).astype(float)

    # 비선형 pIC50 관계 (신경망이 유리)
    pIC50 = (
        6.0
        - 0.005 * (MW - 350) ** 2 / 100
        + 0.3 * logP
        - 0.1 * logP ** 2
        + 0.2 * HBD
        + 0.1 * HBA
        - 0.005 * TPSA
        + 0.15 * AromaticRings
        + np.random.normal(0, 0.4, n_samples)
    )
    pIC50 = np.clip(pIC50, 3.0, 10.0)

    X = np.column_stack([MW, logP, HBD, HBA, TPSA, RotBonds, AromaticRings, HeavyAtoms])
    return X, pIC50, feature_names


# =============================================================================
# TODO 1: nn.Module MLP 모델 정의
# =============================================================================
class MoleculeMLPModel(nn.Module):
    """
    분자 기술자로 pIC50를 예측하는 Multi-Layer Perceptron.

    아키텍처:
      Input (n_features) -> Linear(128) -> ReLU -> Linear(64) -> ReLU -> Linear(1)

    Parameters
    ----------
    n_features : int
        입력 기술자 수

    힌트:
      - __init__에서:
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

      - forward에서:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    """

    def __init__(self, n_features: int):
        super().__init__()
        # TODO: 레이어를 정의하세요
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: 순전파를 구현하세요
        pass


# =============================================================================
# TODO 2: 모델 학습 함수
# =============================================================================
def train_nn_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    lr: float = 0.001,
    n_epochs: int = 100,
) -> list[float]:
    """
    신경망 모델을 학습합니다.

    Parameters
    ----------
    model : nn.Module
        MoleculeMLPModel 인스턴스
    X_train : torch.Tensor
        학습 기술자 텐서
    y_train : torch.Tensor, shape (n, 1)
        학습 타깃 텐서
    lr : float
        학습률
    n_epochs : int
        에폭 수

    Returns
    -------
    losses : list[float]
        에폭별 손실 기록

    힌트:
      - criterion = nn.MSELoss()
      - optimizer = torch.optim.Adam(model.parameters(), lr=lr)
      - 매 에폭:
        1. optimizer.zero_grad()
        2. y_pred = model(X_train)
        3. loss = criterion(y_pred, y_train)
        4. loss.backward()
        5. optimizer.step()
    """
    # TODO: 학습 루프를 구현하세요
    pass


# =============================================================================
# TODO 3: 모델 평가
# =============================================================================
def evaluate_nn_model(
    model: nn.Module, X_test: torch.Tensor, y_test: np.ndarray
) -> dict[str, float]:
    """
    학습된 신경망 모델의 성능을 평가합니다.

    Parameters
    ----------
    model : nn.Module
        학습된 모델
    X_test : torch.Tensor
        테스트 기술자
    y_test : np.ndarray
        테스트 타깃 (NumPy)

    Returns
    -------
    metrics : dict[str, float]
        {"r2": ..., "mae": ...}

    힌트:
      - model.eval()
      - with torch.no_grad():
          y_pred = model(X_test).numpy().flatten()
      - r2 = r2_score(y_test, y_pred)
      - mae = mean_absolute_error(y_test, y_pred)
    """
    # TODO: 예측 및 평가 지표 계산
    pass


# =============================================================================
# TODO 4: sklearn RF 모델과 비교
# =============================================================================
def train_and_evaluate_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Random Forest 모델을 학습하고 동일한 지표로 평가합니다.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        학습 데이터
    X_test, y_test : np.ndarray
        테스트 데이터
    random_state : int
        시드

    Returns
    -------
    metrics : dict[str, float]
        {"r2": ..., "mae": ...}

    힌트:
      - rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
      - rf.fit(X_train, y_train)
      - y_pred = rf.predict(X_test)
      - r2 = r2_score(y_test, y_pred)
    """
    # TODO: RF 모델 학습 및 평가
    pass


# =============================================================================
# 메인 실행 및 자가 검증
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("Exercise 12: Feedforward 신경망 vs Random Forest QSAR")
    print("=" * 60)

    # 1. 데이터 준비
    X, y, feature_names = generate_qsar_data(500)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

    print(f"\n[데이터] 학습: {X_train.shape[0]}개, 테스트: {X_test.shape[0]}개")
    print(f"  기술자: {feature_names}")
    print(f"  pIC50 범위: {y.min():.2f} ~ {y.max():.2f}")

    # 2. 신경망 모델 생성 및 학습
    model = MoleculeMLPModel(n_features=8)
    assert model is not None, "MoleculeMLPModel 초기화 실패"

    # 파라미터 수 확인
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[MLP 모델] 파라미터 수: {n_params:,}")

    losses = train_nn_model(model, X_train_t, y_train_t, lr=0.001, n_epochs=100)
    assert losses is not None, "train_nn_model()이 None을 반환했습니다."
    assert len(losses) == 100
    assert losses[-1] < losses[0], "학습이 진행되면서 손실이 감소해야 합니다."
    print(f"  초기 손실: {losses[0]:.4f}")
    print(f"  최종 손실: {losses[-1]:.4f}")

    # 3. 신경망 평가
    nn_metrics = evaluate_nn_model(model, X_test_t, y_test)
    assert nn_metrics is not None, "evaluate_nn_model()이 None을 반환했습니다."
    assert "r2" in nn_metrics and "mae" in nn_metrics
    print(f"\n[MLP 결과]")
    print(f"  R²:  {nn_metrics['r2']:.4f}")
    print(f"  MAE: {nn_metrics['mae']:.4f} pIC50")

    # 4. RF 비교
    rf_metrics = train_and_evaluate_rf(X_train_scaled, y_train, X_test_scaled, y_test)
    assert rf_metrics is not None, "train_and_evaluate_rf()가 None을 반환했습니다."
    print(f"\n[Random Forest 결과]")
    print(f"  R²:  {rf_metrics['r2']:.4f}")
    print(f"  MAE: {rf_metrics['mae']:.4f} pIC50")

    # 5. 비교
    print(f"\n[비교]")
    winner = "MLP" if nn_metrics["r2"] > rf_metrics["r2"] else "Random Forest"
    print(f"  R² 기준 우수 모델: {winner}")
    print(f"  MLP R²={nn_metrics['r2']:.4f} vs RF R²={rf_metrics['r2']:.4f}")

    assert nn_metrics["r2"] > 0.0, "MLP R²가 너무 낮습니다."
    assert rf_metrics["r2"] > 0.0, "RF R²가 너무 낮습니다."

    print("\n" + "=" * 60)
    print("모든 검증을 통과했습니다!")
    print("=" * 60)
