"""
Exercise 06: 첫 번째 QSAR 모델 — 용해도(logS) 예측

목표:
  - 분자 기술자(descriptor)를 이용한 QSAR 회귀 모델 구축
  - Linear Regression과 Random Forest Regressor 비교
  - R², MAE, RMSE로 모델 성능 평가
  - 특성 중요도(feature importance) 분석

배경:
  수용해도(aqueous solubility, logS)는 약물의 흡수와 직결되는 핵심 물리화학적 성질입니다.
  Lipinski가 제안한 분자 기술자(MW, logP, HBD, HBA, TPSA)를 사용하여
  logS를 예측하는 QSAR 모델을 구축합니다.

사용 라이브러리:
  numpy, sklearn (LinearRegression, RandomForestRegressor, train_test_split,
  mean_absolute_error, mean_squared_error, r2_score)
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def generate_solubility_data(n_samples: int = 300) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    분자 기술자와 용해도(logS) 데이터를 생성합니다.

    실제 QSAR에서는 RDKit으로 기술자를 계산하지만,
    여기서는 현실적인 분포를 따르는 시뮬레이션 데이터를 사용합니다.

    Parameters
    ----------
    n_samples : int
        생성할 분자 수

    Returns
    -------
    X : np.ndarray, shape (n_samples, 5)
        분자 기술자 [MW, logP, HBD, HBA, TPSA]
    y : np.ndarray, shape (n_samples,)
        logS 값 (용해도, 단위: log mol/L)
    feature_names : list[str]
        기술자 이름 목록
    """
    np.random.seed(42)
    feature_names = ["MW", "logP", "HBD", "HBA", "TPSA"]

    # 현실적인 분자 기술자 범위
    MW = np.random.uniform(150, 550, n_samples)       # 분자량 (Da)
    logP = np.random.normal(2.5, 1.5, n_samples)      # 지용성
    HBD = np.random.randint(0, 6, n_samples).astype(float)  # 수소결합 공여체
    HBA = np.random.randint(1, 11, n_samples).astype(float)  # 수소결합 수용체
    TPSA = np.random.uniform(20, 180, n_samples)      # 위상 극성 표면적 (Å²)

    # logS ≈ 현실적 경험식 + 노이즈
    # 일반적으로 MW↑, logP↑ → 용해도↓ / HBD↑, TPSA↑ → 용해도↑
    logS = (
        0.5
        - 0.01 * MW
        - 0.5 * logP
        + 0.2 * HBD
        + 0.1 * HBA
        + 0.005 * TPSA
        + np.random.normal(0, 0.5, n_samples)
    )

    X = np.column_stack([MW, logP, HBD, HBA, TPSA])
    return X, logS, feature_names


# =============================================================================
# TODO 1: 데이터 분할 함수
# =============================================================================
def split_data(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    데이터를 학습/테스트 세트로 분할합니다.

    Parameters
    ----------
    X : np.ndarray
        특성 행렬
    y : np.ndarray
        타깃 벡터 (logS)
    test_size : float
        테스트 세트 비율
    random_state : int
        재현성을 위한 시드

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple of np.ndarray

    힌트: sklearn.model_selection.train_test_split 사용
    """
    # TODO: train_test_split을 사용하여 데이터를 분할하세요
    pass


# =============================================================================
# TODO 2: 선형 회귀 QSAR 모델 학습
# =============================================================================
def train_linear_regression(
    X_train: np.ndarray, y_train: np.ndarray
) -> LinearRegression:
    """
    Linear Regression 모델을 학습합니다.

    Parameters
    ----------
    X_train : np.ndarray
        학습용 특성 행렬
    y_train : np.ndarray
        학습용 logS 값

    Returns
    -------
    model : LinearRegression
        학습된 모델

    힌트: LinearRegression().fit(X_train, y_train)
    """
    # TODO: LinearRegression 모델을 생성하고 학습시키세요
    pass


# =============================================================================
# TODO 3: Random Forest 회귀 모델 학습
# =============================================================================
def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray,
    n_estimators: int = 100, random_state: int = 42
) -> RandomForestRegressor:
    """
    Random Forest Regressor 모델을 학습합니다.

    Parameters
    ----------
    X_train : np.ndarray
        학습용 특성 행렬
    y_train : np.ndarray
        학습용 logS 값
    n_estimators : int
        트리 개수
    random_state : int
        재현성을 위한 시드

    Returns
    -------
    model : RandomForestRegressor
        학습된 모델

    힌트: RandomForestRegressor(n_estimators=..., random_state=...).fit(...)
    """
    # TODO: RandomForestRegressor 모델을 생성하고 학습시키세요
    pass


# =============================================================================
# TODO 4: 모델 평가 함수
# =============================================================================
def evaluate_model(
    model, X_test: np.ndarray, y_test: np.ndarray
) -> dict[str, float]:
    """
    회귀 모델의 성능을 R², MAE, RMSE로 평가합니다.

    Parameters
    ----------
    model : sklearn estimator
        학습된 모델 (predict 메서드 필요)
    X_test : np.ndarray
        테스트 특성 행렬
    y_test : np.ndarray
        테스트 logS 값

    Returns
    -------
    metrics : dict[str, float]
        {"r2": ..., "mae": ..., "rmse": ...}

    힌트:
      - model.predict(X_test)로 예측값 계산
      - r2_score, mean_absolute_error, mean_squared_error 사용
      - RMSE = sqrt(MSE) → np.sqrt(mean_squared_error(...))
    """
    # TODO: 예측 후 R², MAE, RMSE를 계산하여 딕셔너리로 반환하세요
    pass


# =============================================================================
# TODO 5: 특성 중요도 추출
# =============================================================================
def get_feature_importance(
    model: RandomForestRegressor, feature_names: list[str]
) -> list[tuple[str, float]]:
    """
    Random Forest 모델에서 특성 중요도를 추출합니다.

    Parameters
    ----------
    model : RandomForestRegressor
        학습된 Random Forest 모델
    feature_names : list[str]
        기술자 이름 목록 (예: ["MW", "logP", "HBD", "HBA", "TPSA"])

    Returns
    -------
    importance_list : list[tuple[str, float]]
        (기술자 이름, 중요도) 쌍을 중요도 내림차순으로 정렬한 리스트

    힌트:
      - model.feature_importances_ 속성 사용
      - zip(feature_names, importances)로 묶은 후 sorted(..., key=..., reverse=True)
    """
    # TODO: 특성 중요도를 추출하고 내림차순으로 정렬하세요
    pass


# =============================================================================
# 메인 실행 및 자가 검증
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 06: 첫 번째 QSAR 모델 — 용해도(logS) 예측")
    print("=" * 60)

    # 1. 데이터 생성
    X, y, feature_names = generate_solubility_data(300)
    print(f"\n[데이터] 분자 수: {X.shape[0]}, 기술자 수: {X.shape[1]}")
    print(f"  기술자: {feature_names}")
    print(f"  logS 범위: {y.min():.2f} ~ {y.max():.2f}")

    # 2. 데이터 분할
    result = split_data(X, y)
    assert result is not None, "split_data()가 None을 반환했습니다. TODO를 구현하세요."
    X_train, X_test, y_train, y_test = result
    assert X_train.shape[0] == 240, f"학습 세트 크기가 240이어야 하지만 {X_train.shape[0]}입니다."
    assert X_test.shape[0] == 60, f"테스트 세트 크기가 60이어야 하지만 {X_test.shape[0]}입니다."
    print(f"\n[분할] 학습: {X_train.shape[0]}개, 테스트: {X_test.shape[0]}개")

    # 3. Linear Regression
    lr_model = train_linear_regression(X_train, y_train)
    assert lr_model is not None, "train_linear_regression()이 None을 반환했습니다."
    assert hasattr(lr_model, "coef_"), "모델이 학습되지 않았습니다 (coef_ 속성 없음)."
    lr_metrics = evaluate_model(lr_model, X_test, y_test)
    assert lr_metrics is not None, "evaluate_model()이 None을 반환했습니다."
    assert "r2" in lr_metrics and "mae" in lr_metrics and "rmse" in lr_metrics
    print(f"\n[Linear Regression]")
    print(f"  R² = {lr_metrics['r2']:.4f}")
    print(f"  MAE = {lr_metrics['mae']:.4f} log mol/L")
    print(f"  RMSE = {lr_metrics['rmse']:.4f} log mol/L")

    # 4. Random Forest
    rf_model = train_random_forest(X_train, y_train)
    assert rf_model is not None, "train_random_forest()가 None을 반환했습니다."
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    print(f"\n[Random Forest]")
    print(f"  R² = {rf_metrics['r2']:.4f}")
    print(f"  MAE = {rf_metrics['mae']:.4f} log mol/L")
    print(f"  RMSE = {rf_metrics['rmse']:.4f} log mol/L")

    # 5. 특성 중요도
    importance = get_feature_importance(rf_model, feature_names)
    assert importance is not None, "get_feature_importance()가 None을 반환했습니다."
    assert len(importance) == 5, f"5개 특성의 중요도가 필요하지만 {len(importance)}개입니다."
    assert importance[0][1] >= importance[-1][1], "중요도가 내림차순이어야 합니다."
    print(f"\n[특성 중요도 — Random Forest]")
    for name, imp in importance:
        print(f"  {name:>6s}: {imp:.4f}")

    # 6. 모델 비교 (R² 기반)
    assert lr_metrics["r2"] > 0.5, "Linear Regression R²가 너무 낮습니다. 구현을 확인하세요."
    assert rf_metrics["r2"] > 0.5, "Random Forest R²가 너무 낮습니다. 구현을 확인하세요."

    print("\n" + "=" * 60)
    print("모든 검증을 통과했습니다!")
    print("=" * 60)
