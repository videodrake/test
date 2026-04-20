"""
Exercise 09: 분자 기술자 특성 엔지니어링

목표:
  - 분산 필터(Variance Threshold)로 정보 없는 기술자 제거
  - 상관관계 필터로 중복 기술자 제거
  - RFE(Recursive Feature Elimination)로 최적 기술자 선택
  - Pipeline으로 전처리-선택-모델을 하나로 묶기

배경:
  RDKit 등으로 수백 개의 분자 기술자를 계산할 수 있지만,
  모든 기술자가 유용한 것은 아닙니다. 분산이 없거나, 다른 기술자와
  높은 상관관계를 갖는 중복 기술자는 모델 성능을 저하시킬 수 있습니다.
  체계적인 특성 선택은 QSAR 모델의 해석 가능성과 성능을 모두 향상시킵니다.

사용 라이브러리:
  numpy, sklearn (VarianceThreshold, RandomForestClassifier, RFE,
  Pipeline, StandardScaler, cross_val_score)
"""

import numpy as np
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


def generate_descriptor_data(
    n_samples: int = 400,
    n_informative: int = 20,
    n_redundant: int = 30,
    n_noisy: int = 100,
    n_constant: int = 50,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    다양한 특성을 가진 200개 분자 기술자 데이터를 생성합니다.

    - 정보성 기술자(informative): 활성 예측에 실제로 유용
    - 중복 기술자(redundant): 정보성 기술자의 선형 조합
    - 노이즈 기술자(noisy): 순수 랜덤, 예측에 무관
    - 상수 기술자(constant): 분산이 거의 0

    Parameters
    ----------
    n_samples : int
        분자 수
    n_informative, n_redundant, n_noisy, n_constant : int
        각 유형의 기술자 수

    Returns
    -------
    X : np.ndarray, shape (n_samples, 200)
        분자 기술자 행렬
    y : np.ndarray, shape (n_samples,)
        활성 레이블 (0 또는 1)
    total_features : int
        총 기술자 수 (200)
    """
    np.random.seed(42)
    total_features = n_informative + n_redundant + n_noisy + n_constant

    # 정보성 기술자 → 활성과 관련됨
    X_informative = np.random.randn(n_samples, n_informative)
    # 활성 레이블: 정보성 기술자의 가중합 기반
    weights = np.random.randn(n_informative)
    logit = X_informative @ weights
    prob = 1 / (1 + np.exp(-logit))
    y = (prob > 0.5).astype(int)

    # 중복 기술자: 정보성 기술자의 선형 조합 + 소량의 노이즈
    mixing = np.random.randn(n_informative, n_redundant) * 0.5
    X_redundant = X_informative @ mixing + np.random.randn(n_samples, n_redundant) * 0.1

    # 노이즈 기술자: 순수 랜덤
    X_noisy = np.random.randn(n_samples, n_noisy) * 2.0

    # 상수 기술자: 분산 거의 0
    X_constant = np.ones((n_samples, n_constant)) * 5.0
    X_constant += np.random.randn(n_samples, n_constant) * 0.001

    X = np.hstack([X_informative, X_redundant, X_noisy, X_constant])
    return X, y, total_features


# =============================================================================
# TODO 1: 분산 필터 (Variance Threshold)
# =============================================================================
def apply_variance_filter(
    X: np.ndarray, threshold: float = 0.01
) -> tuple[np.ndarray, int, int]:
    """
    분산이 낮은 기술자를 제거합니다.

    상수이거나 거의 변하지 않는 기술자는 모델에 정보를 제공하지 않습니다.

    Parameters
    ----------
    X : np.ndarray
        기술자 행렬
    threshold : float
        최소 분산 임계값 (이보다 낮으면 제거)

    Returns
    -------
    X_filtered : np.ndarray
        필터링된 기술자 행렬
    n_original : int
        원래 기술자 수
    n_remaining : int
        남은 기술자 수

    힌트:
      - selector = VarianceThreshold(threshold=threshold)
      - X_filtered = selector.fit_transform(X)
    """
    # TODO: VarianceThreshold를 적용하세요
    pass


# =============================================================================
# TODO 2: 상관관계 필터
# =============================================================================
def apply_correlation_filter(
    X: np.ndarray, threshold: float = 0.95
) -> tuple[np.ndarray, list[int]]:
    """
    높은 상관관계를 가진 기술자 쌍에서 하나를 제거합니다.

    Parameters
    ----------
    X : np.ndarray
        기술자 행렬
    threshold : float
        상관계수 임계값 (이 이상이면 하나 제거)

    Returns
    -------
    X_filtered : np.ndarray
        중복 제거된 기술자 행렬
    removed_indices : list[int]
        제거된 기술자의 인덱스 목록

    힌트:
      1. corr_matrix = np.corrcoef(X, rowvar=False) → 상관 행렬 (특성×특성)
      2. 상삼각행렬만 검사: np.triu_indices_from(corr_matrix, k=1)
      3. |상관계수| >= threshold인 쌍에서 뒤쪽 인덱스를 제거 목록에 추가
      4. 제거 인덱스 중복 제거 후, np.delete(X, removed_indices, axis=1)
    """
    # TODO: 상관관계 필터를 구현하세요
    pass


# =============================================================================
# TODO 3: RFE (Recursive Feature Elimination)
# =============================================================================
def apply_rfe(
    X: np.ndarray, y: np.ndarray, n_features_to_select: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """
    RFE로 최적의 기술자 부분집합을 선택합니다.

    Parameters
    ----------
    X : np.ndarray
        기술자 행렬
    y : np.ndarray
        활성 레이블
    n_features_to_select : int
        선택할 기술자 수

    Returns
    -------
    X_selected : np.ndarray
        선택된 기술자 행렬
    selected_mask : np.ndarray
        선택된 기술자의 boolean 마스크 (True/False)

    힌트:
      - estimator = RandomForestClassifier(n_estimators=50, random_state=42)
      - rfe = RFE(estimator, n_features_to_select=n_features_to_select, step=10)
      - rfe.fit(X, y)
      - X_selected = rfe.transform(X)
      - selected_mask = rfe.support_
    """
    # TODO: RFE를 적용하여 기술자를 선택하세요
    pass


# =============================================================================
# TODO 4: Pipeline 구축
# =============================================================================
def build_pipeline(random_state: int = 42) -> Pipeline:
    """
    StandardScaler → VarianceThreshold → LogisticRegression Pipeline을 구축합니다.

    Parameters
    ----------
    random_state : int
        재현성 시드

    Returns
    -------
    pipe : Pipeline
        3단계 파이프라인

    힌트:
      Pipeline([
          ("scaler", StandardScaler()),
          ("var_filter", VarianceThreshold(threshold=0.01)),
          ("model", LogisticRegression(random_state=..., max_iter=1000))
      ])
    """
    # TODO: Pipeline을 구축하세요
    pass


# =============================================================================
# TODO 5: Pipeline 교차 검증
# =============================================================================
def evaluate_pipeline(
    pipe: Pipeline, X: np.ndarray, y: np.ndarray, cv: int = 5
) -> dict[str, float]:
    """
    Pipeline에 대해 교차 검증을 수행합니다.

    Parameters
    ----------
    pipe : Pipeline
        평가할 파이프라인
    X : np.ndarray
        전체 특성 행렬
    y : np.ndarray
        전체 레이블
    cv : int
        폴드 수

    Returns
    -------
    result : dict[str, float]
        {"mean_accuracy": float, "std_accuracy": float}

    힌트:
      - scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
      - mean_accuracy = scores.mean()
      - std_accuracy = scores.std()
    """
    # TODO: 교차 검증을 수행하세요
    pass


# =============================================================================
# 메인 실행 및 자가 검증
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 09: 분자 기술자 특성 엔지니어링")
    print("=" * 60)

    # 1. 데이터 생성
    X, y, total_features = generate_descriptor_data()
    print(f"\n[데이터] 분자: {X.shape[0]}, 기술자: {X.shape[1]}")
    print(f"  활성: {np.sum(y==1)}, 비활성: {np.sum(y==0)}")

    # 2. 분산 필터
    var_result = apply_variance_filter(X, threshold=0.01)
    assert var_result is not None, "apply_variance_filter()가 None을 반환했습니다."
    X_var, n_orig, n_remain = var_result
    assert n_orig == 200, f"원래 기술자 수가 200이어야 하지만 {n_orig}입니다."
    assert n_remain < n_orig, "분산 필터로 기술자가 줄어야 합니다."
    assert n_remain == X_var.shape[1]
    print(f"\n[분산 필터] {n_orig} → {n_remain} 기술자 ({n_orig - n_remain}개 제거)")

    # 3. 상관관계 필터
    corr_result = apply_correlation_filter(X_var, threshold=0.95)
    assert corr_result is not None, "apply_correlation_filter()가 None을 반환했습니다."
    X_corr, removed = corr_result
    assert X_corr.shape[1] <= X_var.shape[1], "상관관계 필터 후 기술자가 줄어야 합니다."
    print(f"[상관 필터] {X_var.shape[1]} → {X_corr.shape[1]} 기술자 ({len(removed)}개 제거)")

    # 4. RFE
    rfe_result = apply_rfe(X_corr, y, n_features_to_select=20)
    assert rfe_result is not None, "apply_rfe()가 None을 반환했습니다."
    X_rfe, mask = rfe_result
    assert X_rfe.shape[1] == 20, f"RFE 후 20개 기술자여야 하지만 {X_rfe.shape[1]}개입니다."
    assert mask.sum() == 20
    print(f"[RFE] {X_corr.shape[1]} → {X_rfe.shape[1]} 기술자 선택")

    # 5. Pipeline
    pipe = build_pipeline()
    assert pipe is not None, "build_pipeline()이 None을 반환했습니다."
    assert isinstance(pipe, Pipeline), "Pipeline 객체여야 합니다."
    assert len(pipe.steps) == 3, f"3단계 파이프라인이어야 하지만 {len(pipe.steps)}단계입니다."
    print(f"\n[Pipeline] 단계: {[name for name, _ in pipe.steps]}")

    # 6. 교차 검증
    cv_result = evaluate_pipeline(pipe, X, y, cv=5)
    assert cv_result is not None, "evaluate_pipeline()이 None을 반환했습니다."
    assert "mean_accuracy" in cv_result and "std_accuracy" in cv_result
    assert 0 <= cv_result["mean_accuracy"] <= 1
    print(f"[교차검증] 정확도: {cv_result['mean_accuracy']:.4f} ± {cv_result['std_accuracy']:.4f}")

    # 7. 특성 선택 전후 비교
    pipe_full = build_pipeline()
    pipe_selected = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=42, max_iter=1000))
    ])
    scores_full = cross_val_score(pipe_full, X, y, cv=5, scoring="accuracy")
    scores_selected = cross_val_score(pipe_selected, X_rfe, y, cv=5, scoring="accuracy")
    print(f"\n[비교] 전체 기술자 ({X.shape[1]}개): {scores_full.mean():.4f} ± {scores_full.std():.4f}")
    print(f"[비교] 선택 기술자 ({X_rfe.shape[1]}개): {scores_selected.mean():.4f} ± {scores_selected.std():.4f}")

    print("\n" + "=" * 60)
    print("모든 검증을 통과했습니다!")
    print("=" * 60)
