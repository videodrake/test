"""
Exercise 07: 활성/비활성 분자 분류기

목표:
  - 이진 분류 문제로서의 약물 활성 예측 이해
  - LogisticRegression, DecisionTree, RandomForest 비교
  - 불균형 데이터(30% 활성, 70% 비활성) 처리 경험
  - 분류 보고서(classification_report) 해석

배경:
  High-Throughput Screening(HTS)에서 대량의 화합물을 테스트하면
  대부분 비활성(inactive)이고 소수만 활성(active)입니다.
  분자 기술자를 사용하여 활성/비활성을 분류하는 모델을 구축합니다.

사용 라이브러리:
  numpy, sklearn (LogisticRegression, DecisionTreeClassifier,
  RandomForestClassifier, train_test_split, accuracy_score,
  classification_report, StandardScaler)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


def generate_activity_data(
    n_samples: int = 500,
    active_ratio: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    분자 기술자와 활성 레이블 데이터를 생성합니다.

    활성 분자는 일반적으로 특정 물리화학적 프로파일을 가집니다:
    - 적절한 분자량 (200~450)
    - 적절한 지용성 (logP 1~4)
    - 높은 약물 유사성 (drug-likeness)

    Parameters
    ----------
    n_samples : int
        총 분자 수
    active_ratio : float
        활성 분자 비율 (0.3 = 30%)

    Returns
    -------
    X : np.ndarray, shape (n_samples, 6)
        분자 기술자 [MW, logP, HBD, HBA, TPSA, RotBonds]
    y : np.ndarray, shape (n_samples,)
        활성 레이블 (1=active, 0=inactive)
    feature_names : list[str]
        기술자 이름
    """
    np.random.seed(42)
    feature_names = ["MW", "logP", "HBD", "HBA", "TPSA", "RotBonds"]

    n_active = int(n_samples * active_ratio)
    n_inactive = n_samples - n_active

    # 활성 분자: drug-like 프로파일
    active_MW = np.random.normal(350, 60, n_active)
    active_logP = np.random.normal(2.5, 0.8, n_active)
    active_HBD = np.random.choice([1, 2, 3], n_active, p=[0.4, 0.4, 0.2])
    active_HBA = np.random.choice([3, 4, 5, 6], n_active)
    active_TPSA = np.random.normal(80, 20, n_active)
    active_RotBonds = np.random.choice([3, 4, 5, 6], n_active)

    # 비활성 분자: 더 넓은 범위, drug-like 벗어나는 경향
    inactive_MW = np.random.normal(400, 120, n_inactive)
    inactive_logP = np.random.normal(3.5, 2.0, n_inactive)
    inactive_HBD = np.random.choice([0, 1, 2, 3, 4, 5], n_inactive)
    inactive_HBA = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9], n_inactive)
    inactive_TPSA = np.random.normal(100, 50, n_inactive)
    inactive_RotBonds = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n_inactive)

    X_active = np.column_stack([
        active_MW, active_logP, active_HBD.astype(float),
        active_HBA.astype(float), active_TPSA, active_RotBonds.astype(float)
    ])
    X_inactive = np.column_stack([
        inactive_MW, inactive_logP, inactive_HBD.astype(float),
        inactive_HBA.astype(float), inactive_TPSA, inactive_RotBonds.astype(float)
    ])

    X = np.vstack([X_active, X_inactive])
    y = np.array([1] * n_active + [0] * n_inactive)

    # 셔플
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    return X, y, feature_names


# =============================================================================
# TODO 1: 데이터 전처리 파이프라인
# =============================================================================
def preprocess_and_split(
    X: np.ndarray, y: np.ndarray,
    test_size: float = 0.2, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    데이터를 분할하고 특성 스케일링을 적용합니다.

    주의: 스케일러는 반드시 학습 데이터에만 fit하고,
    테스트 데이터에는 transform만 적용해야 합니다 (데이터 누출 방지).

    Parameters
    ----------
    X : np.ndarray
        특성 행렬
    y : np.ndarray
        레이블 벡터
    test_size : float
        테스트 세트 비율
    random_state : int
        시드

    Returns
    -------
    X_train_scaled, X_test_scaled, y_train, y_test, scaler

    힌트:
      1. train_test_split으로 분할
      2. StandardScaler()로 scaler 생성
      3. scaler.fit_transform(X_train) → X_train_scaled
      4. scaler.transform(X_test) → X_test_scaled (fit 하지 않음!)
    """
    # TODO: 데이터 분할 + 스케일링을 구현하세요
    pass


# =============================================================================
# TODO 2: 세 가지 분류 모델 학습
# =============================================================================
def train_classifiers(
    X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42
) -> dict[str, object]:
    """
    LogisticRegression, DecisionTree, RandomForest 세 모델을 학습합니다.

    Parameters
    ----------
    X_train : np.ndarray
        학습용 특성 행렬 (스케일링 완료)
    y_train : np.ndarray
        학습용 레이블

    Returns
    -------
    models : dict[str, object]
        {"LogisticRegression": model1, "DecisionTree": model2, "RandomForest": model3}

    힌트:
      - LogisticRegression(random_state=..., max_iter=1000)
      - DecisionTreeClassifier(random_state=...)
      - RandomForestClassifier(n_estimators=100, random_state=...)
      - 각 모델에 .fit(X_train, y_train) 호출
    """
    # TODO: 세 모델을 학습하여 딕셔너리로 반환하세요
    pass


# =============================================================================
# TODO 3: 모델 비교 평가
# =============================================================================
def compare_models(
    models: dict[str, object],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, dict[str, float]]:
    """
    모든 모델의 정확도(accuracy)를 계산하고 분류 보고서를 출력합니다.

    Parameters
    ----------
    models : dict[str, object]
        학습된 모델 딕셔너리
    X_test : np.ndarray
        테스트 특성 행렬
    y_test : np.ndarray
        테스트 레이블

    Returns
    -------
    results : dict[str, dict[str, float]]
        {모델명: {"accuracy": float}} 형식의 결과

    힌트:
      - 각 모델에 대해:
        1. y_pred = model.predict(X_test)
        2. acc = accuracy_score(y_test, y_pred)
        3. print(classification_report(y_test, y_pred)) 로 상세 보고서 출력
    """
    # TODO: 각 모델의 accuracy를 계산하고 classification_report를 출력하세요
    pass


# =============================================================================
# TODO 4: class_weight='balanced' 적용 비교
# =============================================================================
def train_with_balanced_weight(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    random_state: int = 42,
) -> tuple[float, float]:
    """
    RandomForest에 class_weight='balanced'를 적용하여
    기본 모델과 recall(활성 클래스) 차이를 비교합니다.

    Parameters
    ----------
    X_train, y_train : 학습 데이터
    X_test, y_test : 테스트 데이터
    random_state : int

    Returns
    -------
    recall_default : float
        기본 RandomForest의 활성(클래스 1) recall
    recall_balanced : float
        class_weight='balanced' RandomForest의 활성(클래스 1) recall

    힌트:
      - 기본 모델: RandomForestClassifier(n_estimators=100, random_state=...)
      - 균형 모델: RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=...)
      - classification_report(y_test, y_pred, output_dict=True)['1']['recall']로 recall 추출
    """
    # TODO: 기본 vs balanced 모델의 활성 클래스 recall을 비교하세요
    pass


# =============================================================================
# 메인 실행 및 자가 검증
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 07: 활성/비활성 분자 분류기")
    print("=" * 60)

    # 1. 데이터 생성
    X, y, feature_names = generate_activity_data(500, active_ratio=0.3)
    print(f"\n[데이터] 총 분자: {len(y)}")
    print(f"  활성: {np.sum(y == 1)} ({np.mean(y == 1)*100:.0f}%)")
    print(f"  비활성: {np.sum(y == 0)} ({np.mean(y == 0)*100:.0f}%)")
    print(f"  기술자: {feature_names}")

    # 2. 전처리
    result = preprocess_and_split(X, y)
    assert result is not None, "preprocess_and_split()이 None을 반환했습니다."
    X_train, X_test, y_train, y_test, scaler = result
    assert X_train.shape[0] == 400, f"학습 세트 크기 오류: {X_train.shape[0]}"
    assert X_test.shape[0] == 100, f"테스트 세트 크기 오류: {X_test.shape[0]}"
    # 스케일링 검증: 학습 데이터의 평균 ≈ 0
    assert np.abs(X_train.mean()) < 0.1, "스케일링이 올바르지 않습니다."
    print(f"\n[전처리] 학습: {X_train.shape[0]}개, 테스트: {X_test.shape[0]}개")
    print(f"  스케일링 후 학습 데이터 평균: {X_train.mean():.4f} (≈0이어야 함)")

    # 3. 모델 학습
    models = train_classifiers(X_train, y_train)
    assert models is not None, "train_classifiers()가 None을 반환했습니다."
    assert len(models) == 3, f"3개 모델이 필요하지만 {len(models)}개입니다."
    for name in ["LogisticRegression", "DecisionTree", "RandomForest"]:
        assert name in models, f"'{name}' 모델이 없습니다."
    print("\n[모델 학습 완료]")

    # 4. 모델 비교
    print("\n[모델 비교]")
    results = compare_models(models, X_test, y_test)
    assert results is not None, "compare_models()가 None을 반환했습니다."
    for name, metrics in results.items():
        assert "accuracy" in metrics, f"'{name}'에 accuracy가 없습니다."
        assert metrics["accuracy"] > 0.5, f"'{name}' accuracy가 랜덤보다 낮습니다."
        print(f"  {name}: accuracy = {metrics['accuracy']:.4f}")

    # 5. class_weight 비교
    balanced_result = train_with_balanced_weight(X_train, y_train, X_test, y_test)
    assert balanced_result is not None, "train_with_balanced_weight()가 None을 반환했습니다."
    recall_default, recall_balanced = balanced_result
    print(f"\n[class_weight 효과]")
    print(f"  기본 RF 활성 recall: {recall_default:.4f}")
    print(f"  balanced RF 활성 recall: {recall_balanced:.4f}")

    print("\n" + "=" * 60)
    print("모든 검증을 통과했습니다!")
    print("=" * 60)
