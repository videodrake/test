"""
Exercise 08: 불균형 데이터에서의 모델 평가 심화

목표:
  - 극심한 불균형 (10% 활성, 90% 비활성)에서 적절한 평가 지표 선택
  - Confusion Matrix, ROC-AUC, Precision-Recall 곡선 이해
  - class_weight='balanced' 효과 확인
  - 결정 임계값(threshold) 최적화

배경:
  실제 HTS 데이터에서 활성 hit rate은 보통 1~10% 수준입니다.
  이런 극심한 불균형에서 accuracy만으로는 모델의 유용성을 판단할 수 없습니다.
  활성 분자를 놓치지 않으면서도 거짓양성을 줄이는 최적의 균형점을 찾아야 합니다.

사용 라이브러리:
  numpy, sklearn (RandomForestClassifier, confusion_matrix, roc_auc_score,
  precision_recall_curve, roc_curve, classification_report, f1_score)
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    classification_report,
    f1_score,
)


def generate_imbalanced_data(
    n_samples: int = 1000,
    active_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    극심한 불균형 (10% 활성, 90% 비활성) 스크리닝 데이터를 생성합니다.

    Parameters
    ----------
    n_samples : int
        총 분자 수
    active_ratio : float
        활성 비율 (0.1 = 10%)

    Returns
    -------
    X : np.ndarray, shape (n_samples, 8)
        분자 기술자
    y : np.ndarray, shape (n_samples,)
        활성 레이블 (1=active, 0=inactive)
    """
    np.random.seed(42)
    n_active = int(n_samples * active_ratio)
    n_inactive = n_samples - n_active

    # 8개 기술자: MW, logP, HBD, HBA, TPSA, RotBonds, AromaticRings, HeavyAtoms
    # 활성: 특정 물리화학적 공간에 밀집
    X_active = np.random.randn(n_active, 8) * 0.8 + np.array(
        [320, 2.0, 2, 4, 70, 4, 2, 22]
    )
    # 비활성: 더 넓은 물리화학적 공간에 분산
    X_inactive = np.random.randn(n_inactive, 8) * 1.5 + np.array(
        [380, 3.0, 1, 5, 100, 6, 1, 28]
    )

    X = np.vstack([X_active, X_inactive])
    y = np.array([1] * n_active + [0] * n_inactive)

    shuffle_idx = np.random.permutation(n_samples)
    return X[shuffle_idx], y[shuffle_idx]


# =============================================================================
# TODO 1: Confusion Matrix 분석
# =============================================================================
def analyze_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, int]:
    """
    Confusion matrix를 계산하고 TP, FP, TN, FN을 반환합니다.

    Parameters
    ----------
    y_true : np.ndarray
        실제 레이블
    y_pred : np.ndarray
        예측 레이블

    Returns
    -------
    result : dict[str, int]
        {"TP": int, "FP": int, "TN": int, "FN": int}

    힌트:
      - cm = confusion_matrix(y_true, y_pred)
      - sklearn의 confusion_matrix는 [[TN, FP], [FN, TP]] 형태
      - 신약개발에서 FN(활성을 놓침)은 매우 치명적 → recall이 중요
    """
    # TODO: confusion matrix를 계산하고 TP, FP, TN, FN을 딕셔너리로 반환하세요
    pass


# =============================================================================
# TODO 2: ROC-AUC와 최적 임계값
# =============================================================================
def compute_roc_analysis(
    y_true: np.ndarray, y_prob: np.ndarray
) -> dict[str, float]:
    """
    ROC-AUC를 계산하고 Youden's J statistic 기반 최적 임계값을 찾습니다.

    Youden's J = sensitivity + specificity - 1 = TPR - FPR

    Parameters
    ----------
    y_true : np.ndarray
        실제 레이블
    y_prob : np.ndarray
        활성 클래스(1)에 대한 예측 확률 (predict_proba[:, 1])

    Returns
    -------
    result : dict[str, float]
        {"roc_auc": float, "optimal_threshold": float}

    힌트:
      - roc_auc = roc_auc_score(y_true, y_prob)
      - fpr, tpr, thresholds = roc_curve(y_true, y_prob)
      - J = tpr - fpr → np.argmax(J)로 최적 인덱스 찾기
      - optimal_threshold = thresholds[optimal_idx]
    """
    # TODO: ROC-AUC와 최적 임계값을 계산하세요
    pass


# =============================================================================
# TODO 3: Precision-Recall 분석
# =============================================================================
def compute_pr_analysis(
    y_true: np.ndarray, y_prob: np.ndarray
) -> dict[str, float]:
    """
    Precision-Recall 곡선에서 F1-score를 최대화하는 임계값을 찾습니다.

    불균형 데이터에서는 ROC-AUC보다 PR-AUC가 더 유의미할 수 있습니다.

    Parameters
    ----------
    y_true : np.ndarray
        실제 레이블
    y_prob : np.ndarray
        활성 클래스 예측 확률

    Returns
    -------
    result : dict[str, float]
        {"best_f1": float, "best_threshold": float,
         "precision_at_best": float, "recall_at_best": float}

    힌트:
      - precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
      - F1 = 2 * precision * recall / (precision + recall)
        주의: precision_recall_curve의 반환에서 precision, recall 길이 = len(thresholds) + 1
        → F1 계산 시 precision[:-1], recall[:-1] 사용
      - np.argmax(f1_scores)로 최적 인덱스 찾기
    """
    # TODO: PR 곡선 기반 최적 F1-score와 임계값을 계산하세요
    pass


# =============================================================================
# TODO 4: class_weight='balanced' 비교
# =============================================================================
def compare_balanced_vs_default(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """
    기본 RF vs balanced RF의 주요 지표를 비교합니다.

    Parameters
    ----------
    X_train, y_train, X_test, y_test : 데이터
    random_state : int

    Returns
    -------
    comparison : dict
        {
            "default": {"accuracy": float, "roc_auc": float, "recall_active": float},
            "balanced": {"accuracy": float, "roc_auc": float, "recall_active": float}
        }

    힌트:
      - 기본: RandomForestClassifier(n_estimators=100, random_state=...)
      - balanced: RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=...)
      - recall_active: classification_report(output_dict=True)['1']['recall']
      - roc_auc: roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    """
    # TODO: 기본 vs balanced RF 비교를 구현하세요
    pass


# =============================================================================
# TODO 5: 임계값 기반 예측 함수
# =============================================================================
def predict_with_threshold(
    model: RandomForestClassifier,
    X: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    사용자 지정 임계값으로 예측을 수행합니다.

    기본 predict()는 0.5를 임계값으로 사용하지만,
    불균형 데이터에서는 더 낮은 임계값이 활성 분자 탐지에 유리할 수 있습니다.

    Parameters
    ----------
    model : RandomForestClassifier
        학습된 모델
    X : np.ndarray
        특성 행렬
    threshold : float
        활성으로 분류할 확률 임계값 (0~1)

    Returns
    -------
    y_pred : np.ndarray
        예측 레이블 (0 또는 1)

    힌트:
      - prob = model.predict_proba(X)[:, 1]
      - y_pred = (prob >= threshold).astype(int)
    """
    # TODO: 지정 임계값으로 예측을 수행하세요
    pass


# =============================================================================
# 메인 실행 및 자가 검증
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 08: 불균형 데이터에서의 모델 평가 심화")
    print("=" * 60)

    # 1. 데이터 생성 및 분할
    X, y = generate_imbalanced_data(1000, active_ratio=0.1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[데이터] 총: {len(y)}, 활성: {np.sum(y==1)} ({np.mean(y==1)*100:.0f}%)")
    print(f"  학습: {len(y_train)}, 테스트: {len(y_test)}")

    # 모델 학습 (기본)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    # 2. Confusion Matrix
    cm_result = analyze_confusion_matrix(y_test, y_pred)
    assert cm_result is not None, "analyze_confusion_matrix()가 None을 반환했습니다."
    assert all(k in cm_result for k in ["TP", "FP", "TN", "FN"])
    assert cm_result["TP"] + cm_result["FN"] + cm_result["FP"] + cm_result["TN"] == len(y_test)
    print(f"\n[Confusion Matrix]")
    print(f"  TP={cm_result['TP']}, FP={cm_result['FP']}")
    print(f"  FN={cm_result['FN']}, TN={cm_result['TN']}")

    # 3. ROC 분석
    roc_result = compute_roc_analysis(y_test, y_prob)
    assert roc_result is not None, "compute_roc_analysis()가 None을 반환했습니다."
    assert "roc_auc" in roc_result and "optimal_threshold" in roc_result
    assert 0 <= roc_result["roc_auc"] <= 1
    assert 0 <= roc_result["optimal_threshold"] <= 1
    print(f"\n[ROC 분석]")
    print(f"  ROC-AUC: {roc_result['roc_auc']:.4f}")
    print(f"  최적 임계값 (Youden's J): {roc_result['optimal_threshold']:.4f}")

    # 4. PR 분석
    pr_result = compute_pr_analysis(y_test, y_prob)
    assert pr_result is not None, "compute_pr_analysis()가 None을 반환했습니다."
    assert "best_f1" in pr_result and "best_threshold" in pr_result
    assert 0 <= pr_result["best_f1"] <= 1
    print(f"\n[Precision-Recall 분석]")
    print(f"  최적 F1: {pr_result['best_f1']:.4f}")
    print(f"  최적 임계값: {pr_result['best_threshold']:.4f}")
    print(f"  Precision@best: {pr_result['precision_at_best']:.4f}")
    print(f"  Recall@best: {pr_result['recall_at_best']:.4f}")

    # 5. balanced 비교
    comparison = compare_balanced_vs_default(X_train, y_train, X_test, y_test)
    assert comparison is not None, "compare_balanced_vs_default()가 None을 반환했습니다."
    assert "default" in comparison and "balanced" in comparison
    print(f"\n[Default vs Balanced 비교]")
    for key in ["default", "balanced"]:
        m = comparison[key]
        print(f"  {key:>10s}: accuracy={m['accuracy']:.4f}, "
              f"roc_auc={m['roc_auc']:.4f}, recall_active={m['recall_active']:.4f}")

    # 6. 임계값 기반 예측
    y_pred_custom = predict_with_threshold(rf, X_test, threshold=0.3)
    assert y_pred_custom is not None, "predict_with_threshold()가 None을 반환했습니다."
    assert y_pred_custom.shape == y_test.shape
    assert set(np.unique(y_pred_custom)).issubset({0, 1})
    # 낮은 임계값 → 더 많은 양성 예측
    n_positive_default = np.sum(y_pred)
    n_positive_custom = np.sum(y_pred_custom)
    print(f"\n[임계값 효과]")
    print(f"  기본 (0.5): 양성 예측 {n_positive_default}개")
    print(f"  사용자 (0.3): 양성 예측 {n_positive_custom}개")

    print("\n" + "=" * 60)
    print("모든 검증을 통과했습니다!")
    print("=" * 60)
