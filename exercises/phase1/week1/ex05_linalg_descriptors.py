"""
Exercise 05: 분자 기술자 행렬에서 PCA 직접 구현
Phase 1 > Week 1 > Day 5

목표:
1. 랜덤 분자 기술자 행렬 생성 (50 분자 x 20 기술자)
2. PCA를 NumPy로 직접 구현 (고유값 분해)
3. sklearn PCA와 결과 비교
4. 설명된 분산비 그래프 (elbow plot)
5. 2D 주성분 공간에 분자 매핑
"""

import numpy as np


# === 샘플 데이터 ===
def generate_descriptor_matrix(n_molecules=50, n_descriptors=20):
    """
    분자 기술자 행렬을 생성합니다.
    일부 기술자는 서로 상관관계가 있도록 설계합니다.

    Returns:
        X: (n_molecules, n_descriptors) 행렬
        labels: 분자 클래스 (0, 1, 2)
    """
    np.random.seed(42)

    # 3개 클래스의 분자 생성 (클러스터가 보이도록)
    n_per_class = n_molecules // 3
    remainder = n_molecules - n_per_class * 3

    centers = [
        np.random.randn(n_descriptors) * 2,
        np.random.randn(n_descriptors) * 2 + 3,
        np.random.randn(n_descriptors) * 2 - 2,
    ]

    X_parts = []
    labels_parts = []
    for i, center in enumerate(centers):
        n = n_per_class + (1 if i < remainder else 0)
        X_parts.append(np.random.randn(n, n_descriptors) * 0.8 + center)
        labels_parts.append(np.full(n, i))

    X = np.vstack(X_parts)
    labels = np.concatenate(labels_parts)

    # 셔플
    idx = np.random.permutation(len(X))
    return X[idx], labels[idx]


# === 과제 1: PCA 직접 구현 ===
def pca_from_scratch(X: np.ndarray, n_components: int = 2) -> dict:
    """
    NumPy만 사용하여 PCA를 구현합니다.

    단계:
    1. 데이터 중심화 (평균 0으로)
    2. 공분산 행렬 계산
    3. 고유값 분해
    4. 상위 n_components개 주성분 선택
    5. 데이터 투영

    Returns:
        {
            'transformed': (N, n_components) 변환된 데이터,
            'eigenvalues': 전체 고유값 (내림차순),
            'eigenvectors': 전체 고유벡터,
            'explained_variance_ratio': 각 주성분의 설명된 분산비,
        }
    """
    # TODO: 구현하세요
    pass


# === 과제 2: sklearn 비교 ===
def compare_with_sklearn(X: np.ndarray, n_components: int = 2) -> dict:
    """
    직접 구현한 PCA와 sklearn PCA의 결과를 비교합니다.

    Returns:
        {
            'scratch_result': pca_from_scratch 결과,
            'sklearn_result': sklearn PCA 결과 (같은 형식),
            'variance_ratio_match': bool (설명된 분산비가 유사한지),
        }
    """
    # TODO: 구현하세요
    # 힌트: sklearn.decomposition.PCA 사용
    # 주의: 고유벡터의 부호는 다를 수 있음 (방향은 같지만 부호 반전 가능)
    pass


# === 과제 3: Elbow Plot 데이터 ===
def compute_elbow_data(X: np.ndarray) -> dict:
    """
    모든 주성분에 대한 설명된 분산비와 누적 분산비를 계산합니다.

    Returns:
        {
            'explained_variance_ratio': 각 주성분의 분산비 배열,
            'cumulative_variance': 누적 분산비 배열,
            'n_for_95': 95% 분산을 설명하는 데 필요한 최소 주성분 수,
        }
    """
    # TODO: 구현하세요
    pass


# === 자체 검증 ===
if __name__ == "__main__":
    X, labels = generate_descriptor_matrix()
    print(f"데이터: {X.shape} (분자 x 기술자)\n")

    # 테스트 1: PCA 직접 구현
    result = pca_from_scratch(X, n_components=2)
    if result is not None:
        assert result["transformed"].shape == (50, 2), "Shape 오류"
        evr = result["explained_variance_ratio"]
        assert len(evr) == 20, "고유값 수 오류"
        assert abs(sum(evr) - 1.0) < 0.01, "분산비 합은 1이어야 함"
        assert all(evr[i] >= evr[i + 1] for i in range(len(evr) - 1)), "내림차순이어야 함"
        print(f"[PASS] PCA 변환: {result['transformed'].shape}")
        print(f"  PC1: {evr[0]*100:.1f}%, PC2: {evr[1]*100:.1f}%")

    # 테스트 2: sklearn 비교
    comparison = compare_with_sklearn(X, n_components=2)
    if comparison is not None:
        print(f"[PASS] 분산비 일치: {comparison['variance_ratio_match']}")

    # 테스트 3: Elbow 데이터
    elbow = compute_elbow_data(X)
    if elbow is not None:
        print(f"[PASS] 95% 분산에 {elbow['n_for_95']}개 주성분 필요")
        print(f"  누적: {elbow['cumulative_variance'][:5]}")

    print("\n모든 테스트 통과!")
