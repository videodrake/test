"""
Exercise 10: 화합물 라이브러리 클러스터링

목표:
  - K-Means 클러스터링 및 Elbow plot 분석
  - DBSCAN으로 밀도 기반 클러스터링 및 노이즈 식별
  - 각 클러스터에서 대표 분자 선택
  - PCA로 클러스터 시각화

배경:
  신약개발에서 화합물 라이브러리의 화학적 다양성을 분석하고
  구조적으로 유사한 분자들을 그룹화하는 것은 필수적입니다.
  클러스터링을 통해:
  - 라이브러리의 화학 공간 커버리지 평가
  - 각 클러스터에서 대표 분자를 선택하여 효율적인 스크리닝
  - 구조적 이상치(scaffold hopper) 식별

사용 라이브러리:
  numpy, sklearn (KMeans, DBSCAN, PCA, silhouette_score,
  pairwise_distances)
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler


def generate_compound_library(
    n_molecules: int = 500, n_clusters: int = 5, n_descriptors: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """
    5개의 구조적 클러스터를 가진 화합물 라이브러리를 생성합니다.

    각 클러스터는 서로 다른 화학 scaffold를 대표합니다.
    일부 분자는 클러스터 사이에 위치하는 이상치(outlier)입니다.

    Parameters
    ----------
    n_molecules : int
        총 분자 수
    n_clusters : int
        실제 클러스터 수
    n_descriptors : int
        분자 기술자 차원 수

    Returns
    -------
    X : np.ndarray, shape (n_molecules, n_descriptors)
        분자 기술자 행렬
    true_labels : np.ndarray, shape (n_molecules,)
        실제 클러스터 레이블 (0~4, 노이즈는 -1)
    """
    np.random.seed(42)

    molecules_per_cluster = (n_molecules - 20) // n_clusters  # 20개는 노이즈
    centers = np.random.randn(n_clusters, n_descriptors) * 3

    X_list = []
    labels_list = []

    for i in range(n_clusters):
        # 클러스터별로 분산이 다름 (현실적)
        spread = 0.5 + np.random.rand() * 0.5
        cluster_data = centers[i] + np.random.randn(molecules_per_cluster, n_descriptors) * spread
        X_list.append(cluster_data)
        labels_list.extend([i] * molecules_per_cluster)

    # 노이즈 분자 (어떤 클러스터에도 속하지 않음)
    n_noise = n_molecules - molecules_per_cluster * n_clusters
    X_noise = np.random.randn(n_noise, n_descriptors) * 5
    X_list.append(X_noise)
    labels_list.extend([-1] * n_noise)

    X = np.vstack(X_list)
    true_labels = np.array(labels_list)

    # 셔플
    shuffle_idx = np.random.permutation(len(true_labels))
    X = X[shuffle_idx]
    true_labels = true_labels[shuffle_idx]

    return X, true_labels


# =============================================================================
# TODO 1: K-Means Elbow 분석
# =============================================================================
def kmeans_elbow_analysis(
    X: np.ndarray, k_range: range = range(2, 11), random_state: int = 42
) -> list[tuple[int, float]]:
    """
    다양한 k 값에 대한 inertia를 계산하여 elbow plot 데이터를 생성합니다.

    Inertia: 각 포인트와 해당 클러스터 중심 간 거리 제곱합 (낮을수록 좋음)

    Parameters
    ----------
    X : np.ndarray
        기술자 행렬
    k_range : range
        시도할 k 값의 범위
    random_state : int
        시드

    Returns
    -------
    results : list[tuple[int, float]]
        [(k, inertia), ...] 형태의 리스트

    힌트:
      - 각 k에 대해:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X)
        inertia = km.inertia_
    """
    # TODO: k 범위에 대한 inertia를 계산하세요
    pass


# =============================================================================
# TODO 2: 최적 K-Means 클러스터링
# =============================================================================
def perform_kmeans(
    X: np.ndarray, n_clusters: int = 5, random_state: int = 42
) -> tuple[np.ndarray, float]:
    """
    K-Means 클러스터링을 수행하고 silhouette score를 계산합니다.

    Silhouette score: -1~1, 높을수록 클러스터가 잘 분리됨

    Parameters
    ----------
    X : np.ndarray
        기술자 행렬
    n_clusters : int
        클러스터 수
    random_state : int
        시드

    Returns
    -------
    labels : np.ndarray
        클러스터 레이블
    sil_score : float
        Silhouette score

    힌트:
      - km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
      - labels = km.fit_predict(X)
      - sil_score = silhouette_score(X, labels)
    """
    # TODO: K-Means 클러스터링과 silhouette score를 계산하세요
    pass


# =============================================================================
# TODO 3: DBSCAN 클러스터링
# =============================================================================
def perform_dbscan(
    X: np.ndarray, eps: float = 3.0, min_samples: int = 5
) -> tuple[np.ndarray, int, int]:
    """
    DBSCAN 밀도 기반 클러스터링을 수행합니다.

    DBSCAN의 장점: 클러스터 수를 미리 지정하지 않고,
    노이즈(어떤 클러스터에도 속하지 않는 분자)를 자동 식별합니다.

    Parameters
    ----------
    X : np.ndarray
        기술자 행렬
    eps : float
        이웃 탐색 반경
    min_samples : int
        코어 포인트가 되기 위한 최소 이웃 수

    Returns
    -------
    labels : np.ndarray
        클러스터 레이블 (노이즈는 -1)
    n_clusters : int
        발견된 클러스터 수 (노이즈 제외)
    n_noise : int
        노이즈 포인트 수

    힌트:
      - db = DBSCAN(eps=eps, min_samples=min_samples)
      - labels = db.fit_predict(X)
      - n_clusters = len(set(labels) - {-1})
      - n_noise = np.sum(labels == -1)
    """
    # TODO: DBSCAN 클러스터링을 수행하세요
    pass


# =============================================================================
# TODO 4: 대표 분자 선택
# =============================================================================
def select_representatives(
    X: np.ndarray, labels: np.ndarray
) -> dict[int, int]:
    """
    각 클러스터에서 중심에 가장 가까운 분자를 대표로 선택합니다.

    신약개발에서 대표 분자 선택은:
    - 각 scaffold 그룹의 대표 화합물 확보
    - 효율적인 실험 계획 (모든 분자를 테스트할 수 없을 때)
    - 화합물 라이브러리 최적화

    Parameters
    ----------
    X : np.ndarray
        기술자 행렬
    labels : np.ndarray
        클러스터 레이블 (노이즈=-1 포함 가능)

    Returns
    -------
    representatives : dict[int, int]
        {클러스터_레이블: 대표_분자_인덱스} (노이즈 클러스터 -1은 제외)

    힌트:
      - 각 클러스터 label에 대해 (label != -1인 것만):
        1. 해당 클러스터의 분자 인덱스: indices = np.where(labels == label)[0]
        2. 클러스터 중심: centroid = X[indices].mean(axis=0)
        3. 각 분자와 중심 간 거리: np.linalg.norm(X[indices] - centroid, axis=1)
        4. 가장 가까운 분자: indices[np.argmin(distances)]
    """
    # TODO: 각 클러스터의 대표 분자를 선택하세요
    pass


# =============================================================================
# TODO 5: PCA 시각화 데이터 준비
# =============================================================================
def prepare_pca_visualization(
    X: np.ndarray, labels: np.ndarray, n_components: int = 2
) -> tuple[np.ndarray, float]:
    """
    PCA로 차원 축소하여 2D 시각화용 데이터를 준비합니다.

    Parameters
    ----------
    X : np.ndarray
        기술자 행렬
    labels : np.ndarray
        클러스터 레이블
    n_components : int
        축소할 차원 수 (기본 2)

    Returns
    -------
    X_pca : np.ndarray, shape (n_samples, n_components)
        PCA 변환된 좌표
    explained_variance : float
        처음 n_components 주성분의 누적 설명 분산비 (0~1)

    힌트:
      - scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
      - pca = PCA(n_components=n_components)
      - X_pca = pca.fit_transform(X_scaled)
      - explained_variance = pca.explained_variance_ratio_.sum()
    """
    # TODO: PCA 변환과 설명 분산비를 계산하세요
    pass


# =============================================================================
# 메인 실행 및 자가 검증
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 10: 화합물 라이브러리 클러스터링")
    print("=" * 60)

    # 1. 데이터 생성
    X, true_labels = generate_compound_library(500, n_clusters=5)
    print(f"\n[데이터] 분자: {X.shape[0]}, 기술자: {X.shape[1]}")
    unique, counts = np.unique(true_labels, return_counts=True)
    for label, count in zip(unique, counts):
        name = "노이즈" if label == -1 else f"클러스터 {label}"
        print(f"  {name}: {count}개")

    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Elbow 분석
    elbow = kmeans_elbow_analysis(X_scaled)
    assert elbow is not None, "kmeans_elbow_analysis()가 None을 반환했습니다."
    assert len(elbow) == 9, f"k=2~10 결과가 9개여야 하지만 {len(elbow)}개입니다."
    assert elbow[0][1] > elbow[-1][1], "k가 증가하면 inertia가 감소해야 합니다."
    print(f"\n[Elbow 분석]")
    for k, inertia in elbow:
        bar = "█" * int(inertia / elbow[0][1] * 30)
        print(f"  k={k:2d}: inertia={inertia:10.1f} {bar}")

    # 3. K-Means (k=5)
    km_labels, sil = perform_kmeans(X_scaled, n_clusters=5)
    assert km_labels is not None, "perform_kmeans()가 None을 반환했습니다."
    assert km_labels.shape[0] == 500
    assert len(np.unique(km_labels)) == 5
    assert -1 <= sil <= 1
    print(f"\n[K-Means k=5]")
    print(f"  Silhouette score: {sil:.4f}")
    unique_km, counts_km = np.unique(km_labels, return_counts=True)
    for label, count in zip(unique_km, counts_km):
        print(f"  클러스터 {label}: {count}개")

    # 4. DBSCAN
    db_labels, n_clusters_db, n_noise_db = perform_dbscan(X_scaled, eps=3.0, min_samples=5)
    assert db_labels is not None, "perform_dbscan()가 None을 반환했습니다."
    assert db_labels.shape[0] == 500
    assert n_clusters_db >= 1, "최소 1개 클러스터가 필요합니다."
    print(f"\n[DBSCAN eps=3.0, min_samples=5]")
    print(f"  발견된 클러스터: {n_clusters_db}개")
    print(f"  노이즈 분자: {n_noise_db}개")

    # 5. 대표 분자 선택
    reps = select_representatives(X_scaled, km_labels)
    assert reps is not None, "select_representatives()가 None을 반환했습니다."
    assert len(reps) == 5, f"5개 클러스터 대표가 필요하지만 {len(reps)}개입니다."
    assert all(isinstance(idx, (int, np.integer)) for idx in reps.values())
    print(f"\n[대표 분자]")
    for label, idx in sorted(reps.items()):
        print(f"  클러스터 {label} → 분자 #{idx}")

    # 6. PCA 시각화
    pca_result = prepare_pca_visualization(X, km_labels)
    assert pca_result is not None, "prepare_pca_visualization()가 None을 반환했습니다."
    X_pca, explained = pca_result
    assert X_pca.shape == (500, 2), f"PCA 결과 shape 오류: {X_pca.shape}"
    assert 0 < explained <= 1
    print(f"\n[PCA 시각화]")
    print(f"  2D 설명 분산비: {explained:.4f} ({explained*100:.1f}%)")
    print(f"  PC1 범위: [{X_pca[:, 0].min():.2f}, {X_pca[:, 0].max():.2f}]")
    print(f"  PC2 범위: [{X_pca[:, 1].min():.2f}, {X_pca[:, 1].max():.2f}]")

    print("\n" + "=" * 60)
    print("모든 검증을 통과했습니다!")
    print("=" * 60)
