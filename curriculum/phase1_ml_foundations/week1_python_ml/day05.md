# Day 5: 선형대수와 분자 기술자

## 학습 목표
- 행렬 연산, 고유값 분해, SVD의 개념을 복습한다
- PCA(주성분 분석)를 분자 기술자에 적용하는 원리를 이해한다
- PCA를 직접 구현하고 sklearn 결과와 비교한다

## 이론 (1시간)

### 분자 기술자와 차원의 저주
분자 하나를 수백~수천 개의 숫자로 표현할 수 있습니다 (기술자 = descriptors):
- **Morgan fingerprint**: 1024~2048 비트 (구조적 조각)
- **RDKit descriptors**: ~200개 물리화학적 성질
- **3D descriptors**: 수백 개 (형태, 전하 분포)
- **MACCS keys**: 166 비트 (사전 정의된 구조 패턴)

이 고차원 데이터에서 패턴을 찾으려면 **차원 축소**가 필수입니다:
- 시각화 (2D/3D로 축소)
- 노이즈 제거 (중요 성분만 보존)
- 다중공선성 해결 (상관된 기술자들을 독립 성분으로)
- 계산 효율성 (모델 학습 속도 향상)

### 핵심 선형대수 개념

#### 벡터와 행렬로 보는 분자 데이터
```python
import numpy as np

# 분자 기술자 행렬: N개 분자 × M개 기술자
# 각 행 = 하나의 분자, 각 열 = 하나의 기술자
X = np.array([
    [180.16, 1.19, 63.60, 1, 3],   # 아스피린
    [151.16, 0.46, 49.33, 2, 2],   # 아세트아미노펜
    [206.28, 3.18, 49.33, 0, 1],   # 이부프로펜
])
# X.shape = (3, 5) → 3개 분자, 5개 기술자
```

#### 공분산 행렬
기술자 간의 관계를 행렬로 표현합니다.

```python
# 중심화 (평균 0)
X_centered = X - X.mean(axis=0)

# 공분산 행렬 (M × M)
# C[i,j] = 기술자 i와 j가 함께 변하는 정도
cov_matrix = np.cov(X_centered, rowvar=False)
print(f"공분산 행렬 shape: {cov_matrix.shape}")  # (5, 5)
```

#### 고유값 분해 (Eigendecomposition)
공분산 행렬을 분해하여 데이터의 주요 방향(주성분)을 찾습니다.

```
C = V Λ V^T

C: 공분산 행렬
V: 고유벡터 행렬 (주성분의 방향)
Λ: 고유값 대각행렬 (각 방향의 분산 크기)
```

### PCA의 수학적 기반 - 단계별 구현

```python
import numpy as np

def pca_step_by_step(X, n_components=2):
    """PCA를 단계별로 직접 구현"""

    # Step 1: 중심화
    mean = X.mean(axis=0)
    X_centered = X - mean

    # Step 2: 공분산 행렬
    n = X.shape[0]
    cov_matrix = (X_centered.T @ X_centered) / (n - 1)

    # Step 3: 고유값 분해
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Step 4: 큰 고유값 순으로 정렬
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Step 5: 설명된 분산비
    explained_variance_ratio = eigenvalues / eigenvalues.sum()

    # Step 6: 상위 k개 주성분으로 투영
    W = eigenvectors[:, :n_components]  # 투영 행렬
    X_transformed = X_centered @ W      # 차원 축소된 데이터

    return {
        'transformed': X_transformed,
        'eigenvalues': eigenvalues,
        'explained_variance_ratio': explained_variance_ratio,
        'components': eigenvectors[:, :n_components],
    }
```

### 설명된 분산비와 주성분 수 선택
```python
explained_variance_ratio = eigenvalues / eigenvalues.sum()
cumulative = np.cumsum(explained_variance_ratio)

# 95% 분산을 설명하는 데 필요한 주성분 수
n_components_95 = np.argmax(cumulative >= 0.95) + 1
print(f"95% 분산 설명에 {n_components_95}개 주성분 필요")

# Elbow plot으로 시각화
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance_ratio)+1), explained_variance_ratio)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative)+1), cumulative, 'o-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95%')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.legend()
```

### SVD와의 관계
PCA는 SVD(특이값 분해)로도 구현할 수 있으며, 수치적으로 더 안정적입니다.

```python
# SVD를 이용한 PCA
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
# 주성분 = Vt의 행들
# 특이값과 고유값의 관계: eigenvalue = S^2 / (n-1)
```

### 신약개발에서의 활용

| 활용 | 설명 | 예시 |
|------|------|------|
| 화학 공간 시각화 | 2D PCA로 화합물 매핑 | 히트 vs 미스 분포 확인 |
| 특성 압축 | 200+ 기술자 → 10-20개 PC | ML 모델 입력 크기 축소 |
| 다중공선성 제거 | 상관된 기술자 → 독립 PC | 회귀 모델 안정성 향상 |
| 이상치 탐지 | PC 공간에서 극단값 | 측정 오류 화합물 식별 |
| 다양성 분석 | PC 공간 커버리지 | 스크리닝 라이브러리 다양성 평가 |

## 실습 (1.5시간)

### Exercise 05: 분자 기술자 PCA 구현

파일: `exercises/phase1/week1/ex05_linalg_descriptors.py`

**과제:**
1. 랜덤 분자 기술자 행렬 생성 (50 분자 × 20 기술자, 3개 클래스)
2. PCA를 NumPy로 직접 구현 (고유값 분해 사용)
3. sklearn.decomposition.PCA와 결과 비교 (분산비가 동일한지 확인)
4. Elbow 데이터 계산 (각 주성분의 설명된 분산비, 누적 분산비, 95% 도달 주성분 수)

**도전 과제:**
- SVD 기반 PCA도 구현하여 고유값 분해 결과와 비교
- t-SNE(`sklearn.manifold.TSNE`)도 적용하여 PCA와 시각적으로 비교

## 이번 주 요약

이번 주에 배운 내용:
| Day | 주제 | 핵심 도구 |
|-----|------|----------|
| 1 | NumPy 벡터화, 분자량 계산, Tanimoto 유사도 | `numpy` |
| 2 | Pandas로 분자 데이터 전처리와 분석 | `pandas` |
| 3 | Matplotlib/Seaborn으로 과학 시각화 | `matplotlib`, `seaborn` |
| 4 | 통계학: 상관분석, 가설 검정, Lipinski | `scipy.stats` |
| 5 | 선형대수: PCA, 차원 축소 | `numpy.linalg`, `sklearn` |

**다음 주 (Week 2) 예고**: 이 기초 위에 **scikit-learn으로 실제 QSAR 모델**을 만들기 시작합니다! 분자 기술자로 활성/비활성을 분류하고, 물성을 예측하는 ML 모델을 직접 구축합니다.

## 참고 자료
- 3Blue1Brown: "Essence of Linear Algebra" (YouTube, 강력 추천)
- sklearn PCA 문서
- "Dimensionality Reduction in Chemical Space" - 화학 공간 차원 축소 리뷰
- Bishop, Pattern Recognition and Machine Learning, Ch.12 (PCA)
