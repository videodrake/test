# Day 5: 선형대수와 분자 기술자

## 학습 목표
- 행렬 연산, 고유값 분해, SVD의 개념을 복습한다
- PCA(주성분 분석)를 분자 기술자에 적용하는 원리를 이해한다
- PCA를 직접 구현하고 sklearn 결과와 비교한다

## 이론 (1시간)

### 분자 기술자와 차원의 저주
분자 하나를 수백~수천 개의 숫자로 표현할 수 있습니다 (기술자 = descriptors):
- Morgan fingerprint: 1024~2048 비트
- RDKit descriptors: ~200개 물리화학적 성질
- 3D descriptors: 수백 개

이 고차원 데이터에서 패턴을 찾으려면 **차원 축소**가 필수입니다.

### PCA의 수학적 기반

```python
import numpy as np

# 1. 데이터 중심화
X_centered = X - X.mean(axis=0)

# 2. 공분산 행렬
cov_matrix = np.cov(X_centered, rowvar=False)

# 3. 고유값 분해
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# 4. 큰 고유값 순으로 정렬
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# 5. 상위 k개 주성분으로 투영
k = 2
principal_components = X_centered @ eigenvectors[:, :k]
```

### 설명된 분산비
```python
explained_variance_ratio = eigenvalues / eigenvalues.sum()
cumulative = np.cumsum(explained_variance_ratio)

# 95% 분산을 설명하는 데 필요한 주성분 수
n_components_95 = np.argmax(cumulative >= 0.95) + 1
print(f"95% 분산 설명에 {n_components_95}개 주성분 필요")
```

### 신약개발에서의 활용
- **화학 공간 시각화**: 2D PCA로 화합물 라이브러리 매핑
- **특성 압축**: 200+ 기술자를 10-20개 주성분으로 축소
- **다중공선성 제거**: 상관관계 높은 기술자들을 독립적 주성분으로 변환

## 실습 (1.5시간)

### Exercise 05: 분자 기술자 PCA 구현

파일: `exercises/phase1/week1/ex05_linalg_descriptors.py`

**과제:**
1. 랜덤 분자 기술자 행렬 생성 (50 분자 x 20 기술자)
2. PCA를 NumPy로 직접 구현 (고유값 분해 사용)
3. sklearn.decomposition.PCA와 결과 비교
4. 설명된 분산비 그래프 (elbow plot) 작성
5. 2D 주성분 공간에 분자 매핑

## 이번 주 요약
이번 주에 배운 내용:
- **Day 1**: NumPy 벡터화, 분자량 계산, Tanimoto 유사도
- **Day 2**: Pandas로 분자 데이터 전처리와 분석
- **Day 3**: Matplotlib/Seaborn으로 과학 시각화
- **Day 4**: 통계학 기초와 QSAR 통계 분석
- **Day 5**: 선형대수, PCA, 차원 축소

다음 주에는 이 기초 위에 **scikit-learn으로 실제 QSAR 모델**을 만들기 시작합니다!

## 참고 자료
- 3Blue1Brown: "Essence of Linear Algebra" (YouTube)
- sklearn PCA 문서
- "화학 공간의 차원 축소" 관련 논문
