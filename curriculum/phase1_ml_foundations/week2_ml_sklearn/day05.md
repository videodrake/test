# Day 5: 비지도 학습 — 화학 공간 클러스터링

## 학습 목표
- K-Means, DBSCAN 클러스터링 알고리즘의 원리를 이해한다
- 화합물 라이브러리를 구조적 유사성으로 그룹화한다
- 클러스터링을 활용한 다양성 분석과 대표 화합물 선택을 수행한다

## 이론 (1시간)

### 왜 클러스터링인가?
신약개발에서 비지도 학습의 활용:
- **라이브러리 다양성 분석**: 스크리닝 라이브러리가 화학 공간을 잘 커버하는가?
- **대표 화합물 선택**: 각 클러스터에서 1개씩 선택 → 최소한의 실험으로 최대 커버리지
- **히트 클러스터 분석**: 스크리닝 히트들이 어떤 구조적 그룹에 모여있는가?
- **scaffold 분류**: 자동으로 화학 계열(chemical series) 식별

### K-Means 클러스터링

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# 각 클러스터 크기
for i in range(5):
    print(f"Cluster {i}: {sum(labels == i)} 화합물")
```

#### 최적 K 선택: Elbow Method + Silhouette Score

```python
from sklearn.metrics import silhouette_score

scores = []
for k in range(2, 15):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    scores.append({
        'k': k,
        'inertia': km.inertia_,
        'silhouette': silhouette_score(X_scaled, labels),
    })
# Silhouette가 최대인 k 선택 (0~1, 높을수록 좋은 클러스터)
```

### DBSCAN — 밀도 기반 클러스터링

K-Means와 달리 **클러스터 수를 미리 지정하지 않아도** 됩니다.

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=1.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = sum(labels == -1)
print(f"클러스터: {n_clusters}개, 노이즈: {n_noise}개")
```

| 파라미터 | 의미 | 화학 데이터 팁 |
|---------|------|-------------|
| `eps` | 이웃 반경 | 기술자 공간 스케일에 따라 조정 |
| `min_samples` | 최소 이웃 수 | 작으면 작은 클러스터 허용, 크면 노이즈 많아짐 |

### 클러스터링 활용: 대표 화합물 선택

```python
# 각 클러스터의 중심에 가장 가까운 화합물 선택
representatives = []
for i in range(n_clusters):
    mask = labels == i
    cluster_data = X_scaled[mask]
    center = cluster_data.mean(axis=0)
    distances = np.linalg.norm(cluster_data - center, axis=1)
    rep_idx = np.where(mask)[0][distances.argmin()]
    representatives.append(rep_idx)

print(f"대표 화합물 {len(representatives)}개 선택")
```

### 클러스터링 시각화 (PCA + 색상)

```python
from sklearn.decomposition import PCA

coords = PCA(n_components=2).fit_transform(X_scaled)
plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
plt.scatter(coords[representatives, 0], coords[representatives, 1],
            c='red', marker='*', s=200, label='대표')
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.legend(); plt.title('Chemical Space Clustering')
```

### Week 2 요약

| Day | 주제 | 핵심 |
|-----|------|------|
| 1 | 지도학습, QSAR | train/test split, 교차 검증 |
| 2 | 분류 모델 | LogReg, DT, RF, 특성 중요도 |
| 3 | 모델 평가 | ROC-AUC, F1, 불균형 처리 |
| 4 | 특성 엔지니어링 | 스케일링, 선택, Pipeline |
| 5 | 비지도 학습 | K-Means, DBSCAN, 다양성 분석 |

**다음 주 (Week 3) 예고**: PyTorch로 **딥러닝 기초**에 진입합니다. 텐서, 자동미분, 신경망 구축부터 분자 물성 예측 NN까지!

## 실습 (1.5시간)

### Exercise 10: 화학 공간 클러스터링

파일: `exercises/phase1/week2/ex10_chemical_clustering.py`

**과제:**
1. 500개 분자 기술자 데이터 생성 (5개 클래스)
2. K-Means: k=2~12에 대해 Elbow plot + Silhouette score
3. DBSCAN: eps 파라미터 탐색
4. 각 클러스터에서 대표 화합물 선택
5. PCA 2D 시각화 (클러스터 색상 + 대표 강조)

## 참고 자료
- scikit-learn: Clustering
- "Diversity Selection in Drug Discovery" 리뷰
- Butina 클러스터링 논문 (Taylor, 1977)
