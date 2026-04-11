# Day 3: 과학 데이터 시각화

## 학습 목표
- Matplotlib과 Seaborn으로 분자 데이터를 시각화한다
- 논문 수준의 과학 그래프를 작성한다
- 화학 공간(Chemical Space)을 PCA/t-SNE로 시각화한다

## 이론 (1시간)

### 왜 시각화가 중요한가?
신약개발에서 시각화는 단순한 보조 도구가 아닙니다:
- **화학 공간 탐색**: 후보 물질이 어디에 위치하는지 직관적 파악
- **SAR 패턴 발견**: 구조와 활성의 관계를 시각적으로 포착
- **이상치 탐지**: 비정상적인 물성 값을 가진 화합물 식별

### 핵심 시각화 유형

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 물성 분포 (히스토그램)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, prop in zip(axes, ['MW', 'logP', 'TPSA']):
    ax.hist(df[prop], bins=50, color='#6366f1', alpha=0.7)
    ax.set_xlabel(prop)
    ax.set_ylabel('Count')
plt.tight_layout()

# 2. 물성 상관관계 (산점도)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(df['MW'], df['logP'],
                      c=df['activity'], cmap='RdYlGn',
                      alpha=0.6, s=20)
plt.colorbar(scatter, label='Activity (pIC50)')
plt.xlabel('Molecular Weight')
plt.ylabel('logP')
plt.title('MW vs logP colored by Activity')

# 3. 화학 공간 PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
coords = pca.fit_transform(descriptor_matrix)
plt.scatter(coords[:, 0], coords[:, 1], c=activities, cmap='viridis', s=10)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
```

### 논문 수준 그래프 팁
- 축 레이블에 **단위** 포함 (예: "MW (Da)", "logP")
- **색상**: 색맹 친화적 팔레트 사용 (viridis, cividis)
- **해상도**: `dpi=300` 이상
- **폰트**: 충분히 큰 크기 (`fontsize=12+`)

## 실습 (1.5시간)

### Exercise 03: 분자 데이터 시각화

파일: `exercises/phase1/week1/ex03_visualization.py`

**과제:**
1. 주요 분자 물성(MW, logP, TPSA)의 분포 히스토그램 3개 패널
2. MW vs logP 산점도 (활성도로 색상 매핑)
3. 상관관계 히트맵
4. 간단한 PCA 화학 공간 시각화 (랜덤 데이터 사용 가능)

## 참고 자료
- Matplotlib 갤러리
- Seaborn 튜토리얼
- "화학 공간 시각화" 관련 리뷰 논문
