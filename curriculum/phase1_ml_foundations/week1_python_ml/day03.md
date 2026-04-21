# Day 3: 과학 데이터 시각화

## 학습 목표
- Matplotlib과 Seaborn으로 분자 데이터를 시각화한다
- 논문 수준의 과학 그래프를 작성한다
- 화학 공간(Chemical Space)을 PCA/t-SNE로 시각화한다

## 이론 (1시간)

### 왜 시각화가 중요한가?
신약개발에서 시각화는 단순한 보조 도구가 아닙니다:
- **화학 공간 탐색**: 후보 물질이 알려진 약물 대비 어디에 위치하는지 직관적 파악
- **SAR 패턴 발견**: 구조와 활성의 관계를 시각적으로 포착
- **이상치 탐지**: 비정상적인 물성 값을 가진 화합물 식별
- **의사결정 지원**: 스크리닝 결과를 팀에게 효과적으로 전달

### 핵심 시각화 유형

#### 1. 물성 분포 (히스토그램/KDE)
화합물 라이브러리의 물성 분포를 파악하는 첫 단계입니다.

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1x3 패널 히스토그램
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
properties = {'MW (Da)': mw_values, 'logP': logp_values, 'TPSA (Å²)': tpsa_values}

for ax, (name, values) in zip(axes, properties.items()):
    ax.hist(values, bins=40, color='#6366f1', alpha=0.7, edgecolor='white')
    ax.axvline(np.mean(values), color='red', linestyle='--', label=f'Mean: {np.mean(values):.1f}')
    ax.set_xlabel(name, fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.legend()

plt.tight_layout()
plt.savefig('property_distributions.png', dpi=300, bbox_inches='tight')
```

#### 2. 산점도 (Scatter Plot)
두 물성 간의 관계를 활성도와 함께 3차원으로 표현합니다.

```python
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(df['MW'], df['logP'],
                     c=df['pIC50'], cmap='viridis',
                     alpha=0.6, s=25, edgecolors='none')
plt.colorbar(scatter, label='pIC50')

# Lipinski 기준선
ax.axhline(y=5, color='red', linestyle=':', alpha=0.5, label='logP = 5')
ax.axvline(x=500, color='red', linestyle=':', alpha=0.5, label='MW = 500')

ax.set_xlabel('Molecular Weight (Da)', fontsize=12)
ax.set_ylabel('logP', fontsize=12)
ax.legend()
ax.set_title('Chemical Space: MW vs logP')
```

#### 3. 상관관계 히트맵
물성 간 상관관계를 한눈에 파악합니다. 다중공선성(multicollinearity) 진단에도 유용합니다.

```python
import pandas as pd

corr = df[['MW', 'logP', 'TPSA', 'HBD', 'HBA', 'pIC50']].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax,
            square=True, linewidths=0.5)
ax.set_title('Property Correlation Matrix')
```

#### 4. 화학 공간 PCA 시각화
고차원 기술자를 2D로 축소하여 화합물의 분포와 클러스터를 관찰합니다.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 표준화 → PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(descriptor_matrix)

pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(coords[:, 0], coords[:, 1],
                     c=activities, cmap='RdYlGn', s=15, alpha=0.7)
plt.colorbar(scatter, label='Activity')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('Chemical Space (PCA)')
```

### 논문 수준 그래프 체크리스트
| 항목 | 기준 |
|------|------|
| 축 라벨 | 물성명 + 단위 (예: "MW (Da)") |
| 폰트 크기 | 최소 12pt (축 라벨), 10pt (눈금) |
| 색상 | 색맹 친화적 (viridis, cividis, plasma) |
| 해상도 | 최소 300 dpi |
| 범례 | 간결하고 위치 적절 |
| 그리드 | 필요시 `alpha=0.3`으로 은은하게 |

### 신약개발에서 자주 쓰는 시각화

| 시각화 유형 | 용도 | 라이브러리 |
|------------|------|-----------|
| 분포 히스토그램 | 물성 분포 파악 | Matplotlib, Seaborn |
| 산점도 행렬 | 다변량 관계 탐색 | Seaborn pairplot |
| 상관 히트맵 | 기술자 간 관계 | Seaborn heatmap |
| PCA/t-SNE | 화학 공간 매핑 | sklearn + Matplotlib |
| 박스플롯 | 그룹별 물성 비교 | Seaborn boxplot |
| 바이올린 플롯 | 분포 + 밀도 시각화 | Seaborn violinplot |

## 실습 (1.5시간)

### Exercise 03: 분자 데이터 시각화

파일: `exercises/phase1/week1/ex03_visualization.py`

**과제:**
1. 주요 분자 물성(MW, logP, TPSA)의 분포 히스토그램 3개 패널
2. MW vs logP 산점도 (활성도로 색상 매핑, Lipinski 기준선 포함)
3. 상관관계 히트맵 (annot=True, 발산형 colormap)
4. PCA 화학 공간 시각화 (표준화 → 2D 투영)

**힌트:**
- `fig, axes = plt.subplots(1, 3)` 으로 서브플롯 생성
- `ax.axhline()`, `ax.axvline()`으로 기준선 추가
- sklearn 없이도 Day 5에서 배울 PCA 직접 구현 가능

## 참고 자료
- Matplotlib 갤러리: 다양한 예제 코드
- Seaborn 튜토리얼: 통계 시각화 가이드
- "Visualizing Chemical Space" - 화학 공간 시각화 리뷰 논문
