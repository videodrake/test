# Day 4: 특성 엔지니어링과 선택

## 학습 목표
- 분자 기술자의 특성 엔지니어링 기법을 익힌다
- 특성 선택(Feature Selection) 방법을 이해하고 적용한다
- 다중공선성과 중복 기술자 제거의 중요성을 파악한다

## 이론 (1시간)

### 분자 기술자: 양날의 검
RDKit은 200+개 기술자를 계산할 수 있지만, 모두 유용한 것은 아닙니다:
- **다중공선성**: MW와 Heavy Atom Count는 거의 같은 정보
- **노이즈**: 일부 기술자는 타겟과 무관한 노이즈
- **차원의 저주**: 기술자 > 샘플 수이면 과적합 위험

### 특성 스케일링

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler: 평균=0, 표준편차=1 → 대부분의 알고리즘에 적합
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # test는 train 기��으로 변환!

# MinMaxScaler: 0~1 범위 → 신경망에 적합
# RobustScaler: 이상치에 강건 → 화학 데이터에서 유용
```

> **핵심 규칙**: `fit`은 train에만, `transform`은 train/test 모두에. test에 `fit_transform`을 쓰면 data leakage!

### 특성 선택 방법

#### 1. 분산 기반 필터
```python
from sklearn.feature_selection import VarianceThreshold

# 분산이 0인 (모든 값이 같은) 기술자 제거
selector = VarianceThreshold(threshold=0.01)
X_filtered = selector.fit_transform(X)
print(f"{X.shape[1]} → {X_filtered.shape[1]} 기술자")
```

#### 2. 상관관계 기반 필터
```python
import pandas as pd
import numpy as np

def remove_correlated(df, threshold=0.95):
    """상관관계가 threshold 이상인 기술자 중 하나를 제거"""
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=to_drop), to_drop

X_clean, dropped = remove_correlated(pd.DataFrame(X, columns=feature_names))
print(f"제거된 기술자: {dropped}")
```

#### 3. 모델 기반 선택 (Recursive Feature Elimination)
```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(rf, n_features_to_select=10, step=5)
rfe.fit(X_train, y_train)

selected = [name for name, sel in zip(feature_names, rfe.support_) if sel]
print(f"선택된 기술자: {selected}")
```

#### 4. 특성 중요도 (Feature Importance)
```python
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=feature_names)
top10 = importances.nlargest(10)

# 시각화
top10.plot.barh()
plt.xlabel('Importance')
plt.title('Top 10 Molecular Descriptors')
```

### 파이프라인으로 전처리 자동화

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', RFE(RandomForestClassifier(n_estimators=50, random_state=42),
                     n_features_to_select=15)),
    ('model', RandomForestClassifier(n_estimators=200, random_state=42)),
])

pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)
```

> **Pipeline의 장점**: 전처리와 모델을 하나의 객체로 묶어서 교차 검증 시 data leakage를 방지합니다.

## 실습 (1.5시간)

### Exercise 09: 특성 엔지니어링과 선택

파일: `exercises/phase1/week2/ex09_feature_engineering.py`

**과제:**
1. 200개 기술자가 있는 샘플 데이터에서 분산 0인 기술자 제거
2. 상관관계 0.95 이상인 중복 기술자 제거
3. Random Forest 특성 중요도로 상위 20개 선택
4. 전체 기술자 vs 선택된 기술자로 모델 성능 비교
5. Pipeline으로 전처리+모델을 하나로 묶기

## 참고 자료
- scikit-learn: Feature Selection
- "Molecular Descriptors for Chemoinformatics" (Todeschini & Consonni)
