# Day 1: 지도학습과 QSAR 개요

## 학습 목표
- 지도학습(분류/회귀)의 핵심 개념을 이해한다
- QSAR(정량적 구조-활성 관계)의 원리와 신약개발에서의 역할을 파악한다
- scikit-learn으로 첫 번째 QSAR 회귀 모델을 구축한다

## 이론 (1시간)

### QSAR이란?
QSAR(Quantitative Structure-Activity Relationship)은 **분자의 구조적 특성(X)으로 생물학적 활성(Y)을 예측**하는 수학 모델입니다.

```
분자 구조 → [기술자 계산] → 수치 벡터(X) → [ML 모델] → 예측 활성(ŷ)
```

신약개발에서 QSAR의 역할:
- **가상 스크리닝**: 수백만 화합물 중 활성 후보를 사전 예측
- **리드 최적화**: 어떤 구조 변경이 활성을 높이는지 안내
- **ADMET 예측**: 독성, 대사 안정성 등을 실험 전에 추정
- **비용 절감**: 합성/시험 비용 없이 후보군 축소

### 지도학습의 두 가지 유형

| | 분류 (Classification) | 회귀 (Regression) |
|---|---|---|
| 목표 | 범주 예측 (활성/비활성) | 연속값 예측 (pIC50) |
| 예시 | "이 화합물이 hERG 억제제인가?" | "이 화합물의 용해도는?" |
| 지표 | Accuracy, ROC-AUC, F1 | RMSE, R², MAE |
| sklearn | `classifier.predict()` | `regressor.predict()` |

### 모델 학습의 핵심 흐름

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. 데이터 분할 — 반드시 학습 전에!
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 3. 예측
y_pred = model.predict(X_test)

# 4. 평가
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.3f}, R²: {r2:.3f}")
```

### Train/Test Split — 왜 필수인가?

같은 데이터로 학습하고 평가하면 **과적합(overfitting)**을 감지할 수 없습니다.

```python
# 나쁜 예: 동일 데이터로 학습 & 평가 → 낙관적 결과
model.fit(X, y)
y_pred = model.predict(X)  # R²가 높아 보이지만 실제 성능은 다름

# 좋은 예: 분리된 테스트 세트로 평가
model.fit(X_train, y_train)
y_pred = model.predict(X_test)  # 처음 보는 데이터에 대한 진짜 성능
```

> **신약개발 팁**: 화학 데이터에서는 단순 랜덤 분할보다 **scaffold split**(골격 기반 분할)이 더 현실적입니다. 같은 골격의 분자가 train/test에 모두 들어가면 유사도에 의한 "data leakage"가 발생합니다. 이건 Week 2 Day 3에서 더 다룹니다.

### 교차 검증 (Cross-Validation)

단일 분할은 운에 따라 결과가 달라집니다. K-fold 교차 검증으로 안정적인 평가:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"5-Fold R²: {scores.mean():.3f} ± {scores.std():.3f}")
```

### 첫 번째 QSAR: 용해도 예측

ESOL(Estimated SOLubility)은 대표적인 QSAR 벤치마크입니다:
- **입력(X)**: 분자 기술자 (MW, logP, HBD, HBA, TPSA, 회전 가능 결합 수 등)
- **출력(y)**: logS (수용해도, log mol/L)
- **의미**: logS가 높을수록 잘 녹음 → 경구 약물에 유리

```python
from sklearn.ensemble import RandomForestRegressor

# Random Forest — 분자 데이터에 강한 앙상블 모델
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(f"RF RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
print(f"RF R²:   {r2_score(y_test, y_pred):.3f}")
```

## 실습 (1.5시간)

### Exercise 06: 첫 번째 QSAR 모델 — 용해도 예측

파일: `exercises/phase1/week2/ex06_first_qsar.py`

**과제:**
1. 샘플 분자 기술자 데이터 로드 (MW, logP, HBD, HBA, TPSA → logS)
2. Train/Test 분할 (80/20, random_state=42)
3. LinearRegression과 RandomForest 두 모델 학습
4. RMSE, R² 로 두 모델 성능 비교
5. 5-fold 교차 검증 수행

## 참고 자료
- scikit-learn 공식 튜토리얼: Supervised Learning
- Delaney, J.S. "ESOL: Estimating Aqueous Solubility Directly from Molecular Structure" (2004)
- 화학 데이터 분할 전략 리뷰
