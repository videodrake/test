# Day 2: 분류 모델 — 활성/비활성 화합물

## 학습 목표
- 분류 알고리즘(로지스틱 회귀, 결정 트리, 랜덤 포레스트)의 원리를 이해한다
- 분자 활성 분류(active/inactive) 모델을 구축한다
- 결정 트리의 해석 가능성을 신약개발 맥락에서 활용한다

## 이론 (1시간)

### 신약개발에서의 분류 문제
| 문제 | 클래스 | 실무 활용 |
|------|--------|----------|
| 활성 스크리닝 | active / inactive | 히트 발견 |
| BBB 투과성 | BBB+ / BBB- | CNS 약물 설계 |
| hERG 독성 | toxic / non-toxic | 안전성 필터 |
| CYP 억제 | inhibitor / non-inhibitor | 대사 예측 |

### 로지스틱 회귀 (Logistic Regression)
선형 모델의 분류 버전. 출력이 확률(0~1)입니다.

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

# 클래스 예측
y_pred = lr.predict(X_test)

# 확률 예측 — 신약개발에서 중요! 순위 매기기에 사용
y_prob = lr.predict_proba(X_test)[:, 1]  # 활성일 확률
```

> **왜 확률이 중요한가?** 가상 스크리닝에서는 "활성/비활성" 이분법보다 **확률 순위**가 더 유용합니다. 상위 100개를 실험한다면 확률이 높은 순으로 선택합니다.

### 결정 트리 (Decision Tree)
규칙 기반 모델. **해석 가능**하다는 것이 큰 장점입니다.

```python
from sklearn.tree import DecisionTreeClassifier, export_text

dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)

# 트리 규칙 출력 — 화학적 해석 가능!
rules = export_text(dt, feature_names=['MW', 'logP', 'HBD', 'HBA', 'TPSA'])
print(rules)
# 예: "logP <= 3.5이고 MW <= 400이면 → active (신뢰도 85%)"
```

신약개발에서 결정 트리의 가치:
- **SAR 규칙 발견**: "logP가 3 이하이고 HBD가 2 이하일 때 활성"
- **의약화학자와 소통**: 블랙박스 모델보다 규칙이 직관적
- **필터 설계**: 간단한 규칙을 빠른 사전 필터로 사용

### 랜덤 포레스트 (Random Forest)
여러 결정 트리의 **앙상블(ensemble)**. 단일 트리보다 훨씬 강력합니다.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,       # 트리 200개
    max_depth=None,         # 깊이 제한 없음
    min_samples_leaf=5,     # 잎 노드 최소 5개 샘플
    random_state=42,
    n_jobs=-1,              # 모든 CPU 코어 사용
)
rf.fit(X_train, y_train)

# 특성 중요도 — 어떤 기술자가 활성에 가장 영향을 주는가?
importances = rf.feature_importances_
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.3f}")
```

### 데이터 전처리: 스케일링 (미리보기)
로지스틱 회귀는 기술자의 **스케일에 민감**합니다. MW(수백)와 HBD(0~5)의 범위가 다르면 큰 값이 지배합니다. `StandardScaler`로 모든 기술자를 평균=0, 표준편차=1로 맞춰야 합니다. (Day 4에서 더 자세히 다룹니다.)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # train에서 fit
X_test_scaled = scaler.transform(X_test)         # test는 transform만!
```

### 세 모델 비교

| 모델 | 장점 | 단점 | 신약개발 활용 |
|------|------|------|-------------|
| 로지스틱 회귀 | 빠름, 확률 출력, 해석 가능 | 비선형 관계 못 잡음 | 빠른 베이스라인, 대규모 스크리닝 |
| 결정 트리 | 규칙 추출, 시각화 | 과적합 경향 | SAR 규칙 발견, 필터 설계 |
| 랜덤 포레스트 | 높은 정확도, 과적합 억제 | 느림, 해석 어려움 | 최종 예측 모델 |

## 실습 (1.5시간)

### Exercise 07: 활성/비활성 분류기

파일: `exercises/phase1/week2/ex07_activity_classifier.py`

**과제:**
1. 샘플 분자 활성 데이터 생성 (활성 30% / 비활성 70%)
2. Logistic Regression, Decision Tree, Random Forest 세 모델 학습
3. 각 모델의 accuracy, 확률 예측 비교
4. 결정 트리 규칙 텍스트 출력 및 해석
5. Random Forest 특성 중요도 막대그래프

## 참고 자료
- scikit-learn: Decision Trees
- Breiman, L. "Random Forests" (2001)
- 분자 활성 분류의 베스트 프랙티스 리뷰
