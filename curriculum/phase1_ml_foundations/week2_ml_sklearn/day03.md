# Day 3: 모델 평가와 불균형 데이터

## 학습 목표
- 분류 모델의 다양한 평가 지표(ROC-AUC, Precision-Recall, F1)를 이해한다
- 클래스 불균형 문제와 해결 전략을 익힌다
- 신약개발 스크리닝 맥락에서 올바른 지표를 선택한다

## 이론 (1시간)

### 왜 Accuracy만으로 부족한가?
약물 스크리닝 데이터는 극도로 **불균형**합니다:
- 히트율(hit rate): 보통 0.1~1%
- 100개 중 1개만 활성이면, "전부 비활성" 예측으로도 99% accuracy

```python
# 함정: accuracy가 높아 보이지만 쓸모없는 모델
import numpy as np
y_true = np.array([0]*990 + [1]*10)  # 1% 활성
y_pred = np.zeros(1000)               # 전부 비활성 예측
accuracy = (y_true == y_pred).mean()   # 0.99 (99%!) 하지만 활성 하나도 못 찾음
```

### 혼동 행렬 (Confusion Matrix)

```
                예측: Inactive    예측: Active
실제: Inactive      TN              FP (false alarm)
실제: Active        FN (missed)     TP (hit!)
```

```python
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred, target_names=['Inactive', 'Active']))
```

### 핵심 지표

| 지표 | 계산 | 의미 | 신약개발 맥락 |
|------|------|------|-------------|
| **Precision** | TP / (TP+FP) | 예측한 활성 중 진짜 활성 비율 | 합성할 화합물의 성공률 |
| **Recall (Sensitivity)** | TP / (TP+FN) | 실제 활성 중 찾아낸 비율 | 활성 화합물을 놓치지 않는 능력 |
| **F1 Score** | 2 × (P×R)/(P+R) | Precision과 Recall의 조화 평균 | 균형 잡힌 단일 지표 |
| **ROC-AUC** | 곡선 아래 면적 | 임계값에 무관한 전체 성능 | 모델 간 비교에 최적 |

### 신약개발에서의 지표 선택

```python
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
import matplotlib.pyplot as plt

# ROC-AUC: 모델 비교에 가장 표준적
auc = roc_auc_score(y_test, y_prob)

# Precision-Recall: 불균형 데이터에서 ROC-AUC보다 민감
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

# 실무 시나리오별 선택:
# "상위 100개를 합성할 예산" → Precision@100 (상위 100개 중 활성 비율)
# "활성 화합물을 놓치면 안 됨" → Recall 중시
# "균형 잡힌 성능" → F1 또는 ROC-AUC
```

### 불균형 데이터 해결 전략

#### 1. 클래스 가중치 조정
```python
# 소수 클래스에 더 큰 가중치 → 모델이 활성을 더 중시
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
# 또는 수동 지정: class_weight={0: 1, 1: 10}
```

#### 2. SMOTE (합성 오버샘플링)
소수 클래스의 합성 샘플을 생성합니다.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print(f"Before: {sum(y_train==1)}/{len(y_train)}")
print(f"After:  {sum(y_resampled==1)}/{len(y_resampled)}")
```

> **주의**: SMOTE는 **train 세트에만** 적용합니다. test 세트에 적용하면 평가가 오염됩니다.

#### 3. 임계값 조정
기본 임계값 0.5 대신 최적 임계값을 찾습니다.

```python
from sklearn.metrics import f1_score

best_f1, best_threshold = 0, 0.5
for t in np.arange(0.1, 0.9, 0.05):
    y_pred_t = (y_prob >= t).astype(int)
    f1 = f1_score(y_test, y_pred_t)
    if f1 > best_f1:
        best_f1, best_threshold = f1, t

print(f"최적 임계값: {best_threshold:.2f} (F1: {best_f1:.3f})")
```

### Scaffold Split — 화학적으로 공정한 평가

랜덤 분할은 유사한 분자가 train/test에 모두 포함되어 성능이 과대평가됩니다.

```python
# 개념적 코드 (Phase 2에서 RDKit으로 본격 구현)
# Scaffold split: 같은 골격의 분자는 모두 같은 세트에
from collections import defaultdict

scaffold_groups = defaultdict(list)
for i, scaffold in enumerate(scaffolds):
    scaffold_groups[scaffold].append(i)

# 골격 단위로 train/test 분배
train_idx, test_idx = [], []
for scaffold, indices in scaffold_groups.items():
    if len(train_idx) < len(X) * 0.8:
        train_idx.extend(indices)
    else:
        test_idx.extend(indices)
```

## 실습 (1.5시간)

### Exercise 08: 모델 평가와 불균형 처리

파일: `exercises/phase1/week2/ex08_evaluation.py`

**과제:**
1. 불균형 데이터 생성 (활성 10%, 비활성 90%)
2. 기본 RF 모델의 confusion matrix, classification report 출력
3. class_weight='balanced' 적용 전/후 비교
4. ROC 곡선과 Precision-Recall 곡선 그리기
5. 최적 임계값 탐색 (F1 기준)

## 참고 자료
- scikit-learn: Metrics and scoring
- imbalanced-learn 문서 (SMOTE)
- "Handling Imbalanced Datasets in Drug Discovery" 리뷰
