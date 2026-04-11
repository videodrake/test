# Day 4: ML을 위한 통계학

## 학습 목표
- ML 모델링에 필요한 핵심 통계 개념을 복습한다
- 가설 검정과 상관분석을 분자 데이터에 적용한다
- QSAR(정량적 구조-활성 관계) 모델링의 통계적 기반을 이해한다

## 이론 (1시간)

### ML과 통계학의 연결
QSAR 모델은 본질적으로 통계 모델입니다. "분자의 구조적 특성(X)으로 생물학적 활성(Y)을 예측한다"는 것은 회귀/분류 문제의 정형입니다.

### 핵심 개념

#### 1. 분포와 정규성
많은 ML 알고리즘은 데이터의 정규 분포를 가정합니다.

```python
from scipy import stats

# IC50 값은 보통 로그 정규 분포
ic50_values = np.array([100, 250, 50, 1000, 75, 500])
pic50 = -np.log10(ic50_values * 1e-9)  # pIC50 변환

# 정규성 검정 (Shapiro-Wilk)
stat, p_value = stats.shapiro(pic50)
print(f"Shapiro-Wilk p-value: {p_value:.4f}")
```

#### 2. 상관분석
기술자 간, 기술자-활성 간 관계를 파악합니다.

```python
# Pearson 상관계수: 선형 관계
r, p = stats.pearsonr(df['logP'], df['activity'])

# Spearman 상관계수: 단조 관계 (비선형도 포착)
rho, p = stats.spearmanr(df['logP'], df['activity'])
```

#### 3. 가설 검정: 활성 vs 비활성 비교
```python
# 활성 그룹과 비활성 그룹의 logP 차이 검정
active = df[df['active'] == 1]['logP']
inactive = df[df['active'] == 0]['logP']

# t-test (정규 분포 가정)
t_stat, p_value = stats.ttest_ind(active, inactive)

# Mann-Whitney U test (비모수적)
u_stat, p_value = stats.mannwhitneyu(active, inactive)
```

#### 4. Lipinski's Rule of Five 검증
```python
# 각 규칙의 위반 비율 계산
rules = {
    'MW <= 500': df['MW'] <= 500,
    'logP <= 5': df['logP'] <= 5,
    'HBD <= 5': df['HBD'] <= 5,
    'HBA <= 10': df['HBA'] <= 10,
}

for name, mask in rules.items():
    pct = mask.mean() * 100
    print(f"{name}: {pct:.1f}% 준수")

# 위반 개수와 활성의 상관관계
df['violations'] = sum(~v for v in rules.values())
r, p = stats.spearmanr(df['violations'], df['activity'])
```

## 실습 (1.5시간)

### Exercise 04: 약물 유사성 데이터셋 통계 분석

파일: `exercises/phase1/week1/ex04_statistics.py`

**과제:**
1. 샘플 약물 데이터에서 주요 물성의 기술통계량 계산
2. logP와 활성 간의 상관관계 분석 (Pearson & Spearman)
3. 활성/비활성 그룹 간 물성 차이 t-test
4. Lipinski 규칙 위반 횟수와 약물 성공률의 관계 분석

## 참고 자료
- SciPy stats 모듈 문서
- QSAR의 통계적 기초 리뷰
