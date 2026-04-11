# Day 4: ML을 위한 통계학

## 학습 목표
- ML 모델링에 필요한 핵심 통계 개념을 복습한다
- 가설 검정과 상관분석을 분자 데이터에 적용한다
- QSAR(정량적 구조-활성 관계) 모델링의 통계적 기반을 이해한다

## 이론 (1시간)

### ML과 통계학의 연결
QSAR 모델은 본질적으로 통계 모델입니다. "분자의 구조적 특성(X)으로 생물학적 활성(Y)을 예측한다"는 것은 회귀/분류 문제의 정형입니다.

신약개발에서 통계가 필수적인 이유:
- **데이터 품질 진단**: 측정 노이즈, 이상치, 배치 효과 감지
- **특성 선별**: 수천 개 기술자 중 의미 있는 것만 선택
- **모델 검증**: 과적합 판단, 예측 신뢰구간 설정
- **의사결정**: "이 화합물이 활성일 확률은 얼마인가?"

### 핵심 개념

#### 1. 분포와 정규성
많은 ML 알고리즘은 데이터의 정규 분포를 가정합니다.

```python
from scipy import stats
import numpy as np

# IC50 값은 보통 로그 정규 분포 → pIC50로 변환
ic50_values = np.array([100, 250, 50, 1000, 75, 500])  # nM
pic50 = -np.log10(ic50_values * 1e-9)  # pIC50 변환
print(f"IC50: {ic50_values}")
print(f"pIC50: {np.round(pic50, 2)}")

# 정규성 검정 (Shapiro-Wilk)
stat, p_value = stats.shapiro(pic50)
print(f"Shapiro-Wilk p-value: {p_value:.4f}")
# p > 0.05이면 정규 분포 가정을 기각하지 못함
```

> **신약개발 팁**: IC50, EC50 같은 활성 값은 항상 로그 변환(-log10)하여 pIC50, pEC50로 사용합니다. 로그 변환 전에는 극단적으로 치우친 분포이지만, 변환 후에는 정규분포에 가까워져 ML 모델에 적합합니다.

#### 2. 상관분석
기술자 간, 기술자-활성 간 관계를 파악합니다.

```python
# Pearson 상관계수: 선형 관계 (정규 분포 가정)
r, p = stats.pearsonr(df['logP'], df['activity'])
print(f"Pearson r = {r:.3f}, p = {p:.2e}")

# Spearman 상관계수: 단조 관계 (비선형도 포착, 분포 무관)
rho, p = stats.spearmanr(df['logP'], df['activity'])
print(f"Spearman ρ = {rho:.3f}, p = {p:.2e}")
```

| 상관계수 | 사용 시점 | 해석 |
|---------|----------|------|
| Pearson r | 두 변수 모두 연속적이고 선형 관계 | r=0.7 → 강한 양의 선형관계 |
| Spearman ρ | 순위 기반, 비선형 단조 관계 | ρ=0.7 → 한쪽이 크면 다른쪽도 큰 경향 |
| Kendall τ | 작은 데이터, 동점 많을 때 | τ=0.5 → 중등도 단조 관계 |

#### 3. 가설 검정: 활성 vs 비활성 비교
```python
# 활성 그룹과 비활성 그룹의 logP 차이가 통계적으로 유의한가?
active = df[df['active'] == 1]['logP']
inactive = df[df['active'] == 0]['logP']

# 모수적 검정: t-test (정규 분포 가정)
t_stat, p_value = stats.ttest_ind(active, inactive)
print(f"t-test: t={t_stat:.2f}, p={p_value:.2e}")

# 비모수적 검정: Mann-Whitney U (분포 가정 없음)
u_stat, p_value = stats.mannwhitneyu(active, inactive, alternative='two-sided')
print(f"Mann-Whitney: U={u_stat:.0f}, p={p_value:.2e}")

# 효과 크기 (Cohen's d)
d = (active.mean() - inactive.mean()) / np.sqrt(
    (active.std()**2 + inactive.std()**2) / 2
)
print(f"Cohen's d = {d:.2f}")  # |d| > 0.8 → 큰 효과
```

> **핵심**: p-value만으로 판단하지 마세요. p < 0.05라도 효과 크기(effect size)가 작으면 실질적 의미가 없을 수 있습니다. 특히 대규모 스크리닝 데이터에서는 표본이 커서 아주 작은 차이도 유의하게 나옵니다.

#### 4. Lipinski's Rule of Five 검증
```python
# 각 규칙의 준수 비율 계산
rules = {
    'MW ≤ 500': df['MW'] <= 500,
    'logP ≤ 5': df['logP'] <= 5,
    'HBD ≤ 5': df['HBD'] <= 5,
    'HBA ≤ 10': df['HBA'] <= 10,
}

for name, mask in rules.items():
    pct = mask.mean() * 100
    print(f"{name}: {pct:.1f}% 준수")

# 위반 횟수 계산
violations = sum(~v for v in rules.values())
df['n_violations'] = violations

# 위반 횟수와 활성의 상관관계
rho, p = stats.spearmanr(df['n_violations'], df['activity'])
print(f"위반 수 vs 활성: ρ={rho:.3f}, p={p:.2e}")
```

#### 5. 다중 검정 보정
수천 개 기술자를 한 번에 검정하면 위양성이 대량 발생합니다.

```python
from statsmodels.stats.multitest import multipletests

# 각 기술자와 활성 간의 상관관계 p-value
p_values = []
for col in descriptor_columns:
    _, p = stats.pearsonr(df[col], df['activity'])
    p_values.append(p)

# Benjamini-Hochberg 보정 (FDR 제어)
reject, corrected_p, _, _ = multipletests(p_values, method='fdr_bh', alpha=0.05)
n_significant = sum(reject)
print(f"보정 전 유의한 기술자: {sum(np.array(p_values) < 0.05)}")
print(f"보정 후 유의한 기술자: {n_significant}")
```

## 실습 (1.5시간)

### Exercise 04: 약물 유사성 데이터셋 통계 분석

파일: `exercises/phase1/week1/ex04_statistics.py`

**과제:**
1. 샘플 약물 데이터에서 주요 물성의 기술통계량 계산 (mean, std, median, min, max)
2. logP와 활성 간의 Pearson & Spearman 상관관계 분석
3. 활성/비활성 그룹 간 MW, logP 차이를 t-test와 Mann-Whitney U로 검정
4. Lipinski 규칙 위반 횟수별 평균 활성 분석 + 상관관계

**도전 과제:**
- Cohen's d 효과 크기도 계산해보세요
- 다중 검정 보정(Bonferroni 또는 FDR)을 적용해보세요

## 참고 자료
- SciPy stats 모듈 공식 문서
- "Statistical Methods in QSAR" 리뷰
- 다중 검정 보정: Benjamini & Hochberg (1995)
