# Day 2: Pandas로 분자 데이터 분석

## 학습 목표
- Pandas DataFrame으로 분자 물성 데이터를 로드하고 탐색한다
- 결측치 처리, 그룹별 통계 등 데이터 전처리를 수행한다
- 분자 데이터에 특화된 EDA(탐색적 데이터 분석) 기법을 익힌다

## 이론 (1시간)

### 분자 데이터의 특성
신약개발 데이터는 일반적인 표형 데이터와 다른 특성이 있습니다:
- **SMILES 문자열**: 분자 구조를 텍스트로 표현
- **다양한 물성**: MW, logP, TPSA, HBD, HBA 등
- **활성 데이터**: IC50, EC50, Ki 값 (종종 로그 스케일로 변환)
- **불균형**: 활성 화합물 < 비활성 화합물 (히트율 ~0.1-1%)

### Pandas 핵심 연산

```python
import pandas as pd

# 분자 데이터 로드
df = pd.read_csv('molecules.csv')

# 기본 탐색
df.info()            # 컬럼 타입, 결측치 확인
df.describe()        # 수치형 컬럼 통계 요약
df['logP'].hist()    # logP 분포 확인

# Lipinski's Rule of Five 필터링
drug_like = df[
    (df['MW'] <= 500) &
    (df['logP'] <= 5) &
    (df['HBD'] <= 5) &
    (df['HBA'] <= 10)
]
print(f"Drug-like: {len(drug_like)}/{len(df)} ({len(drug_like)/len(df)*100:.1f}%)")
```

### 결측치 처리 전략
분자 데이터에서 결측치는 흔합니다:
- **계산 불가**: 일부 기술자가 특정 분자에서 정의되지 않음
- **실험 미수행**: 활성 측정이 안 된 화합물
- **파싱 실패**: SMILES 문자열 오류

```python
# 결측치 확인
df.isnull().sum()

# 전략 1: 제거 (데이터 충분할 때)
df_clean = df.dropna(subset=['activity'])

# 전략 2: 중앙값 대체 (물성 기술자)
df['logP'].fillna(df['logP'].median(), inplace=True)

# 전략 3: 그룹별 대체 (화학 클래스별)
df['logP'] = df.groupby('scaffold')['logP'].transform(
    lambda x: x.fillna(x.median())
)
```

### GroupBy로 SAR 분석
구조-활성 관계(SAR)를 Pandas로 빠르게 분석할 수 있습니다:

```python
# 스캐폴드별 활성 통계
sar = df.groupby('scaffold').agg(
    count=('activity', 'count'),
    mean_activity=('activity', 'mean'),
    std_activity=('activity', 'std'),
    mean_logP=('logP', 'mean'),
).sort_values('mean_activity', ascending=False)
```

## 실습 (1.5시간)

### Exercise 02: 분자 물성 데이터 분석

파일: `exercises/phase1/week1/ex02_pandas_moldata.py`

**과제:**
1. 샘플 분자 데이터를 DataFrame으로 생성 (또는 CSV 로드)
2. 기본 통계 요약 및 결측치 확인
3. Lipinski's Rule of Five 위반 화합물 필터링
4. 화학 클래스별 logP 분포 비교
5. 결과를 정리된 요약 테이블로 출력

## 참고 자료
- Pandas 공식 문서: GroupBy
- Lipinski's Rule of Five 논문 (Lipinski et al., 1997)
