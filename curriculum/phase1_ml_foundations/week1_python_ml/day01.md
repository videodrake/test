# Day 1: NumPy와 분자 데이터 처리

## 학습 목표
- NumPy의 벡터화 연산과 브로드캐스팅을 이해한다
- 분자 데이터(원자량, 비트 벡터)를 NumPy로 효율적으로 처리한다
- Tanimoto 유사도를 NumPy로 구현한다

## 이론 (1시간)

### 왜 NumPy인가?
신약개발에서는 수십만~수백만 개의 화합물을 다루게 됩니다. Python의 기본 리스트로는 이 규모의 데이터를 처리하기 어렵습니다. NumPy는 C로 구현된 배열 연산을 제공하여 **100~1000배** 빠른 계산이 가능합니다.

### 핵심 개념

#### 1. 벡터화 (Vectorization)
반복문 대신 배열 전체에 연산을 적용하는 패러다임입니다.

```python
import numpy as np

# 느린 방법: 반복문
masses = [12.011, 1.008, 14.007, 15.999]
result = []
for m in masses:
    result.append(m * 2)

# 빠른 방법: 벡터화
masses = np.array([12.011, 1.008, 14.007, 15.999])
result = masses * 2  # 한 번에 모든 원소에 적용
```

신약개발 맥락에서 분자량 계산을 예로 들면:
```python
# 아스피린 (C9H8O4)의 분자량
atomic_weights = {'C': 12.011, 'H': 1.008, 'O': 15.999}
formula = {'C': 9, 'H': 8, 'O': 4}

weights = np.array([atomic_weights[a] for a in formula])
counts = np.array([formula[a] for a in formula])
molecular_weight = np.sum(weights * counts)  # 180.159
```

#### 2. 브로드캐스팅 (Broadcasting)
서로 다른 크기의 배열 간 연산을 가능하게 합니다.

```python
# 여러 분자의 물성을 한 번에 정규화
properties = np.array([
    [180.16, 1.19, 63.60],   # 아스피린: MW, logP, TPSA
    [151.16, 0.46, 49.33],   # 아세트아미노펜
    [206.28, 3.18, 49.33],   # 이부프로펜
])

mean = properties.mean(axis=0)  # 각 물성의 평균
std = properties.std(axis=0)    # 각 물성의 표준편차
normalized = (properties - mean) / std  # 브로드캐스팅으로 정규화
```

#### 3. Tanimoto 유사도
두 분자의 구조적 유사성을 측정하는 핵심 지표입니다. 분자 지문(fingerprint)을 비트 벡터로 표현한 후 계산합니다.

$$T(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{c}{a + b - c}$$

- a: A에서 1인 비트 수
- b: B에서 1인 비트 수
- c: 둘 다 1인 비트 수

```python
def tanimoto_similarity(fp1, fp2):
    """두 비트 벡터 간의 Tanimoto 유사도 계산"""
    fp1, fp2 = np.asarray(fp1), np.asarray(fp2)
    intersection = np.sum(fp1 & fp2)
    union = np.sum(fp1 | fp2)
    if union == 0:
        return 0.0
    return intersection / union
```

### 실무에서의 활용
- **가상 스크리닝**: 수백만 화합물의 유사도를 행렬로 계산
- **클러스터링**: 화합물 라이브러리를 구조적 유사성으로 그룹화
- **기술자 계산**: 분자의 물리화학적 성질을 수치 벡터로 변환

## 실습 (1.5시간)

### Exercise 01: NumPy로 분자량 계산 및 Tanimoto 유사도 구현

파일: `exercises/phase1/week1/ex01_numpy_chem.py`

**과제:**
1. 원자량 배열과 원소 개수 배열을 입력받아 분자량을 계산하는 함수 작성
2. 여러 분자의 분자량을 한 번에 계산하는 벡터화된 함수 작성
3. Tanimoto 유사도를 NumPy로 구현 (단일 쌍 + 배치)
4. 10개 약물의 유사도 행렬(10x10) 계산

**힌트:**
- `np.dot()`으로 비트 벡터의 교집합을 효율적으로 계산할 수 있습니다
- 행렬 곱으로 모든 쌍의 유사도를 한 번에 구할 수 있습니다

## 참고 자료
- NumPy 공식 문서: Broadcasting
- "NumPy를 활용한 과학 컴퓨팅" (한국어 자료)
- Tanimoto coefficient 위키피디아
