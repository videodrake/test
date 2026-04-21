"""
Exercise 02: Pandas로 분자 물성 데이터 분석
Phase 1 > Week 1 > Day 2

목표:
1. 분자 데이터를 DataFrame으로 생성/로드
2. 기본 통계 요약 및 결측치 처리
3. Lipinski's Rule of Five 위반 필터링
4. 화학 클래스별 logP 분포 비교
5. 결과 요약 테이블 출력
"""

import numpy as np
import pandas as pd


# === 샘플 데이터 생성 ===
def create_sample_data() -> pd.DataFrame:
    """
    분석할 샘플 분자 데이터를 생성합니다.

    Returns:
        컬럼: name, MW, logP, HBD, HBA, TPSA, scaffold, activity
    """
    np.random.seed(42)
    n = 100

    data = {
        "name": [f"MOL_{i:04d}" for i in range(n)],
        "MW": np.random.normal(350, 120, n).clip(100, 800),
        "logP": np.random.normal(2.5, 1.8, n),
        "HBD": np.random.randint(0, 8, n),
        "HBA": np.random.randint(0, 14, n),
        "TPSA": np.random.normal(75, 35, n).clip(0, 200),
        "scaffold": np.random.choice(
            ["benzene", "pyridine", "indole", "piperazine", "quinoline"], n
        ),
        "activity": np.random.choice([0, 1], n, p=[0.7, 0.3]),
    }

    df = pd.DataFrame(data)
    # 의도적으로 결측치 삽입 (~10%)
    for col in ["logP", "TPSA"]:
        mask = np.random.random(n) < 0.1
        df.loc[mask, col] = np.nan

    return df


# === 과제 1: 기본 탐색 ===
def explore_data(df: pd.DataFrame) -> dict:
    """
    DataFrame의 기본 통계를 반환합니다.

    Returns:
        {
            'shape': (행, 열),
            'missing': 컬럼별 결측치 수 dict,
            'numeric_summary': describe() 결과
        }
    """
    # TODO: 구현하세요
    pass


# === 과제 2: 결측치 처리 ===
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    결측치를 scaffold 그룹별 중앙값으로 대체합니다.
    그룹 중앙값도 없으면 전체 중앙값을 사용합니다.

    Returns:
        결측치가 처리된 DataFrame (원본 수정하지 말 것)
    """
    # TODO: 구현하세요
    # 힌트: df.copy(), groupby().transform() 사용
    pass


# === 과제 3: Lipinski 필터 ===
def lipinski_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lipinski's Rule of Five를 모두 만족하는 화합물만 반환합니다.
    - MW <= 500, logP <= 5, HBD <= 5, HBA <= 10

    Returns:
        필터링된 DataFrame
    """
    # TODO: 구현하세요
    pass


# === 과제 4: 스캐폴드별 분석 ===
def scaffold_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    스캐폴드(scaffold)별로 다음을 집계합니다:
    - 화합물 수 (count)
    - 평균 logP (mean_logP)
    - 활성 화합물 비율 (hit_rate)

    Returns:
        스캐폴드별 요약 DataFrame, hit_rate 내림차순 정렬
    """
    # TODO: 구현하세요
    pass


# === 자체 검증 ===
if __name__ == "__main__":
    df = create_sample_data()
    print(f"데이터 생성: {df.shape}")

    # 테스트 1: 탐색
    info = explore_data(df)
    if info is not None:
        print(f"[PASS] Shape: {info['shape']}, 결측치: {info['missing']}")

    # 테스트 2: 결측치 처리
    df_clean = handle_missing(df)
    if df_clean is not None:
        assert df_clean.isnull().sum().sum() == 0, "결측치가 남아있습니다"
        assert len(df_clean) == len(df), "행 수가 변하면 안 됩니다"
        print(f"[PASS] 결측치 처리 완료 (결측: 0)")

    # 테스트 3: Lipinski 필터
    df_druglike = lipinski_filter(df_clean if df_clean is not None else df)
    if df_druglike is not None:
        print(f"[PASS] Drug-like: {len(df_druglike)}/{len(df)} 화합물")

    # 테스트 4: 스캐폴드 분석
    summary = scaffold_analysis(df_clean if df_clean is not None else df)
    if summary is not None:
        assert "count" in summary.columns, "count 컬럼 필요"
        assert "hit_rate" in summary.columns, "hit_rate 컬럼 필요"
        print(f"[PASS] 스캐폴드별 요약:\n{summary}")

    print("\n모든 테스트 통과!")
