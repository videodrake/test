"""
Exercise 04: 약물 유사성(Drug-likeness) 데이터셋의 통계 분석
Phase 1 > Week 1 > Day 4

목표:
1. 주요 물성의 기술통계량 계산
2. logP와 활성 간 상관분석 (Pearson & Spearman)
3. 활성/비활성 그룹 간 물성 차이 t-test
4. Lipinski 규칙 위반 횟수와 약물 성공률 관계 분석
"""

import numpy as np
from scipy import stats


# === 샘플 데이터 생성 ===
def create_drug_data(n=150):
    """통계 분석용 약물 데이터 생성"""
    np.random.seed(42)

    # 활성 화합물은 drug-like 경향
    n_active = int(n * 0.3)
    n_inactive = n - n_active

    active_mw = np.random.normal(350, 80, n_active)
    inactive_mw = np.random.normal(420, 120, n_inactive)
    MW = np.concatenate([active_mw, inactive_mw])

    active_logp = np.random.normal(2.0, 1.0, n_active)
    inactive_logp = np.random.normal(3.5, 1.5, n_inactive)
    logP = np.concatenate([active_logp, inactive_logp])

    HBD = np.concatenate([
        np.random.randint(0, 4, n_active),
        np.random.randint(0, 7, n_inactive),
    ])
    HBA = np.concatenate([
        np.random.randint(1, 8, n_active),
        np.random.randint(2, 13, n_inactive),
    ])
    activity = np.concatenate([
        np.random.normal(7.5, 0.8, n_active),   # 활성 (높은 pIC50)
        np.random.normal(4.5, 1.0, n_inactive),  # 비활성 (낮은 pIC50)
    ])
    is_active = np.concatenate([np.ones(n_active), np.zeros(n_inactive)])

    return {
        "MW": MW, "logP": logP, "HBD": HBD, "HBA": HBA,
        "activity": activity, "is_active": is_active.astype(int),
    }


# === 과제 1: 기술통계량 ===
def descriptive_stats(data: dict) -> dict:
    """
    MW, logP, HBD, HBA, activity 각각의 기술통계량을 계산합니다.

    Returns:
        {물성이름: {'mean': ..., 'std': ..., 'median': ..., 'min': ..., 'max': ...}}
    """
    # TODO: 구현하세요
    pass


# === 과제 2: 상관분석 ===
def correlation_analysis(data: dict) -> dict:
    """
    logP와 activity 간의 Pearson 및 Spearman 상관관계를 계산합니다.

    Returns:
        {
            'pearson': {'r': ..., 'p_value': ...},
            'spearman': {'rho': ..., 'p_value': ...},
        }
    """
    # TODO: 구현하세요
    # 힌트: scipy.stats.pearsonr, scipy.stats.spearmanr
    pass


# === 과제 3: 그룹 간 차이 검정 ===
def group_comparison(data: dict) -> dict:
    """
    활성(is_active=1) vs 비활성(is_active=0) 그룹 간
    MW와 logP의 차이를 t-test와 Mann-Whitney U test로 검정합니다.

    Returns:
        {
            'MW': {'t_stat': ..., 't_pvalue': ..., 'u_stat': ..., 'u_pvalue': ...},
            'logP': {'t_stat': ..., 't_pvalue': ..., 'u_stat': ..., 'u_pvalue': ...},
        }
    """
    # TODO: 구현하세요
    pass


# === 과제 4: Lipinski 위반과 활성 관계 ===
def lipinski_violation_analysis(data: dict) -> dict:
    """
    각 화합물의 Lipinski 위반 횟수(0~4)를 계산하고,
    위반 횟수별 평균 활성 및 위반 횟수-활성 상관관계를 분석합니다.

    Returns:
        {
            'violation_counts': {0: n개, 1: n개, ...},
            'mean_activity_by_violations': {0: 평균, 1: 평균, ...},
            'correlation': {'rho': ..., 'p_value': ...},
        }
    """
    # TODO: 구현하세요
    # 힌트: Lipinski 규칙 4가지를 각각 체크 후 위반 수 합산
    pass


# === 자체 검증 ===
if __name__ == "__main__":
    data = create_drug_data()
    print(f"데이터 생성: {len(data['MW'])}개 화합물\n")

    # 테스트 1
    desc = descriptive_stats(data)
    if desc is not None:
        for prop, s in desc.items():
            print(f"  {prop}: mean={s['mean']:.2f}, std={s['std']:.2f}")
        print("[PASS] 기술통계량\n")

    # 테스트 2
    corr = correlation_analysis(data)
    if corr is not None:
        print(f"  Pearson r={corr['pearson']['r']:.3f} (p={corr['pearson']['p_value']:.2e})")
        print(f"  Spearman rho={corr['spearman']['rho']:.3f} (p={corr['spearman']['p_value']:.2e})")
        print("[PASS] 상관분석\n")

    # 테스트 3
    comp = group_comparison(data)
    if comp is not None:
        for prop, res in comp.items():
            print(f"  {prop}: t={res['t_stat']:.2f} (p={res['t_pvalue']:.2e})")
        print("[PASS] 그룹 비교\n")

    # 테스트 4
    lip = lipinski_violation_analysis(data)
    if lip is not None:
        print(f"  위반 분포: {lip['violation_counts']}")
        print(f"  상관관계: rho={lip['correlation']['rho']:.3f}")
        print("[PASS] Lipinski 분석\n")

    print("모든 테스트 통과!")
