"""
Exercise 03: 분자 데이터 시각화
Phase 1 > Week 1 > Day 3

목표:
1. 주요 분자 물성 분포 히스토그램 (3 패널)
2. MW vs logP 산점도 (활성도 색상 매핑)
3. 물성 상관관계 히트맵
4. PCA 화학 공간 시각화
"""

import numpy as np

# matplotlib, seaborn이 없으면 pip install 필요
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("설치 필요: pip install matplotlib seaborn")
    raise


# === 샘플 데이터 ===
def generate_mol_data(n=200):
    """시각화용 샘플 분자 데이터 생성"""
    np.random.seed(42)
    data = {
        "MW": np.random.normal(380, 100, n).clip(150, 700),
        "logP": np.random.normal(2.5, 1.5, n),
        "TPSA": np.random.normal(75, 30, n).clip(0, 180),
        "HBD": np.random.randint(0, 6, n).astype(float),
        "HBA": np.random.randint(1, 12, n).astype(float),
        "activity": np.random.normal(6.5, 1.2, n).clip(4, 10),  # pIC50
    }
    return data


# === 과제 1: 물성 분포 히스토그램 ===
def plot_property_distributions(data: dict, save_path: str | None = None):
    """
    MW, logP, TPSA의 분포를 1x3 히스토그램 패널로 그립니다.

    요구사항:
    - 서브플롯 1x3 배치
    - 각 축에 물성 이름과 단위 라벨
    - bins=40, 반투명 색상
    - 각 히스토그램에 평균선(수직 점선) 추가
    """
    # TODO: 구현하세요
    pass


# === 과제 2: MW vs logP 산점도 ===
def plot_mw_vs_logp(data: dict, save_path: str | None = None):
    """
    MW vs logP 산점도를 활성도(pIC50)로 색상 매핑합니다.

    요구사항:
    - colorbar 추가 (label: "pIC50")
    - Lipinski 기준선 표시 (MW=500 세로선, logP=5 가로선)
    - 색맹 친화적 colormap (viridis)
    """
    # TODO: 구현하세요
    pass


# === 과제 3: 상관관계 히트맵 ===
def plot_correlation_heatmap(data: dict, save_path: str | None = None):
    """
    물성 간 상관관계를 히트맵으로 시각화합니다.

    요구사항:
    - 숫자형 물성 (MW, logP, TPSA, HBD, HBA, activity)
    - 각 셀에 상관계수 값 표시 (annot=True)
    - 발산형 colormap (coolwarm 또는 RdBu_r)
    """
    # TODO: 구현하세요
    pass


# === 과제 4: PCA 화학 공간 ===
def plot_chemical_space_pca(data: dict, save_path: str | None = None):
    """
    물성 기술자를 PCA로 2D 축소 후 화학 공간을 시각화합니다.

    요구사항:
    - MW, logP, TPSA, HBD, HBA를 표준화 후 PCA
    - 활성도로 색상 매핑
    - 축 라벨에 설명된 분산비(%) 표시
    """
    # TODO: 구현하세요
    # 힌트: sklearn.preprocessing.StandardScaler, sklearn.decomposition.PCA
    pass


# === 자체 검증 ===
if __name__ == "__main__":
    data = generate_mol_data()
    print(f"데이터 생성 완료: {len(data['MW'])}개 분자")

    plot_property_distributions(data)
    print("[DONE] 과제 1: 물성 분포 히스토그램")

    plot_mw_vs_logp(data)
    print("[DONE] 과제 2: MW vs logP 산점도")

    plot_correlation_heatmap(data)
    print("[DONE] 과제 3: 상관관계 히트맵")

    plot_chemical_space_pca(data)
    print("[DONE] 과제 4: PCA 화학 공간")

    plt.show()
    print("\n모든 시각화 완료!")
