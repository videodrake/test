"""
Exercise 01: NumPy로 분자량 계산 및 Tanimoto 유사도 구현
Phase 1 > Week 1 > Day 1

목표:
1. 원자량 배열로 분자량 계산하는 함수 작성
2. 여러 분자의 분자량을 벡터화하여 한 번에 계산
3. Tanimoto 유사도 구현 (단일 쌍 + 배치)
4. 10개 약물의 유사도 행렬(10x10) 계산
"""

import numpy as np

# === 원자량 데이터 ===
ATOMIC_WEIGHTS = {
    'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
    'F': 18.998, 'P': 30.974, 'S': 32.065, 'Cl': 35.453,
    'Br': 79.904
}

# === 과제 1: 분자량 계산 함수 ===
def calculate_molecular_weight(formula: dict) -> float:
    """
    분자식 딕셔너리로부터 분자량을 계산합니다.

    Args:
        formula: {'C': 9, 'H': 8, 'O': 4} 형태의 분자식

    Returns:
        분자량 (Da)

    예시:
        >>> calculate_molecular_weight({'C': 9, 'H': 8, 'O': 4})  # 아스피린
        180.159
    """
    # TODO: NumPy를 사용하여 구현하세요
    pass


# === 과제 2: 벡터화된 분자량 계산 ===
def batch_molecular_weight(formulas: list[dict]) -> np.ndarray:
    """
    여러 분자의 분자량을 한 번에 계산합니다.

    Args:
        formulas: 분자식 딕셔너리들의 리스트

    Returns:
        분자량 배열
    """
    # TODO: 벡터화하여 구현하세요
    pass


# === 과제 3: Tanimoto 유사도 ===
def tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """
    두 비트 벡터 간의 Tanimoto 유사도를 계산합니다.

    Args:
        fp1, fp2: 이진 비트 벡터 (0과 1로 이루어진 배열)

    Returns:
        유사도 (0.0 ~ 1.0)
    """
    # TODO: 구현하세요
    # 힌트: np.sum, 비트 연산 &, | 사용
    pass


# === 과제 4: 유사도 행렬 ===
def tanimoto_matrix(fingerprints: np.ndarray) -> np.ndarray:
    """
    N개 분자 지문의 NxN 유사도 행렬을 계산합니다.

    Args:
        fingerprints: (N, M) shape의 이진 행렬 (N개 분자, M비트 지문)

    Returns:
        (N, N) 유사도 행렬
    """
    # TODO: 행렬 연산으로 효율적으로 구현하세요
    # 힌트: np.dot으로 교집합, 합집합은 a + b - c
    pass


# === 자체 검증 ===
if __name__ == "__main__":
    # 테스트 1: 분자량 계산
    aspirin = {'C': 9, 'H': 8, 'O': 4}
    mw = calculate_molecular_weight(aspirin)
    if mw is not None:
        assert abs(mw - 180.159) < 0.01, f"아스피린 MW 오류: {mw}"
        print(f"[PASS] 아스피린 분자량: {mw:.3f} Da")

    # 테스트 2: 배치 분자량
    molecules = [
        {'C': 9, 'H': 8, 'O': 4},   # 아스피린
        {'C': 8, 'H': 9, 'N': 1, 'O': 2},  # 아세트아미노펜
        {'C': 13, 'H': 18, 'O': 2},  # 이부프로펜
    ]
    mws = batch_molecular_weight(molecules)
    if mws is not None:
        print(f"[PASS] 배치 분자량: {mws}")

    # 테스트 3: Tanimoto 유사도
    fp_a = np.array([1, 1, 0, 1, 0, 1, 0, 0])
    fp_b = np.array([1, 0, 0, 1, 1, 1, 0, 0])
    sim = tanimoto_similarity(fp_a, fp_b)
    if sim is not None:
        expected = 3 / 5  # 교집합 3, 합집합 5
        assert abs(sim - expected) < 0.01, f"Tanimoto 오류: {sim}"
        print(f"[PASS] Tanimoto 유사도: {sim:.3f}")

    # 테스트 4: 유사도 행렬
    fps = np.random.randint(0, 2, size=(10, 64))
    sim_matrix = tanimoto_matrix(fps)
    if sim_matrix is not None:
        assert sim_matrix.shape == (10, 10), f"Shape 오류: {sim_matrix.shape}"
        assert np.allclose(np.diag(sim_matrix), 1.0), "대각선은 1이어야 함"
        assert np.allclose(sim_matrix, sim_matrix.T), "대칭 행렬이어야 함"
        print(f"[PASS] 10x10 유사도 행렬 생성 완료")

    print("\n모든 테스트 통과!")
