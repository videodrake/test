"""
Exercise 15: CNN으로 SMILES 문자열 기반 분자 속성 예측

목표:
  - SMILES 문자열을 원-핫 인코딩으로 변환
  - 1D CNN 기반 SMILES 분류/회귀 모델 구현
  - SMILES 인코딩 + CNN 학습 파이프라인 완성
  - 문자 수준 특징 학습의 원리 이해

배경:
  SMILES(Simplified Molecular Input Line Entry System)는 분자를
  문자열로 표현하는 표준 방법입니다. 예: 아스피린 = CC(=O)Oc1ccccc1C(=O)O
  CNN은 SMILES 문자열에서 국소적 패턴(작용기, 고리 구조 등)을
  자동으로 학습할 수 있습니다. 이 접근법은 분자 기술자 설계 없이
  end-to-end 학습이 가능한 장점이 있습니다.

사용 라이브러리:
  torch, torch.nn, numpy
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# SMILES에서 사용되는 주요 문자들
SMILES_CHARS = [
    'C', 'c', 'N', 'n', 'O', 'o', 'S', 's', 'F', 'P',
    'B', 'I', 'H', '(', ')', '[', ']', '=', '#', '+',
    '-', '/', '\\', '@', '.', '0', '1', '2', '3', '4',
    '5', '6', '7', '8', '9', ' '  # 패딩 문자
]
CHAR_TO_IDX = {c: i for i, c in enumerate(SMILES_CHARS)}
N_CHARS = len(SMILES_CHARS)


def generate_smiles_data(n_samples: int = 300) -> tuple[list[str], np.ndarray]:
    """
    약물 유사 SMILES 문자열과 활성(pIC50) 데이터를 생성합니다.

    Parameters
    ----------
    n_samples : int
        분자 수

    Returns
    -------
    smiles_list : list[str]
        SMILES 문자열 리스트
    activities : np.ndarray
        pIC50 값
    """
    np.random.seed(42)

    # 실제 약물 유사 SMILES 템플릿
    scaffolds = [
        "c1ccccc1",           # 벤젠
        "c1ccncc1",           # 피리딘
        "c1cc2ccccc2cc1",     # 나프탈렌
        "C1CCCCC1",           # 시클로헥산
        "c1ccc2[nH]ccc2c1",  # 인돌
    ]
    substituents = [
        "O", "N", "F", "C(=O)O", "C(=O)N", "OC", "NC",
        "CC", "C(C)C", "C(=O)", "S", "C(F)(F)F", "Cl",
    ]

    smiles_list = []
    activities = []

    for _ in range(n_samples):
        scaffold = np.random.choice(scaffolds)
        n_subs = np.random.randint(1, 4)
        subs = np.random.choice(substituents, n_subs, replace=True)

        smiles = scaffold
        for sub in subs:
            pos = np.random.randint(0, len(smiles))
            smiles = smiles[:pos] + sub + smiles[pos:]

        # 활성은 SMILES 길이와 특정 패턴에 기반 (시뮬레이션)
        activity = 5.0
        activity += 0.05 * len(smiles)
        if 'N' in smiles or 'n' in smiles:
            activity += 0.5
        if 'F' in smiles:
            activity += 0.3
        if 'O' in smiles:
            activity -= 0.2
        activity += np.random.normal(0, 0.4)
        activity = np.clip(activity, 3.0, 10.0)

        smiles_list.append(smiles)
        activities.append(activity)

    return smiles_list, np.array(activities, dtype=np.float32)


# =============================================================================
# TODO 1: SMILES 원-핫 인코딩
# =============================================================================
def smiles_to_onehot(
    smiles: str, max_len: int = 100
) -> np.ndarray:
    """
    SMILES 문자열을 원-핫 인코딩 행렬로 변환합니다.

    Parameters
    ----------
    smiles : str
        SMILES 문자열
    max_len : int
        최대 문자열 길이 (패딩 적용)

    Returns
    -------
    onehot : np.ndarray, shape (N_CHARS, max_len)
        원-핫 인코딩 행렬 (채널 x 길이)
        - CNN 입력 형식: (채널, 길이)

    힌트:
      - onehot = np.zeros((N_CHARS, max_len), dtype=np.float32)
      - for i, char in enumerate(smiles[:max_len]):
          if char in CHAR_TO_IDX:
              onehot[CHAR_TO_IDX[char], i] = 1.0
          else:
              onehot[CHAR_TO_IDX[' '], i] = 1.0  # 미지 문자는 패딩
      - 나머지 위치는 패딩(' ') 인덱스에 1.0
    """
    # TODO: SMILES를 원-핫 인코딩 행렬로 변환하세요
    pass


# =============================================================================
# TODO 2: 배치 인코딩
# =============================================================================
def encode_smiles_batch(
    smiles_list: list[str], max_len: int = 100
) -> torch.Tensor:
    """
    SMILES 리스트를 배치 텐서로 변환합니다.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES 문자열 리스트
    max_len : int
        최대 길이

    Returns
    -------
    batch_tensor : torch.Tensor, shape (n_samples, N_CHARS, max_len)
        인코딩된 배치 텐서

    힌트:
      - encoded = [smiles_to_onehot(s, max_len) for s in smiles_list]
      - torch.tensor(np.array(encoded), dtype=torch.float32)
    """
    # TODO: 배치 인코딩을 구현하세요
    pass


# =============================================================================
# TODO 3: SMILES CNN 모델 정의
# =============================================================================
class SMILESCNN(nn.Module):
    """
    SMILES 원-핫 인코딩을 입력으로 받는 1D CNN 모델.

    아키텍처:
      Conv1d(N_CHARS, 64, kernel_size=5, padding=2) -> ReLU -> MaxPool1d(2)
      Conv1d(64, 128, kernel_size=3, padding=1) -> ReLU -> MaxPool1d(2)
      AdaptiveAvgPool1d(1) -> Flatten
      Linear(128, 64) -> ReLU -> Dropout(0.3)
      Linear(64, 1)

    kernel_size=5는 5개 연속 문자 패턴을 감지합니다.
    이는 작용기(예: C(=O)O 카르복실기)의 길이에 해당합니다.

    힌트:
      - nn.Conv1d(in_channels, out_channels, kernel_size, padding)
      - nn.MaxPool1d(kernel_size)
      - nn.AdaptiveAvgPool1d(output_size)
        -> 입력 길이에 관계없이 고정 크기 출력
      - nn.Flatten()
    """

    def __init__(self):
        super().__init__()
        # TODO: CNN 레이어를 정의하세요
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, N_CHARS, max_len)

        Returns
        -------
        out : torch.Tensor, shape (batch, 1)
        """
        # TODO: 순전파를 구현하세요
        pass


# =============================================================================
# TODO 4: CNN 학습 함수
# =============================================================================
def train_smiles_cnn(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    n_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
) -> dict[str, list[float]]:
    """
    SMILES CNN 모델을 학습합니다.

    Parameters
    ----------
    model : nn.Module
    X_train : torch.Tensor, shape (n_train, N_CHARS, max_len)
    y_train : torch.Tensor, shape (n_train, 1)
    X_val, y_val : 검증 데이터
    n_epochs : int
    batch_size : int
    lr : float

    Returns
    -------
    history : dict
        {"train_losses": [...], "val_losses": [...]}

    힌트:
      - DataLoader 사용
      - 학습: model.train() -> 배치 루프 -> backward + step
      - 검증: model.eval() -> torch.no_grad()
    """
    # TODO: CNN 학습 루프를 구현하세요
    pass


# =============================================================================
# 메인 실행 및 자가 검증
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("Exercise 15: CNN으로 SMILES 기반 분자 속성 예측")
    print("=" * 60)

    # 1. 데이터 생성
    smiles_list, activities = generate_smiles_data(300)
    print(f"\n[데이터] 분자 수: {len(smiles_list)}")
    print(f"  SMILES 예시: {smiles_list[:3]}")
    print(f"  pIC50 범위: {activities.min():.2f} ~ {activities.max():.2f}")
    print(f"  SMILES 길이 범위: {min(len(s) for s in smiles_list)} ~ "
          f"{max(len(s) for s in smiles_list)}")

    # 2. 원-핫 인코딩 테스트
    test_smiles = "c1ccccc1O"
    onehot = smiles_to_onehot(test_smiles, max_len=50)
    assert onehot is not None, "smiles_to_onehot()이 None을 반환했습니다."
    assert onehot.shape == (N_CHARS, 50), f"shape이 ({N_CHARS}, 50)이어야 하지만 {onehot.shape}입니다."
    assert onehot.sum(axis=0).max() <= 1.0, "각 위치에서 최대 하나의 문자만 활성화되어야 합니다."
    print(f"\n[원-핫 인코딩] '{test_smiles}' -> shape: {onehot.shape}")

    # 3. 배치 인코딩
    MAX_LEN = 80
    batch_tensor = encode_smiles_batch(smiles_list, max_len=MAX_LEN)
    assert batch_tensor is not None, "encode_smiles_batch()가 None을 반환했습니다."
    assert batch_tensor.shape == (300, N_CHARS, MAX_LEN), \
        f"배치 shape 오류: {batch_tensor.shape}"
    print(f"[배치 인코딩] 전체 shape: {batch_tensor.shape}")

    # 4. 학습/검증 분할
    n_train = 240
    X_train = batch_tensor[:n_train]
    y_train = torch.tensor(activities[:n_train], dtype=torch.float32).reshape(-1, 1)
    X_val = batch_tensor[n_train:]
    y_val = torch.tensor(activities[n_train:], dtype=torch.float32).reshape(-1, 1)

    # 5. 모델 생성
    model = SMILESCNN()
    assert model is not None

    # 순전파 테스트
    test_input = torch.randn(2, N_CHARS, MAX_LEN)
    test_output = model(test_input)
    assert test_output is not None, "모델 순전파가 None을 반환했습니다."
    assert test_output.shape == (2, 1), f"출력 shape이 (2,1)이어야 하지만 {test_output.shape}입니다."

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[SMILES CNN 모델] 파라미터 수: {n_params:,}")

    # 6. 학습
    history = train_smiles_cnn(
        model, X_train, y_train, X_val, y_val,
        n_epochs=50, batch_size=32, lr=0.001
    )
    assert history is not None, "train_smiles_cnn()이 None을 반환했습니다."
    assert "train_losses" in history and "val_losses" in history
    assert len(history["train_losses"]) == 50

    print(f"\n[학습 결과]")
    print(f"  초기 학습 손실: {history['train_losses'][0]:.4f}")
    print(f"  최종 학습 손실: {history['train_losses'][-1]:.4f}")
    print(f"  최종 검증 손실: {history['val_losses'][-1]:.4f}")

    assert history["train_losses"][-1] < history["train_losses"][0], \
        "학습 손실이 감소해야 합니다."

    print("\n" + "=" * 60)
    print("모든 검증을 통과했습니다!")
    print("=" * 60)
