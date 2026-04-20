"""
Exercise 18: 어텐션 메커니즘 - Self-Attention과 SMILES 어텐션

목표:
  - Scaled Dot-Product Attention을 직접 구현
  - SMILES 문자열에 어텐션을 적용하여 중요한 부분 구조 식별
  - 어텐션 가중치를 히트맵으로 시각화 (데이터 생성)
  - 어텐션이 분자의 핵심 작용기에 집중하는지 분석

배경:
  어텐션 메커니즘은 Transformer의 핵심이며, 분자 표현 학습에서
  점점 중요해지고 있습니다. SMILES 기반 어텐션 모델은 분자의
  어떤 부분(원자, 작용기)이 예측에 중요한지 해석할 수 있게 합니다.
  이는 신약개발에서 SAR(Structure-Activity Relationship) 분석과
  직접 연결됩니다.

사용 라이브러리:
  torch, torch.nn, numpy
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# SMILES 문자 사전
SMILES_VOCAB = [
    'C', 'c', 'N', 'n', 'O', 'o', 'S', 's', 'F', 'P',
    'B', 'I', 'H', '(', ')', '[', ']', '=', '#', '+',
    '-', '/', '\\', '@', '.', '0', '1', '2', '3', '4',
    '5', '6', '7', '8', '9', '<PAD>'
]
VOCAB_TO_IDX = {c: i for i, c in enumerate(SMILES_VOCAB)}
VOCAB_SIZE = len(SMILES_VOCAB)
PAD_IDX = VOCAB_TO_IDX['<PAD>']


def smiles_to_indices(smiles: str, max_len: int = 80) -> np.ndarray:
    """SMILES 문자열을 인덱스 시퀀스로 변환합니다."""
    indices = np.full(max_len, PAD_IDX, dtype=np.int64)
    for i, char in enumerate(smiles[:max_len]):
        indices[i] = VOCAB_TO_IDX.get(char, PAD_IDX)
    return indices


def generate_attention_data(n_samples: int = 200) -> tuple[list[str], np.ndarray]:
    """
    어텐션 학습용 SMILES와 활성 데이터를 생성합니다.

    Returns
    -------
    smiles_list : list[str]
    activities : np.ndarray (pIC50)
    """
    np.random.seed(42)

    scaffolds = ["c1ccccc1", "c1ccncc1", "c1ccc2ccccc2c1", "C1CCNCC1", "c1ccoc1"]
    active_groups = ["N(=O)=O", "C(=O)N", "S(=O)(=O)", "C(F)(F)F", "NC(=O)"]
    inactive_groups = ["CC", "CCC", "C(C)C", "OC", "CCCC"]

    smiles_list = []
    activities = []

    for _ in range(n_samples):
        scaffold = np.random.choice(scaffolds)
        is_active = np.random.random() < 0.5

        if is_active:
            group = np.random.choice(active_groups)
            activity = np.random.uniform(6.5, 9.0)
        else:
            group = np.random.choice(inactive_groups)
            activity = np.random.uniform(3.5, 6.0)

        smiles = scaffold + group
        # 추가 치환기
        if np.random.random() < 0.3:
            smiles += np.random.choice(["O", "N", "F", "C"])

        smiles_list.append(smiles)
        activities.append(activity)

    return smiles_list, np.array(activities, dtype=np.float32)


# =============================================================================
# TODO 1: Scaled Dot-Product Attention
# =============================================================================
def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    mask: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled Dot-Product Attention을 구현합니다.

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Parameters
    ----------
    Q : torch.Tensor, shape (..., seq_len_q, d_k)
        Query 행렬
    K : torch.Tensor, shape (..., seq_len_k, d_k)
        Key 행렬
    V : torch.Tensor, shape (..., seq_len_k, d_v)
        Value 행렬
    mask : torch.Tensor, optional, shape (..., seq_len_q, seq_len_k)
        패딩 마스크 (True인 위치는 -inf로 마스킹)

    Returns
    -------
    output : torch.Tensor, shape (..., seq_len_q, d_v)
        어텐션 가중 합
    attention_weights : torch.Tensor, shape (..., seq_len_q, seq_len_k)
        어텐션 가중치 (softmax 결과)

    힌트:
      - d_k = Q.shape[-1]
      - scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
      - if mask is not None:
          scores = scores.masked_fill(mask, float('-inf'))
      - weights = F.softmax(scores, dim=-1)
      - output = weights @ V
    """
    # TODO: Scaled Dot-Product Attention을 구현하세요
    pass


# =============================================================================
# TODO 2: SMILES 어텐션 모델
# =============================================================================
class SMILESAttentionModel(nn.Module):
    """
    SMILES 문자열에 Self-Attention을 적용하는 모델.

    아키텍처:
      Embedding(VOCAB_SIZE, d_model=64)
      -> Self-Attention (Q, K, V를 학습)
      -> 가중 평균으로 분자 벡터 생성
      -> Linear(64, 32) -> ReLU -> Linear(32, 1)

    힌트:
      - __init__에서:
        self.embedding = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_IDX)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc1 = nn.Linear(d_model, 32)
        self.fc2 = nn.Linear(32, 1)

      - forward에서:
        1. emb = self.embedding(x)  # (batch, seq_len, d_model)
        2. Q, K, V = self.W_q(emb), self.W_k(emb), self.W_v(emb)
        3. 패딩 마스크 생성: mask = (x == PAD_IDX)
        4. output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        5. 패딩이 아닌 위치의 평균: mol_vector = output의 평균
        6. MLP: fc1 -> relu -> fc2
    """

    def __init__(self, d_model: int = 64):
        super().__init__()
        # TODO: 레이어를 정의하세요
        pass

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len)
            SMILES 인덱스 시퀀스

        Returns
        -------
        prediction : torch.Tensor, shape (batch, 1)
            예측 pIC50
        attention_weights : torch.Tensor, shape (batch, seq_len, seq_len)
            어텐션 가중치 (시각화용)
        """
        # TODO: 순전파를 구현하세요
        pass


# =============================================================================
# TODO 3: 어텐션 히트맵 데이터 생성
# =============================================================================
def get_attention_heatmap(
    model: nn.Module, smiles: str, max_len: int = 80
) -> tuple[np.ndarray, list[str]]:
    """
    단일 SMILES에 대한 어텐션 가중치를 추출합니다.

    Parameters
    ----------
    model : SMILESAttentionModel
    smiles : str
        SMILES 문자열
    max_len : int

    Returns
    -------
    attention_matrix : np.ndarray, shape (smiles_len, smiles_len)
        실제 SMILES 문자에 대한 어텐션 가중치 (패딩 제외)
    characters : list[str]
        SMILES의 각 문자 리스트

    힌트:
      - model.eval()
      - indices = smiles_to_indices(smiles, max_len)
      - x = torch.tensor(indices).unsqueeze(0)  # 배치 차원 추가
      - with torch.no_grad():
          _, attn = model(x)
      - smiles_len = len(smiles)
      - attention_matrix = attn[0, :smiles_len, :smiles_len].numpy()
    """
    # TODO: 어텐션 히트맵 데이터를 추출하세요
    pass


# =============================================================================
# TODO 4: 학습 함수
# =============================================================================
def train_attention_model(
    model: nn.Module,
    smiles_list: list[str],
    activities: np.ndarray,
    max_len: int = 80,
    n_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
) -> list[float]:
    """
    SMILES 어텐션 모델을 학습합니다.

    Parameters
    ----------
    model : SMILESAttentionModel
    smiles_list : list[str]
    activities : np.ndarray
    max_len : int
    n_epochs : int
    batch_size : int
    lr : float

    Returns
    -------
    losses : list[float]
        에폭별 학습 손실

    힌트:
      - 인덱스 변환: [smiles_to_indices(s, max_len) for s in smiles_list]
      - torch.tensor로 변환 후 DataLoader 사용
      - 학습 루프에서 model(X_batch) -> (pred, attn)
      - 손실은 pred에 대해서만 계산
    """
    # TODO: 학습 루프를 구현하세요
    pass


# =============================================================================
# 메인 실행 및 자가 검증
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("Exercise 18: 어텐션 메커니즘 - SMILES 어텐션")
    print("=" * 60)

    # 1. Scaled Dot-Product Attention 테스트
    print("\n--- Scaled Dot-Product Attention 테스트 ---")
    Q = torch.randn(2, 5, 16)  # (batch=2, seq=5, d_k=16)
    K = torch.randn(2, 5, 16)
    V = torch.randn(2, 5, 16)

    attn_output, attn_weights = scaled_dot_product_attention(Q, K, V)
    assert attn_output is not None, "scaled_dot_product_attention()이 None을 반환했습니다."
    assert attn_output.shape == (2, 5, 16), f"출력 shape 오류: {attn_output.shape}"
    assert attn_weights.shape == (2, 5, 5), f"가중치 shape 오류: {attn_weights.shape}"
    # 어텐션 가중치의 합은 1이어야 함
    weight_sums = attn_weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
        "어텐션 가중치의 합이 1이 아닙니다."
    print(f"  출력 shape: {attn_output.shape}")
    print(f"  가중치 shape: {attn_weights.shape}")
    print(f"  가중치 합 (첫 번째 행): {attn_weights[0, 0].sum().item():.4f}")

    # 마스크 테스트
    mask = torch.zeros(2, 5, 5, dtype=torch.bool)
    mask[:, :, 3:] = True  # 위치 3, 4를 마스킹
    _, masked_weights = scaled_dot_product_attention(Q, K, V, mask=mask)
    assert torch.allclose(masked_weights[:, :, 3:], torch.zeros(2, 5, 2), atol=1e-6), \
        "마스킹된 위치의 어텐션 가중치가 0이 아닙니다."
    print(f"  마스크 테스트 통과 (마스킹 위치 가중치 ≈ 0)")

    # 2. 데이터 생성
    smiles_list, activities = generate_attention_data(200)
    print(f"\n[데이터] 분자: {len(smiles_list)}")
    print(f"  예시: {smiles_list[:3]}")

    # 3. 모델 생성 및 테스트
    model = SMILESAttentionModel(d_model=64)
    assert model is not None

    test_idx = torch.tensor([smiles_to_indices("c1ccccc1NC(=O)", max_len=80)])
    pred, attn = model(test_idx)
    assert pred is not None, "모델 순전파 실패"
    assert pred.shape == (1, 1), f"예측 shape 오류: {pred.shape}"
    assert attn.shape[0] == 1 and attn.shape[1] == 80 and attn.shape[2] == 80

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[SMILES Attention 모델] 파라미터: {n_params:,}")

    # 4. 학습
    losses = train_attention_model(
        model, smiles_list, activities,
        max_len=80, n_epochs=50, batch_size=32, lr=0.001
    )
    assert losses is not None, "train_attention_model()이 None을 반환했습니다."
    assert len(losses) == 50
    assert losses[-1] < losses[0], "학습 손실이 감소해야 합니다."
    print(f"\n[학습 결과]")
    print(f"  초기 손실: {losses[0]:.4f}")
    print(f"  최종 손실: {losses[-1]:.4f}")

    # 5. 어텐션 히트맵
    test_smiles = "c1ccccc1NC(=O)N"
    heatmap_result = get_attention_heatmap(model, test_smiles, max_len=80)
    assert heatmap_result is not None, "get_attention_heatmap()이 None을 반환했습니다."
    attn_matrix, chars = heatmap_result
    assert attn_matrix.shape == (len(test_smiles), len(test_smiles)), \
        f"히트맵 shape 오류: {attn_matrix.shape}"
    assert len(chars) == len(test_smiles)
    print(f"\n[어텐션 히트맵] '{test_smiles}'")
    print(f"  히트맵 shape: {attn_matrix.shape}")
    print(f"  문자: {chars}")
    print(f"  최대 어텐션 위치: {np.unravel_index(attn_matrix.argmax(), attn_matrix.shape)}")

    print("\n" + "=" * 60)
    print("모든 검증을 통과했습니다!")
    print("=" * 60)
