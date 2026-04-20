"""
Exercise 16: 그래프 신경망(GNN) 입문 - 분자 그래프와 GCN

목표:
  - 분자를 그래프(노드=원자, 엣지=결합)로 표현하는 방법 이해
  - 간단한 GCN(Graph Convolutional Network) 레이어를 직접 구현
  - 그래프 수준 분류를 위한 readout(global pooling) 구현
  - 메시지 패싱의 원리 이해

배경:
  분자는 자연스럽게 그래프 구조를 가집니다. 원자가 노드, 화학 결합이
  엣지가 됩니다. GNN은 이 그래프 구조를 직접 입력으로 받아 분자의
  성질을 예측할 수 있습니다. 기존 분자 기술자(MW, logP 등)와 달리
  GNN은 분자의 위상적 구조를 그대로 활용하므로, 사람이 설계하지 못한
  구조적 특징도 자동으로 학습할 수 있습니다.

사용 라이브러리:
  torch, torch.nn, numpy
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_molecule_graphs(
    n_molecules: int = 200,
) -> list[dict]:
    """
    간단한 분자 그래프 데이터를 생성합니다.

    각 분자는 다음을 포함합니다:
    - node_features: (n_atoms, n_atom_features) 원자 특성
    - adjacency: (n_atoms, n_atoms) 인접 행렬
    - label: 0 또는 1 (활성/비활성)

    원자 특성: [원자 번호(정규화), 원자가 전자, 방향족 여부]

    Returns
    -------
    molecules : list[dict]
        각 dict는 {"node_features", "adjacency", "label"} 포함
    """
    np.random.seed(42)
    molecules = []

    for i in range(n_molecules):
        n_atoms = np.random.randint(5, 15)

        # 원자 특성 생성 (3차원: 원자번호(정규화), 원자가전자, 방향족)
        atom_numbers = np.random.choice([6, 7, 8, 9, 16], n_atoms)  # C, N, O, F, S
        valence = np.random.randint(1, 5, n_atoms)
        aromatic = np.random.randint(0, 2, n_atoms)

        node_features = np.column_stack([
            atom_numbers / 20.0,  # 정규화
            valence / 4.0,
            aromatic.astype(float),
        ]).astype(np.float32)

        # 인접 행렬 생성 (연결된 분자 그래프)
        adjacency = np.zeros((n_atoms, n_atoms), dtype=np.float32)
        # 최소 스패닝 트리 (연결 보장)
        for j in range(1, n_atoms):
            k = np.random.randint(0, j)
            adjacency[j, k] = 1.0
            adjacency[k, j] = 1.0
        # 추가 엣지 (고리 구조 등)
        n_extra = np.random.randint(0, n_atoms // 2)
        for _ in range(n_extra):
            a, b = np.random.randint(0, n_atoms, 2)
            if a != b:
                adjacency[a, b] = 1.0
                adjacency[b, a] = 1.0

        # 레이블 (질소/황 포함 + 방향족 비율에 기반)
        n_ratio = np.mean(atom_numbers == 7) + np.mean(atom_numbers == 16)
        aromatic_ratio = np.mean(aromatic)
        prob = 1.0 / (1.0 + np.exp(-(n_ratio * 3 + aromatic_ratio * 2 - 1.5)))
        label = int(np.random.random() < prob)

        molecules.append({
            "node_features": node_features,
            "adjacency": adjacency,
            "label": label,
        })

    return molecules


# =============================================================================
# TODO 1: 간단한 GCN 레이어 구현
# =============================================================================
class SimpleGCNLayer(nn.Module):
    """
    GCN 레이어를 직접 구현합니다.

    GCN 공식: H' = sigma(D^(-1/2) * A_hat * D^(-1/2) * H * W)
    간소화: H' = sigma(A_hat * H * W)  (degree normalization 생략)

    여기서:
    - A_hat = A + I (자기 루프 추가된 인접 행렬)
    - H: 현재 노드 특성 행렬
    - W: 학습 가능한 가중치 행렬
    - sigma: 활성화 함수 (ReLU)

    Parameters
    ----------
    in_features : int
        입력 특성 차원
    out_features : int
        출력 특성 차원

    힌트:
      - __init__에서:
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

      - forward에서:
        # 자기 루프 추가: A_hat = A + I
        I = torch.eye(adj.size(0), device=adj.device)
        A_hat = adj + I
        # 메시지 패싱: 이웃 특성 집계 + 선형 변환
        support = x @ self.weight + self.bias  # (n_atoms, out_features)
        output = A_hat @ support               # 이웃 특성 합산
        return F.relu(output)
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # TODO: 가중치와 편향을 정의하세요
        pass

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (n_atoms, in_features)
        adj : torch.Tensor, shape (n_atoms, n_atoms)

        Returns
        -------
        out : torch.Tensor, shape (n_atoms, out_features)
        """
        # TODO: GCN 순전파를 구현하세요
        pass


# =============================================================================
# TODO 2: 그래프 수준 분류 모델
# =============================================================================
class MoleculeGCN(nn.Module):
    """
    분자 그래프 분류를 위한 GCN 모델.

    아키텍처:
      GCNLayer(3, 32) -> GCNLayer(32, 64) -> Global Mean Pool
      -> Linear(64, 32) -> ReLU -> Linear(32, 1) -> Sigmoid

    힌트:
      - __init__에서:
        self.gcn1 = SimpleGCNLayer(n_atom_features, 32)
        self.gcn2 = SimpleGCNLayer(32, 64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

      - forward에서:
        1. GCN 레이어 통과: h = self.gcn1(x, adj) -> self.gcn2(h, adj)
        2. Global mean pooling: graph_emb = h.mean(dim=0)
        3. MLP: fc1 -> relu -> fc2 -> sigmoid
    """

    def __init__(self, n_atom_features: int = 3):
        super().__init__()
        # TODO: GCN 레이어와 분류 MLP를 정의하세요
        pass

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (n_atoms, n_atom_features)
        adj : torch.Tensor, shape (n_atoms, n_atoms)

        Returns
        -------
        prob : torch.Tensor, shape (1,)
            활성 확률
        """
        # TODO: 순전파를 구현하세요
        pass


# =============================================================================
# TODO 3: 단일 분자 그래프 학습 스텝
# =============================================================================
def train_step(
    model: nn.Module,
    molecule: dict,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """
    단일 분자에 대한 학습 스텝을 수행합니다.

    Parameters
    ----------
    model : MoleculeGCN
    molecule : dict
        {"node_features", "adjacency", "label"}
    optimizer : torch.optim.Optimizer
    criterion : 손실 함수

    Returns
    -------
    loss_value : float

    힌트:
      - x = torch.tensor(molecule["node_features"])
      - adj = torch.tensor(molecule["adjacency"])
      - label = torch.tensor([molecule["label"]], dtype=torch.float32)
      - optimizer.zero_grad()
      - pred = model(x, adj)
      - loss = criterion(pred, label)
      - loss.backward()
      - optimizer.step()
    """
    # TODO: 학습 스텝을 구현하세요
    pass


# =============================================================================
# TODO 4: 평가 함수
# =============================================================================
def evaluate_model(
    model: nn.Module, molecules: list[dict]
) -> dict[str, float]:
    """
    분자 리스트에 대한 정확도와 평균 손실을 계산합니다.

    Parameters
    ----------
    model : MoleculeGCN
    molecules : list[dict]

    Returns
    -------
    metrics : dict
        {"accuracy": float, "avg_loss": float}

    힌트:
      - model.eval()
      - with torch.no_grad():
          for mol in molecules:
              pred = model(x, adj)
              predicted_label = (pred > 0.5).int().item()
              correct += (predicted_label == mol["label"])
    """
    # TODO: 정확도와 평균 손실을 계산하세요
    pass


# =============================================================================
# 메인 실행 및 자가 검증
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("Exercise 16: GNN 입문 - 분자 그래프와 GCN")
    print("=" * 60)

    # 1. 데이터 생성
    molecules = generate_molecule_graphs(200)
    print(f"\n[데이터] 분자 수: {len(molecules)}")
    mol0 = molecules[0]
    print(f"  예시 분자: 원자 수={mol0['node_features'].shape[0]}, "
          f"특성 차원={mol0['node_features'].shape[1]}, "
          f"레이블={mol0['label']}")
    labels = [m["label"] for m in molecules]
    print(f"  활성 비율: {sum(labels)}/{len(labels)} "
          f"({sum(labels)/len(labels)*100:.1f}%)")

    # 2. GCN 레이어 테스트
    gcn_layer = SimpleGCNLayer(3, 16)
    assert gcn_layer is not None
    test_x = torch.tensor(mol0["node_features"])
    test_adj = torch.tensor(mol0["adjacency"])
    out = gcn_layer(test_x, test_adj)
    assert out is not None, "SimpleGCNLayer forward가 None을 반환했습니다."
    assert out.shape == (mol0["node_features"].shape[0], 16), \
        f"GCN 출력 shape 오류: {out.shape}"
    print(f"\n[GCN 레이어] 입력: {test_x.shape} -> 출력: {out.shape}")

    # 3. 전체 모델 테스트
    model = MoleculeGCN(n_atom_features=3)
    assert model is not None
    pred = model(test_x, test_adj)
    assert pred is not None, "MoleculeGCN forward가 None을 반환했습니다."
    assert pred.shape == (1,), f"모델 출력 shape이 (1,)이어야 하지만 {pred.shape}입니다."
    assert 0 <= pred.item() <= 1, f"확률이 0~1 범위여야 합니다: {pred.item()}"
    print(f"[MoleculeGCN] 예측 확률: {pred.item():.4f}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  파라미터 수: {n_params:,}")

    # 4. 학습
    train_mols = molecules[:160]
    test_mols = molecules[160:]

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    print(f"\n[학습] 학습 분자: {len(train_mols)}, 테스트 분자: {len(test_mols)}")
    epoch_losses = []
    for epoch in range(30):
        total_loss = 0.0
        for mol in train_mols:
            loss = train_step(model, mol, optimizer, criterion)
            assert loss is not None, "train_step()이 None을 반환했습니다."
            total_loss += loss
        avg_loss = total_loss / len(train_mols)
        epoch_losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  에폭 {epoch+1:3d}: 평균 손실 = {avg_loss:.4f}")

    assert epoch_losses[-1] < epoch_losses[0], "학습 손실이 감소해야 합니다."

    # 5. 평가
    metrics = evaluate_model(model, test_mols)
    assert metrics is not None, "evaluate_model()이 None을 반환했습니다."
    assert "accuracy" in metrics and "avg_loss" in metrics
    print(f"\n[테스트 결과]")
    print(f"  정확도: {metrics['accuracy']*100:.1f}%")
    print(f"  평균 손실: {metrics['avg_loss']:.4f}")

    assert metrics["accuracy"] >= 0.4, \
        f"정확도가 40% 이상이어야 합니다: {metrics['accuracy']*100:.1f}%"

    print("\n" + "=" * 60)
    print("모든 검증을 통과했습니다!")
    print("=" * 60)
