"""
Exercise 17: PyTorch Geometric으로 분자 그래프 학습

목표:
  - PyG Data 객체로 분자 그래프 표현
  - GCNConv 기반 분자 속성 예측 모델 구현
  - PyG DataLoader로 배치 학습
  - GCN vs GAT 성능 비교

배경:
  PyTorch Geometric(PyG)은 GNN 구현을 위한 표준 라이브러리입니다.
  직접 인접 행렬을 다루는 대신 edge_index와 Data 객체를 사용하여
  효율적으로 분자 그래프를 처리합니다.
  실제 신약개발에서는 PyG를 활용하여 대규모 화합물 라이브러리에 대한
  가상 스크리닝(virtual screening)을 수행합니다.

사용 라이브러리:
  torch, torch_geometric (try/except로 감싸기), numpy
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("[경고] torch_geometric이 설치되지 않았습니다.")
    print("  pip install torch_geometric")
    print("  PyG 없이도 구조를 학습할 수 있도록 스텁 코드가 실행됩니다.")


def generate_pyg_molecules(n_molecules: int = 300) -> list:
    """
    PyG Data 형식의 분자 그래프 데이터를 생성합니다.

    각 분자는 5~15개의 원자와 원자 간 결합을 가집니다.
    원자 특성: [원자 번호(정규화), 형식 전하, 원자가 전자, 방향족 여부]

    Returns
    -------
    data_list : list[Data] (PyG 설치 시) 또는 list[dict] (미설치 시)
    """
    np.random.seed(42)
    data_list = []

    atom_types = {6: "C", 7: "N", 8: "O", 9: "F", 16: "S"}

    for i in range(n_molecules):
        n_atoms = np.random.randint(5, 16)

        # 원자 특성 (4차원)
        atom_nums = np.random.choice(list(atom_types.keys()), n_atoms)
        features = np.column_stack([
            atom_nums / 20.0,                           # 원자 번호 (정규화)
            np.random.choice([-1, 0, 1], n_atoms) / 1.0,  # 형식 전하
            np.random.randint(1, 5, n_atoms) / 4.0,     # 원자가 전자
            np.random.randint(0, 2, n_atoms).astype(float),  # 방향족
        ]).astype(np.float32)

        # 엣지 (결합) 생성
        edges_src, edges_dst = [], []
        for j in range(1, n_atoms):
            k = np.random.randint(0, j)
            edges_src.extend([j, k])  # 양방향
            edges_dst.extend([k, j])
        n_extra = np.random.randint(0, n_atoms // 2)
        for _ in range(n_extra):
            a, b = np.random.randint(0, n_atoms, 2)
            if a != b:
                edges_src.extend([a, b])
                edges_dst.extend([b, a])

        # 타깃: pIC50 (회귀)
        n_nitrogen = np.sum(atom_nums == 7)
        n_aromatic = np.sum(features[:, 3] > 0.5)
        pIC50 = (
            5.0
            + 0.3 * n_nitrogen
            + 0.2 * n_aromatic
            - 0.05 * n_atoms
            + np.random.normal(0, 0.3)
        )
        pIC50 = np.clip(pIC50, 3.0, 9.0)

        if HAS_PYG:
            data = Data(
                x=torch.tensor(features, dtype=torch.float32),
                edge_index=torch.tensor([edges_src, edges_dst], dtype=torch.long),
                y=torch.tensor([pIC50], dtype=torch.float32),
            )
            data_list.append(data)
        else:
            data_list.append({
                "x": features,
                "edge_index": np.array([edges_src, edges_dst]),
                "y": pIC50,
                "n_atoms": n_atoms,
            })

    return data_list


# =============================================================================
# TODO 1: PyG Data 객체 이해 및 검사
# =============================================================================
def inspect_pyg_data(data) -> dict:
    """
    PyG Data 객체의 속성을 검사하여 요약 정보를 반환합니다.

    Parameters
    ----------
    data : torch_geometric.data.Data 또는 dict

    Returns
    -------
    info : dict
        {
            "n_atoms": int,       -- 원자(노드) 수
            "n_bonds": int,       -- 결합(엣지) 수 (양방향이므로 실제 결합 수의 2배)
            "n_features": int,    -- 원자 특성 차원
            "has_label": bool,    -- 타깃 레이블 존재 여부
        }

    힌트 (PyG 설치 시):
      - data.num_nodes 또는 data.x.shape[0]
      - data.num_edges 또는 data.edge_index.shape[1]
      - data.x.shape[1]
      - data.y is not None
    """
    # TODO: Data 객체의 속성을 검사하세요
    pass


# =============================================================================
# TODO 2: GCN 기반 분자 속성 예측 모델
# =============================================================================
if HAS_PYG:
    class MoleculeGCNModel(nn.Module):
        """
        PyG GCNConv를 사용한 분자 속성 예측 모델.

        아키텍처:
          GCNConv(4, 64) -> ReLU -> GCNConv(64, 128) -> ReLU
          -> global_mean_pool
          -> Linear(128, 64) -> ReLU -> Dropout(0.2) -> Linear(64, 1)

        힌트:
          - __init__에서:
            self.conv1 = GCNConv(n_features, 64)
            self.conv2 = GCNConv(64, 128)
            self.fc1 = nn.Linear(128, 64)
            self.fc2 = nn.Linear(64, 1)
            self.dropout = nn.Dropout(0.2)

          - forward에서:
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = global_mean_pool(x, batch)  # 그래프별 집계
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
        """

        def __init__(self, n_features: int = 4):
            super().__init__()
            # TODO: GCNConv 레이어와 MLP를 정의하세요
            pass

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            batch: torch.Tensor,
        ) -> torch.Tensor:
            """
            Parameters
            ----------
            x : (total_atoms, n_features) 전체 배치의 노드 특성
            edge_index : (2, total_edges) 전체 배치의 엣지
            batch : (total_atoms,) 각 노드의 그래프 소속 인덱스

            Returns
            -------
            out : (n_graphs, 1) 각 분자의 예측 pIC50
            """
            # TODO: 순전파를 구현하세요
            pass

    # =========================================================================
    # TODO 3: GAT 기반 모델 (GCN과 비교용)
    # =========================================================================
    class MoleculeGATModel(nn.Module):
        """
        GAT(Graph Attention Network) 기반 분자 속성 예측 모델.

        아키텍처:
          GATConv(4, 32, heads=2) -> ELU -> GATConv(64, 64, heads=1) -> ELU
          -> global_mean_pool
          -> Linear(64, 32) -> ReLU -> Linear(32, 1)

        힌트:
          - GATConv(in_channels, out_channels, heads=2)
            -> 출력 차원 = out_channels * heads (concat=True 기본)
          - 두 번째 GATConv의 in_channels = 32 * 2 = 64
        """

        def __init__(self, n_features: int = 4):
            super().__init__()
            # TODO: GATConv 레이어와 MLP를 정의하세요
            pass

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            batch: torch.Tensor,
        ) -> torch.Tensor:
            # TODO: 순전파를 구현하세요
            pass


# =============================================================================
# TODO 4: 학습 및 평가 함수
# =============================================================================
def train_and_evaluate_pyg(
    model: nn.Module,
    train_loader,
    test_loader,
    n_epochs: int = 50,
    lr: float = 0.001,
) -> dict:
    """
    PyG 모델을 학습하고 평가합니다.

    Parameters
    ----------
    model : nn.Module
    train_loader : PyG DataLoader
    test_loader : PyG DataLoader
    n_epochs : int
    lr : float

    Returns
    -------
    result : dict
        {
            "train_losses": list[float],
            "test_mae": float,  -- 테스트 MAE (pIC50 단위)
            "test_r2": float,
        }

    힌트:
      - criterion = nn.MSELoss()
      - optimizer = torch.optim.Adam(model.parameters(), lr=lr)
      - 학습 루프:
        model.train()
        for batch in train_loader:
            pred = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(pred, batch.y.reshape(-1, 1))
      - 평가:
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                pred = model(batch.x, batch.edge_index, batch.batch)
    """
    # TODO: PyG 모델 학습 및 평가를 구현하세요
    pass


# =============================================================================
# 메인 실행 및 자가 검증
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("Exercise 17: PyTorch Geometric 분자 그래프 학습")
    print("=" * 60)

    # 1. 데이터 생성
    data_list = generate_pyg_molecules(300)
    print(f"\n[데이터] 분자 수: {len(data_list)}")

    # 2. Data 검사
    info = inspect_pyg_data(data_list[0])
    assert info is not None, "inspect_pyg_data()가 None을 반환했습니다."
    assert "n_atoms" in info and "n_bonds" in info and "n_features" in info
    print(f"[첫 번째 분자]")
    print(f"  원자 수: {info['n_atoms']}")
    print(f"  결합 수: {info['n_bonds']} (양방향)")
    print(f"  특성 차원: {info['n_features']}")
    print(f"  레이블 존재: {info['has_label']}")

    if not HAS_PYG:
        print("\n[PyG 미설치] GCN/GAT 모델 학습은 PyG 설치 후 실행하세요.")
        print("  pip install torch_geometric")
        print("\n" + "=" * 60)
        print("PyG 없이 가능한 검증을 모두 통과했습니다!")
        print("=" * 60)
    else:
        # 3. 학습/테스트 분할 및 DataLoader
        train_data = data_list[:240]
        test_data = data_list[240:]
        train_loader = PyGDataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = PyGDataLoader(test_data, batch_size=32, shuffle=False)

        batch = next(iter(train_loader))
        print(f"\n[DataLoader 배치]")
        print(f"  x shape: {batch.x.shape}")
        print(f"  edge_index shape: {batch.edge_index.shape}")
        print(f"  batch shape: {batch.batch.shape}")
        print(f"  y shape: {batch.y.shape}")

        # 4. GCN 모델
        print(f"\n--- GCN 모델 ---")
        gcn_model = MoleculeGCNModel(n_features=4)
        assert gcn_model is not None
        gcn_params = sum(p.numel() for p in gcn_model.parameters())
        print(f"  파라미터 수: {gcn_params:,}")

        gcn_result = train_and_evaluate_pyg(
            gcn_model, train_loader, test_loader, n_epochs=50, lr=0.001
        )
        assert gcn_result is not None, "train_and_evaluate_pyg()가 None을 반환했습니다."
        assert "test_mae" in gcn_result and "test_r2" in gcn_result
        print(f"  테스트 MAE: {gcn_result['test_mae']:.4f} pIC50")
        print(f"  테스트 R²: {gcn_result['test_r2']:.4f}")

        # 5. GAT 모델
        print(f"\n--- GAT 모델 ---")
        torch.manual_seed(42)
        gat_model = MoleculeGATModel(n_features=4)
        assert gat_model is not None
        gat_params = sum(p.numel() for p in gat_model.parameters())
        print(f"  파라미터 수: {gat_params:,}")

        gat_result = train_and_evaluate_pyg(
            gat_model, train_loader, test_loader, n_epochs=50, lr=0.001
        )
        assert gat_result is not None
        print(f"  테스트 MAE: {gat_result['test_mae']:.4f} pIC50")
        print(f"  테스트 R²: {gat_result['test_r2']:.4f}")

        # 6. 비교
        print(f"\n[GCN vs GAT 비교]")
        print(f"  GCN MAE: {gcn_result['test_mae']:.4f}, R²: {gcn_result['test_r2']:.4f}")
        print(f"  GAT MAE: {gat_result['test_mae']:.4f}, R²: {gat_result['test_r2']:.4f}")

        print("\n" + "=" * 60)
        print("모든 검증을 통과했습니다!")
        print("=" * 60)
