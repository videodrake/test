# Day 2: PyTorch Geometric 실습

## 학습 목표
- PyTorch Geometric(PyG)의 Data 객체와 배치 처리를 이해한다
- PyG의 내장 GNN 레이어(GCNConv, GATConv)를 사용한다
- 분자 물성 예측 GNN을 PyG로 구축한다

## 이론 (1시간)

### PyTorch Geometric 설치와 데이터 구조

```python
# pip install torch-geometric
from torch_geometric.data import Data, Batch

# 하나의 분자 그래프
data = Data(
    x=torch.FloatTensor([[6,4],[8,2],[7,3]]),  # 노드 특성 (3 원자)
    edge_index=torch.LongTensor([[0,1,1,2],[1,0,2,1]]),  # 엣지 (양방향)
    y=torch.FloatTensor([3.5]),  # 타겟 (용해도)
)
print(data)  # Data(x=[3, 2], edge_index=[2, 4], y=[1])
```

### PyG GNN 모델 구축

```python
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

class MolGNN(nn.Module):
    def __init__(self, n_node_features, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(n_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # Readout: 그래프 내 모든 노드의 평균
        x = global_mean_pool(x, batch)  # (num_graphs, hidden_dim)
        return self.fc(x).squeeze(-1)
```

### 배치 처리

```python
from torch_geometric.loader import DataLoader

# 여러 그래프를 하나의 큰 그래프로 결합
dataset = [data1, data2, data3, ...]
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    out = model(batch)  # batch.batch가 각 노드의 소속 그래프를 추적
```

### GNN 변형: GAT (Graph Attention Network)

```python
from torch_geometric.nn import GATConv

# GAT: 이웃마다 다른 가중치(attention) 적용
class MolGAT(nn.Module):
    def __init__(self, n_features, hidden=64, heads=4):
        super().__init__()
        self.conv1 = GATConv(n_features, hidden, heads=heads)
        self.conv2 = GATConv(hidden * heads, hidden, heads=1)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, ei))
        x = F.relu(self.conv2(x, ei))
        x = global_mean_pool(x, batch)
        return self.fc(x).squeeze(-1)
```

> **GCN vs GAT**: GCN은 모든 이웃을 동등하게 취급. GAT는 각 이웃의 "중요도"를 학습. 약물-표적 상호작용처럼 특정 원자가 더 중요한 경우 GAT가 유리합니다.

## 실습 (1.5시간)

### Exercise 17: PyG 분자 GNN

파일: `exercises/phase1/week4/ex17_pyg_molecules.py`

**과제:**
1. 랜덤 분자 그래프 데이터셋 생성 (Data 객체 리스트)
2. GCNConv 기반 MolGNN 모델 정의
3. DataLoader로 배치 학습
4. GCN vs GAT 성능 비교

## 참고 자료
- PyTorch Geometric 공식 튜토리얼
- Veličković et al. "Graph Attention Networks" (2018)
