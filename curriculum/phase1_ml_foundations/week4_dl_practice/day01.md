# Day 1: 그래프 신경망(GNN) 개요

## 학습 목표
- 분자를 그래프로 표현하는 방법을 이해한다
- 메시지 패싱(Message Passing)의 원리를 파악한다
- 간단한 GCN(Graph Convolutional Network) 레이어를 직접 구현한다

## 이론 (1시간)

### 왜 분자에 GNN인가?

분자는 본질적으로 **그래프**입니다:
- **노드(Node)** = 원자 (C, N, O, ...)
- **엣지(Edge)** = 결합 (단일, 이중, 방향족, ...)

지금까지 사용한 표현 방식의 한계:
- **기술자**: 구조 정보 손실 (MW=180인 분자는 수천 개)
- **지문**: 고정 길이, 구조 일부만 인코딩
- **SMILES**: 같은 분자를 다르게 표현 가능, 순서 의존적

GNN은 **분자 그래프 구조를 직접 입력**으로 받습니다. 원자 간 연결 관계를 그대로 학습합니다.

### 분자 그래프의 구성 요소

```python
# 아스피린 (C9H8O4) 그래프 표현
node_features = [
    [6, 4, 0],  # C: 원자번호=6, 결합 수=4, 전하=0
    [6, 3, 0],  # C
    [8, 1, 0],  # O (카르보닐)
    # ... 나머지 원자
]

edge_index = [
    [0, 1], [1, 0],  # C-C 결합 (양방향)
    [1, 2], [2, 1],  # C=O 결합
    # ... 나머지 결합
]

edge_features = [
    [1, 0, 0],  # 단일 결합
    [0, 1, 0],  # 이중 결합
    [0, 0, 1],  # 방향족 결합
]
```

### 메시지 패싱 (Message Passing)

GNN의 핵심 메커니즘. 각 원자가 이웃 원자의 정보를 모아서 자신을 업데이트합니다.

```
1단계: 각 이웃이 "메시지" 전송
2단계: 메시지를 합산(aggregation)
3단계: 자신의 특성과 합쳐서 업데이트
```

```python
# 개념적 의사코드
for each node v:
    messages = [W @ h_u for u in neighbors(v)]  # 이웃의 특성을 변환
    aggregated = sum(messages)                    # 합산
    h_v_new = ReLU(aggregated + W_self @ h_v)    # 자신과 합쳐서 업데이트
```

> **비유**: 단백질 folding에서 각 아미노산이 주변 잔기들의 영향을 받아 최종 구조가 결정되는 것과 유사합니다. GNN의 각 레이어는 "1-hop 이웃"의 정보를 수집합니다. 3개 레이어 → 3-hop 이웃까지 고려.

### GCN 레이어 직접 구현

```python
import torch
import torch.nn as nn

class SimpleGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        """
        x: (N, in_features) — 노드 특성 행렬
        adj: (N, N) — 인접 행렬 (정규화됨)
        """
        # 이웃 정보 합산: A @ X
        support = self.linear(x)       # 특성 변환
        output = torch.mm(adj, support) # 이웃 합산
        return torch.relu(output)

class SimpleGCN(nn.Module):
    def __init__(self, n_node_features, hidden_dim, n_classes):
        super().__init__()
        self.gcn1 = SimpleGCNLayer(n_node_features, hidden_dim)
        self.gcn2 = SimpleGCNLayer(hidden_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, n_classes)

    def forward(self, x, adj):
        h = self.gcn1(x, adj)
        h = self.gcn2(h, adj)
        # 그래프 레벨 예측: 모든 노드의 평균 (단일 그래프용)
        graph_embedding = h.mean(dim=0)
        return self.readout(graph_embedding)

# 주의: 이 구현은 한 번에 하나의 그래프만 처리합니다.
# 여러 그래프를 배치로 처리하려면 PyG의 Batch + global_mean_pool()이 필요합니다.
# → Day 2에서 PyG로 배치 처리를 배웁니다.
```

### 인접 행렬 정규화

```python
def normalize_adj(adj):
    """D^(-1/2) @ A @ D^(-1/2) 정규화"""
    degree = adj.sum(dim=1)
    d_inv_sqrt = torch.diag(1.0 / torch.sqrt(degree + 1e-8))
    return d_inv_sqrt @ adj @ d_inv_sqrt
```

## 실습 (1.5시간)

### Exercise 16: GCN 직접 구현

파일: `exercises/phase1/week4/ex16_gnn_intro.py`

**과제:**
1. 간단한 분자 그래프 데이터 생성 (노드 특성 + 인접 행렬)
2. SimpleGCNLayer 구현 (Linear + 인접 행렬 곱)
3. 2-layer GCN으로 그래프 분류 수행
4. 메시지 패싱 과정 시각화 (1-hop, 2-hop 이웃)

## 참고 자료
- Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks" (2017)
- "Neural Message Passing for Quantum Chemistry" (Gilmer et al., 2017)
