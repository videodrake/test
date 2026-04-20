# Day 5: CNN과 분자 시퀀스

## 학습 목표
- 1D CNN(Convolutional Neural Network)의 원리를 이해한다
- SMILES 문자열을 문자 수준으로 인코딩하여 CNN 입력으로 사용한다
- CNN 기반 분자 물성 예측 모델을 구축한다

## 이론 (1시간)

### 분자를 시퀀스로 — SMILES와 CNN

SMILES는 분자를 **문자열**로 표현합니다. 이를 문자 단위로 인코딩하면 1D CNN에 입력 가능합니다:

```
아스피린: CC(=O)Oc1ccccc1C(=O)O
         → [C, C, (, =, O, ), O, c, 1, c, c, c, c, c, 1, C, (, =, O, ), O]
         → 원-핫 인코딩 → (seq_len, vocab_size) 행렬
         → 1D CNN
```

### SMILES 문자 인코딩

```python
VOCAB = {
    'C': 1, 'c': 2, 'N': 3, 'n': 4, 'O': 5, 'o': 6, 'S': 7, 's': 8,
    'F': 9, 'P': 10, 'I': 11, 'B': 12, 'H': 13,
    '(': 14, ')': 15, '[': 16, ']': 17, '=': 18, '#': 19,
    '1': 20, '2': 21, '3': 22, '4': 23,
    '+': 24, '-': 25, '/': 26, '\\': 27, '@': 28, '.': 29,
    '<PAD>': 0,
}

def encode_smiles(smiles, max_len=100):
    """SMILES를 정수 시퀀스로 인코딩 + 패딩"""
    encoded = [VOCAB.get(c, 0) for c in smiles[:max_len]]
    padded = encoded + [0] * (max_len - len(encoded))
    return padded
```

### 1D CNN 모델

```python
class SMILESCNN(nn.Module):
    def __init__(self, vocab_size=30, embed_dim=32, max_len=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len) 정수 시퀀스
        x = self.embedding(x)        # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)        # (batch, embed_dim, seq_len) — Conv1d 입력
        x = torch.relu(self.conv1(x)) # (batch, 64, seq_len)
        x = torch.relu(self.conv2(x)) # (batch, 128, seq_len)
        x = self.pool(x).squeeze(-1)  # (batch, 128) — 전역 최대 풀링
        return self.fc(x).squeeze(-1)  # (batch,)
```

### CNN이 분자에서 포착하는 것

Conv1d 필터는 SMILES 문자열의 **지역 패턴**을 인식합니다:
- `kernel_size=3`: `C(=O)` → 카르보닐기
- `kernel_size=5`: `c1ccc(` → 방향족 고리 시작
- 여러 층의 필터가 합쳐져 → 전체 구조적 특성 포착

> **한계**: SMILES는 같은 분자가 여러 문자열로 표현될 수 있습니다 (canonical vs non-canonical). Week 4의 GNN은 이 문제를 해결합니다.

### Week 3 요약

| Day | 주제 | 핵심 |
|-----|------|------|
| 1 | 텐서, 자동미분 | torch.Tensor, autograd, backward() |
| 2 | 피드포워드 NN | nn.Module, nn.Sequential, 활성 함수 |
| 3 | 학습 루프 | DataLoader, 옵티마이저, Early Stopping |
| 4 | 정규화 | Dropout, Weight Decay, BatchNorm |
| 5 | CNN + SMILES | 1D Conv, Embedding, 시퀀스 인코딩 |

**다음 주 (Week 4) 예고**: 분자를 **그래프**로 표현하는 GNN(Graph Neural Network), Attention 메커니즘, 전이학습으로 Phase 1을 마무리합니다!

## 실습 (1.5시간)

### Exercise 15: SMILES CNN 물성 예측

파일: `exercises/phase1/week3/ex15_cnn_smiles.py`

**과제:**
1. SMILES 문자열 인코딩 함수 구현
2. SMILESCNN 모델 정의
3. 샘플 SMILES 데이터로 물성 예측 학습
4. 학습 곡선 시각화

## 참고 자료
- "Convolutional Networks on Graphs for Learning Molecular Fingerprints" (Duvenaud et al.)
- PyTorch: Conv1d, Embedding 문서
