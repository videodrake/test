# Day 3: Attention 메커니즘

## 학습 목표
- Self-Attention의 수학적 원리(Q, K, V)를 이해한다
- SMILES 토큰에 대한 Attention을 직접 구현한다
- Attention 가중치를 시각화하여 모델의 "관심 영역"을 해석한다

## 이론 (1시간)

### Attention이란?
"입력의 어떤 부분에 집중할지"를 학습하는 메커니즘입니다.

```
입력: CC(=O)Oc1ccccc1C(=O)O  (아스피린)
Attention: 카르보닐기(=O)와 에스터 결합(O)에 높은 가중치
→ 이 부분이 용해도에 가장 영향을 준다고 모델이 학습
```

### Self-Attention 수학

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Q (Query):  "내가 찾는 것"
K (Key):    "내가 가진 것"
V (Value):  "내가 전달할 것"
d_k:        Key의 차원 (스케일링)
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, L, head_dim)

        scale = self.head_dim ** 0.5
        attn = (q @ k.transpose(-2, -1)) / scale  # (B, heads, L, L)
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out(out), attn
```

### 분자에서 Attention의 의미

```python
# Attention 가중치 시각화
model.eval()
with torch.no_grad():
    output, attn_weights = model(smiles_tokens)

# attn_weights[0, 0] = 첫 번째 분자, 첫 번째 헤드의 attention map
# 높은 가중치 = 모델이 중요하다고 판단한 토큰
```

약물 설계에서의 가치:
- **해석 가능성**: 어떤 부분 구조가 활성에 기여하는지 시각화
- **SAR 발견**: Attention이 높은 원자/작용기 = 핵심 약물작용단(pharmacophore)
- **신뢰성**: 화학자가 직관과 비교하여 모델을 검증

### Transformer 맛보기

Attention + FFN + 잔차 연결 = Transformer 블록:

```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads=4):
        super().__init__()
        self.attn = SelfAttention(embed_dim, n_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + attn_out       # 잔차 연결
        x = x + self.ffn(self.norm2(x))
        return x, attn_weights
```

## 실습 (1.5시간)

### Exercise 18: SMILES Attention

파일: `exercises/phase1/week4/ex18_attention.py`

**과제:**
1. Single-head Self-Attention 직접 구현 (Q, K, V)
2. SMILES 토큰 시퀀스에 Attention 적용
3. Attention 가중치 히트맵 시각화
4. Attention 기반 분자 물성 예측 모델 구축

## 참고 자료
- Vaswani et al. "Attention Is All You Need" (2017)
- "Molecular Transformer" (Schwaller et al., 2019)
