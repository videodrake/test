# Day 4: 전이학습과 사전학습

## 학습 목표
- 전이학습(Transfer Learning)의 개념과 화학 데이터에서의 가치를 이해한다
- 사전학습 모델을 파인튜닝하는 방법을 익힌다
- 작은 화학 데이터셋에서 전이학습의 효과를 실험한다

## 이론 (1시간)

### 전이학습이 필요한 이유

신약개발 데이터의 근본적 문제:
- 특정 타겟의 활성 데이터: **수백~수천 개** (매우 적음)
- 레이블 없는 분자 데이터: **수백만 개** (ChEMBL, ZINC)

전이학습 전략:
```
1. 대규모 데이터로 일반적인 분자 표현 학습 (사전학습)
2. 소규모 타겟 데이터로 미세조정 (파인튜닝)
```

### 파인튜닝 패턴

```python
# 1. 사전학습 모델 로드
pretrained = MolGNN(n_features=9, hidden_dim=64)
pretrained.load_state_dict(torch.load('pretrained_model.pt'))

# 2. 마지막 층만 교체
pretrained.fc = nn.Linear(64, 1)  # 새로운 타겟에 맞게

# 3. 전략 A: 전체 파인튜닝
optimizer = torch.optim.Adam(pretrained.parameters(), lr=1e-4)  # 작은 학습률!

# 3. 전략 B: 특성 추출 (마지막 층만 학습)
for param in pretrained.parameters():
    param.requires_grad = False
pretrained.fc.requires_grad_(True)
optimizer = torch.optim.Adam(pretrained.fc.parameters(), lr=1e-3)
```

### 사전학습 방식

| 방식 | 설명 | 예시 |
|------|------|------|
| 지도 사전학습 | 대규모 레이블 데이터로 학습 | QM9 양자역학 물성 |
| 자기지도 학습 | 레이블 없이 구조 자체에서 학습 | SMILES 마스킹, 원자 예측 |
| 대조 학습 | 유사/비유사 쌍으로 표현 학습 | 같은 분자의 다른 augmentation |

```python
# 자기지도 사전학습 개념 (실제 구현은 Phase 3에서)
# "분자의 일부 원자를 마스킹하고 예측하게 학습"
# → 분자 구조의 일반적인 패턴을 파악
```

### 전이학습 실험 설계

```python
# 공정한 비교를 위한 실험 구조
results = {}

# A: 처음부터 학습 (baseline)
model_scratch = MolGNN(n_features=9, hidden_dim=64)
train(model_scratch, small_dataset, epochs=200)
results['scratch'] = evaluate(model_scratch, test_set)

# B: 사전학습 + 파인튜닝
model_pretrained = load_pretrained()
model_pretrained.fc = nn.Linear(64, 1)
train(model_pretrained, small_dataset, epochs=50, lr=1e-4)
results['pretrained'] = evaluate(model_pretrained, test_set)

# C: 사전학습 + 특성 추출
model_frozen = load_pretrained()
freeze_backbone(model_frozen)
model_frozen.fc = nn.Linear(64, 1)
train(model_frozen, small_dataset, epochs=100)
results['frozen'] = evaluate(model_frozen, test_set)
```

### 작은 데이터에서의 효과

일반적으로:
- **데이터 < 100개**: 전이학습 효과 매우 큼 (R² 0.3 → 0.7)
- **데이터 100~1000개**: 중간 정도 효과
- **데이터 > 5000개**: 효과 작음 (처음부터 학습해도 충분)

## 실습 (1.5시간)

### Exercise 19: 전이학습 실험

파일: `exercises/phase1/week4/ex19_transfer_learning.py`

**과제:**
1. 큰 데이터셋(2000개)으로 사전학습 모델 훈련
2. 작은 데이터셋(100개)에서 세 가지 접근법 비교: scratch / finetune / frozen
3. 데이터 크기별(50, 100, 200, 500) 전이학습 효과 그래프
4. 결과 분석 및 해석

## 참고 자료
- Hu et al. "Strategies for Pre-training Graph Neural Networks" (2020)
- "Transfer Learning in Drug Discovery" 리뷰
