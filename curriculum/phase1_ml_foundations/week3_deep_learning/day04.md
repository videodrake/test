# Day 4: 정규화 기법

## 학습 목표
- 과적합(Overfitting)의 원인과 감지 방법을 이해한다
- Dropout, Weight Decay, Batch Normalization을 적용한다
- 정규화 전/후 학습 곡선을 비교 분석한다

## 이론 (1시간)

### 과적합 — 화학 데이터의 근본적 문제

신약개발 데이터는 과적합에 취약합니다:
- **작은 데이터**: 실험 데이터는 수백~수천 개 (ImageNet의 100만+ 대비)
- **높은 차원**: 기술자 수 > 샘플 수인 경우도 빈번
- **노이즈**: 실험 측정값의 오차 (IC50의 3-5배 변동은 흔함)

### 과적합 감지

```python
# train_loss는 감소하지만 val_loss가 증가 → 과적합!
# Epoch 50: train=0.01, val=0.05  ← 양호
# Epoch 100: train=0.001, val=0.15 ← 과적합!
```

### Dropout — 랜덤 뉴런 비활성화

```python
class RegularizedNet(nn.Module):
    def __init__(self, n_features, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),     # 30% 뉴런 랜덤 비활성화
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
```

> **비유**: Dropout은 연구팀에서 매번 다른 조합의 팀원만 참여시키는 것과 같습니다. 어떤 조합이든 결론이 같으면 그건 진짜 결론입니다. 특정 뉴런(팀원)에 과도하게 의존하는 것을 방지합니다.

### Weight Decay (L2 정규화)

```python
# 가중치가 너무 커지지 않도록 패널티
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01  # λ: 클수록 가중치 축소 강화
)
```

### Batch Normalization

```python
class BNNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),    # 배치 정규화
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
```

BatchNorm의 효과:
- 학습 속도 향상 (더 큰 학습률 사용 가능)
- 내부 공변량 이동(Internal Covariate Shift) 감소
- 약한 정규화 효과

### 정규화 조합 전략

| 데이터 크기 | 권장 전략 |
|-----------|----------|
| < 500 | Dropout(0.5) + Weight Decay(0.01) + Early Stopping |
| 500~5000 | Dropout(0.3) + BatchNorm + Weight Decay(0.001) |
| > 5000 | BatchNorm + 약한 Weight Decay, Dropout 선택적 |

## 실습 (1.5시간)

### Exercise 14: 정규화 비교 실험

파일: `exercises/phase1/week3/ex14_regularization.py`

**과제:**
1. 기본 모델(정규화 없음)과 정규화 모델 정의
2. 정규화 없는 모델 학습 → train/val 곡선 기록
3. Dropout(0.3) + Weight Decay(0.01) 적용 후 재학습
4. 두 모델의 학습 곡선 비교 시각화
5. 테스트 세트에서 최종 성능 비교

## 참고 자료
- Srivastava et al. "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (2014)
- PyTorch: BatchNorm1d 문서
