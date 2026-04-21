# Day 5: Phase 1 종합 복습

## 학습 목표
- Phase 1 (4주)의 핵심 개념을 정리한다
- 각 주차의 도구와 기법을 연결하여 전체 파이프라인을 조망한다
- Phase 1 마일스톤 프로젝트를 소개하고 접근법을 설계한다

## 이론 (1시간)

### Phase 1 전체 지도

```
Week 1: 데이터 기초        Week 2: 전통 ML          Week 3: 딥러닝            Week 4: 분자 DL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NumPy (벡터화)          sklearn (QSAR)          PyTorch (텐서/NN)        GNN (그래프)
Pandas (데이터)         분류/회귀                학습 루프                 PyG
Matplotlib (시각화)     모델 평가 (AUC)         정규화                    Attention
SciPy (통계)           특성 엔지니어링          CNN (SMILES)             전이학습
PCA (차원축소)          클러스터링               Early Stopping           사전학습
```

### 핵심 체크리스트

#### 데이터 처리 (Week 1)
- [ ] NumPy 벡터화로 대량 분자 데이터 처리
- [ ] Pandas로 분자 물성 EDA, 결측치 처리
- [ ] Tanimoto 유사도 계산 (단일 + 행렬)
- [ ] PCA로 화학 공간 시각화

#### 전통 ML (Week 2)
- [ ] Train/Test 분할과 교차 검증
- [ ] 분류: LogReg, DT, RF + 확률 예측
- [ ] 평가: ROC-AUC, F1, Precision-Recall
- [ ] 불균형 처리: class_weight, SMOTE
- [ ] Pipeline으로 전처리+모델 통합

#### 딥러닝 기초 (Week 3)
- [ ] PyTorch 텐서, 자동미분
- [ ] nn.Module로 네트워크 정의
- [ ] 완전한 학습 루프 (DataLoader, Early Stopping)
- [ ] 정규화: Dropout, BatchNorm, Weight Decay

#### 분자 DL (Week 4)
- [ ] 분자 그래프 표현 (노드, 엣지, 인접 행렬)
- [ ] GCN 메시지 패싱 원리
- [ ] PyG로 GNN 구축
- [ ] Attention으로 해석 가능한 예측

### 신약개발 ML 파이프라인 — 전체 그림

```python
# Phase 1에서 배운 것으로 구축 가능한 파이프라인

# 1. 데이터 준비 (Week 1)
df = pd.read_csv('molecules.csv')
df = handle_missing_values(df)

# 2. 특성 엔지니어링 (Week 1 + Week 2)
X = compute_descriptors(df)  # 또는 GNN이면 그래프로 변환
X_selected = feature_selection(X)

# 3. 데이터 분할 (Week 2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. 모델 선택과 학습 (Week 2-4)
model = choose_model(data_size, data_type)
# 작은 데이터 + 기술자 → RF
# 큰 데이터 + SMILES → CNN
# 그래프 구조 중요 → GNN

# 5. 평가 (Week 2-3)
evaluate_model(model, X_test, y_test)

# 6. 해석 (Week 2 + Week 4)
feature_importance(model)  # RF
attention_visualization(model)  # GNN/Attention
```

### 모델 선택 가이드

| 상황 | 추천 모델 | 이유 |
|------|----------|------|
| 데이터 < 500, 기술자 있음 | Random Forest | 과적합 적고 해석 가능 |
| 데이터 500~5000 | RF → GNN 비교 | 데이터 충분하면 DL이 유리 |
| 데이터 > 5000 | GNN 또는 Transformer | 복잡한 패턴 학습 가능 |
| 해석 필요 | DT + Attention GNN | 규칙/가중치 시각화 |
| 빠른 스크리닝 | RF (학습), LR (추론) | 수백만 화합물 실시간 처리 |

### Phase 1 마일스톤 프로젝트: 분자 물성 예측기

**목표**: ESOL 용해도 데이터셋에서 Random Forest와 GNN을 비교하는 완전한 ML 파이프라인 구축

**요구사항:**
1. 데이터 로드 및 전처리 (결측치 처리, 스케일링)
2. 기술자 기반 Random Forest 모델
3. 간단한 GNN 모델 (또는 피드포워드 NN)
4. 5-fold 교차 검증으로 공정한 비교
5. 결과 시각화 (예측 vs 실제 산점도, 잔차 분포)
6. 프로젝트 README 작성

**디렉토리**: `projects/phase1_molecular_property_predictor/`

## 이번 주 학습 확인

### Week 4 요약
| Day | 주제 | 핵심 |
|-----|------|------|
| 1 | GNN 개요 | 메시지 패싱, GCN 직접 구현 |
| 2 | PyG 실습 | GCNConv, DataLoader, GAT |
| 3 | Attention | Self-Attention, Q/K/V, 시각화 |
| 4 | 전이학습 | 파인튜닝, 특성 추출, 소규모 데이터 |
| 5 | 종합 복습 | 파이프라인, 모델 선택, 마일스톤 |

## Phase 2 예고
다음 Phase에서는 **RDKit으로 실제 분자를 다루기 시작**합니다! SMILES 파싱, 분자 지문 계산, 3D 배좌 생성 등 화학정보학(Cheminformatics)의 핵심 기술을 배웁니다. Phase 1의 ML 기초 위에 화학 도메인 지식이 결합됩니다.

## 참고 자료
- "Machine Learning in Drug Discovery" (Vamathevan et al., 2019)
- MoleculeNet 벤치마크 논문 (Wu et al., 2018)
- "A Practical Guide to ML in Drug Discovery" (Stokes et al.)
