# Day 4: Transformer와 Foundation Model — 분자의 BERT

> 이전 학습(Phase 2 Week 1 Day 3)에서 **GNN 기반 3세대 QSAR**을 다뤘습니다. 분자 그래프를 직접 입력해 task에 맞춘 표현을 학습한다는 점이 핵심이었고, 데이터 N이 5,000~10,000을 넘어야 GNN의 우위가 시작된다는 한계도 함께 확인했습니다. 오늘은 그 한계를 정면 돌파하는 **4세대 QSAR — Transformer 기반 분자 Foundation Model**을 학습합니다. 라벨이 거의 없어도, 1억 개 단위의 unlabeled SMILES로 사전학습한 모델을 가져와 fine-tuning하는 패러다임입니다.

## 개요

**분자 Foundation Model**은 자연어처리(NLP)의 BERT·GPT 패러다임을 화학에 이식한 접근입니다. PubChem·ZINC·ChEMBL에서 수집한 **수천만~수억 개의 라벨 없는 SMILES**로 Transformer를 자기지도학습(self-supervised learning)시킨 뒤, 특정 ADMET·생리활성 task에 fine-tuning합니다. 2020년 **ChemBERTa**(Chithrananda et al.)·**MolBERT**(Fabian et al.)가 첫 신호를 보인 뒤, 2022년 IBM의 **MolFormer-XL**과 DP Technology의 **Uni-Mol**이 MoleculeNet·TDC 벤치마크에서 SOTA를 차지하며 패러다임의 가능성을 입증했습니다. 이 접근의 결정적 가치는 — **다운스트림 task의 라벨이 수백~수천 개 수준이어도 GNN을 압도하는 성능을 낸다는 점**입니다. 약학 전공자에게 익숙한 in vitro 어세이 데이터(보통 N=500-3,000)의 규모에서 가장 큰 이점이 발생합니다. 단, 사전학습 비용이 GPU 기준 수만 달러 수준이라 자체 학습은 비현실적이고, HuggingFace에 공개된 사전학습 체크포인트를 가져다 쓰는 것이 표준 워크플로우입니다.

## 핵심 개념

### 1) 왜 Transformer인가 — SMILES를 자연어처럼 다루는 발상

Transformer의 핵심은 **self-attention**입니다. 입력 시퀀스의 모든 토큰 쌍이 직접 상호작용 가중치를 학습하므로, GNN의 layer 수에 따라 receptive field가 제한되는 한계가 사라집니다. SMILES `C1=CC=C(C=C1)C(=O)O`(benzoic acid)에서 1번 탄소와 7번 카르복실 탄소는 그래프 거리 6이지만, attention은 단 1 layer로 둘을 직접 연결합니다. 약학적으로는 — **pharmacophore의 두 끝점, 분자 내 H-bond donor/acceptor 쌍, intramolecular conformational constraint** 같은 장거리 상호작용을 한 번에 포착할 수 있다는 의미입니다.

또 하나 결정적인 차이는 **사전학습-파인튜닝(pre-train/fine-tune) 분리**입니다. GNN은 보통 task별로 처음부터 학습합니다. Transformer 기반 Foundation Model은 일단 라벨 없는 수억 개 SMILES로 **분자의 일반 문법**을 학습한 뒤, 작은 라벨 데이터로 task에 맞춰 미세조정합니다. NLP에서 BERT가 단어 빈도·문법·의미를 사전학습한 것과 동일한 구조입니다.

### 2) 사전학습 목표 — Masked Language Modeling과 그 변형

화학 Foundation Model의 사전학습 목표는 NLP의 그것을 거의 그대로 가져옵니다.

| 목표 | 설명 | 대표 모델 |
|------|------|---------|
| **Masked Language Modeling (MLM)** | SMILES의 15% 토큰을 [MASK]로 가리고 원래 토큰 예측 | ChemBERTa, MolBERT |
| **Contrastive Learning** | 같은 분자의 다른 SMILES 표기(canonical/randomized)를 같은 임베딩으로 | MolCLR, SELFormer |
| **Span Prediction** | 연속된 여러 토큰을 가리고 한꺼번에 복원 | MolFormer |
| **3D-MLM + Coord Prediction** | 원자 타입 마스킹 + 3D 좌표 회귀 | Uni-Mol |

MLM은 단순하지만 효과적입니다. 모델이 "벤젠고리의 다음 토큰은 무엇이어야 하는가", "이 위치에 어떤 작용기가 합리적인가"를 1억 번 학습하면서, 사람이 명시적으로 가르치지 않은 **화학적 valence·aromaticity·작용기 호환성**을 암묵적으로 익히게 됩니다. 약학 전공자가 직관적으로 이해하기 쉬운 비유는 — **약학 교과서 1,000권을 통째로 외운 학생이 새로운 약물의 SAR을 직관적으로 추정하는 것**과 같습니다.

### 3) 대표 모델 — ChemBERTa, MolFormer, Uni-Mol, MolE

| 모델 | 발표 | 사전학습 데이터 | 파라미터 | 특징 |
|------|------|--------------|---------|------|
| **ChemBERTa** | Chithrananda 2020 | PubChem 77M | 12M | 첫 화학 BERT, 베이스라인용 |
| **MolBERT** | Fabian 2020 | GuacaMol 1.6M | ~90M | MLM + 물성 예측 보조 task |
| **MolFormer-XL** | IBM 2022 | PubChem + ZINC 1.1B | 47M | 선형 어텐션, 효율적 학습 |
| **Uni-Mol** | DP Tech 2022 | 209M 분자 + 209M 단백질 포켓 | 100M | **3D 정보 사전학습**, MoleculeNet SOTA |
| **MolE** | Roche 2023 | 842M PubChem | 21M | disentangled attention, GNN보다 가벼움 |
| **SELFormer** | 2023 | 2M 분자 | 87M | SELFIES 기반(문법 오류 방지) |

산업적으로 **MolFormer-XL과 Uni-Mol**이 가장 자주 인용되는 두 축입니다. MolFormer-XL은 IBM이 HuggingFace에 공개해 라이선스 부담이 적고, Uni-Mol은 3D 좌표를 직접 활용해 결합 친화도 예측에서 강점이 있습니다. 약학 전공 창업자가 첫 베이스라인을 잡을 때 — **ADMET·물성 회귀는 MolFormer-XL, docking/binding 친화도는 Uni-Mol**이 합리적 출발점입니다.

### 4) 왜 작은 데이터에서 강한가 — Transfer Learning의 정량 증거

MolFormer 논문(Ross et al., 2022, *Nature Machine Intelligence*)의 핵심 표는 task별 데이터 크기와 모델 성능을 정리합니다. 데이터 N이 500-2,000인 작은 ADMET task에서 — Chemprop(D-MPNN)이 RMSE 0.85일 때, MolFormer-XL은 0.71 수준으로 일관되게 우위를 보였습니다(BBBP, Tox21, ClinTox 등). 직관적 이유는 — 사전학습이 분자의 일반 표현을 이미 완성해뒀기 때문에, fine-tuning 단계에서는 task별 미세 조정만 하면 됩니다. GNN은 매번 표현을 처음부터 학습하므로 데이터가 부족하면 표현 자체가 미완성으로 끝납니다.

약학적 함의는 분명합니다. 신생 스타트업이 보유하는 실험 데이터는 보통 N=500-5,000 수준입니다. 이 구간이 Foundation Model의 sweet spot이라는 점은 — **자체 데이터가 적은 한국형 스타트업의 첫 모델 선택**에 정량적 근거를 제공합니다.

### 5) Foundation Model의 약점 — 해석성, 비용, 분포 외 일반화

만능은 아닙니다. 세 가지 한계가 명확합니다.

- **해석성**: attention weight 시각화로 부분적으로 해석할 수 있으나, GNN의 GNNExplainer나 트리의 SHAP보다 신뢰도가 낮습니다. 규제 제출 맥락에서는 여전히 보조 도구일 뿐입니다.
- **사전학습 비용**: MolFormer-XL은 GPU 16장 × 약 2주 학습이 필요했습니다. 비용 환산 5만 달러 이상. 자체 사전학습은 빅테크·연구소만 가능합니다.
- **Out-of-distribution 일반화**: PubChem 분포 내에서는 강하지만, 매우 신규한 chemotype(예: PROTAC, macrocycle)에서는 GNN 대비 큰 우위가 사라진다는 보고가 누적되고 있습니다(2023-2024년).

약학 전공 창업자가 가져야 할 판단 기준은 — **HuggingFace에서 가져다 fine-tuning하는 것은 표준이지만, 도메인 특화 사전학습(예: 한국 임상 적합도가 높은 분자 분포로 추가 사전학습)을 누가 어떻게 수행하느냐가 차별점**이라는 것입니다.

## 작동 원리와 아키텍처

### MolFormer-XL 파인튜닝 표준 파이프라인

```
[1. 사전학습된 체크포인트 로드]
   HuggingFace: ibm/MoLFormer-XL-both-10pct
   토크나이저: regex 기반 SMILES tokenizer (vocab ≈ 2,300)

[2. 데이터 준비]
   SMILES 표준화: RDKit canonicalize + tautomer 통일
   토큰화: max_length=202, padding/truncation
   train/valid/test = scaffold split 8:1:1

[3. 모델 구조]
   Embedding(2,300 → 768)
   × 12 layers Transformer Encoder
       - Multi-Head Linear Attention (12 heads)
       - FFN (3,072 hidden)
       - LayerNorm + Residual
   Pooling: [CLS] 토큰 또는 mean pooling
   Task Head: Linear(768 → 1) for regression
              Linear(768 → C) for classification

[4. Fine-tuning]
   AdamW, lr=1e-5 ~ 5e-5
   배치 32, epoch 20, early stopping(patience 3)
   GPU 1장 기준 5,000 분자 ≈ 30분
   freeze 옵션: encoder 동결 후 head만 학습(데이터 N<500일 때)

[5. 평가]
   회귀: RMSE / Spearman ρ
   분류: ROC-AUC / PR-AUC / EF@1%

[6. 해석]
   Attention rollout / IG (Integrated Gradients)
   분자 시각화: 토큰별 기여도 RDKit highlight
```

### Foundation Model 선택 의사결정 표

| 상황 | 추천 모델 | 이유 |
|------|----------|------|
| ADMET 회귀, N=500-5,000 | **MolFormer-XL** | 작은 데이터에서 GNN보다 강함 |
| 친화도/결합 예측, 3D 가용 | **Uni-Mol** | 3D 좌표를 사전학습에 활용 |
| 매우 작은 데이터(N<300) | MolFormer 동결 + linear head | 과적합 방지 |
| GPU 비용 제약 | **MolE** 또는 ChemBERTa | 파라미터 작음, 추론 빠름 |
| 신규 chemotype 발굴 | Chemprop + MolFormer 앙상블 | 분포 외 보완 |
| 합성 가능성 / 문법 보장 | **SELFormer** | SELFIES로 invalid SMILES 방지 |

이 표는 Claude Code에 첫 fine-tuning 코드를 요청할 때 그대로 명세로 쓸 수 있습니다. "N=1,500 hERG block 분류, scaffold split 5 seed, MolFormer-XL fine-tuning + Chemprop 베이스라인 병행"이라는 한 문장 프롬프트로 표준 파이프라인이 생성됩니다.

## 신약개발 적용

### 사례 1 — IBM MolFormer-XL의 MoleculeNet 벤치마크

Ross *et al.* (2022)은 PubChem·ZINC 약 11억 분자로 사전학습한 MolFormer-XL을 MoleculeNet 11개 task에 fine-tuning해, 평균적으로 Chemprop·SchNet·GROVER 대비 8개 task에서 SOTA를 갱신했습니다. 특히 **BBBP(Blood-Brain Barrier Permeability)**, **Tox21**, **ClinTox** 같은 N<10,000의 작은 ADMET task에서 우위가 두드러졌습니다. 약학 전공 창업자에게 이 사례의 의미는 — **사전학습 비용을 들이지 않고 공개 체크포인트만으로도 GNN을 능가하는 ADMET 모델을 구축할 수 있다**는 것입니다. IBM은 MolFormer-XL을 Apache 2.0 라이선스로 HuggingFace에 공개했습니다.

### 사례 2 — DP Technology Uni-Mol과 구조 기반 약물 설계

**DP Technology**(중국)의 **Uni-Mol**(2022)은 분자 2억 개와 단백질 결합 포켓 2억 개를 동시에 사전학습한 3D-aware Foundation Model입니다. 결합 친화도(binding affinity) 예측·docking pose ranking에서 — 기존 AutoDock Vina 대비 약 30% 정확도 향상을 보고했고, MoleculeNet의 3D 의존 task(QM9, HIV 등)에서 SOTA를 기록했습니다. DP Technology는 이 기술 위에 **Bohrium·Hermite**라는 SaaS를 구축해 중국 제약사·연구소에 판매하고 있습니다(2023-2024 IR 자료, 정확한 매출 규모는 확인 필요). 한국 시장에는 동급 한국형 Foundation Model이 아직 부재하며, 이는 명확한 시장 기회입니다.

### 사례 3 — 빅파마의 도메인 특화 사전학습

**Pfizer**·**AstraZeneca**·**Genentech** 등은 2023-2024년 자체 내부 데이터로 추가 사전학습한 도메인 특화 Foundation Model을 보유하고 있다고 알려져 있습니다(NeurIPS LMRL 워크숍, AI4Science 워크숍 발표 기준). 공개 모델 위에 **internal corporate ELN(전자 실험 노트) 데이터**를 추가로 사전학습해 자체 화학 공간에 특화된 표현을 만드는 전략입니다. 한국 스타트업의 시사점은 — **인프라(아키텍처)는 평준화됐고, 도메인 특화 데이터로 사전학습을 추가하는 능력이 차별점**이라는 것입니다.

### 사례 4 — Korean Pharmaceutical FM의 부재와 기회

한국 AI 신약개발 기업 중 자체 Foundation Model을 사전학습한 사례는 2024년 기준 매우 제한적입니다(스탠다임·디어젠·온코크로스 등이 자체 ML 스택을 보유하지만, 1억 분자 수준의 본격 FM 사전학습 보도는 확인되지 않음, 확인 필요). 이는 — **한국 임상 환자 적합도가 높은 분자 분포(예: 한국인 CYP2D6 변이형에 최적화된 분자군)로 추가 사전학습한 K-MolFormer**가 명확한 차별화 포지셔닝이 될 수 있다는 점을 시사합니다. 약학 전공자의 강점은 — 어떤 분자 분포가 한국 임상에서 실제 차이를 만드는지 정의할 수 있다는 데 있습니다.

## 창업 관점

Phase 2 단계에서 창업 관점은 세 갈래로 짚습니다. 첫째, **공개 Foundation Model 위의 fine-tuning SaaS**: HuggingFace의 MolFormer-XL·Uni-Mol·ChemBERTa를 백엔드로 두고, 사용자가 자신의 in vitro 어세이 데이터(엑셀 1장)를 업로드하면 3개 모델을 자동 fine-tuning해 결과를 비교 제시하는 도구는, 한국 중소 제약사·바이오테크가 즉시 필요로 하는 가장 명확한 제품 유형입니다. 둘째, **도메인 특화 추가 사전학습 서비스(Continued Pre-training as a Service)**: 고객사의 ELN·HTS 데이터를 받아 공개 FM 위에 5-10% 추가 사전학습해 도메인 특화 체크포인트를 납품하는 모델입니다. GPU 비용을 명확히 견적 가능한 컨설팅+제품 결합형 비즈니스가 됩니다. 셋째, **K-MolFM(Korean Molecular Foundation Model)**: 한국 임상 적합도가 높은 분자 분포로 추가 사전학습한 한국 특화 모델은 — 정부 지원과제(KISTI·KIST·한국연구재단의 AI 신약개발 사업)와 직접 연계 가능한 명분 있는 포지셔닝입니다. 약학 전공 창업자의 도메인 정의 능력이 그대로 기술적 차별점으로 전환됩니다.

## 오늘의 과제

1. **MolFormer 논문 핵심 표 정리 (40분)**: Ross, J. *et al.* (2022), "Large-Scale Chemical Language Representations Capture Molecular Structure and Properties," *Nature Machine Intelligence*, 4, 1256-1264. 의 결과 표(Tables 1-3)에서 MolFormer-XL이 Chemprop·SchNet·GROVER 대비 task별로 얼마나 향상되었는지를 데이터 크기 축(N<5,000 / 5,000-50,000 / >50,000)으로 재정리합니다. 본인이 fine-tuning 모델 선택을 정당화할 때 직접 인용할 근거가 됩니다.

2. **HuggingFace Foundation Model 카탈로그 작성 (40분)**: huggingface.co에서 "molecule" "chemistry" "SMILES" 키워드로 검색해 사전학습된 분자 모델 5개 이상의 카탈로그를 표로 정리합니다. 항목: 모델명·발표 연도·사전학습 데이터 규모·파라미터 수·라이선스·HuggingFace ID·주요 task. 본인의 첫 fine-tuning 실험에서 베이스라인 후보군이 됩니다.

3. **K-MolFM 기획서 1쪽 작성 (40분)**: 본인이 가정한 한국 특화 도메인(예: 한국인 CYP2D6 변이형 대응 약물, K-PMS 한국 임상 데이터 정합 분자군, KFDA 허가 의약품 분포) 한 가지를 골라 — 어떤 분자 분포로 추가 사전학습을 할지, 베이스 모델은 무엇으로 할지, 예상 GPU 비용·기간은 얼마인지, 어떤 다운스트림 task에서 성능 향상이 기대되는지 — 4개 항목으로 구성된 1쪽 기획서를 작성합니다. 향후 정부 과제 제안서·VC 피칭의 기술 차별화 슬라이드로 직접 활용 가능한 자료입니다.

## 참고 자료

- Ross, J. *et al.* (2022). "Large-Scale Chemical Language Representations Capture Molecular Structure and Properties." *Nature Machine Intelligence*, 4, 1256-1264. — MolFormer-XL 원논문. 11억 분자 사전학습 + MoleculeNet SOTA를 입증한 핵심 레퍼런스입니다.
- Chithrananda, S. *et al.* (2020). "ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction." *arXiv:2010.09885*. — 첫 화학 BERT. Foundation Model 패러다임의 출발점이며 베이스라인 비교군으로 자주 인용됩니다.
- Zhou, G. *et al.* (2023). "Uni-Mol: A Universal 3D Molecular Representation Learning Framework." *ICLR 2023*. — 3D 사전학습 모델. docking·binding affinity에서 강점을 보였습니다.
- HuggingFace Hub(huggingface.co/models?search=molecule), MolFormer-XL 체크포인트(huggingface.co/ibm/MoLFormer-XL-both-10pct), Uni-Mol GitHub(github.com/deepmodeling/Uni-Mol) — 공개 Foundation Model의 표준 진입점. Claude Code 프롬프트에 직접 명시할 핵심 리소스입니다.
