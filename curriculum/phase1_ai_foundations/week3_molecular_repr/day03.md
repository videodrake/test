# Day 3: 분자 그래프 — 원자는 노드, 결합은 엣지

> 이전 학습(Day 2)에서 분자를 고정 길이 비트 벡터로 환원하는 ECFP 지문과 RDKit 200개 기술자를 다뤘습니다. 오늘은 그 비트화 단계를 거치지 않고 **분자의 그래프 구조를 그대로 모델에 넣는 표현** — 분자 그래프(molecular graph)와 이를 학습하는 그래프 신경망(Graph Neural Network, GNN)의 표현 계층을 학습합니다. 이 표현은 ECFP의 해시 충돌과 동일 비트 다의성 문제를 원천적으로 우회하며, 2017년 이후 분자 물성 예측·생성 모델의 사실상 표준이 됩니다.

## 개요

분자 그래프(molecular graph)는 분자를 **원자 = 노드(node), 결합 = 엣지(edge)** 로 두는 표현 방식입니다. 추상적이지만 화학자가 종이에 그리는 구조식과 가장 가까운 데이터 표현이며, 원자별·결합별 특징을 손실 없이 그대로 담을 수 있습니다. 이 표현 위에서 작동하는 **그래프 신경망**은 각 원자가 이웃 원자의 정보를 단계적으로 모으면서 자신의 표현(embedding)을 갱신하는 메시지 패싱(message passing) 방식으로 분자 임베딩을 학습합니다. 2017년 Gilmer 등이 MPNN(Message Passing Neural Network)으로 양자화학 물성 예측에서 사람이 설계한 기술자를 넘어선 이후, GNN은 ChemProp·SchNet·Uni-Mol 등 화학 AI의 표준 골격이 되었습니다. 약학 전공자에게 이 표현이 결정적인 이유는 **GNN의 k-hop 이웃 집계가 약학에서 말하는 약물발현단(pharmacophore) 인식과 구조적으로 동일한 연산**이기 때문입니다 — 모델 내부 동작에 약학적 의미를 부여할 수 있다는 점이 ECFP의 해석 불가능한 비트와 갈리는 지점입니다.

## 핵심 개념

### 분자 그래프의 구성요소

분자 그래프 $G = (V, E, X, R)$는 네 요소로 정의됩니다.

| 구성요소 | 의미 | 표현 |
|---------|------|------|
| **V (노드 집합)** | 원자 목록 | 분자당 N개 원자 |
| **E (엣지 집합)** | 결합 목록 | 인접 행렬 또는 엣지 리스트 |
| **X (노드 특징)** | 원자별 특징 벡터 | 원소·차수·전하·방향족성 등 (보통 10~80차원) |
| **R (엣지 특징)** | 결합별 특징 벡터 | 결합 차수·고리 여부·conjugation 등 (보통 6~12차원) |

표준 노드 특징은 다음을 포함합니다 — 원소 one-hot(C/N/O/S/F/Cl/Br/I 등), formal charge, 결합 가능한 수소 수, 혼성화(sp/sp²/sp³), 방향족성, 고리 소속, chirality 표기. 엣지 특징은 결합 차수(단/이중/삼/방향족), 입체결합 표기(E/Z), 고리 결합 여부 등입니다. RDKit으로 SMILES를 파싱해 위 특징을 추출하면 한 분자가 즉시 그래프로 변환됩니다.

ECFP가 모든 부분구조 정보를 한 번에 비트 위치로 해싱하는 **lossy 인코딩**이라면, 분자 그래프는 정보 손실 없이 구조를 보존한다는 점이 결정적 차이입니다. 다만 분자마다 노드 수가 다르므로 **고정 길이 입력을 가정하는 일반 신경망(MLP)에 그대로 넣을 수 없다**는 점이 GNN이라는 별도 아키텍처를 필요로 한 이유입니다.

### 메시지 패싱 — GNN이 분자를 이해하는 방식

**메시지 패싱(message passing)** 은 각 노드가 이웃 노드로부터 정보를 받아 자신의 표현을 갱신하는 절차이며, 한 번의 갱신을 한 **레이어(layer)** 라고 부릅니다. 한 레이어의 동작을 약학 전공자에게 친숙한 언어로 풀면 다음과 같습니다.

```
각 원자 v에 대해:
   1. 메시지 수집: 이웃 원자 u의 현재 표현 h_u와 결합 특징 e_uv를 가져옴
   2. 메시지 집계: 모든 이웃 메시지를 합/평균/최대로 묶음 (aggregation)
   3. 자기 갱신: 자기 이전 표현 h_v와 집계된 메시지를 결합 → 새 표현 h_v'
한 레이어 종료 → 다음 레이어 반복
```

레이어 수 k가 늘어날수록 각 원자는 **k-hop 거리 안의 이웃 정보**를 모으게 됩니다. 보통 k = 3~5 레이어를 사용하며, 이는 의약화학에서 **pharmacophore가 3~5개 원자 거리 내의 작용기 조합으로 정의되는 사실과 정확히 일치**합니다 — 예를 들어 GPCR 결합에 결정적인 "양전하성 질소 + 2-3개 원자 떨어진 방향족 고리" 패턴은 3-hop 메시지 패싱으로 정확히 포착됩니다. 이 일치가 우연이 아닌 이유는, 약물의 활성이 국소적 분자 환경에 의해 결정되며 GNN의 receptive field가 이 국소성을 직접 모델링하기 때문입니다.

### GNN 변형들 — GCN, GAT, MPNN, GIN

분자 GNN은 메시지 집계 방식의 차이로 여러 변형이 존재합니다.

| 변형 | 집계 방식 | 강점 | 약점 |
|------|---------|------|------|
| **GCN** (Kipf & Welling 2017) | 이웃의 평균 (정규화) | 단순, 빠름 | 모든 이웃을 동등 취급 |
| **GAT** (Veličković et al. 2018) | 어텐션 가중 평균 | 중요 이웃에 집중 | 파라미터 많음 |
| **MPNN** (Gilmer et al. 2017) | 엣지 특징을 메시지 함수에 포함 | 결합 차수·고리 정보 활용 | 메모리 사용량 큼 |
| **GIN** (Xu et al. 2019) | sum + MLP (Weisfeiler-Lehman 등가) | 이론적 표현력 최대 | 정규화 어려움 |

약학 응용에서 결합 차수와 입체화학을 잘 다루는 **MPNN과 그 후속 D-MPNN(Directed MPNN, Yang et al. 2019)** 이 ChemProp의 기본 골격이 되었고, 산업 표준 baseline의 한 축을 차지합니다.

### Readout — 분자 단위 표현으로 묶기

원자별 임베딩이 끝나면 분자 전체를 대표하는 **분자 임베딩(graph-level embedding)** 으로 묶어야 합니다. 이를 **readout(또는 graph pooling)** 이라 하며 다음 방식이 일반적입니다.

- **Sum/Mean/Max pooling**: 모든 원자 임베딩을 합·평균·최대로 환원 (단순, 가장 흔함)
- **Attention pooling**: 학습된 가중치로 원자별 중요도를 다르게 부여
- **Set2Set** (Vinyals 2016): 순서 무관한 집합을 다루는 LSTM 기반 readout
- **Virtual node**: 모든 원자와 연결된 가상 노드를 추가해 그 임베딩을 분자 대표로 사용

분자 임베딩은 보통 128~512차원이며, 이 벡터가 다운스트림 헤드(MLP)를 거쳐 pIC50·logP·hERG 같은 표적 값을 출력합니다.

## 작동 원리와 아키텍처

### 표준 GNN 파이프라인

```
[SMILES 문자열]
   ↓ 1. RDKit 파싱: SMILES → Mol 객체 → 원자/결합 리스트
[분자 그래프 (V, E, X, R)]
   ↓ 2. 노드/엣지 특징 추출 (RDKit 헬퍼 또는 PyG/DGL의 데이터로더)
[(노드행렬 N×F_n, 엣지 인덱스 2×E, 엣지행렬 E×F_e)]
   ↓ 3. 메시지 패싱 레이어 × k (보통 3~5)
[원자 임베딩 N×D]
   ↓ 4. Readout (sum/attention pooling)
[분자 임베딩 1×D]
   ↓ 5. 예측 헤드 (2-3 layer MLP)
[예측값: pIC50, 용해도, 독성 확률 등]
```

이 파이프라인을 **바이브코딩으로 Claude Code에게 "PyG로 ChemProp 스타일 D-MPNN을 만들어 ChEMBL의 hERG 데이터로 학습하는 스크립트를 짜줘"라고 지시하면 한 번에 작동하는 결과물이 나옵니다**. 약학 전공자의 역할은 코드 한 줄을 짜는 것이 아니라, (1) hERG 양성/음성의 IC50 임계값을 어떻게 정의할지(보통 10 µM 또는 30 µM이 임상적 임계), (2) train/test split을 무작위로 할지 scaffold split으로 할지(후자가 신약 일반화 성능의 실제 척도), (3) 어떤 분자 부분구조가 모델 예측에 기여했는지를 GNN attention map으로 해석하는 것입니다.

### 핵심 설계 결정

| 결정 사항 | 선택지 | 권장 (소형 분자, 데이터 1만~10만) | 이유 |
|---------|--------|---------|------|
| 라이브러리 | PyG / DGL / ChemProp | ChemProp (D-MPNN) | 화학 특화 기본값 |
| 레이어 수 | 2 / 3 / 5 / 10 | 3-5 | pharmacophore 거리와 일치 |
| 노드 임베딩 차원 | 64 / 128 / 256 | 128-256 | 표현력과 과적합 균형 |
| Readout | mean / sum / attention | sum + attention | 합산이 분자 크기 정보 보존 |
| 데이터 split | random / scaffold | **scaffold split** | 신약 일반화 검증의 표준 |

**scaffold split**의 중요성을 강조할 필요가 있습니다 — 무작위 분할은 학습셋과 평가셋에 같은 골격(scaffold) 분자가 섞이게 만들어 성능을 과대평가합니다. 실제 신약개발은 "기존에 없던 새로운 골격에서 활성을 예측"하는 과제이므로 scaffold split이 진짜 일반화 능력을 측정하며, MoleculeNet도 이 기준을 사용합니다.

## 신약개발 적용

### 산업 표준이 된 ChemProp과 그 임팩트

**ChemProp**(Yang et al. 2019, *Journal of Chemical Information and Modeling*)은 MIT Coley 그룹이 공개한 D-MPNN 기반 분자 물성 예측 도구로, **데이터셋이 1만 분자 수준일 때 ECFP+RF/XGBoost 베이스라인을 일관되게 상회**한다는 결과가 다수 보고되었습니다. ChemProp의 가장 유명한 적용 사례는 2020년 MIT의 *Cell* 논문 *A Deep Learning Approach to Antibiotic Discovery*(Stokes et al. 2020)로, 약 2,335개 화합물의 *E. coli* 성장 억제 데이터로 D-MPNN을 학습한 뒤 약 1억 7백만 분자의 ZINC15 라이브러리를 가상 스크리닝해 새로운 항생제 후보 **할리신(halicin)** 을 발굴했습니다 — 기존 항생제와 구조적으로 매우 다른 신규 골격이며, 다제내성 균주에도 활성을 보였다는 점에서 GNN의 일반화 능력을 보여준 사례로 자주 인용됩니다.

### Foundation Model로의 확장

2022~2024년에는 GNN 위에 사전학습(pre-training) 패러다임이 결합됩니다 — **Uni-Mol**(Zhou et al. 2023, *ICLR*), **GROVER**(Rong et al. 2020), **MolFormer**(IBM 2022) 등이 수억~수십억 분자로 자기지도 학습된 모델을 공개했고, ChEMBL의 작은 ADMET 데이터셋에 fine-tuning하는 방식이 표준이 되었습니다. 2024년 Isomorphic Labs와 DeepMind는 AlphaFold3와 같은 라인의 분자 표현 모델을 공개했습니다(확인 필요 — 구체적 모델명·수치는 사내 도구 단계). 이 흐름은 Phase 2 Week 1(QSAR)과 Phase 3 Week 3(Foundation Model)에서 본격적으로 다룹니다.

### 한계 — GNN이 실패하는 지점

GNN의 약점도 명확합니다. 첫째, **데이터가 수백 개 수준이면 ECFP+RF가 GNN보다 안정적**입니다 — Day 2에서 다룬 "데이터가 적으면 전통 ML이 우세"라는 명제의 실증적 근거입니다. 둘째, **3D conformer 정보를 기본 그래프는 표현하지 못합니다** — 회전이성질체, 활성 conformer 선택, 단백질 결합 포즈는 Day 4에서 다룰 3D 표현이 필요합니다. 셋째, **over-smoothing** 문제 — 레이어가 10 이상으로 깊어지면 모든 원자 임베딩이 비슷해져 구분력을 잃는 현상이 보고됩니다(Li et al. 2018). 분자가 작아 3-5 레이어로 충분한 약물 도메인에서는 이 문제가 임상적으로 부각되지 않습니다.

## 창업 관점

GNN은 분자 AI의 표준 도구이므로 그 자체가 차별점이 되진 않습니다. 다만 **데이터 양에 따라 ECFP+XGBoost와 GNN을 적절히 조합하는 판단력**이 약학 전공자에게는 실용적 무기가 됩니다 — 희귀질환·신생 표적 영역에서는 데이터가 수백~수천 수준이라 거대 GNN보다 ECFP 베이스라인이 우세할 수 있고, 이 사실을 알고 있는 창업자는 "거대 모델이 항상 좋다"는 빅테크 마케팅의 함정을 피해 자원을 효율적으로 배분합니다. Phase 5 Week 2(비즈니스 모델)에서 다룰 "표적 좁히기 전략"의 기술적 근거가 됩니다.

## 오늘의 과제

1. **분자 그래프 손으로 그리기 (40분)**: 본인이 잘 아는 약물 3개(예: 이부프로펜·시메티딘·아세트아미노펜)에 대해 노드(원자) 리스트와 엣지(결합) 리스트를 직접 표로 적습니다. 각 노드에 (원소, 차수, 방향족성, 수소 수) 4개 특징을, 각 엣지에 (결합 차수, 고리 결합 여부) 2개 특징을 적습니다. Day 2의 ECFP 비트와 비교해 "구조 정보 보존"이 무슨 의미인지 체감하는 과제입니다.
2. **할리신 사례 분석 (40분)**: Stokes et al. (2020) *A Deep Learning Approach to Antibiotic Discovery*(*Cell*)의 Abstract와 Discussion을 읽고, (a) 학습 데이터 규모와 양성/음성 비율, (b) 가상 스크리닝 라이브러리 규모, (c) 실험 검증한 hit 수, (d) 할리신의 화학적 신규성에 대한 평가를 1쪽으로 정리합니다. 약학 전공자 관점에서 "MIC, 다제내성, 작용 기전" 항목도 함께 정리하면 Phase 3·5와 연결됩니다.
3. **scaffold split의 의미 정리 (30분)**: ChemProp 문서 또는 RDKit의 Murcko scaffold 설명을 읽고, scaffold split이 무작위 split보다 신약개발 일반화 성능을 더 정직하게 측정하는 이유를 1쪽으로 정리합니다. "내가 만든 ADMET 도구를 빅파마에 팔 때 어떤 split의 성능 지표를 제시해야 신뢰를 얻을 수 있는가"의 답이 됩니다.

## 참고 자료

- Gilmer, J. *et al.* (2017). "Neural Message Passing for Quantum Chemistry." *International Conference on Machine Learning (ICML)*. — MPNN의 통합 프레임워크. 이후 분자 GNN의 표준 표기법.
- Yang, K. *et al.* (2019). "Analyzing Learned Molecular Representations for Property Prediction." *Journal of Chemical Information and Modeling*, 59(8), 3370-3388. — ChemProp(D-MPNN) 원전. 산업 표준 baseline.
- Stokes, J. M. *et al.* (2020). "A Deep Learning Approach to Antibiotic Discovery." *Cell*, 180(4), 688-702. — 할리신 발굴. GNN 기반 신약 발굴의 대표 사례.
- ChemProp (chemprop.readthedocs.io) — MIT의 오픈소스 D-MPNN 구현. PyG 기반 ChemProp v2가 2023년 공개되어 fine-tuning이 쉬워졌으며, 본인 관심 적응증의 ChEMBL 데이터로 직접 학습 가능.
