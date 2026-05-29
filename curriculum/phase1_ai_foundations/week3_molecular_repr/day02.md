# Day 2: 분자 지문과 기술자 — 구조를 숫자로

> 이전 학습(Day 1)에서 분자를 문자열로 적는 SMILES와 그 약점·대안(SELFIES, InChI)을 다뤘습니다. 오늘은 그 문자열에서 한 발 더 들어가, **분자를 고정 길이 숫자 벡터로 환원하는 두 가지 고전 도구** — 분자 지문(molecular fingerprint)과 분자 기술자(molecular descriptor)를 학습합니다. 이 두 표현은 Random Forest·XGBoost 같은 전통 머신러닝 모델의 입력이 되며, 약학 전공자가 매일 보던 logP·MW·HBA 같은 숫자가 실제로 어떻게 모델에 들어가는지를 보여줍니다.

## 개요

분자 지문(molecular fingerprint)과 분자 기술자(molecular descriptor)는 분자를 **고정 길이의 숫자 벡터**로 변환하는 두 가지 전통적 표현 방식입니다. 지문은 "이 구조 조각이 있는가/없는가"를 0과 1의 비트 벡터(보통 1024 또는 2048비트)로 적는 **구조 인코딩**이고, 기술자는 분자량·logP·수소 결합 공여자 수처럼 **물리화학적 속성을 직접 계산한 실수 벡터**입니다. 둘 다 1990년대 화학정보학(cheminformatics)의 산물이지만, 2024~2025년의 최신 GNN·Transformer 모델 시대에도 여전히 산업 현장의 1차 베이스라인으로 살아남았습니다. **데이터가 적은 경우(수백~수천 분자)에는 GNN보다 지문+Random Forest가 더 좋은 결과를 내는 경우도 흔합니다**. 약학 전공자에게는 이 표현이 특별한 이유가 있습니다 — Lipinski의 Rule of Five 같은 약학 지식이 그대로 기술자 벡터의 일부로 모델에 주입되기 때문입니다.

## 핵심 개념

### 분자 지문 — 구조를 비트로

**분자 지문(molecular fingerprint)** 은 분자의 부분구조(substructure)를 비트 벡터로 인코딩한 표현입니다. 가장 널리 쓰이는 두 계열이 있습니다.

| 지문 종류 | 원리 | 길이 | 특성 |
|---------|------|------|------|
| **MACCS 키** | 사전 정의된 166개 구조 패턴의 유무 | 166비트 | 해석 가능, 작음 |
| **ECFP/Morgan** | 각 원자의 r-hop 이웃 환경을 해싱 | 보통 1024/2048비트 | 비편향, 산업 표준 |
| **RDKit Fingerprint** | 경로(path) 기반 부분구조 해싱 | 가변 | RDKit 기본 |
| **PubChem 지문** | 881개 사전 정의 패턴 | 881비트 | DB 검색용 |

산업 표준은 **ECFP(Extended Connectivity Fingerprint)**, 정확히는 **Morgan 알고리즘 기반 ECFP4/ECFP6**입니다. ECFP4의 "4"는 직경(diameter) 4, 즉 반경 2-hop까지의 이웃을 본다는 뜻입니다. 알고리즘은 단순합니다 — (1) 각 원자에 초기 식별자(원소·차수·전하 등)를 부여하고, (2) 매 반복마다 자기와 이웃 원자의 식별자를 합쳐 해싱해 새 식별자를 만들고, (3) 반경 r회까지 반복한 뒤 모든 식별자를 해싱해 비트 위치를 결정합니다. 결과는 1024 또는 2048비트의 sparse한 0/1 벡터입니다.

**Tanimoto 유사도(Tanimoto coefficient)** 는 지문 간 유사도의 표준입니다: $T(A,B) = |A \cap B| / |A \cup B|$. 두 분자의 지문이 공유하는 비트 수를 합집합 비트 수로 나눈 값입니다. **Tanimoto 0.7 이상이면 "유사 화합물(analog)"** 로 간주하는 것이 1990년대부터의 경험적 임계값이며, 가상 스크리닝 1차 필터링에서 여전히 쓰입니다.

### 분자 기술자 — 약학 지식이 그대로 들어가는 곳

**분자 기술자(molecular descriptor)** 는 분자의 물리화학적·구조적 속성을 수치화한 값입니다. RDKit만으로 약 200개, Mordred 라이브러리는 약 1,800개를 계산할 수 있습니다. 약학 전공자에게 친숙한 핵심 기술자는 다음과 같습니다.

| 기술자 | 의미 | 약학적 중요성 |
|-------|------|-------------|
| **MW** (분자량) | 분자량 g/mol | Lipinski RoF: ≤500 |
| **logP** | 옥탄올-물 분배계수 | 친유성 — BBB 투과, 경구 흡수의 핵심 |
| **HBA / HBD** | 수소결합 수용체/공여자 수 | RoF: HBA ≤10, HBD ≤5 |
| **TPSA** | 위상학적 극성 표면적 (Ų) | 경구 흡수 ≤140, BBB 투과 ≤90 |
| **RotBonds** | 회전 가능 결합 수 | 경구 생체이용률, 결합 엔트로피 |
| **QED** | 약물성(drug-likeness) 정량 지표 | Bickerton et al. 2012, 0~1 점수 |
| **SAS** | 합성 접근성 점수 | 1(쉬움)~10(어려움) |

여기서 **약학 전공자의 결정적 우위가 드러납니다** — Lipinski의 Rule of Five(MW ≤500, logP ≤5, HBA ≤10, HBD ≤5)는 1997년 화이자 Christopher Lipinski가 2,245개 경구 약물을 분석해 도출한 규칙입니다. 비약학 전공자가 모델 입력으로 RDKit 200개 기술자를 무작정 넣을 때, 약학 전공자는 (1) 어떤 기술자가 적응증·투여 경로에 따라 의미를 갖는지(예: 주사제는 RoF 무관, BBB 약물은 TPSA가 결정적), (2) 어떤 기술자가 서로 강하게 상관되어 다중공선성을 일으키는지(MW와 분자체적은 상관 0.9 이상), (3) 어떤 임계값이 임상적 의미인지를 사전 판단해 **feature selection의 의사결정 비용을 결정적으로 낮춥니다**.

### 지문 vs 기술자 — 언제 무엇을 쓰나

| 비교 항목 | 지문 (ECFP) | 기술자 (RDKit 200개) |
|---------|-------------|----------------------|
| 인코딩 대상 | 부분구조 패턴 | 물리화학적 속성 |
| 값 형태 | 0/1 비트 | 실수 (정규화 필요) |
| 해석 가능성 | 낮음 (해시 비트) | 높음 (의미 있는 변수) |
| 강점 과제 | 유사도 검색, SAR | ADMET, 약물성 평가 |
| 약점 | 동일 비트의 충돌(hash collision) | 새 구조 발견 약함 |

실무에서는 **둘을 연결(concatenation)해 함께 입력**하는 것이 가장 흔합니다 — ECFP4(2048비트) + RDKit 기술자(약 200개)를 이어 붙여 2,248차원 벡터로 만들고 XGBoost에 넣는 식입니다. Phase 2 Week 1에서 다룰 전통 QSAR의 표준 입력 형태입니다.

## 작동 원리와 아키텍처

### 표준 파이프라인

```
[SMILES 문자열]
   ↓ 1. RDKit 파싱: SMILES → Mol 객체
   ↓ 2a. AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
   ↓ 2b. Descriptors.MolWt(mol), MolLogP(mol), NumHDonors(mol), ...
[ECFP4 비트 벡터(2048d)] + [기술자 벡터(약 200d)]
   ↓ 3. 정규화: 기술자는 StandardScaler (지문은 그대로)
   ↓ 4. Feature selection: 분산 0인 비트 제거, VIF 기반 다중공선성 제거
[정제된 특징 벡터(약 1,500~2,200d)]
   ↓ 5. 모델: Random Forest / XGBoost / LightGBM
[예측값: pIC50, logS, hERG 활성 등]
```

이 파이프라인의 단계 1~2는 RDKit으로 수십 줄 안에 구현되며, **바이브코딩으로 Claude Code에게 "ChEMBL CSV를 받아 ECFP4 + RDKit 200 기술자를 계산해 XGBoost로 pIC50 예측하는 스크립트를 만들어줘"라고 지시하면 한 번에 작동하는 베이스라인이 나옵니다**. 이것이 Phase 1 학습이 끝나는 시점의 학습자가 가장 먼저 만들 수 있어야 할 실용적 도구입니다.

### 주요 설계 결정

| 결정 사항 | 선택지 | 권장 | 이유 |
|---------|--------|------|------|
| 지문 반경 | r=1,2,3 (ECFP2/4/6) | 2 (ECFP4) | 작용기 크기와 일치 |
| 비트 길이 | 1024 / 2048 / 4096 | 2048 | 충돌과 메모리 균형 |
| 기술자 수 | 핵심 10개 / RDKit 200 / Mordred 1800 | RDKit 200 | 노이즈 대비 표준 |
| 결측치 처리 | 평균 대치 / 0 대치 / 제거 | 0 대치 + 결측 인디케이터 | QSAR 관례 |

## 신약개발 적용

### 산업 현장의 표준 베이스라인

ECFP+RF/XGBoost 조합은 화학정보학 30년 표준이며, MoleculeNet(Wu et al. 2018, *Chemical Science*) 벤치마크의 다수 과제에서 GNN 대비 경쟁력 있는 성능을 보입니다. 특히 데이터가 1만 분자 미만인 적응증에서는 GNN의 데이터 갈증을 메우지 못해 ECFP+XGBoost가 더 좋거나 동등합니다. Recursion·Atomwise 같은 회사들도 1차 가상 스크리닝의 빠른 필터로 지문 기반 유사도 검색을 사용한다고 보고되어 왔으며, ECFP+Tanimoto 검색은 GPU 없이 CPU만으로 수억 분자를 수 시간 내 처리할 수 있다는 비용 효율이 결정적입니다.

### Lipinski Rule of Five — 기술자가 만든 산업 표준

Lipinski의 1997년 논문 *Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings*(Advanced Drug Delivery Reviews)는 약학사에서 가장 인용된 논문 중 하나이며, 그 본질은 **4개의 분자 기술자(MW, logP, HBA, HBD)에 임계값을 설정한 단순 규칙**입니다. 이 규칙이 산업 표준이 된 것 자체가 "단순한 기술자로 임상 성공 확률을 의미 있게 예측할 수 있다"는 명제의 가장 강력한 증거이며, 오늘날 모든 분자 생성·필터링 파이프라인의 마지막 단계에 들어갑니다.

### 한계와 이행

지문의 본질적 약점은 **해시 충돌**과 **동일 비트의 의미가 분자마다 다를 수 있다**는 점입니다. 또한 부분구조 패턴 중심이므로 **3D 구조·입체화학·conformer 다양성**을 표현하지 못합니다. 이 한계는 Day 3(분자 그래프)와 Day 4(3D 표현)에서 그래프 신경망과 3D 인코더로 보완되는 과정을 다룹니다. Phase 2 Week 1(QSAR)에서는 ECFP+XGBoost 베이스라인이 GNN과 어떻게 비교되는지를 본격적으로 다룹니다.

## 창업 관점

지문·기술자는 분자 표현의 하부 인프라이지만, **약학 전공자 창업자에게 두 가지 짧은 관전 포인트**가 있습니다. 첫째, RoF·QED 같은 약물성 점수의 임상적 의미를 정확히 알고 있다는 것은, 향후 SaaS 형태의 ADMET·약물성 평가 도구를 설계할 때 사용자(메디시널 케미스트)에게 의미 있는 임계값과 시각화를 정의할 수 있는 차별점입니다. 둘째, 적은 데이터에서 ECFP+XGBoost가 GNN보다 잘 작동한다는 실증적 사실은 **희귀질환·신생 표적 영역에서 거대 모델을 가진 빅테크 대비 작은 모트를 만들 수 있는 근거**가 됩니다 — Phase 5에서 다룰 포지셔닝 전략의 한 줄기입니다.

## 오늘의 과제

1. **약물 10개의 RoF 직접 점검 (40분)**: 본인이 잘 아는 경구 약물 10개(이부프로펜·아토르바스타틴·아세트아미노펜·메트포르민·시메티딘 등)에 대해 PubChem에서 MW, XLogP3, HBA, HBD, TPSA를 적어 표로 만들고, Lipinski RoF·Veber 규칙(TPSA ≤140, RotBonds ≤10)에 위반되는 항목을 표시합니다. 위반에도 시판된 약물이 있다면(예: 아토르바스타틴의 MW) 그 임상적 배경을 1문단으로 설명합니다. 약학 지식이 기술자 해석에 어떻게 결합되는지 체감하는 과제입니다.
2. **ECFP Tanimoto 유사도 검증 (30분)**: 카페인(`CN1C=NC2=C1C(=O)N(C(=O)N2C)C`)과 (a) 테오필린, (b) 테오브로민, (c) 아스피린의 ECFP4 Tanimoto 유사도를 RDKit으로 계산해 표로 만듭니다. 결과가 약학적 직관(잔틴 유도체끼리 유사)과 일치하는지 검증합니다. Claude Code로 5줄 짜리 RDKit 스크립트를 짜게 시켜도 됩니다 — 핵심은 결과 해석입니다.
3. **지문 vs 기술자 의사결정 1쪽 메모 (30분)**: 본인이 관심 있는 적응증(예: 특정 암종·중추신경계 질환)에 대해 데이터 1,000개·10,000개·100,000개 시나리오 각각에서 "지문만 / 기술자만 / 둘 다 / GNN"의 어느 조합을 선택하겠는지, 그 이유와 함께 1쪽으로 정리합니다. Phase 2 Week 1과 직접 연결되는 사전 정리 작업입니다.

## 참고 자료

- Rogers, D. & Hahn, M. (2010). "Extended-Connectivity Fingerprints." *Journal of Chemical Information and Modeling*, 50(5), 742-754. — ECFP의 표준 정의 논문. ECFP4가 산업 표준이 된 근거.
- Lipinski, C. A. *et al.* (1997). "Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings." *Advanced Drug Delivery Reviews*, 23(1-3), 3-25. — Rule of Five 원전. 약학 전공자가 이미 익숙한 그 규칙이 곧 기술자 벡터의 핵심 4개임을 확인.
- Bickerton, G. R. *et al.* (2012). "Quantifying the chemical beauty of drugs." *Nature Chemistry*, 4, 90-98. — QED 점수의 정의. 분자 생성 모델의 평가 지표로 광범위하게 사용.
- RDKit Descriptors 문서(rdkit.org) — 200개 기술자 목록과 계산식. 본인 관심 적응증에 어떤 기술자가 의미 있는지 사전 점검에 활용.
