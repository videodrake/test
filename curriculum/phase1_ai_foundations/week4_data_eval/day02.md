# Day 2: 데이터 품질과 전처리 — 쓰레기를 넣으면 쓰레기가 나온다

> 이전 학습(Week 4 Day 1)에서 ChEMBL·PDB·ZINC의 구조와 규모, 그리고 각 DB의 잘 알려진 한계(어세이 이질성·해상도 편향·합성 가능성 함정)를 정리했습니다. 오늘은 그 원천 데이터를 **모델 학습용 데이터셋**으로 바꾸는 과정 — 분자 표준화(standardization), 활성값 정제, 데이터 분할(split) — 에서 발생하는 실수와 그 약학적 해석을 다룹니다. 모델 성능의 진짜 상한선은 알고리즘이 아니라 이 전처리 결정에서 정해집니다.

## 개요

머신러닝 신약개발에서 "Garbage in, garbage out(GIGO)"은 비유가 아니라 측정 가능한 현실입니다. 동일한 표적·동일한 모델 구조에서도 전처리 방식만 바꾸면 외부 검증 성능(R², AUROC)이 0.1~0.3씩 흔들리는 것이 표준 관찰입니다. 데이터 품질 결함은 크게 (1) **분자 표현의 불일치**(같은 분자가 다른 SMILES로 저장됨), (2) **활성값의 측정 편차와 검열(censoring)**, (3) **데이터 누출(data leakage)을 일으키는 잘못된 분할**의 세 가지에서 발생하며, 이 셋은 모두 약학적 판단을 통해서만 정확히 처리됩니다. 오늘은 각각의 원인·전처리 절차·약학 전공자가 비전공자보다 정확히 판단할 수 있는 지점을 정리합니다.

## 핵심 개념

### 1) 분자 표준화 — 같은 분자를 같다고 인식시키기

원천 DB의 SMILES는 그대로 쓸 수 없습니다. **카페인**(caffeine) 하나만 해도 PubChem·ChEMBL·논문 supplementary에서 최소 6~8가지 서로 다른 SMILES 문자열로 표기됩니다 — 원자 순서가 다르거나, 케토–에놀(keto–enol) 호변이성질체(tautomer) 표기가 다르거나, 방향족성(aromaticity) 표기 규칙이 다르거나, 염(salt) 또는 용매화물(solvate)이 붙어 있기 때문입니다. 모델은 문자열을 보기 때문에 "다른 분자"로 인식합니다.

표준화의 표준 절차는 다음과 같습니다.

| 단계 | 처리 내용 | 약학적 함의 |
|------|---------|---------|
| **Disconnect metals** | 금속–유기 결합 분리 | 시스플라틴(cisplatin) 같은 금속 함유 약물은 별도 처리 |
| **Remove salts/solvates** | 가장 큰 유기 단편만 유지 | mesylate, HCl 등 염 형태 차이가 활성과 무관(생체 내에서 해리) |
| **Normalize functional groups** | 니트로, 카르복실산 등 표준 표기 | 같은 작용기의 일관된 인식 |
| **Tautomer canonicalization** | 호변이성질체 통일 | 구아닌(guanine) 등에서 결정적; 결합 부위에서 실제 다른 형태를 취함 |
| **Stereochemistry** | 입체화학 정보 유지/표준화 | (S)-탈리도마이드(thalidomide) 사례 — 광학 이성질체가 작용 다름 |
| **Canonical SMILES** | RDKit/InChI 기반 표준화 | 동일 분자에 동일 문자열 보장 |

표준화 후에는 **InChIKey**(분자의 27자 해시)로 중복을 통합합니다. ChEMBL의 약 240만 화합물을 InChIKey 기준으로 묶으면 같은 분자가 다중 레코드인 경우가 흔하며, 중복 제거 후 학습 가능 집합은 일반적으로 5~15% 줄어듭니다.

**약학 전공자의 결정적 판단 지점**: 호변이성질체 표준화는 단순한 문자열 처리가 아닙니다. 예를 들어 **HIV 역전사효소 억제제**의 일부 분자는 결합 부위에서 특정 호변이성질체로 고정되는데, 학습 데이터에서 다른 호변이성질체로 표준화하면 모델은 잘못된 약물–표적 상호작용 가설을 학습합니다. 약학에서 익숙한 "이 분자는 생체 pH에서 어떤 형태로 존재하는가" 판단이 직접 적용되는 지점입니다.

### 2) 활성값 정제 — 측정값을 학습 가능한 라벨로

ChEMBL의 활성값은 **그대로 회귀 라벨로 쓸 수 없습니다**. 다음 결함을 순서대로 처리해야 합니다.

**(a) 단위·로그 변환**: IC50·Ki는 nM 또는 µM로 보고되며 분포가 극단적으로 오른쪽으로 치우칩니다(skewed). 학습 라벨로는 **pIC50 = −log10(IC50[M])**, pKi = −log10(Ki[M]) 형태가 표준입니다. ChEMBL의 `pchembl_value` 필드가 이를 이미 계산해 제공합니다. 약학에서 익숙한 pKa·pH 변환과 동일한 로그 변환입니다.

**(b) 검열 데이터(censored data)**: 어세이 농도 한계 때문에 "IC50 > 10,000 nM"로 보고되는 경우가 다수입니다. 이를 단순히 10,000 nM로 처리해 회귀 모델에 넣으면 모델은 "활성 없음"을 "특정한 약한 활성"으로 학습합니다. 대처 방법은 (i) 회귀에서는 `=` 관계만 사용, (ii) 분류(active/inactive)로 변환, (iii) **Tobit 회귀** 같은 검열 인식 모델 사용입니다. 임상에서 익숙한 *생존 분석*의 검열 처리와 본질이 같습니다.

**(c) 중복 측정 통합**: 같은 분자–표적 쌍이 여러 논문에서 측정된 경우 — 일반적인 약학 데이터에서는 같은 화합물의 IC50이 **0.5~1.5 로그 단위**(즉 3~30배)까지 흩어집니다. 단순 평균은 이상치(outlier)에 취약하므로 **중앙값(median) 또는 기하평균** 사용이 표준입니다. 단, 흩어짐(spread)이 1.5 로그 이상이면 어세이 조건 자체가 다른 것이므로 통합하지 않고 별도로 두거나 제외해야 합니다.

**(d) 어세이 조건 분리**: ChEMBL의 `assay_type`이 B(binding)·F(functional)인지, 세포 어세이인지 효소 어세이인지에 따라 IC50의 의미가 다릅니다. 약학 학부에서 배운 **Cheng-Prusoff 보정**(Ki = IC50 / (1 + [S]/Km))을 알면, 다른 기질 농도에서 측정된 IC50들을 비교 가능한 Ki로 환산할지 여부를 즉각 판단할 수 있습니다 — 비약학 전공자가 흔히 놓치는 지점입니다.

### 3) 데이터 분할 — 모델 성능 환상의 가장 큰 원인

**무작위 분할(random split)** 은 분자 데이터셋에서 거의 항상 잘못된 선택입니다. 같은 화학적 골격(scaffold)을 공유하는 유사체(analog)들이 train/test에 동시에 들어가면 모델은 "외운 것을 다시 본 셈"이 되어 외부 데이터에서 급격히 성능이 떨어집니다(generalization gap). 표준 분할 전략은 다음과 같습니다.

| 분할 방법 | 원리 | 적합한 상황 |
|---------|------|----------|
| **무작위(random)** | 인덱스 셔플 후 8:1:1 | 학습 곡선 진단·디버그 한정 |
| **Scaffold split** | Bemis-Murcko scaffold로 그룹화 후 분리 | 신규 화학 골격으로의 일반화 평가 — **사실상 표준** |
| **Temporal split** | 시간 기준(연도) 분할 | 신약개발의 실제 상황과 가장 일치 |
| **Cluster split** | Tanimoto 유사도 군집 분리 | scaffold가 모호한 경우 |
| **Out-of-distribution(OOD)** | 특정 표적·물성 범위 제외 | 분포 외 일반화 측정 |

전형적으로 같은 모델·같은 데이터셋에서 무작위 분할 R² = 0.85가 scaffold split에서 0.55, temporal split에서 0.40으로 떨어지는 양상은 흔합니다. **첫째 수치를 모델 성능으로 보고하면 거의 반드시 외부 데이터에서 실패**합니다.

또한 라이브러리 중복(near-duplicates), 표적이 같으면서 서로 다른 측정 단위·어세이로 들어온 데이터 등 **데이터 누출(data leakage)** 의 형태가 다양하므로 분할 후에 train과 test의 **최대 Tanimoto 유사도가 0.7 이상인 쌍이 얼마나 있는지** 점검하는 것이 필수입니다.

### 4) 화합물 필터 — 학습 전 제외해야 할 분자

원천 데이터에는 **모델 학습에 해로운 분자**들이 섞여 있습니다.

- **PAINS(Pan-Assay Interference Compounds)**: 다중 어세이에서 비특이적으로 활성으로 보고되는 골격(예: rhodanines, curcuminoids). Baell & Holloway(2010)의 약 480개 substructure 필터가 표준이며 RDKit·FILTER에 내장되어 있습니다.
- **REOS(Rapid Elimination of Swill)**: 반응성·독성·합성 비현실 골격 제거.
- **Lipinski Rule of Five 일탈**: 학습 데이터로 쓰면 모델이 비현실적 화학 공간을 탐색하게 됩니다. 단, 항체–약물 결합체(ADC)나 PROTAC처럼 의도적으로 Ro5를 벗어나는 모달리티는 별도 취급해야 합니다.
- **반응성 작용기**: 마이클 수용체(Michael acceptor), 에폭사이드 등은 어세이 위양성(false positive)의 주요 원인.

약학 전공자는 학부에서 배운 **유기화학·약물 설계** 지식으로 이 골격들이 왜 위양성을 일으키는지를 즉시 판단할 수 있어 비전공자보다 적절한 필터 강도를 정합니다.

## 작동 원리와 아키텍처

### 표준 전처리 파이프라인

```
[ChEMBL raw]  표적 X의 모든 activity 레코드
   │
   ├─ Step 1. SMILES 표준화
   │     ─ RDKit: salt strip → tautomer canon → stereo 보존 → canonical SMILES
   │     ─ InChIKey 계산 → 중복 통합
   │
   ├─ Step 2. 활성값 필터
   │     ─ assay_type 고정 (B 또는 F 단일 선택)
   │     ─ standard_relation = "=" 만 (회귀 시)
   │     ─ confidence_score ≥ 7 (단일 표적 확정)
   │     ─ pchembl_value 존재
   │
   ├─ Step 3. 중복 통합
   │     ─ 같은 (InChIKey, target) 쌍의 활성값 → median
   │     ─ 1.5 로그 이상 흩어지면 제외 또는 별도 split
   │
   ├─ Step 4. 화합물 필터
   │     ─ PAINS substructure 제거
   │     ─ MW 100-700, heavy atoms 5-70 등 합리적 범위
   │
   ├─ Step 5. Scaffold split
   │     ─ Bemis-Murcko scaffold 추출
   │     ─ 8 : 1 : 1 (train : valid : test)
   │     ─ train–test 간 max Tanimoto < 0.7 확인
   │
   └─→ [학습 준비 완료 데이터셋]
        ─ 일반적으로 원천의 30-60% 만 남음
```

### 핵심 설계 결정

| 결정 사항 | 선택지 | 권장 | 이유 |
|---------|--------|------|------|
| 라벨 형식 | IC50(nM) / pIC50 / 분류 | pIC50 (회귀) | 로그 분포로 정규성 회복, Cheng-Prusoff 직관과 일치 |
| 검열 처리 | 단순 대입 / 제외 / Tobit | 회귀 시 제외 + 분류 병행 | 정보 손실 vs 편향의 trade-off |
| 중복 통합 | 평균 / 중앙값 / 분산 컷오프 | 중앙값 + 1.5 로그 컷 | 이상치 강건성 |
| 표준화 도구 | RDKit / MolVS / ChemAxon | RDKit + MolVS | 오픈소스, 재현성 |
| 분할 | random / scaffold / temporal | scaffold (기본), temporal (실전 검증) | 실제 신약개발과 일치 |

## 신약개발 적용

### 실패 사례 — 같은 모델, 다른 전처리

**MoleculeNet 재평가(Wu et al. 2018 *Chem. Sci.* 및 후속 연구들)**: 동일한 약물 물성 데이터셋(BBBP, Tox21, HIV)에 대해 무작위 분할 vs scaffold split을 비교한 결과, AUROC가 평균 0.08~0.15 떨어지는 것이 일관되게 보고되었습니다. 일부 모델 비교 논문이 무작위 분할로 보고한 SOTA(state-of-the-art)는 scaffold split에서 다시 측정하면 순위가 뒤바뀌는 경우가 빈번했습니다.

**ChEMBL pIC50 회귀의 현실(Cortés-Ciriano & Bender 2019 *J. Cheminf.*)**: 같은 표적의 IC50 측정 재현성 자체가 **약 0.4 로그(즉 2~3배)** 의 표준편차를 가지므로, 회귀 모델의 RMSE가 0.5 로그 미만이면 "사실상 측정 노이즈 수준"에 도달한 것입니다. 이 천장(ceiling)을 모르고 더 복잡한 모델로 RMSE를 0.45→0.40으로 줄였다고 보고하는 것은 노이즈를 학습하는 과적합(overfitting)입니다. 임상에서 익숙한 *측정의 한계*(limit of quantification) 개념이 그대로 적용됩니다.

**TDC 벤치마크(Huang et al. 2021 *NeurIPS*)**: Therapeutic Data Commons는 ADMET·결합 친화도 등 70+ 데이터셋을 통일된 scaffold/temporal split으로 제공하며, 같은 모델이 무작위 분할 leaderboard와 비교하면 일관되게 낮은 성능을 보입니다. 산업에서는 TDC 점수가 실전 일반화 성능에 더 가깝다고 보는 합의가 형성되고 있습니다(Day 4에서 상세).

### 표 — 약학 지식이 결정적인 전처리 판단

| 상황 | 비약학 데이터 사이언티스트의 함정 | 약학 전공자의 판단 |
|------|--------------------------|---------------|
| 사람 vs 마우스 표적 IC50 통합 | 같은 표적이면 통합 | 종간 차이(예: hERG, CYP) 알고 분리 |
| 같은 약물의 prodrug과 활성형 통합 | InChIKey 다르면 다른 분자 | 생체 내에서 같은 물질이 되는지 판단 |
| 어세이 조건 무시 | standard_type만 보고 묶음 | 세포 어세이 vs 효소 어세이 의미 차이 인식 |
| 활성 임계값 설정 | 임의(예: 1 µM 분류) | 표적별 임상적 의미 있는 농도 사용 |
| PAINS 골격 처리 | 일률적 제거 | 실제 약물 중 PAINS 골격 알고 예외 처리 |

## 창업 관점

Phase 1 단계에서는 짧게만 짚으면 충분합니다 — 다수의 AI 신약개발 스타트업이 모델 아키텍처 차별화에 집중하지만, 실제로는 **전처리 파이프라인의 질**과 **자체 데이터 큐레이션**이 외부 검증 성능을 가르며, 약학 전공자가 데이터 품질 판단에서 비교우위를 갖는 영역입니다. 시장·BM 차원의 분석은 Phase 5에서 다룹니다.

## 오늘의 과제

1. **ChEMBL 데이터 전처리 시뮬레이션 (50분)**: Day 1에서 선택한 표적의 데이터에 대해 단계별 감소를 추정합니다 — (a) 원천 activity 레코드 수, (b) `standard_type=IC50`만 남긴 수, (c) `standard_relation="="` 만 남긴 수, (d) `confidence_score ≥ 7` 만 남긴 수, (e) `pchembl_value` 존재만 남긴 수, (f) InChIKey 기준 중복 통합 후 수. 각 단계에서 데이터가 몇 % 줄어드는지 표로 정리하고, 마지막에 "이 정도 크기로 회귀 모델 학습이 가능한가? 분류로 전환해야 하는가?" 판단을 한 문단으로 적습니다. A4 1쪽.
2. **분할 전략 비교 리서치 (40분)**: Therapeutic Data Commons(tdcommons.ai)에서 ADMET 그룹의 데이터셋 하나(예: Caco2_Wang, BBB_Martins, hERG)를 선택해 그 페이지에 나오는 **무작위 분할 vs scaffold split** 의 리더보드 점수 차이를 표로 정리합니다. 같은 모델이 두 분할에서 얼마나 차이나는지, 그 차이가 약학적으로 무엇을 의미하는지 한 단락으로 적습니다.
3. **전처리 정책 문서 초안 (30분)**: 본인이 AI 신약개발 회사의 CTO라고 가정하고, 회사가 모든 신규 모델 학습에 따라야 할 **데이터 전처리 정책**을 1쪽 분량으로 작성합니다. 포함 항목 — (a) 표준화 도구와 절차, (b) 활성값 라벨 형식, (c) 검열 데이터 처리 원칙, (d) 분할 전략 기본값, (e) 화합물 필터(PAINS 등) 적용 기준, (f) 데이터 누출 점검 방법. 이 문서는 이후 Phase 2~4에서 실제 모델 학습 시 다시 참조합니다.

## 참고 자료

- Wu, Z. *et al.* (2018). "MoleculeNet: a benchmark for molecular machine learning." *Chemical Science*, 9(2), 513–530. — 분자 ML 벤치마크의 출발점. 무작위 vs scaffold split의 영향을 처음 체계적으로 보고.
- Cortés-Ciriano, I. & Bender, A. (2019). "Reliable Prediction Errors for Deep Neural Networks Using Test-Time Dropout." *Journal of Chemical Information and Modeling*, 59(7), 3330–3339. — 활성 측정의 재현성과 모델 성능의 천장 개념.
- Baell, J. B. & Holloway, G. A. (2010). "New substructure filters for removal of pan assay interference compounds (PAINS) from screening libraries." *Journal of Medicinal Chemistry*, 53(7), 2719–2740. — PAINS 필터의 정의 논문. 약 480개 substructure 표준.
- Huang, K. *et al.* (2021). "Therapeutic Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development." *NeurIPS Datasets Track*. — TDC 벤치마크. 분할 전략과 일반화 평가의 산업 표준 후보.
- RDKit(rdkit.org) · MolVS(github.com/mcs07/MolVS) · Therapeutic Data Commons(tdcommons.ai) — 본인 데이터에 직접 표준화·분할 파이프라인을 적용하며 익히는 것이 가장 효과적.
