# Day 2: 전통 ML 기반 QSAR — Random Forest에서 XGBoost까지

> 이전 학습(Phase 2 Week 1 Day 1)에서 QSAR의 60년 역사와 1·2·3세대 구분, OECD 5원칙, 베이스라인 결정 트리를 정리했습니다. 오늘은 그 트리의 중심에 자리잡은 **2세대 QSAR — 분자 지문(fingerprint) + 트리 기반 앙상블** 모델을 깊이 학습합니다. Random Forest와 XGBoost가 왜 지금도 산업 표준 베이스라인인지, 어떤 하이퍼파라미터가 약학적 의미를 갖는지, 어떤 함정에 빠지지 않아야 하는지를 다룹니다.

## 개요

**전통 ML 기반 QSAR**은 사람이 설계한 분자 표현(주로 ECFP4 같은 분자 지문)을 입력으로, **Random Forest(RF), Support Vector Machine(SVM), Gradient Boosting(XGBoost/LightGBM/CatBoost)** 같은 비신경망 머신러닝 모델로 활성·물성·독성을 예측하는 접근입니다. 1990년대 후반부터 산업의 주류 도구가 된 이 접근은 GNN·Transformer 시대(3세대)에도 여전히 **데이터 N이 500~10,000 범위일 때 최강의 가성비**를 보입니다. 학습 시간이 분 단위이고, 해석 가능하며, GPU가 필요 없고, 작은 데이터에서 과적합이 적기 때문입니다. AI 신약개발 스타트업이 외주·내부 어세이로 모은 첫 2,000개 hERG 데이터로 모델을 만들 때 첫 베이스라인은 거의 예외 없이 **ECFP4 + XGBoost**이며, 이 베이스라인을 넘지 못하는 3세대 모델은 채택되지 않는다는 것이 산업의 현실입니다. 오늘은 이 베이스라인을 정확히 설계하는 능력을 만듭니다.

## 핵심 개념

### 1) 분자 지문(Molecular Fingerprint) — 2세대 QSAR의 입력 표현

2세대 QSAR의 첫 단계는 **분자 → 고정 길이 벡터** 변환입니다. 표준은 **Extended Connectivity Fingerprint(ECFP, 일명 Morgan fingerprint)**입니다. ECFP4는 각 원자의 반경 2(즉 직경 4) 이내 부분 구조를 해시해 2,048비트 벡터의 특정 비트를 1로 켭니다(Day 12에서 상세 다룸). 약학적으로 보면, ECFP4가 인코딩하는 "반경 2의 원자 환경"은 **pharmacophore의 3-원자 단위 작용기 배치**와 정확히 같은 스케일입니다. 한 비트는 "방향족 N-C(=O)-알킬"같은 부분 구조 한 종의 존재 유무를 의미하며, 이는 medicinal chemist가 SAR 분석에서 직접 보는 단위와 일치합니다.

ECFP 외에 산업에서 함께 쓰이는 지문은 다음과 같습니다.

| 지문 | 차원 | 특징 | 추천 용도 |
|------|------|------|---------|
| **ECFP4 / ECFP6** | 2,048bit | 원자 환경 해싱, 가장 범용 | 일반 활성·ADMET 예측 1차 선택 |
| **MACCS keys** | 166bit | 사람이 정의한 166개 부분 구조 | 해석성 필요한 경우 |
| **RDKit descriptors** | 200+ 실수값 | logP·TPSA·HBD/HBA 등 물리화학 | RF·XGBoost와 함께 hybrid 입력 |
| **Avalon / Atom-Pair** | 2,048bit | 다른 해싱 전략, ECFP와 보완 | 앙상블 다양성 확보 |

산업에서 자주 쓰는 강한 베이스라인은 **ECFP4(2,048bit) + RDKit 200 descriptors**를 concatenate한 약 2,248차원 입력입니다. 이렇게 하면 ECFP가 잡지 못하는 전체 분자 수준의 물성(logP, TPSA)이 동시에 들어가 트리 모델이 활용할 수 있습니다.

### 2) Random Forest — 약학자가 가장 먼저 배워야 할 모델

**Random Forest(RF)**는 수백 개의 결정 트리를 데이터의 부트스트랩 표본과 무작위 특성 선택으로 학습한 뒤 평균(회귀) 또는 다수결(분류)하는 앙상블 모델입니다. 약학적 직관에 정확히 들어맞는 두 가지 성질이 있습니다.

- **변수 중요도(feature importance)**: 어떤 ECFP 비트가 활성에 가장 기여했는지를 정량화. 약사·medicinal chemist에게 "이 부분 구조가 hERG 차단에 핵심이었다"는 해석을 제공합니다.
- **out-of-bag(OOB) 점수**: 별도 validation set 없이도 추정된 generalization 성능. 작은 데이터셋에서 특히 유용합니다.

RF의 결정적 약점은 **외삽(extrapolation)이 불가능하다**는 점입니다. 트리는 학습 데이터의 leaf 평균값을 출력하므로, 학습 분포 밖의 분자(applicability domain 밖)에 대해서도 학습 범위 내의 값을 내놓습니다. 약학적으로 이는 — "학습 데이터에 IC50<10nM인 분자가 없으면, RF는 절대 10nM 미만의 예측을 출력하지 않는다"는 의미입니다. de novo 생성 분자를 평가할 때 이 한계는 큰 영향을 줍니다.

### 3) Gradient Boosting — XGBoost가 산업 표준이 된 이유

**Gradient Boosting**은 약한 학습기(얕은 결정 트리)를 순차적으로 추가하면서 이전 트리들의 잔차(residual)를 학습하는 부스팅 앙상블입니다. 2014년 Tianqi Chen의 **XGBoost**가 정규화 항(L1/L2), 결측값 자동 처리, 병렬화, GPU 가속을 결합하면서 Kaggle·KDD 시대를 열었고, 동시에 화학정보학 벤치마크에서도 RF를 넘어서기 시작했습니다. **LightGBM**(Microsoft)과 **CatBoost**(Yandex)는 XGBoost의 변형으로, LightGBM은 leaf-wise 성장과 히스토그램 기반 분할로 대규모 데이터에서 더 빠르고, CatBoost는 범주형 변수를 자체적으로 처리합니다. 약학 데이터에서는 세 모델의 성능 차이가 크지 않으나 **XGBoost가 가장 풍부한 문서와 해석 도구(SHAP 호환)를 갖고 있어** 1순위 선택입니다.

QSAR에서 XGBoost의 핵심 하이퍼파라미터와 약학적 함의는 다음과 같습니다.

| 하이퍼파라미터 | 역할 | 약학적 영향 |
|--------------|------|----------|
| `n_estimators` (트리 수) | 모델 용량 | 너무 크면 sparse한 비트의 과적합 |
| `max_depth` | 트리 깊이 | 4-6이 표준, 깊으면 activity cliff 노이즈 학습 |
| `learning_rate` | 잔차 반영 비율 | 0.01-0.1, 작을수록 일반화 향상 |
| `subsample` / `colsample_bytree` | 행/열 무작위 추출 비율 | 0.7-0.8이 ECFP에서 안정 |
| `reg_alpha` (L1) / `reg_lambda` (L2) | 정규화 강도 | ECFP 2,048bit의 희소성 제어에 중요 |
| `scale_pos_weight` | 양성 클래스 가중치 | 희귀 독성(hERG block) 분류에서 결정적 |

### 4) SHAP — 모델 해석성의 표준

**SHAP(SHapley Additive exPlanations)** 값은 게임이론의 Shapley value를 머신러닝에 적용한 것으로, 각 입력 특성(ECFP의 각 비트)이 예측에 얼마나 기여했는지를 부호 있는 값으로 분해합니다. 트리 기반 모델에서는 **TreeSHAP** 알고리즘으로 정확한 SHAP 값을 빠르게 계산할 수 있습니다. 약학적 활용은 두 갈래입니다.

- **분자 수준 설명**: 특정 후보 분자의 예측 hERG 차단 확률이 높은 이유를, 어떤 부분 구조(ECFP 비트)가 얼마나 기여했는지로 분해. medicinal chemist가 분자 수정 방향을 결정할 수 있습니다.
- **전역 패턴 분석**: 학습 데이터 전체에서 어떤 부분 구조가 hERG 차단의 일관된 위험 신호인지를 정량화. SAR 가설 생성에 직접 쓰입니다.

규제 제출용 모델에서는 SHAP 기반 해석성이 OECD 5원칙 5번(기계적 해석 가능성)을 충족하는 표준 방법으로 자리잡고 있습니다(Phase 4 Week 4에서 다시 다룸).

### 5) Scaffold Split — QSAR의 가장 흔한 함정

전통 ML QSAR의 평가에서 **scaffold split**(Day 19에서 다룬 개념)을 사용하지 않으면 거의 모든 성능 지표가 부풀려집니다. 무작위 분할(random split)을 쓰면 학습/검증 셋에 같은 Murcko scaffold를 공유하는 분자가 섞이고, 모델이 사실상 "외운 scaffold의 작은 변형"을 예측하는 것이 되어 실제 신약 발굴 상황(새로운 scaffold에 일반화)을 반영하지 못합니다. 산업 표준은 **scaffold split** 또는 **time-based split**이며, 두 분할 방식 사이에서도 성능 차이가 10-30% 발생하는 것이 정상입니다. 약학 전공자의 강점은 — "이 모델은 어떤 scaffold 다양성의 화학 공간에서 작동하는가"를 평가 단계에서 명시할 수 있다는 점입니다.

## 작동 원리와 아키텍처

### 표준 2세대 QSAR 파이프라인

```
[1. 분자 입력]
   SMILES (학습/검증 분자 N개)
   ↓ 표준화: 염 제거·tautomer 통일·canonicalization

[2. Featurization]
   RDKit으로 ECFP4 (2,048bit) + 200 descriptors 생성
   결측값(failed featurization) 처리
   → X: (N, 2,248) 행렬

[3. 분할 (Split)]
   Scaffold split (Murcko scaffold 기준 80:10:10)
   5개 random seed로 반복 → 5개 분할 세트

[4. 모델 학습]
   기본 베이스라인: RandomForestRegressor(n=500)
   강한 베이스라인: XGBoost(n=1000, lr=0.05, depth=6)
   하이퍼파라미터 탐색: Optuna 50회 시도 (학습 시간 5-30분/seed)

[5. 평가]
   회귀: RMSE, MAE, R², Spearman ρ
   분류: ROC-AUC, PRC-AUC, BEDROC, EF@top-1%
   Cliff metric: activity cliff 분리 보고 (Day 18)
   AD 점수: 평균 Tanimoto 유사도 또는 ensemble variance

[6. 해석 + 출력]
   SHAP 값으로 분자 수준·전역 해석
   각 예측에 AD 점수와 신뢰구간 첨부
```

### 베이스라인 설정의 의사결정 표

같은 데이터셋이라도 task 유형에 따라 권장 베이스라인이 달라집니다.

| Task 유형 | 추천 1차 모델 | 핵심 평가 지표 | 함정 |
|---------|------------|----------|------|
| 회귀 (pIC50, logS) | XGBoost Regressor | RMSE + Spearman ρ | 분포 비대칭일 때 RMSE만 보면 오판 |
| 이진 분류 (hERG block Y/N) | XGBoost Classifier | PRC-AUC + EF@1% | ROC-AUC만 보면 클래스 불균형 무시 |
| 멀티태스크 (활성 + ADMET 30개) | LightGBM + 30 head | task별 ROC-AUC + 평균 | 데이터 양 차이로 일부 task 무시됨 |
| 시간 외 일반화 (특허 데이터) | XGBoost + time split | scaffold + time split 동시 | random split만 보면 산업 적용 시 실패 |

이 표는 바이브코딩으로 Claude Code에 첫 모델 학습 코드를 작성시킬 때, "task 유형과 함정을 명시한 프롬프트"로 직접 활용 가능합니다.

## 신약개발 적용

### 사례 1 — Stokes et al. (2020) Halicin: D-MPNN 이전의 베이스라인이 무엇이었는가

Day 1에서 다룬 **Halicin 발견** 논문에서, 저자들은 D-MPNN을 제안하기 전에 **Random Forest + ECFP4** 베이스라인을 함께 보고했습니다. 보고된 ROC-AUC는 RF가 약 0.88, D-MPNN이 약 0.90 수준으로 큰 차이가 없었으며, 결정적 차이는 **새로운 chemotype의 enrichment**에서 나타났습니다. 이 논문이 보여주는 산업적 통찰은 — 3세대 모델을 채택하기 전에 2세대 베이스라인을 정확히 측정해야 "왜 3세대가 필요한지"를 정량적으로 설명할 수 있다는 점입니다. 투자자·파트너 제약사에게 모델 채택을 설득할 때 결정적인 데이터가 됩니다.

### 사례 2 — Tox21 챌린지와 QSAR 산업 표준의 확립

**Tox21**(NIH·EPA·FDA 공동, 2014~) 챌린지는 약 12,000개 화합물의 12개 독성 endpoint에 대한 QSAR 챌린지였으며, 최종 결과에서 1·2위는 모두 **앙상블 트리 기반 모델**(딥러닝과 트리의 하이브리드)이 차지했습니다. 이 챌린지가 산업에 남긴 유산은 — "환경독성 같이 데이터 N이 수천~수만 규모의 문제에서는 2세대 트리 베이스라인이 여전히 SOTA에 가깝다"는 정량적 증거입니다. 미국 EPA와 OECD가 ToxCast·Tox21 데이터에 기반한 규제 적용 QSAR 모델 가이드라인을 정비할 때, 이 결과가 직접 반영되었습니다.

### 사례 3 — Recursion Pharmaceuticals의 ML 스택과 트리 모델

**Recursion**은 이미지 기반 phenotypic screening을 핵심으로 하지만, 화학 구조 기반 모델 스택에는 LightGBM·CatBoost 같은 트리 부스팅이 깊이 통합되어 있다고 알려져 있습니다(2023~2024년 IR 자료 기반, 구체 비율은 확인 필요). 이는 빅 플레이어조차 새로운 표적·새로운 chemotype에 대해 첫 모델은 트리 기반으로 시작한다는 산업적 관행을 보여줍니다.

### 사례 4 — XGBoost로 만든 PROTAC 분해 효율 예측

2023년 이후 **PROTAC**(표적 단백질 분해제) 데이터셋이 공개되면서 ECFP+XGBoost로 분해 효율을 예측하는 모델들이 학계에서 발표되고 있습니다. PROTAC는 분자량이 700-1,200 Da로 Lipinski 규칙 밖이라 일반 QSAR이 어렵지만, RF/XGBoost의 비선형 학습 능력이 이 영역에서도 합리적 베이스라인을 만든다는 것이 확인되고 있습니다. 약학 전공자에게 시사점은 — 새로운 모달리티가 등장할 때 가장 먼저 시도되는 모델은 여전히 2세대이며, 3세대 모델은 데이터가 충분히 쌓인 뒤 채택된다는 점입니다.

## 창업 관점

Phase 2 단계의 창업 관점은 짧게 짚습니다. **첫 SaaS·내부 도구의 첫 모델은 거의 항상 ECFP+XGBoost 베이스라인**입니다. 이는 학습 비용이 시간당 GPU 대비 1/100 이하, 학습 시간이 분 단위, 해석성이 SHAP으로 자동 확보, 규제 제출 시 OECD 5원칙을 가장 깔끔하게 충족한다는 점에서 그렇습니다. 약학 전공자의 진입점은 — (1) 특정 표적군에 특화된 XGBoost 베이스라인을 **OECD 5원칙 자동 문서화**와 함께 묶어 규제 친화적 SaaS로 제공하는 방향, (2) 약사 워크플로우에 통합된 SHAP 기반 SAR 가설 생성 도구, (3) 작은 데이터(N<2,000)로 학습된 신뢰 가능한 ADMET 모델 라이브러리를 internal API로 묶어 한국 제약사에 공급하는 방향입니다. 세 방향 모두 GPU 인프라가 필요 없어 초기 자본 효율이 매우 높습니다.

## 오늘의 과제

1. **Tox21 챌린지 결과 분석 (40분)**: Tox21 Data Challenge 2014의 최종 결과 페이지 또는 후속 review 논문(Mayr et al., 2016, "DeepTox" 또는 Idakwo et al., 2018 review)에서 상위 5팀의 모델 구성을 확인하고, 트리 기반·신경망·하이브리드 비율을 표로 정리합니다. 그리고 각 모델이 사용한 분자 표현(ECFP·MACCS·기술자·SMILES)을 함께 정리해, 2세대 표현이 어떤 task에서 여전히 강한지 패턴을 도출합니다.

2. **ECFP+XGBoost 베이스라인 설계서 작성 (40분)**: 본인이 관심있는 ADMET 항목 1개(BBB·hERG·CYP3A4 추천)에 대해 — ① 분자 표현(ECFP4 또는 +descriptors), ② 분할 방식(scaffold split + 5 seed), ③ XGBoost 하이퍼파라미터 초기값, ④ 평가 지표 3개, ⑤ SHAP 해석 출력 형식 — 다섯 항목을 명시한 설계서를 1쪽 분량으로 작성합니다. 이 설계서는 Claude Code에 그대로 입력하면 코드가 생성되는 수준이어야 합니다.

3. **2세대 vs 3세대 QSAR 의사결정 트리 보강 (20분)**: Day 1의 베이스라인 결정 트리에 "언제 2세대로 충분한가" 결정 기준을 추가합니다. 데이터 N뿐만 아니라 — scaffold 다양성, 데이터 노이즈(어세이 반복 표준편차), 규제 제출 필요성, 해석성 요구 강도 — 네 축을 추가한 의사결정 표를 작성합니다.

## 참고 자료

- Chen, T., Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *KDD '16*, 785–794. — XGBoost 원논문. 정규화·결측 처리·근사 알고리즘이 왜 산업 표준이 되었는지를 가장 정확히 설명합니다.
- Mayr, A. *et al.* (2016). "DeepTox: Toxicity Prediction using Deep Learning." *Frontiers in Environmental Science*, 3, 80. — Tox21 챌린지 1위 팀의 방법론. 딥러닝과 트리 앙상블의 하이브리드 구성이 핵심.
- Lundberg, S. M., Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*. — SHAP의 원전. TreeSHAP과 함께 트리 기반 QSAR의 해석성 표준을 만든 논문.
- RDKit(rdkit.org)·scikit-learn·XGBoost(xgboost.readthedocs.io) — 2세대 QSAR 베이스라인을 한 시간 안에 구성할 수 있는 표준 스택. Claude Code 프롬프트에서 명시할 핵심 라이브러리.
