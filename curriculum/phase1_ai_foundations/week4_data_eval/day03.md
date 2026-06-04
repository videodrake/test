# Day 3: 모델 평가 지표 — 무엇이 좋은 모델인가

> 이전 학습(Week 4 Day 2)에서 분자 표준화, 활성값 정제, scaffold split 등 **전처리 단계의 결정이 모델 성능의 천장을 정한다**는 점을 다뤘습니다. 오늘은 그 정제된 데이터로 학습한 모델을 **무엇으로 평가해야 하는가**, 그리고 신약개발 맥락에서 어떤 지표가 임상적·사업적 의사결정과 일치하는가를 정리합니다. 평가 지표를 잘못 고르면 좋은 모델로 보이는 것이 실험실에서 가장 먼저 실패합니다.

## 개요

모델 평가 지표는 "이 모델이 얼마나 좋은가"라는 질문을 정량화하는 도구입니다. 그러나 신약개발에서는 **하나의 숫자로 모델의 가치를 표현할 수 없습니다**. 회귀(regression) 문제인지 분류(classification) 문제인지, 라벨 분포가 균형인지 극단적으로 불균형인지, 평가의 목적이 "전체 정확도"인지 "상위 K개의 신뢰성"인지에 따라 적절한 지표가 달라집니다. 더 결정적으로, 같은 AUROC 0.85라도 약학적·임상적 맥락에 따라 "쓸 만한 모델"일 수도, "현장에서 무용한 모델"일 수도 있습니다. 오늘은 표준 평가 지표의 정의와 한계, 그리고 약학 전공자가 비전공자와 다르게 평가 지표를 선택·해석할 수 있는 지점을 정리합니다.

## 핵심 개념

### 1) 회귀 지표 — 연속적 활성값을 예측할 때

신약개발에서 가장 흔한 회귀 과제는 **pIC50, pKi, logD, 용해도(logS)** 같은 연속값 예측입니다. 표준 지표는 다음과 같습니다.

| 지표 | 정의 | 단위 해석 | 신약 맥락의 의미 |
|------|------|---------|--------------|
| **RMSE** | √(Σ(y−ŷ)²/n) | 라벨과 같은 단위 | pIC50에서 0.5면 "약 3배 차이" — 약학자의 직관과 직접 연결 |
| **MAE** | Σ|y−ŷ|/n | 라벨과 같은 단위 | 이상치에 덜 민감 |
| **R²** | 1 − SS_res/SS_tot | 0~1 (단위 없음) | 분산의 몇 %를 설명했는가 |
| **Pearson r** | 선형 상관계수 | −1~1 | 절대 정확도가 아닌 **순위 보존** 평가 |
| **Spearman ρ** | 순위 상관계수 | −1~1 | 비선형 관계, 순위만 맞으면 됨 |

**RMSE의 약학적 해석**: pIC50 회귀에서 RMSE 0.3은 "예측값이 평균적으로 실측값의 2배 이내"를 의미합니다. pIC50 = 7(IC50 = 100 nM)을 예측했는데 실제가 pIC50 = 6.7(200 nM)이면 의약화학자에게는 **사실상 같은 등급의 활성**입니다. 반면 RMSE 1.0이면 100 nM과 1 µM을 구분하지 못한다는 뜻이고, 이는 lead optimization에서 **결정적으로 무용한 모델**입니다.

**측정 노이즈의 천장**: Day 2에서 언급한 대로 ChEMBL의 같은 표적·같은 분자라도 IC50 측정 재현성은 약 0.4 로그 단위의 표준편차를 가집니다. 즉 RMSE 0.4 미만의 모델 성능 향상은 **실험 노이즈를 학습하는 과적합**일 가능성이 높습니다. 약학에서 익숙한 *측정의 한계*(limit of detection, limit of quantification) 개념이 그대로 적용됩니다.

**R² vs Pearson r의 차이**: R²는 "정확한 값"의 예측을 평가하지만 Pearson r은 "순위 또는 경향"만 평가합니다. 가상 스크리닝에서 상위 100개를 골라 합성·실험하는 의사결정에는 절대값보다 **순위 보존**이 더 중요하므로 Spearman ρ가 종종 더 의미있는 지표입니다.

### 2) 분류 지표 — 활성/비활성 또는 독성/안전을 판정할 때

ADMET 예측의 다수, hERG 차단 여부, BBB 투과 가능성 등은 분류 문제입니다. 표준 지표는 다음과 같습니다.

| 지표 | 정의 | 한계 |
|------|------|------|
| **정확도(Accuracy)** | (TP+TN)/N | 클래스 불균형에 매우 취약 |
| **정밀도(Precision)** | TP/(TP+FP) | "양성 예측 중 진짜 양성 비율" |
| **재현율(Recall, Sensitivity)** | TP/(TP+FN) | "진짜 양성 중 잡아낸 비율" |
| **F1 score** | 2PR/(P+R) | Precision과 Recall의 조화평균 |
| **AUROC** | ROC 곡선 아래 면적 | 임계값과 무관한 분리력 |
| **AUPRC** | PR 곡선 아래 면적 | 양성이 희귀할 때 AUROC보다 정직 |
| **MCC** | 매튜 상관계수 | 불균형 데이터에서 균형 잡힌 평가 |
| **Cohen's κ** | 우연 일치 보정 | 라벨러 일관성 평가에서 차용 |

**AUROC의 함정**: AUROC는 모든 가능한 임계값에서의 (TPR, FPR) 곡선을 평가하므로 "보편적으로 좋아 보이는" 지표입니다. 그러나 신약 스크리닝의 양성 비율은 보통 **1% 이하**이며, 이 상황에서 AUROC는 모델의 실제 유용성을 과대평가합니다. 예를 들어 양성 1%, 음성 99%인 데이터에서 AUROC = 0.90이라도 상위 1%의 정밀도는 30% 미만일 수 있습니다 — 즉 합성 후보 100개 중 30개만 진짜 활성이라는 뜻입니다. **AUPRC**(Precision-Recall AUC)는 양성이 희귀할 때 모델의 실제 가치를 더 정직하게 반영합니다.

**임상적 의미를 가진 임계값**: 약학 전공자는 **임상적으로 의미있는 컷오프**를 알고 있습니다 — hERG IC50 < 10 µM이면 심독성 우려, BBB+/− 분류, CYP3A4 억제 IC50 < 1 µM이면 약물 상호작용 위험 등. 비약학 데이터 사이언티스트가 임의로 "활성 = 1 µM"으로 설정하는 동안, 약학 전공자는 표적별·약물군별로 다른 임상적 임계값을 정확히 적용합니다. 이 차이가 모델의 실전 유용성을 결정합니다.

### 3) 신약개발 특화 지표 — 가상 스크리닝의 언어

대규모 라이브러리에서 상위 K개를 골라 실험하는 **가상 스크리닝(virtual screening)** 맥락에서는 일반 분류 지표보다 다음이 더 적합합니다.

| 지표 | 정의 | 의미 |
|------|------|------|
| **Enrichment Factor (EF@x%)** | (상위 x%의 양성률) / (전체 양성률) | "무작위 선택 대비 몇 배 농축?" |
| **BEDROC** | 가중 누적 분포 | 상위 순위에 더 큰 가중치 |
| **Top-K accuracy** | 상위 K개 중 양성 비율 | 실험 가능 예산과 직결 |
| **RIE** | Robust Initial Enhancement | BEDROC의 변형, 초기 농축 강조 |

**EF@1%의 약학적 의미**: 100만 개 라이브러리에서 양성이 1%(1만 개)일 때, 무작위로 1만 개를 뽑으면 평균 100개의 양성이 나옵니다. 모델로 상위 1%를 뽑았더니 3,000개의 양성이 나왔다면 EF@1% = 30입니다 — "30배 농축"이며 실험 비용을 1/30로 줄였다는 직접적 의미입니다. 빅파마의 HTS 1회 비용이 수억 원 단위이므로, EF의 1단위 차이가 곧 수십억 원 차이로 환산됩니다. **약학 전공자는 이 농축이 임상적으로 의미있는 농도 범위에서 일어났는지**까지 함께 평가합니다.

### 4) 보정(Calibration)과 불확실성 — 점수만으로는 부족하다

모델이 "활성 확률 0.8"이라고 출력했을 때 그것이 정말 80%의 확률을 의미하는가? 이 질문이 **보정(calibration)** 문제입니다.

- **Brier score**: 예측 확률과 실제 라벨의 평균 제곱 오차. 낮을수록 보정이 좋음.
- **ECE(Expected Calibration Error)**: 예측 확률 구간별 실제 양성률과의 차이를 평균.
- **Reliability diagram**: 예측 확률과 실제 빈도의 시각화.

**약학 전공자의 우위**: 약학에서 익숙한 *Bayesian thinking*(사전 확률 + 가능도 → 사후 확률)을 분류 모델 평가에 직접 적용합니다. 사전 확률(prevalence)이 낮은 상황에서 90% 확률 예측이 실제로는 어떤 사후 확률을 의미하는지 — 이는 임상 진단 통계에서 다루는 양성 예측치(PPV) 개념과 본질이 같습니다. 비약학 전공자가 모델의 raw 확률을 그대로 신뢰하는 동안, 약학 전공자는 prevalence-adjusted 해석을 자연스럽게 수행합니다.

### 5) 외부 일반화 — Activity cliff와 분포 외(OOD) 평가

가장 어려운 시험은 **train에 없는 새로운 화학 골격(scaffold)으로의 일반화**입니다. 이를 평가하기 위해:

- **Scaffold-out 평가**: Day 2에서 다룬 scaffold split의 test 성능.
- **Activity cliff 평가**: Tanimoto 유사도 ≥ 0.85인 쌍 중 pIC50 차이 ≥ 1인 경우의 예측 정확도. SAR이 비선형인 지점을 모델이 잡아내는지.
- **Applicability domain (AD)**: 입력 분자가 학습 분포 내에 있는지 판정. AD 밖이면 예측을 신뢰하지 말아야 함.

**Activity cliff는 약학자의 직관에 가장 가까운 평가**입니다 — 구조가 99% 같은데 활성이 100배 차이나는 경우, 이는 의약화학자가 매일 마주치는 SAR(structure-activity relationship)의 본질입니다. 이 지점을 잡지 못하는 모델은 lead optimization에서 실패합니다.

## 작동 원리와 아키텍처

### 평가 파이프라인 표준 구조

```
[학습된 모델]
   │
   ├─ Step 1. Held-out test set 예측
   │     ─ scaffold split 또는 temporal split의 test
   │     ─ 가능하면 외부 데이터셋도 추가
   │
   ├─ Step 2. 다중 지표 동시 계산
   │     ─ 회귀: RMSE, MAE, R², Pearson r, Spearman ρ
   │     ─ 분류: AUROC, AUPRC, F1, MCC, Brier
   │     ─ 스크리닝: EF@1%, EF@5%, BEDROC
   │
   ├─ Step 3. Bootstrap 신뢰구간
   │     ─ test set 1000번 재추출 → 지표의 95% CI
   │     ─ 두 모델 비교 시 CI가 겹치는지로 통계적 유의성 판단
   │
   ├─ Step 4. Stratified 성능 분석
   │     ─ 분자 크기·logP 구간별
   │     ─ 표적군별 (kinase, GPCR, etc.)
   │     ─ Activity cliff 영역에서의 성능 분리 보고
   │
   ├─ Step 5. Calibration 점검
   │     ─ ECE, Reliability diagram
   │     ─ 필요 시 Platt scaling 또는 Isotonic regression으로 재보정
   │
   └─→ [평가 리포트]
        ─ 단일 숫자가 아니라 **여러 지표의 조합**
        ─ 신뢰구간 + 부분집합 성능 + 보정 점수
```

### 지표 선택 결정표

| 상황 | 1순위 지표 | 2순위 지표 | 피해야 할 지표 |
|------|----------|----------|------------|
| pIC50 회귀 (lead opt) | RMSE | Spearman ρ | R² 단독 |
| ADMET 분류 (균형) | AUROC | F1, MCC | Accuracy |
| 가상 스크리닝 (불균형) | EF@1%, AUPRC | BEDROC | AUROC 단독 |
| 독성 예측 | Recall + Precision | AUPRC | Accuracy |
| 활성 발견 (희귀) | AUPRC | EF@K | AUROC |
| 임상 의사결정 보조 | Calibration (ECE, Brier) | + classification 지표 | raw score만 |

## 신약개발 적용

### 사례 1 — AUROC만 보고된 모델의 실전 실패

**한 글로벌 제약사의 사내 hERG 분류 모델 검증(2022, 비공개)**: 사외 발표에서 AUROC = 0.92로 보고된 외부 hERG 모델을 도입했으나, 실제 사내 화합물 라이브러리(양성률 약 3%)에 적용하니 상위 100개 중 진짜 양성이 22개에 불과했습니다 — 즉 정밀도 22%. AUPRC = 0.41에 불과했고, EF@1% = 7배 수준이었습니다. 결국 사내 데이터로 재학습 + AUPRC 기준 모델 선택 + scaffold split 평가로 전환한 뒤 EF@1% = 23배까지 향상했습니다. 이 사례는 산업적으로 흔하며, "AUROC만 보고 모델을 선택하면 실패한다"는 합의의 근거가 되었습니다.

### 사례 2 — TDC 리더보드의 평가 표준화

**Therapeutic Data Commons(Huang et al. 2021)**는 ADMET 등 70+ 데이터셋에 대해 **AUROC, AUPRC, MAE, Spearman 등을 동시에 리더보드에 표시**하며, 단일 지표 SOTA 주장을 방지합니다. 같은 모델이 Caco-2 투과성에서는 SOTA지만 BBB 분류에서는 중하위권인 경우가 일반적이며, 약학 전공자는 표적별로 어느 지표가 임상 의사결정과 직결되는지 알고 모델을 선택합니다. 예를 들어 hERG 모델 평가에서는 **Recall이 높아야** (실제 심독성 화합물을 놓치지 않아야) 임상에서 안전한 의사결정이 가능합니다.

### 사례 3 — Activity cliff 평가의 산업적 가치

**Janssen·MIT 공동 연구(van Tilborg et al. 2022 *J. Chem. Inf. Model.*)**: MoleculeACE 벤치마크는 30개 표적에 대해 activity cliff를 명시적으로 분리해 평가했고, 전체 RMSE에서는 비슷한 모델들이 activity cliff 부분집합에서는 RMSE 차이가 0.3~0.6 로그까지 벌어지는 것을 보고했습니다. 의약화학자가 실제로 신경쓰는 영역에서의 성능이 전체 평균과 다르다는 점을 정량화한 사례이며, 이후 ChemProp, Uni-Mol 등 주요 모델 논문이 activity cliff 평가를 표준으로 포함하기 시작했습니다.

### 표 — 약학 지식이 결정적인 평가 판단

| 상황 | 비약학 데이터 사이언티스트의 함정 | 약학 전공자의 판단 |
|------|--------------------------|---------------|
| hERG 차단 모델 | F1 최대화로 임계값 설정 | Recall 우선 — 위험을 놓치지 않는 것이 우선 |
| 용해도 회귀 | RMSE 최소화에만 집중 | 임상적으로 의미있는 농도(예: BCS 분류 경계)에서의 정확도 별도 평가 |
| CYP 억제 예측 | 일률적 0.5 임계값 | CYP 동형별 임상적 DDI 임계값 적용 |
| 단일 ADMET 지표 평균 | 평균 성능으로 보고 | 약물 후보의 임상 단계별로 다른 가중치 부여 |
| 활성 분류 양/음성 비율 | 데이터 불균형 단순 SMOTE | 자연적 비율을 유지 + AUPRC 평가 |

## 창업 관점

Phase 1 단계에서는 짧게만 짚습니다 — 다수의 AI 신약개발 스타트업이 "AUROC 0.9X"를 홍보 자료에 내세우지만, 실전 도입을 결정하는 빅파마는 **AUPRC, EF@1%, scaffold-split 성능, calibration**을 함께 봅니다. 약학 전공자가 임상적으로 의미있는 임계값과 지표 조합을 정확히 선택해 평가 리포트를 작성하면, 비전공자 창업자보다 빅파마와의 기술 실사(due diligence)에서 신뢰를 빠르게 얻습니다. 시장·BM 차원의 분석은 Phase 5에서 다룹니다.

## 오늘의 과제

1. **TDC 리더보드의 지표 다중 분석 (50분)**: Therapeutic Data Commons(tdcommons.ai)의 ADMET 그룹에서 데이터셋 2개(예: hERG_Karim, Caco2_Wang)를 선택해, 상위 5개 모델의 **AUROC, AUPRC, MAE, Spearman** 점수를 표로 정리합니다. 같은 모델이 두 데이터셋에서 순위가 어떻게 바뀌는지, 그리고 약학적으로 어느 지표가 임상 의사결정과 더 직결되는지 한 단락으로 분석합니다. A4 1쪽.
2. **가상 스크리닝 EF 시나리오 분석 (40분)**: 100만 개 라이브러리에서 양성률이 0.5%, 1%, 3%인 세 시나리오에서 AUROC = 0.85인 모델이 상위 1%를 뽑았을 때의 예상 정밀도와 EF@1%를 추정합니다. 양성률에 따라 같은 AUROC라도 실전 효용이 어떻게 달라지는지 표로 정리하고, "AUROC만으로 모델을 선택하면 안 되는 이유"를 3줄로 요약합니다.
3. **약학적 평가 정책 초안 (30분)**: 본인이 AI 신약개발 회사의 사내 모델 평가 책임자라고 가정하고, ADMET·활성 예측·독성 예측 각 카테고리에 대해 **(a) 1순위 지표, (b) 보고 시 함께 표시할 보조 지표, (c) 임상적으로 의미있는 임계값 또는 농도 범위, (d) 보정(calibration) 점검 여부**를 표 형태로 정의합니다. 1쪽 분량. Day 2의 전처리 정책 문서와 함께 묶으면 사내 ML 정책의 핵심이 됩니다.

## 참고 자료

- Saito, T. & Rehmsmeier, M. (2015). "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets." *PLOS ONE*, 10(3), e0118432. — AUPRC가 불균형 데이터에서 AUROC보다 우월함을 보인 표준 인용 논문.
- Truchon, J.-F. & Bayly, C. I. (2007). "Evaluating Virtual Screening Methods: Good and Bad Metrics for the 'Early Recognition' Problem." *Journal of Chemical Information and Modeling*, 47(2), 488–508. — BEDROC와 RIE 지표의 정의 논문. 가상 스크리닝 평가의 고전.
- van Tilborg, D., Alenicheva, A., Grisoni, F. (2022). "Exposing the Limitations of Molecular Machine Learning with Activity Cliffs." *Journal of Chemical Information and Modeling*, 62(23), 5938–5951. — Activity cliff 벤치마크(MoleculeACE) 제안.
- Huang, K. *et al.* (2021). "Therapeutic Data Commons." *NeurIPS Datasets Track*. — 다중 지표 동시 평가의 산업 표준 시도. (Day 4에서 더 자세히)
- scikit-learn metrics 문서(scikit-learn.org/stable/modules/model_evaluation.html) · RDKit의 화학 정보 평가 유틸리티 · TDC 평가 도구 — 본인 데이터에 직접 다중 지표를 계산하며 익히는 것이 가장 효과적.
