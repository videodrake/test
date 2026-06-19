# Day 1: ADMET 개요 — 임상 실패의 60%를 예방하는 기술

> 이전 학습(Phase 2 Week 2)에서 De Novo 분자 생성 패러다임과 통합 의사결정 트리를 다뤘습니다. 오늘은 이를 바탕으로, 생성된 후보 분자에 적용할 **ADMET 게이트의 약학적 의미와 AI 기반 예측 체계** 전반을 학습합니다. Week 3 전체는 — Week 2에서 만든 cascade 파이프라인의 각 단계에 약학적 임계값을 어떻게 코드화할지를 다룹니다.

## 개요

**ADMET**(Absorption, Distribution, Metabolism, Excretion, Toxicity)은 약물 후보의 흡수·분포·대사·배설·독성을 정량적으로 평가하는 약학의 핵심 분과이며, AI 신약개발에서는 가장 약학 전공자의 우위가 직접 발현되는 영역입니다. 임상 1상에 진입한 후보 중 약 90%가 시판에 도달하지 못하며, 이 실패의 약 30~40%가 안전성(독성), 약 20~30%가 약물동태(PK) 문제 — 즉 ADMET 관련 실패가 전체의 약 50~60%를 차지한다는 분석이 반복적으로 보고됩니다(Hay et al. 2014, Sun et al. 2022 등). 따라서 ADMET 예측은 — 임상 실패 비용(평균 후보당 약 1억~10억 달러)을 발견 단계에서 절감하는 가장 효율적 지렛대입니다. 본 Day는 ADMET의 5개 축, 임상 실패 통계의 정확한 해석, AI 예측 모델의 표준 아키텍처, 그리고 2024년 산업 표준 도구를 정리합니다.

## 핵심 개념

### 1) ADMET 5축 — 약학적 정의와 임상적 의미

ADMET 각 축의 정의와 임상 단계에서의 핵심 판단 지표를 정리합니다.

| 축 | 정의 | 핵심 측정 지표 | 임상 실패 시 결과 |
|----|------|------------|--------------|
| **A — 흡수(Absorption)** | 투여 부위에서 전신 순환으로 진입하는 효율 | 용해도(solubility), 투과성(Caco-2, PAMPA), 생체이용률(F%) | 경구 투여 불가, 정맥주사로 한정 |
| **D — 분포(Distribution)** | 혈장-조직 간 약물 이동과 분포 양상 | 분포용적(Vd), 단백질 결합률(PPB), 뇌혈관장벽 통과(BBB) | 표적 조직 도달 실패, 중추 부작용 |
| **M — 대사(Metabolism)** | 주로 간에서의 효소적 변환 | CYP450 억제/유도, 대사 안정성(t1/2), 활성 대사체 | 약물 상호작용, 노출 변동 |
| **E — 배설(Excretion)** | 신장·담즙을 통한 체외 배출 | 청소율(CL), 신장 청소율(CLr), 반감기 | 축적 독성, 노인·신부전 환자 위험 |
| **T — 독성(Toxicity)** | 표적외 작용으로 인한 유해 반응 | hERG 차단, 간독성(DILI), 변이원성(Ames), 심독성, 면역원성 | 시판 후 회수(예: Vioxx, Rezulin) |

약학 전공자의 우위는 — 각 축의 측정값이 **어떤 임상 단계에서 go/no-go**를 결정하는지 정확히 안다는 점입니다. 예를 들어 hERG IC50 > 10 μM은 일반적 hard cutoff이지만, 항부정맥제처럼 표적이 심장 채널 자체인 경우엔 정반대 기준이 적용됩니다. 비약학 전공자가 일률적 임계값을 적용하는 동안, 약학 전공자는 적응증·표적·투여 경로에 따라 임계값을 조정합니다.

### 2) "임상 실패의 60%" — 통계의 정확한 해석

이 숫자는 종종 인용되지만 — 근거 데이터를 정확히 이해해야 본인의 모델 설계에 반영할 수 있습니다.

| 데이터 출처 | 분석 시기 | 핵심 결론 |
|----------|---------|---------|
| Kola & Landis (Nature Rev Drug Discov, 2004) | 1990년대 | PK/생체이용률 문제가 임상 실패의 약 39% (PK가 압도적 1위) |
| Waring et al. (Nature Rev Drug Discov, 2015) | AstraZeneca 2005~2010 | PK는 5% 미만으로 감소(전임상 ADMET 강화), 안전성·효능 문제가 주된 사유 |
| Hay et al. (Nature Biotechnol, 2014) | 2003~2011 | Phase II 실패 51%가 효능, 19%가 안전성 |
| Sun et al. (Acta Pharm Sin B, 2022) | 2010~2017 | 임상 실패의 약 30%가 독성/안전성, 약 10~15%가 PK 관련 |

**핵심 해석** — "60%"는 (a) PK 문제(흡수·분포·배설)와 (b) 독성(T)을 합산했을 때의 역사적 추정이며, 2010년대 이후 전임상 ADMET이 발전하면서 비중이 약 30~50%로 감소했습니다. 즉, AI ADMET 예측의 가치는 "60%를 막는 것"이 아니라 — **이미 줄어든 비중을 더 줄이는 한계 비용 절감**과 **희귀한 독성(특이 약물반응, idiosyncratic DILI 등)을 조기 식별**하는 데 있습니다. 비약학 전공자는 옛 통계를 그대로 가져다 쓰지만, 약학 전공자는 — 최신 분석에서 어떤 ADMET 축이 여전히 가장 큰 실패 사유인지 정확히 짚을 수 있습니다.

### 3) AI 기반 ADMET 예측의 데이터 지형

ADMET AI 모델은 — Week 1(QSAR)에서 학습한 GNN·Transformer·RF/XGBoost가 그대로 적용되지만, **데이터셋 특성이 매우 달라** 별도 설계가 필요합니다.

| 특성 | QSAR(활성 예측) | ADMET 예측 |
|------|------------|---------|
| 라벨 데이터 규모 | 표적당 수천~수십만 | 평가항목당 수백~수천 (소량) |
| 데이터 일관성 | 표적 정의 명확 | 실험 프로토콜·종(species)·세포주 다양 |
| 표준 벤치마크 | MoleculeNet, OGB | **Therapeutic Data Commons(TDC) ADMET Benchmark** (22개 task) |
| 분포 편향 | 표적별 chemotype 집중 | drug-like 화합물 강한 편향 |
| 임상 외삽 가능성 | in vitro→in vivo 격차 큼 | 일부는 직접 in vivo 라벨(예: 클리어런스) |

이 차이가 — AI ADMET 모델 설계의 3대 도전 과제로 이어집니다: (a) 소량 데이터에서의 일반화, (b) 분포 외(out-of-distribution) 신규 chemotype 예측, (c) 실험 노이즈와 종간 변환의 보정.

## 작동 원리와 아키텍처

산업 표준 ADMET AI 시스템의 구조 — 2024년 기준:

```
[ADMET 예측 통합 시스템]

1. 입력 표현
   - SMILES (canonical) + RDKit 표준화(salt 제거, tautomer)
   - 분자 그래프 + 3D conformer (필요 시 ETKDG)

2. 표현 학습 (encoder)
   - GNN (D-MPNN: Chemprop) — 분자 그래프 직접 학습
   - 사전학습 Transformer (ChemBERTa, MolFormer-XL) — 22B SMILES 사전학습
   - 멀티태스크 헤드 — 22개 ADMET task 동시 학습으로 데이터 증강

3. 예측 헤드 (task별)
   - 분류: hERG inhibition, Ames mutagenicity, DILI
   - 회귀: solubility (logS), permeability (logPapp), clearance, t1/2
   - 다중 라벨: CYP450 isoform 5종 (1A2, 2C9, 2C19, 2D6, 3A4)

4. 불확실성 정량화 (필수 — 임상 의사결정용)
   - MC Dropout, Deep Ensemble로 95% 신뢰구간
   - Tanimoto 기반 도메인 적용성(applicability domain) 점수

5. 약학적 해석 레이어
   - SHAP / Integrated Gradients로 기여 부분구조 추출
   - 임상 cutoff 적용 (hERG > 10μM, DILI 확률 < 0.5 등)
   - 적응증·투여경로별 가중치 조정 — 약학 전공자의 차별 영역

6. 출력 — Multi-Parameter Optimization(MPO) 점수
   - Optibrium StarDrop 방식의 가중 desirability score
   - cascade filter 결과(pass/fail + score)
```

핵심 설계 결정 — 약학 전공자가 차별화할 수 있는 지점:

| 결정 | 일반 접근 | 약학적 차별화 |
|------|--------|-----------|
| Task 우선순위 | 22개 균등 | 적응증·표적 기반 가중치 |
| Cutoff 임계값 | 문헌 기본값 | 투여 경로(경구/주사)·환자군별 조정 |
| 종간 변환 | 무시 | 인간 간세포·iPSC 데이터 우선 가중 |
| 불확실성 처리 | 점수만 출력 | 신뢰구간 + 임상 의사결정 권고 |

이 설계 결정이 — Day 5(MPO와 약학적 판단)에서 종합되는 핵심 의사결정 프레임으로 이어집니다.

## 신약개발 적용

ADMET AI는 — Phase 2 Week 1의 활성 예측보다 산업 도입이 더 광범위하게 정착된 영역입니다. 2024년 기준 주요 도구와 회사:

| 회사·도구 | 카테고리 | 차별 포인트 | 2024년 동향 |
|---------|--------|-------|----------|
| **Simulations Plus (GastroPlus, ADMET Predictor)** | 상용 SaaS | PBPK 시뮬레이션 통합, FDA 자주 참조 | 상장사, 연매출 약 6,000만 달러 |
| **Optibrium (StarDrop)** | 상용 데스크톱+클라우드 | MPO desirability 표준화 | 2023년 Inpharmatica·Cresset 통합 |
| **Schrödinger (QikProp, Maestro)** | 상용 통합 플랫폼 | 물리 기반 + ML 하이브리드 | 자체 파이프라인 약 20개 |
| **Chemaxon (cxcalc)** | 상용 SaaS | ChemAxon 화학구조 표준 | JChem과 통합 |
| **ADMET-AI (Pande Lab, Stanford)** | 오픈소스 | TDC 기반 22개 task 통합 모델 | 2023년 *Bioinformatics* 발표 |
| **DeepPurpose** | 오픈소스 라이브러리 | DTI·ADMET 통합 | 학계 표준 |
| **Chemprop (MIT)** | 오픈소스 | D-MPNN 사실상 표준 구현 | 2024년 v2 발표 |
| **Valence Labs (Recursion)** | 사내 + 협업 | MolPhenix·MolGPS 사전학습 | 2023년 Recursion 인수 |

대표적 성과 사례 — **Sun et al. (2022)** 메타분석에 따르면 AI ADMET 도구 도입 이후 발견 단계 후보의 평균 ADMET pass rate가 약 15%에서 약 35%로 향상되었으며, **전임상 단계 평균 비용이 약 30% 감소**(약 2,500만 달러 → 1,800만 달러)했다고 보고됩니다. **Insilico Medicine**의 INS018_055는 자체 Chemistry42 + ADMET 예측 모듈로 발견부터 임상 1상 진입까지 약 30개월에 도달했으며 — 전통적 평균 약 60개월의 절반입니다. **Recursion**의 LOWE 시스템은 ADMET 예측을 phenomic screening과 통합해 임상 후보 4개를 동시에 운영합니다.

기존 방식 대비 개선점 — 정량 비교:

| 항목 | 전통 in vitro 패널 | AI ADMET 예측 |
|------|--------------|-----------|
| 분자 1개당 평가 비용 | 약 5,000~50,000 달러 | 약 0.01~1 달러 (계산만) |
| 처리 속도 | 수일~수주 | 수초 (분자당) |
| 평가 가능 분자 수 | 수십~수백 | 수백만~수십억 |
| 결과 정확도 (실측 대비 R²) | 1.0 (정의상) | 0.5~0.8 (task별) |
| 임상 외삽 가능성 | 직접 측정 | 종간·in vitro-in vivo 격차 있음 |

핵심 통찰 — AI ADMET은 **in vitro 패널을 대체하지 않습니다.** 수십억 후보를 1차 필터링해 in vitro로 보낼 수십~수백 개를 선별하는 cascade의 첫 단계이며, 정확도와 처리량의 trade-off를 약학적으로 설계하는 것이 차별점입니다.

## 창업 관점

ADMET 예측은 — Phase 2의 다른 영역(QSAR, 분자 생성) 대비 약학 전공자의 진입 장벽이 가장 낮고 도메인 깊이의 가치가 가장 높은 영역입니다. 1인 약학 창업자의 합리적 진입 경로는 — TDC ADMET Benchmark의 22개 task에 대해 Chemprop v2 + ChemBERTa 사전학습을 활용한 wrapper SaaS로 시작하되, **적응증·투여경로·환자군별 cutoff 자동 조정**과 **약학적 의사결정 권고**를 차별 포인트로 가져가는 것입니다. 시장 규모는 ADMET 예측 소프트웨어 부문이 2024년 약 4억 달러로 추정되며(MarketsandMarkets 2024, 확인 필요), 연 12~15% 성장 중입니다. Simulations Plus·Schrödinger·Optibrium의 가격대(연 5만~50만 달러/사용자)는 빅파마용이며 — 중소 바이오·CRO 대상의 월 1,000~5,000 달러 SaaS 공백이 진입 지점입니다. 약학 전공자의 해자는 도메인 코드화(임상 cutoff, 종간 변환, 적응증별 가중치)이며 — Week 3 Day 2~5에서 다루는 각 ADMET 축의 약학적 판단이 이 해자의 구체적 내용이 됩니다.

## 오늘의 과제

1. **ADMET 5축의 적응증 특화 cutoff 매트릭스 작성 (40분)**: 본인이 Phase 2 Week 2에서 선택한 표적의 적응증(예: 항암, 항감염, CNS, 만성 대사질환 중 1개)에 대해 — ADMET 5축의 각 측정 지표별 hard cutoff와 soft penalty 임계값을 약학적 근거와 함께 표로 정리하세요. 예를 들어 CNS 적응증이면 BBB 통과는 hard requirement(logBB > -1), 항암제면 분포용적 큰 값이 종양 침투에 유리할 수 있음 등. 표 마지막에 — "비약학 전공자가 일반 임계값으로 만든 모델 대비 본인의 매트릭스가 어떤 후보를 추가로 필터링하거나 살릴 수 있는지" 3개 구체적 시나리오를 작성하세요.

2. **AI ADMET 도구 비교 분석 (50분)**: ADMET-AI(Stanford), Chemprop v2, Simulations Plus ADMET Predictor 3개를 — (a) 지원 task 수, (b) 학습 데이터 출처, (c) 불확실성 정량화 여부, (d) 임상 cutoff 내장 여부, (e) 사용 비용(오픈소스/상용), (f) API/SaaS 가능성의 6축으로 비교 분석하세요. 1인 약학 창업자가 wrapper SaaS의 백엔드로 어떤 조합을 선택할지 결정 근거 3가지와 함께 결론을 1문단으로 작성하세요.

3. **임상 실패 통계 검증 리서치 (30분)**: 본 Day에서 인용한 임상 실패 통계(Kola & Landis 2004, Waring et al. 2015, Hay et al. 2014, Sun et al. 2022) 중 2개 이상의 원문 abstract를 찾아 — (a) 분석한 회사·시기·분자 수, (b) ADMET이 차지하는 비중에 대한 정확한 표현, (c) 본 Day에서의 인용이 정확한지 확인하세요. 차이점이 있다면 — 본인의 향후 콘텐츠/사업계획서에서 어떻게 정확하게 인용할지 1문단으로 정리하세요.

## 참고 자료

- **벤치마크**: Huang, K. et al. "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development." *NeurIPS 2021 Datasets and Benchmarks*. — TDC ADMET 22-task의 표준 벤치마크.
- **종합 분석**: Sun, D. et al. "Why 90% of clinical drug development fails and how to improve it?" *Acta Pharmaceutica Sinica B*, 2022. — 최신 임상 실패 분석.
- **고전 분석**: Kola, I., Landis, J. "Can the pharmaceutical industry reduce attrition rates?" *Nature Reviews Drug Discovery*, 2004. — "ADMET이 임상 실패의 주범" 통계의 원전.
- **AI ADMET 모델**: Swanson, K. et al. "ADMET-AI: a machine learning ADMET platform for evaluation of large-scale chemical libraries." *Bioinformatics*, 2023. — Stanford Pande Lab 오픈소스.
- **오픈소스**: Chemprop v2 (chemprop/chemprop), TDC (mims-harvard/TDC), ADMET-AI (swansonk14/admet_ai). — 1인 창업자가 즉시 활용 가능한 표준 스택.
- **시장 보고서**: MarketsandMarkets, "ADMET Testing Market", 2024. — ADMET 소프트웨어 시장 규모(확인 필요).
