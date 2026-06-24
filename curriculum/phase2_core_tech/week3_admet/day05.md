# Day 5: Week 3 종합 — 다중 파라미터 최적화(MPO)와 약학적 판단

> 이전 학습(Phase 2 Week 3 Day 4)에서 hERG·DILI·Ames·Tox21을 다루는 독성 cascade 예측과 ICH M7·E14·S7B 규제 외삽 레이어를 다뤘습니다. 오늘은 Week 3의 종합으로 — Day 1~4에서 학습한 ADMET 5축(A·D·M·E·T)의 개별 예측을 **하나의 의사결정 점수로 통합하는 다중 파라미터 최적화(Multi-Parameter Optimization, MPO)** 원리와, AI 점수만으로 답할 수 없는 영역을 메우는 **약학 전공자의 판단 레이어**를 정리합니다. 동시에 다음 Week(Phase 2 Week 4 — 단백질 구조 기반 설계, AlphaFold)와의 연결점을 예고합니다.

## 개요

ADMET 5축을 개별적으로 예측하는 것까지는 Day 1~4에서 다뤘습니다. 그러나 **실제 발견 단계의 결정은 항상 다축 동시 평가**입니다 — 용해도는 좋지만 hERG가 위험한 분자, BBB 투과는 우수하지만 CYP3A4 억제가 강한 분자처럼 — 단일 축에서의 우열이 전체 후보의 가치를 결정하지 못합니다. **MPO 점수**는 ADMET·물성·활성을 가중 결합해 단일 의사결정 지표로 환산하는 방법이며, Pfizer의 CNS MPO(Wager 2010)·GSK의 Property Forecast Index(PFI) 등 산업 표준이 존재합니다. 약학 전공자의 우위는 — (a) 적응증·투여 경로에 맞는 가중치 설계, (b) AI가 출력하지 못하는 임상·규제·환자 맥락의 결합, (c) MPO 점수의 한계를 인지하고 도메인 판단으로 보정하는 영역에서 발현됩니다. 본 Day는 MPO의 수학적 원리, 산업 표준 사례, 약학 전공자의 차별 레이어, Week 3 종합 정리, Week 4 예고로 구성됩니다.

## 핵심 개념

### 1) MPO의 정의와 수학적 원리

**다중 파라미터 최적화(MPO)**는 — 여러 ADMET·활성·물성 파라미터를 단일 desirability 점수(0~1)로 결합하는 의사결정 방법론입니다. 핵심 아이디어는 — 각 파라미터를 **desirability function**으로 변환한 뒤, 가중 평균(또는 곱)을 취하는 것입니다.

**Derringer-Suich desirability function**(1980, 산업 표준)은 — 파라미터 값을 0~1 사이로 매핑합니다:

| 형태 | 정의 | 예시 |
|------|------|------|
| smaller-the-better | 값이 작을수록 1에 근접 | hERG IC50의 *역수*, 분자량 |
| larger-the-better | 값이 클수록 1에 근접 | 용해도, BBB 투과 |
| target value | 특정 값(범위)에서 1, 멀어지면 0 | logP 1~3 |

**최종 MPO 점수 — 두 가지 결합 방식**:

```
가중 산술평균: D = Σ(wᵢ × dᵢ)         (한 축이 0이어도 다른 축으로 보상)
가중 기하평균: D = Π(dᵢ^wᵢ)            (한 축이 0이면 전체 0 — "약점 거부")
```

> **약학적 직관 — 신약 발견에서는 기하평균이 더 타당**합니다. hERG IC50 < 1 μM 분자는 — 다른 축이 아무리 우수해도 후보 자격을 잃으며, 산술평균은 이를 다른 축의 우수성으로 보상해 잘못된 신호를 줍니다.

### 2) 산업 표준 MPO — Pfizer CNS MPO와 그 변형

**Pfizer CNS MPO (Wager 2010, *ACS Chem. Neurosci.*)**는 — 중추신경계(CNS) 약물의 BBB 투과·뇌 노출을 예측하기 위해 — 6개 물성을 결합한 산업 표준입니다.

| 파라미터 | 이상 범위 | 0이 되는 값 |
|---------|--------|----------|
| **ClogP** | ≤ 3 | ≥ 5 |
| **ClogD (pH 7.4)** | ≤ 2 | ≥ 4 |
| **MW** | ≤ 360 | ≥ 500 |
| **TPSA (topological polar surface area)** | 40~90 | < 20 또는 > 120 |
| **HBD (수소결합 공여체)** | ≤ 0.5 | ≥ 3.5 |
| **pKa (가장 염기성)** | ≤ 8 | ≥ 10 |

각 파라미터는 — 0~1 사이 piecewise linear desirability로 변환되며, 단순 **합산(0~6)**이 최종 점수입니다. CNS MPO ≥ 4가 — Pfizer 사내 표준 임계값이며, FDA 승인 CNS 약물의 약 74%가 이 임계값을 만족한다는 검증 데이터가 있습니다.

**GSK Property Forecast Index (PFI) — 단순화된 산업 변형**:

```
PFI = ClogD(pH 7.4) + (방향족 고리 개수)
PFI ≤ 6이 발견 단계의 권고 임계값
```

PFI는 — 발견 단계에서 **저용해도·단백질 결합 과다·CYP 비특이적 결합**의 위험을 한 줄로 가늠하는 간이 지표이며, GSK 사내 약 4만 개 화합물 분석에서 — PFI > 7 분자의 후속 ADMET 실패율이 약 3배 높았다는 결과가 보고되었습니다(Young 2011, *Drug Discov. Today*).

**약학적 차별** — Pfizer/GSK MPO는 **단일 적응증 가정**(CNS 또는 일반 경구)에 최적화되어 있습니다. **흡입제(폐 표적)·국소제(피부)·경구 BCS Class II/IV 약물**에서는 동일 MPO가 부적합하며 — 약학 전공자는 (a) 투여 경로별 desirability range를 재정의하고, (b) 환자군 특성(소아·고령·간장애)을 가중치에 반영할 수 있습니다.

### 3) ADMET Risk Score — Simulations Plus의 통합 점수

**ADMET Predictor (Simulations Plus, 1996~, v11 2024)**의 ADMET Risk Score는 — 약 40개 endpoint를 24개 룰로 통합해 **0~24 위험 점수**로 환산합니다. 룰의 예:

| 룰 | 트리거 조건 | 가중치 |
|----|---------|------|
| Tox_Risk | hepatotoxicity 확률 > 0.5 | 1 |
| Absn_Risk | 인간 fa(분율 흡수) < 0.5 | 1 |
| CYP_Risk | CYP3A4 억제 IC50 < 10 μM | 1 |
| hERG_Risk | hERG IC50 < 10 μM | 2 |

산업 표준 임계값은 — **ADMET Risk ≤ 7이 발견 단계 후보 진입 기준**입니다. 약 30만 분자에 대한 사내 검증에서 — ADMET Risk ≤ 7 분자의 임상 1상 진입률이 약 2.5배 높았다는 백서가 공개되어 있습니다.

**약학 전공자의 차별** — 가중치 2가 부여된 hERG_Risk·DILI_Risk는 — 임상 회수 통계(Day 4)와 일치하지만, **희귀질환·항암제·생명위협 적응증**에서는 위험-이익 균형이 다릅니다(예: kinase inhibitor의 hERG 허용도). 약학 전공자는 — 적응증·환자 위중도에 따라 ADMET Risk 임계값과 룰 가중치를 재조정할 수 있습니다.

### 4) Pareto 최적화 — MPO의 다목적 확장

단일 desirability 점수로 환원할 수 없는 경우 — **Pareto 최적화**가 표준입니다. Pareto front는 — *다른 목표를 희생하지 않고서는 어느 한 목표를 더 개선할 수 없는 분자들의 집합*입니다.

**약학적 예시 — BBB 투과 vs hERG**:

```
A 분자: BBB 0.8, hERG IC50 30 μM     [Pareto front 멤버]
B 분자: BBB 0.9, hERG IC50 15 μM     [Pareto front 멤버]
C 분자: BBB 0.6, hERG IC50 5 μM      [지배됨 — A에 의해]
```

Pareto front에서의 최종 선택은 — *적응증·투여 경로·시장 포지셔닝*에 따른 인간의 판단입니다. AI는 front를 그리지만, **front 위 어느 점을 선택할지는 약학 전공자의 판단**입니다.

### 5) 약학 전공자의 판단 레이어 — AI MPO가 메우지 못하는 영역

MPO 점수가 동등한 두 분자에서 — 다음 영역은 약학 지식 없이는 결정 불가합니다:

| 영역 | AI MPO가 출력하지 못하는 것 | 약학적 판단 |
|------|----------------|----------|
| **환자군 특성** | 소아·고령·간장애·임신 | 용량 조절, 금기 |
| **약물상호작용 맥락** | 동반 약물 종류·복약 순응도 | CYP 억제 위험 평가 |
| **약물경제학** | COGS, 보험 적용 가능성 | scaffold 합성 난이도 정성 평가 |
| **시판 후 모니터링** | 희귀 부작용 신호, REMS 필요성 | 위험관리 계획 사전 설계 |
| **규제 전략** | breakthrough designation, fast track | 적응증 선택과 임상 디자인 |

> **결론** — MPO는 **scaffold 우선순위와 합성 후보 선별**까지가 강점이며, 임상 디자인·시장 전략·환자 맞춤 결정은 — 약학 전공자의 판단이 보완하는 영역입니다. 이 분업의 인식 자체가 — 약학 전공 창업자의 핵심 자산입니다.

## 작동 원리와 아키텍처

Week 3 통합 ADMET MPO 시스템의 표준 구조:

```
[Week 3 통합 — ADMET MPO 의사결정 시스템]

1. 입력 — Day 1 ADMET 개요
   - SMILES 표준화
   - 적응증·투여 경로·환자군 메타데이터

2. 5축 예측 — Day 1~4
   A) Absorption: solubility, Caco-2, fa, HIA
   D) Distribution: BBB, PPB(fu), Vd
   M) Metabolism: CYP 억제·기질, CL
   E) Excretion: t1/2, 신·간 배설 비율
   T) Toxicity: hERG, DILI, Ames, Tox21

3. desirability 변환 — 본 Day
   - 각 축 출력 → 0~1 desirability
   - 투여 경로·적응증별 desirability range 동적 설정

4. MPO 결합 — 본 Day
   - 가중 기하평균(약점 거부) + Pareto front 추출
   - 산업 표준 점수 병기: Pfizer CNS MPO, GSK PFI, ADMET Risk

5. 약학·규제 후처리 — 본 Day의 차별
   - 환자군 보정 (소아·고령·간장애)
   - 약물상호작용 시뮬레이션 (Day 3 CYP 결과 활용)
   - 회수 약물 유사도 경고 (Day 4 DILI/hERG 회수 분자와의 fingerprint 유사도)
   - 규제 외삽 (ICH M7/E14/S7B 위치)
```

핵심 설계 결정 — MPO 구축에서의 차별 지점:

| 결정 | 표준 접근 | 약학·규제 차별 |
|------|--------|--------|
| 결합 방식 | 산술 또는 기하 | 기하(약점 거부) + Pareto 병기 |
| 가중치 | 고정 (Wager 2010 등) | 적응증·투여 경로별 동적 |
| 임계값 | 단일 cut-off | 환자 위중도·시장 포지셔닝 반영 |
| 출력 | 점수만 | 점수 + 회수 약물 유사도 + 규제 권고 |
| 사용자 | 메디시널 케미스트 | 메디시널 케미스트 + 임상약리학자 |

## 신약개발 적용

MPO 기반 ADMET 의사결정은 — 실제 산업에서 분자 후보 우선순위를 결정합니다. 대표 사례:

| 사례·도구 | 적용 영역 | 핵심 결과 |
|---------|--------|--------|
| **Pfizer CNS MPO (Wager 2010, *ACS Chem. Neurosci.*)** | CNS 분자 BBB 투과 우선순위 | 승인 CNS 약물의 약 74%가 CNS MPO ≥ 4 |
| **GSK PFI (Young 2011, *Drug Discov. Today*)** | 발견 단계 ADMET 실패 사전 신호 | PFI > 7의 후속 실패율 약 3배 |
| **Simulations Plus ADMET Risk** | 사내 30만 분자 검증 | Risk ≤ 7의 1상 진입률 약 2.5배 |
| **Insilico Medicine MPO + RL (2023 *Nat. Biotechnol.*)** | INS018_055 IPF 후보 발굴 | 다축 ADMET MPO 강화학습 보상으로 통합 |
| **Schrödinger LiveDesign MPO 워크플로우** | 빅파마 메디시널 케미스트 표준 | 다축 desirability + Pareto 시각화 |
| **ADMET-AI (Swanson 2024, *Bioinformatics*)** | 41개 task 통합 무료 도구 | 사용자 정의 desirability 입력 지원 |

**대표 정량 비교 — 단축 vs MPO 의사결정**:

| 항목 | 단일 축 의사결정 | MPO 통합 의사결정 |
|------|------------|--------------|
| 발견 단계 후보 선별 정확도 | 약 30~40% | 약 60~75% |
| 후기 단계(전임상) 실패율 | 약 40~50% | 약 20~30% |
| 의사결정 분자 수 | 분자당 5~10개 축 별도 검토 | 단일 점수 + 회피 신호 |
| 부서간 소통 | 케미스트·약리학자 분리 협의 | 단일 대시보드 통합 |
| 적응증 맞춤 | 어려움 | desirability 가중치 동적 조정 |

**산업적 통찰** — Sun et al. 2022(*Acta Pharm. Sin. B*) 분석은 — 임상 실패의 약 90%가 **개별 ADMET 축이 아니라 다축 trade-off의 실패**에서 비롯한다고 보고합니다. 즉 hERG만 안전하거나, BBB만 우수한 것은 임상 성공을 보장하지 않으며, **다축 균형이 잡힌 분자가 임상에 도달**합니다. 이는 MPO의 산업적 가치가 — 단순 점수화가 아니라 *trade-off를 명시적으로 시각화*하는 데 있음을 시사합니다.

## 창업 관점

ADMET MPO SaaS는 — Week 3의 4일 학습을 단일 제품으로 통합하는 자연스러운 결과물입니다. 시장 위치는 — Schrödinger LiveDesign·Simulations Plus ADMET Predictor가 연 5만~50만 달러로 빅파마 표준이지만, **적응증·환자군·규제 맞춤 MPO**는 약한 영역입니다. **약사가 만든 MPO** 포지셔닝의 차별 — (a) 투여 경로별 desirability range 프리셋(경구·정맥·흡입·국소·BBB), (b) 환자군 보정(소아·고령·간장애·임신), (c) 회수 약물 유사도 경고(Day 4 데이터), (d) ICH M7/E14 자동 권고문(Day 4)이 결합되면 — 중소 바이오·임상약리학 컨설팅·CRO 시장(월 2,000~10,000 달러)에서 차별이 가능합니다. **핵심 해자**는 — 비약학 경쟁자가 6~12개월 학습으로 따라잡기 어려운 *적응증·규제 맥락 가중치 라이브러리*입니다. 단, **AI MPO 점수를 임상 결정에 직접 사용하지 말 것**이라는 명시적 면책 조항과, 전문가 검토 워크플로우(expert-in-the-loop)가 필수입니다. **Week 3 학습의 결과물** — 본인이 1인 창업자로서 (a) ADMET-AI 또는 Chemprop 베이스라인을 1주, (b) 약학·규제 후처리 레이어를 2~4주, (c) 적응증별 desirability 프리셋을 2~4주에 구축한다면 — **2~3개월 만에 차별화된 MVP**를 만들 수 있다는 것이 본 Week의 실천적 결론입니다.

## Week 3 종합 및 Week 4 예고

**Week 3 학습 정리** — 4일에 걸쳐 ADMET 5축의 AI 예측을 다뤘습니다.

| Day | 주제 | 약학 전공자의 차별 |
|-----|------|--------|
| Day 1 | ADMET 개요·임상 실패 60% | 임상 회수 메커니즘 해석 |
| Day 2 | 흡수·분포(용해도·BBB·PPB) | 자유 분율 계산, 투여 경로 매핑 |
| Day 3 | 대사·배설(CYP·CL) | PK 1구획 자동 구성, DDI 평가 |
| Day 4 | 독성(hERG·DILI·Ames·Tox21) | ICH M7/E14/S7B 규제 외삽 |
| **Day 5** | **MPO·약학적 판단 통합** | **적응증 맞춤 desirability + 환자군 보정** |

**다음 Week 예고 — Phase 2 Week 4: 단백질과 구조 기반 설계** — Week 3까지는 *분자(ligand) 중심*의 ADMET·물성 예측이었습니다. Week 4부터는 — **단백질 표적(target) 측면**으로 시각이 확장됩니다. AlphaFold가 바꾼 단백질 구조 예측의 세계, 단백질-리간드 도킹, AI 도킹(DiffDock), 단백질 언어 모델(ESM)을 다룹니다. ADMET 예측이 *"이 분자는 약이 될 수 있는가"*에 답한다면, 구조 기반 설계는 *"이 분자가 표적에 정말로 결합하는가"*에 답합니다. **두 축이 결합되면 — 발견 단계의 핵심 의사결정이 완성**됩니다.

## 오늘의 과제

1. **Pfizer CNS MPO vs PFI 직접 계산 (40분)**: 본인 관심 적응증의 표준 약물 3개와 비교 약물 2개(총 5개)를 선택하세요. 각 약물에 대해 — (a) PubChem 또는 DrugBank에서 SMILES·MW·ClogP·HBD 등 6개 입력값을 수집, (b) Pfizer CNS MPO 점수와 GSK PFI를 직접 계산, (c) 실제 BBB 투과(brain/plasma 비율 공개 자료)와 MPO 점수의 일치도를 표로 정리하세요. 결론으로 — 본인이 만들 MPO에서 어떤 desirability range를 적응증별로 다르게 설정할지 가설 3개를 명시하세요.

2. **MPO + Pareto 시각화 도구 비교 (40분)**: ADMET-AI(admet.ai.greenstonebio.com), SwissADME(swissadme.ch), pkCSM(biosig.lab.uq.edu.au/pkcsm) — 3개 무료 도구에 동일 분자 3개를 입력해 ADMET 출력을 받으세요. (a) 각 도구가 desirability·MPO·Pareto 시각화를 제공하는지, (b) 어떤 도구가 다축 trade-off를 가장 잘 보여주는지, (c) 본인 SaaS가 추가해야 할 시각화 요소(예: 회수 약물과의 fingerprint 유사도 heatmap)를 1페이지 비교 노트로 정리하세요.

3. **적응증 맞춤 MPO 프리셋 설계 (40분)**: 본인 관심 적응증 1개를 선택해 — (a) 투여 경로(경구·정맥·흡입·국소·CNS), (b) 환자군(성인·소아·고령·임신·간장애), (c) 위중도(생명위협·만성·증상완화)에 따른 desirability range 표를 작성하세요. 각 셀에 — logP 이상범위, MW 상한, hERG safety margin 임계값, DILI 허용도 등 5~7개 파라미터의 구체적 수치를 명시하세요. 이 프리셋 라이브러리가 — 본인 MPO SaaS의 핵심 차별 자산입니다.

## 참고 자료

- **Pfizer CNS MPO**: Wager, T.T. et al. "Moving beyond rules: the development of a central nervous system multiparameter optimization (CNS MPO) approach." *ACS Chem. Neurosci.*, 2010. — CNS MPO의 표준 정의와 6개 파라미터 desirability 함수.
- **GSK PFI**: Young, R.J. et al. "Getting physical in drug discovery II: the impact of chromatographic hydrophobicity measurements and aromaticity." *Drug Discov. Today*, 2011. — PFI의 산업적 검증과 ADMET 실패 예측.
- **Pareto 최적화 in drug discovery**: Nicolaou, C.A., Brown, N. "Multi-objective optimization methods in drug design." *Drug Discov. Today: Technol.*, 2013. — MPO와 Pareto의 신약개발 적용 표준 정리.
- **임상 실패 분석**: Sun, D. et al. "Why 90% of clinical drug development fails and how to improve it?" *Acta Pharm. Sin. B*, 2022. — 다축 trade-off 실패가 임상 실패의 핵심임을 보고.
- **ADMET-AI**: Swanson, K. et al. "ADMET-AI: a machine learning ADMET platform for evaluation of large-scale chemical libraries." *Bioinformatics*, 2024. — 41 task 통합 무료 도구, MPO desirability 입력 지원.
- **산업 표준 도구**: Schrödinger LiveDesign, Simulations Plus ADMET Predictor v11, ChemAxon Calculator — 빅파마 MPO 워크플로우 표준.
- **약학 통합 관점**: Di, L., Kerns, E.H. *Drug-Like Properties: Concepts, Structure Design and Methods*, 2nd ed., Academic Press, 2015 — ADMET 다축 최적화의 약학적 기반 교과서.
- **무료 도구 비교**: ADMET-AI, SwissADME(2017), pkCSM(2015), ProTox-3.0(2024) — 1인 창업자가 즉시 활용 가능한 오픈 스택.
