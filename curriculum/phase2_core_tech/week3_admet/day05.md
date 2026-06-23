# Day 5: Week 3 종합 — 다중 파라미터 최적화(MPO)와 약학적 판단

> 이전 학습(Phase 2 Week 3 Day 1~4)에서 ADMET 5축의 개요, A·D(흡수·분포), M·E(대사·배설), T(독성)의 표준 AI 예측 모델과 데이터셋, 약학·규제 후처리를 다뤘습니다. 오늘은 — 흩어진 ADMET 예측 출력을 **하나의 의사결정으로 통합**하는 **다중 파라미터 최적화(Multi-Parameter Optimization, MPO)**의 프레임워크와, 비약학 전공자가 모방하기 어려운 **약학적 판단 레이어**의 설계를 정리합니다. Week 3의 결산이자 — 다음 Week에서 다룰 **단백질·구조 기반 설계**와 ADMET 게이트가 어떻게 결합되는지의 연결고리입니다.

## 개요

발견 단계 후보 분자는 — 단일 속성 하나만 좋아서는 임상에 진입할 수 없습니다. 활성(potency)이 100배 강해도 hERG IC50가 100 nM이면 회수 위험이 압도적이고, ADMET 5축 모두가 평균적으로 우수해도 — BBB 통과가 필요한 CNS 약물에서 BBB 음성이면 의미가 없습니다. **MPO**는 — 활성, A·D·M·E·T 각 축의 예측 출력, 합성 가능성(SAscore), 신규성(IP)을 — 임상 단계 기준의 가중치로 통합해 단일 desirability 점수 또는 Pareto frontier로 변환하는 약학·통계 프레임워크입니다. Pfizer의 MPO score(2012), Astex의 LLE(lipophilic ligand efficiency), Roche의 CNS MPO(Wager 2010), GSK의 PFI(Property Forecast Index) 등 — 빅파마는 각자 사내 MPO 표준을 운영하고 있으며, 이는 **약학·임상 지식이 코드로 자산화된 가장 직접적 사례**입니다. 본 Day는 (a) MPO의 수학적 형식, (b) 약학적 desirability function 설계, (c) Pareto frontier 분석, (d) 약학 전공자만이 정렬할 수 있는 적응증·표적별 판단 레이어, (e) Week 4 단백질·구조 기반 설계와의 연결을 정리합니다.

## 핵심 개념

### 1) MPO의 수학적 형식 — desirability function

MPO의 핵심은 — 각 속성 x_i를 **desirability function** d_i(x_i) ∈ [0, 1]로 매핑한 뒤, 가중 평균 또는 기하 평균으로 통합하는 것입니다.

**Derringer-Suich desirability**(1980, *J. Quality Tech.*) — MPO의 표준 형식:

```
[가중 평균] D = Σ w_i · d_i(x_i)        — 한 속성이 0이어도 다른 속성으로 보상 가능
[기하 평균] D = Π d_i(x_i)^w_i          — 한 속성이 0이면 전체가 0 (hard cutoff 효과)
```

각 desirability function의 형태 — **단조 감소/증가/구간 최적**:

| 속성 유형 | 예시 | desirability 형태 |
|---------|------|---------------|
| 클수록 좋음 | 용해도, F% | sigmoid (낮을수록 0, 높을수록 1) |
| 작을수록 좋음 | hERG IC50의 *역수*, CYP IC50 | 역 sigmoid |
| 구간 최적 | logP (1~3 영역), MW (200~500) | bell curve / piecewise linear |
| 이진 분류 | Ames 음성, BBB(+/−) | step function 또는 확률 직접 사용 |

**약학적 핵심** — desirability function의 변곡점이 곧 **임상 단계 go/no-go 임계값**입니다. 일반 컴퓨터과학자가 [0, 1] 선형 정규화를 적용하는 동안, 약학 전공자는 — hERG IC50의 변곡점을 1 μM에 두고(임상 cardiotoxicity 임계값), F%의 변곡점을 30%에 두고(경구 약물의 산업 표준), logP의 최적 구간을 1~3에 두는(Lipinski + Veber + Pfizer MPO 합의) 식으로 임상·규제 지식을 함수 곡선에 직접 인코딩합니다.

### 2) 산업 표준 MPO 예시

대표 사내·공개 MPO 시스템들:

| MPO 시스템 | 속성 | 적용 영역 |
|--------|------|------|
| **Pfizer MPO (Wager 2010)** | clogP, clogD7.4, MW, TPSA, HBD, pKa | CNS 약물 BBB 통과 가능성 |
| **Astex LLE/LipE** | LipE = pIC50 − clogP | hit-to-lead 효율 |
| **GSK PFI** | chromlogD7.4 + 방향족 고리 수 | developability |
| **Roche CNS MPO** | 6개 속성의 desirability 통합 | 중추신경계 약물 후보 우선순위 |
| **QED (Bickerton 2012)** | 8개 약물유사성 desirability | drug-likeness 정량 |

**약학 전공자의 깊이** — 이들 MPO는 — 적응증과 투여 경로에 따라 적용되는 식이 다릅니다. CNS 약물에는 Pfizer/Roche CNS MPO, 경구 약물에는 Lipinski + Veber + PFI, 항생제에는 LipE + CNS 회피 등. 비약학 전공자가 QED 단일 값으로 모든 분자를 평가하는 동안, 약학 전공자는 — 적응증별 MPO 식을 분기 적용하고, 사내 데이터로 변곡점을 보정합니다.

### 3) Pareto frontier — 절충(trade-off)의 시각화

MPO 통합값 D 하나로 압축하면 — 분자 간 절충 관계가 사라집니다. **Pareto frontier**(다목적 최적화의 표준)는 — "어떤 다른 분자도 모든 축에서 동시에 우월하지 않은" 분자 집합을 시각화합니다.

대표 활용 — *potency vs. hERG safety margin* 2D Pareto plot:

```
[Pareto frontier 예시]

  pIC50 (활성)
   9 ┤    ★(높은 활성 + 낮은 안전성)
   8 ┤      ●  
   7 ┤        ●(Pareto optimal, 균형)
   6 ┤           ●
   5 ┤             ★(낮은 활성 + 높은 안전성)
   4 ┤                ●
        └─────────────────────
          1 μM   10 μM   100 μM
              hERG IC50 (안전성)
```

Pareto frontier 위 분자들은 — 모두 **합리적 선택지**이며, 어느 분자를 우선 합성할지는 — 적응증·환자군·경쟁 상황의 비즈니스 판단입니다. 약학 전공자는 — frontier의 어느 영역이 임상 가능성이 높은지(예: hERG IC50 > 10 μM 영역만 임상 진입 가능)를 임상 지식으로 판단합니다.

### 4) 약학적 판단 레이어 — 모방 불가능한 차별

ADMET 5축 예측 출력 + MPO 통합 + Pareto frontier 위에 — 약학 전공자만이 추가할 수 있는 판단 레이어:

| 판단 영역 | 약학적 근거 | 비전공자 대비 우위 |
|--------|--------|------------|
| **적응증별 MPO 분기** | CNS/경구/주사/국소에 따른 BBB·F%·반감기 다른 임계값 | 일률 cutoff 회피 |
| **표적·MoA 기반 hard cutoff** | 항부정맥제는 hERG 차단이 표적이므로 일반 cutoff 무효 | 잘못된 제거 방지 |
| **자유 약물(unbound) 보정** | PPB Day 2 + CL Day 3 → 자유 C_max → hERG safety margin | 안전 여유 정확 계산 |
| **임상 단계 게이트 정렬** | Phase I 진입 시점의 go/no-go 기준(hERG, DILI, Ames) | 게이트 누락 방지 |
| **규제 가이드라인 매핑** | ICH M7, S7B, E14, Q3A/B에 따른 한도 | 규제 제출 직접 활용 |
| **DDI·인구약리 보정** | CYP3A4 기질 + 노인 인구 → 용량 조정 권고 | 시판 후 회수 위험 사전 식별 |

이 판단 레이어가 — Week 3 ADMET 통합 SaaS의 **모방 불가능한 자산**입니다. 단일 ADMET 모델은 — TDC 리더보드와 ADMET-AI·ProTox-3.0 무료 도구와 경쟁이 어렵지만, 위 6개 판단 레이어가 결합되면 — 빅파마 사내 표준(Simulations Plus, Schrödinger)과 같은 카테고리에서 경쟁할 수 있습니다.

## 작동 원리와 아키텍처

Week 3 ADMET 통합 의사결정 시스템 — Day 1~4 출력을 단일 의사결정으로 통합:

```
[ADMET 통합 의사결정 파이프라인 — Week 3 결산]

1. 입력층 (Day 1)
   - SMILES + 적응증 + 표적 + 투여 경로 명세
   - Target Product Profile(TPP) 자동 로드

2. 다중 ADMET 예측 (Day 2~4)
   ┌─ A/D 모듈(Day 2): solubility, Caco-2, BBB, PPB
   ├─ M/E 모듈(Day 3): CYP IC50, CL, t1/2 → 1구획 PK
   └─ T 모듈(Day 4): hERG, DILI, Ames, Tox21 cascade

3. 약학적 후처리 (각 모듈의 차별 영역)
   - 자유 약물 보정: PPB · CL → 자유 C_max 추정
   - hERG safety margin = IC50 / 자유 C_max
   - ICH M7 Class 자동 분류
   - DDI risk score (CYP3A4·2D6·1A2 기질 + 억제 통합)

4. MPO 통합층 — Week 3 핵심
   - 적응증·표적 분기 → desirability function 선택
   - Derringer-Suich 통합 (가중/기하 평균)
   - Pareto frontier 자동 추출 (활성 vs. 독성 등)

5. 임상 게이트 평가
   - Phase I 진입 가능성 점수
   - 규제 권고문 자동 생성 (ICH 가이드라인 인용)
   - 합성 우선순위 ranking + Pareto 위 분자 강조

6. 출력층
   - 분자별 ADMET 카드(5축 + 통합 점수 + 임상 게이트)
   - Pareto frontier 인터랙티브 시각화
   - 규제 친화 리포트(PDF)
```

핵심 설계 결정 — Day 1~4 학습이 통합되는 지점:

| 결정 | Week 3 통합 관점 |
|------|--------|
| **단일 점수 vs. Pareto** | 둘 다 출력: 분류는 통합 점수, 의사결정은 Pareto |
| **가중치 결정** | 적응증별 사내 분기 + 임상 가이드라인 인용 |
| **Hard vs. soft cutoff** | hERG/Ames는 hard, F%/PPB는 soft (sigmoid) |
| **불확실성** | 각 ADMET 예측의 prediction interval을 desirability에 반영 |
| **재현성** | MPO 식과 가중치 파일을 버전 관리(YAML) |

## 신약개발 적용

산업 표준 MPO·통합 시스템 사례 — Week 3의 결산 관점에서:

| 사례·도구 | 핵심 결과 | Week 3 통합의 단서 |
|--------|--------|------------|
| **Pfizer MPO (Wager 2010, *ACS Chem. Neurosci.*)** | CNS 약물의 BBB 통과 후보 분리 정확도 약 80% | desirability function의 약학적 변곡점 |
| **GSK 4/400 rule + PFI** | clogP > 4 또는 MW > 400은 임상 실패율 약 2배 | 단순 cutoff도 임상 데이터 누적 시 강력 |
| **Astex LLE/LipE** | hit-to-lead 효율 약 2~3배 향상 | 활성-친유성 절충의 단일 지표화 |
| **Insilico Chemistry42** | RL 보상에 약 15개 ADMET 모듈 통합 | INS018_055 발굴의 핵심 게이트 |
| **Schrödinger LiveDesign MPO** | 빅파마 약 50개 사내 배포 | 약학 컨설팅 + 소프트웨어 결합 BM |
| **ADMET-AI (Stanford 2024)** | 41 task 통합 무료 웹 도구 | 무료 baseline + MPO 약함 → 차별화 여지 |
| **Roche CNS MPO + 사내 PK 모델 (2023)** | 후보 선정 사이클 약 30% 단축 | 적응증별 MPO 분기의 효과 측정 |

**대표 정량 비교 — MPO 도입 효과**:

| 평가 항목 | 단일 속성 우선순위 | MPO 통합 우선순위 |
|--------|--------------|---------------|
| 발견 단계 후보 분자 수 | 약 50,000개 | 약 5,000~10,000개 (1차 필터) |
| 합성 우선순위 정확도 | 약 30~40% (실제 임상 진입 분자 회수율) | 약 60~70% |
| 임상 1상 진입율 | 약 5~10% | 약 15~25% (Pfizer 사내 데이터 보고) |
| 후기 단계 회수 위험 | 약 30~40% | 약 15~20% (hERG·DILI 게이트 효과) |

**산업적 통찰** — Hann·Keserü 2012(*Nature Rev. Drug Discov.*) 분석은 — 단일 속성 최적화가 다른 속성을 악화시키는 "molecular obesity"(분자 비만, MW·logP가 hit-to-lead 과정에서 증가) 현상을 보고했습니다. MPO는 — 이러한 풍선 효과를 정량적으로 제어하는 표준 방법이며, 약학 전공자의 임상·규제 지식이 — 코드 자산으로 변환되는 가장 직접적 영역입니다. **Week 4 예고** — 다음 Week에서는 **단백질 구조와 구조 기반 설계**(AlphaFold, docking, DiffDock, 단백질 언어 모델)를 다룹니다. ADMET 게이트는 — 구조 기반 설계의 출력 분자에도 동일하게 적용되며, Week 3에서 만든 통합 의사결정 파이프라인이 Week 4 구조 기반 설계와 결합되어 **분자 발견의 완전한 cascade**를 이룹니다.

## 창업 관점

ADMET 통합 SaaS는 — Week 3 Day 1~4의 표준 모델을 합쳐도 무료 도구(ADMET-AI, ProTox-3.0)와 정면 경쟁이 어렵습니다. 차별의 핵심은 — 위 약학적 판단 레이어 6개(적응증 분기, 표적 MoA hard cutoff, 자유 약물 보정, 임상 게이트 정렬, 규제 가이드라인 매핑, DDI·인구약리 보정)를 **YAML 기반 desirability 라이브러리**로 자산화하는 것입니다. **1인 창업자의 현실적 로드맵**:

| Step | 기간 | 산출물 |
|------|------|------|
| **MVP** | 약 4~6주 | Chemprop + TDC 31개 ADMET task + ADMET-AI baseline + 단순 QED MPO |
| **차별 v1** | 약 8~12주 | 적응증별 MPO 분기(CNS·경구·주사) + 자유 약물 보정 + ICH M7 자동 분류 |
| **차별 v2** | 약 16~24주 | 규제 권고문 자동 생성 + Pareto 시각화 + 사내 데이터 finetune wrapper |
| **수익화** | 약 6~12개월 | 중소 바이오·임상약리 컨설팅 대상 월 1,000~10,000 달러 SaaS |

**시장 위치** — Simulations Plus(약 4~8억 달러 시장가치)·Schrödinger(약 30~50억 달러)는 빅파마 표준이지만 — 규제 외삽 자동화는 약합니다. "약사가 만든 규제·임상 친화 ADMET MPO" 포지셔닝이 — 중소 바이오·CRO·임상약리 컨설팅 시장에서 가능합니다. **핵심 해자는 desirability 라이브러리** — 본인의 적응증·표적별 임계값, ICH 가이드라인 매핑, 사내 데이터 보정 노하우가 YAML로 자산화되면, 비약학 경쟁자는 6~18개월 학습 격차를 따라잡기 어렵습니다. 단, **AI 단독 예측을 임상·규제 결정에 사용하지 말 것**이라는 면책 조항과 — 전문가 검토(expert-in-the-loop) 워크플로우 통합이 신뢰의 전제 조건입니다.

## 오늘의 과제

1. **본인 적응증 MPO 식 설계 (45분)**: 본인의 관심 적응증(예: NASH, Alzheimer, 항생제 내성)을 하나 선택하세요. 그 적응증의 표준 약물 5개를 정하고, 각 약물의 (a) clogP, MW, TPSA, HBD, F%, hERG IC50, t1/2, BBB(+/−) 값을 ChEMBL·DrugBank에서 수집하세요. 그 다음 — (b) 본인 적응증에 적합한 MPO 식을 직접 설계하세요(어떤 속성을 desirability에 포함할지, 각 변곡점은 어디인지, 가중치는 어떻게 할지). (c) 5개 약물에 본인 MPO 식을 적용해 점수를 계산하고, 임상에서의 성공·실패 패턴과 정렬되는지 검증하세요. 결론으로 — 본인 SaaS의 적응증별 MPO 분기 중 이 적응증의 v1 식을 YAML로 정의하세요.

2. **Pareto frontier 분석 + 빅파마 MPO 비교 (40분)**: ChEMBL에서 동일 표적(예: JAK1, EGFR, BACE1)의 활성 화합물 약 50~100개를 추출하세요. (a) 활성(pIC50)과 hERG IC50를 2D plot으로 그리고 Pareto frontier를 식별하세요(시각화는 Python·matplotlib 또는 ChatGPT/Claude로 빠르게). (b) Pfizer MPO, GSK PFI, Astex LLE, QED 4개 식을 동일 분자에 적용해 — 어떤 식이 임상 진입 분자(DrugBank에서 임상 단계 확인)를 더 잘 분리하는지 비교하세요. (c) 본인의 적응증·표적 조합에서 — 4개 표준 MPO 중 어느 것을 baseline으로 시작할지, 어떤 desirability를 추가/수정할지 1페이지로 정리하세요.

3. **Week 4 예고 리서치 — 단백질·구조 기반 설계와 ADMET 연결 (30분)**: AlphaFold·DiffDock·ESM-2 중 하나를 선택해 — (a) 그 도구의 핵심 출력(단백질 구조, 도킹 포즈, 단백질 임베딩)을 파악하세요. (b) 그 출력이 Week 3 ADMET 게이트와 어떻게 결합되는지(예: 도킹 점수 + ADMET MPO → 종합 후보 우선순위) 의사결정 흐름도를 작성하세요. (c) 본인 SaaS가 Week 3 ADMET MPO 위에 Week 4 구조 기반 설계를 추가한다면 — 어떤 적응증·표적이 첫 번째 우선순위가 되어야 하는지 근거와 함께 정리하세요.

## 참고 자료

- **MPO 표준**: Wager, T.T. et al. "Moving beyond rules: the development of a central nervous system multiparameter optimization (CNS MPO) approach." *ACS Chem. Neurosci.*, 2010. — 빅파마 MPO의 학술적 기반.
- **Desirability function**: Derringer, G., Suich, R. "Simultaneous optimization of several response variables." *J. Quality Technology*, 1980. — MPO의 통계적 표준 (역사적 맥락).
- **QED**: Bickerton, G.R. et al. "Quantifying the chemical beauty of drugs." *Nature Chemistry*, 2012. — drug-likeness 정량의 표준.
- **Molecular obesity 경고**: Hann, M.M., Keserü, G.M. "Finding the sweet spot: the role of nature and nurture in medicinal chemistry." *Nature Rev. Drug Discov.*, 2012. — 단일 속성 최적화의 위험과 MPO 필요성.
- **임상 실패 통합 분석**: Sun, D. et al. "Why 90% of clinical drug development fails and how to improve it?" *Acta Pharm. Sin. B*, 2022. — ADMET 통합 의사결정의 임상적 가치.
- **산업 사례**: Insilico Medicine Chemistry42 white papers(2023~2024), Schrödinger LiveDesign 제품 문서 — 빅파마 MPO 통합 시스템의 현황.
- **오픈 도구**: ADMET-AI(swansonk14/admet_ai, 2024), ProTox-3.0(tox.charite.de), PyTDC ADMET Benchmark Group, RDKit `Descriptors`·`QED` 모듈 — 1인 창업자의 즉시 활용 스택.
