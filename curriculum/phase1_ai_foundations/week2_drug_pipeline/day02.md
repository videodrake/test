# Day 2: 타겟 발굴과 검증 — AI가 표적을 찾는 법

> 이전 학습(Week 2 Day 1)에서 신약개발 5단계 좌표계와 단계별 의사결정·비용 비대칭을 정리했습니다. 오늘은 그 1단계인 **타겟 발굴·검증(target discovery and validation)** 을 깊이 파고들어, AI가 어떤 데이터로 어떤 방법으로 표적을 찾고 검증하는지 — 그리고 약학 전공자가 어디서 결정적 차이를 만드는지를 학습합니다.

## 개요

타겟 발굴은 "질환의 분자적 원인이 되는 단백질·RNA·경로를 식별하고, 그 표적을 약물로 조절(modulate)했을 때 임상적 효과가 나타날 것임을 입증하는 과정"입니다. 임상 II상 실패의 약 **40~50%가 효능 부족(lack of efficacy)** 에서 발생하며, 그 근본 원인은 대부분 타겟 자체가 질환과 인과관계가 없거나 약물화가 어려운 경우입니다(Cook et al., *Nature Reviews Drug Discovery*, 2014 — AstraZeneca 5R framework). AI는 단일 가설에 의존하는 전통 발굴 방식의 한계를 보완하기 위해 다중 오믹스(multi-omics), 지식 그래프(knowledge graph), 문헌 마이닝을 통합하여 **수만 개 후보 표적의 우선순위를 데이터 기반으로 매기는** 역할을 합니다. 이번 Day는 타겟 발굴의 5가지 데이터 축, AI 방법론 4가지, 검증(validation) 3단계, 그리고 실제 회사 사례를 다루며 — Day 1에서 그린 5단계 좌표계의 1단계에 구체적 살을 붙입니다.

## 핵심 개념

### 타겟이란 무엇인가 — 정의와 druggability

**타겟(target)** 은 약물이 결합하여 그 활성을 변화시킴으로써 질환 표현형을 변화시키는 생체분자입니다. 임상에 진입한 약물의 표적은 대부분 단백질이며, FDA 승인 약물의 표적은 약 **890개 단백질**(Santos et al., *Nature Reviews Drug Discovery*, 2017) 수준에 그칩니다. 인간 단백체(proteome)가 약 2만 개임을 고려하면 활용 가능한 표적 공간은 여전히 광대합니다.

타겟 후보를 평가하는 두 가지 핵심 속성:

- **Druggability(약물화 가능성)**: 표적이 small molecule이나 biologic으로 조절 가능한 구조적·생화학적 특성을 가지는가. small molecule의 경우 깊은 결합 포켓(binding pocket)·소수성 표면·구조적 유연성 등이 평가 기준입니다(Hopkins & Groom, *Nature Reviews Drug Discovery*, 2002 — druggable genome 개념의 원전).
- **Disease relevance(질환 연관성)**: 표적의 활성을 변화시키면 실제로 질환 표현형이 개선되는가. 유전학적 증거(GWAS, Mendelian randomization)·발현 변화·기능적 손실/획득 실험으로 평가합니다.

약학 전공자가 즉시 떠올려야 할 점: **이미 승인된 약물의 표적군(GPCR, kinase, ion channel, nuclear receptor)에 약물화 가능성 지식이 집중**되어 있고, transcription factor·RNA·단백질-단백질 상호작용은 전통적으로 "undruggable"로 간주되었습니다. PROTAC·molecular glue·RNA 표적화 약물의 등장으로 이 경계가 무너지고 있으며, AI는 이 신규 표적군(emerging target class)에서 가장 큰 가치를 만들 가능성이 있습니다.

### 타겟 발굴의 5가지 데이터 축

AI 타겟 발굴 시스템이 활용하는 데이터는 5축으로 분류됩니다. 약학 전공자에게 익숙한 데이터부터 출발합니다.

| 데이터 축 | 대표 데이터 | AI가 추출하는 신호 |
|----------|-----------|-----------------|
| 유전학(genetics) | GWAS, ExWAS, rare variant burden, Mendelian randomization | 인과관계(causal evidence) — 유전 변이 → 질환 |
| 전사체(transcriptomics) | bulk RNA-seq, single-cell, GTEx | 질환·정상 조직 발현 차이, 세포 특이성 |
| 단백체(proteomics) | mass spec, affinity proteomics | 활성 단백질·인산화 등 PTM·약물-표적 상호작용 |
| 표현형(phenomics) | CRISPR screen, cell painting, animal model | 기능적 손실/획득의 표현형 변화 |
| 문헌·지식(knowledge) | PubMed, ChEMBL, DrugBank, OmniPath | 보고된 표적-질환 관계, 약물 부작용 신호, 경로 |

각 축은 독립적으로는 노이즈가 많지만 **다축 합의(multi-axis convergence)** — 예를 들어 GWAS hit이면서 동시에 질환 조직에서 발현이 변하고 CRISPR로 표현형이 바뀌는 유전자 — 가 강력한 표적 후보입니다. Open Targets 플랫폼이 이 합의 점수(association score) 알고리즘으로 약 2만 개 유전자에 대해 약 24,000개 질환과의 연관성을 정량화하는 대표적 사례입니다.

### 약학 전공자의 차별적 기여 — 인과추론과 임상 번역

비전공자가 데이터 점수만 신뢰할 때 약학 전공자는 다음 질문을 던집니다.

- **인과성 vs 상관성**: 표적 유전자의 발현이 질환에서 높다면, 그것이 원인인가 결과인가? Mendelian randomization과 약물 부작용 phenome-wide association(PheWAS)이 인과 방향을 추정합니다.
- **조직·세포 특이성**: 표적이 질환 조직 외에 심장·간에서도 강하게 발현되면 부작용 위험이 큽니다(예: hERG potassium channel은 모든 약물의 회피 대상).
- **경로 redundancy**: 같은 경로 내에 paralog 또는 우회 경로가 있으면 표적 억제 효과가 보상(compensation)됩니다.
- **약리학적 modality 적합성**: 표적이 세포 내·외부, 막관통 여부에 따라 small molecule·antibody·ASO 중 적합 약물 형태가 결정됩니다.

이 질문들은 AI 모델의 입력 feature 설계와 점수 통합 가중치에 직접 반영됩니다 — 약학 전공자가 모델 개발자에게 "이 feature를 추가하세요, 이 가중치를 올리세요"라고 지시할 수 있는 영역입니다.

## 작동 원리와 아키텍처

### AI 타겟 발굴의 4가지 방법론

타겟 발굴 AI는 데이터 유형과 모델링 방식으로 4가지 패러다임으로 정리됩니다.

| 방법론 | 입력 | 모델 | 출력 |
|--------|------|------|------|
| 1. 차등 발현·CRISPR 통합 | bulk/sc RNA-seq + CRISPR 의존성 | 통계 + 머신러닝 분류기 | 질환 대비 우선순위 점수 |
| 2. 지식 그래프 임베딩 | 표적-질환-약물-경로 노드/엣지 | TransE, RotatE, GNN | 노드 임베딩 + 링크 예측 |
| 3. 문헌 NLP | PubMed abstract, 임상 보고 | LLM, BERT 계열 | 표적-질환 관계 추출 |
| 4. 인과추론(causal ML) | GWAS, PheWAS, eQTL | Mendelian randomization, 인과 그래프 | 인과 방향성 + 효과 크기 |

이들은 경쟁이 아니라 **앙상블**로 결합됩니다. Open Targets의 association score가 이 결합의 단순 가중합 버전이고, BenevolentAI·Insilico의 PandaOmics는 그래프 임베딩과 LLM을 더 정교하게 통합한 사유 시스템입니다.

### 지식 그래프 기반 표적 발굴 — 아키텍처 개요

가장 대표적인 AI 타겟 발굴 시스템의 구조를 의사코드 수준으로 살펴봅니다.

```
[입력] 노드: 유전자 2만 + 질환 2만 + 약물 1만 + 경로 5천 + 부작용 1만
        엣지: "유전자 → 질환 연관"(GWAS), "약물 → 표적"(ChEMBL),
              "유전자 → 경로"(Reactome), "약물 → 부작용"(FAERS)

[처리]
1. 그래프 구축: ~6만 노드, ~수백만 엣지
2. 임베딩 학습: 각 노드 → 128~512 차원 벡터
   - 손실: 실제 엣지의 유사도는 높게, 무작위 엣지는 낮게
3. 링크 예측: 질환 노드와 유전자 노드의 임베딩 유사도로
   "아직 보고되지 않은 표적-질환 관계" 점수화
4. 약학적 필터: druggability·tissue specificity·safety 필터 적용

[출력] 질환별 표적 후보 랭킹 리스트 (Top-100 등)
```

이 시스템의 강점은 **새로운 표적-질환 관계의 발굴**입니다 — 학습 시 보지 않은 조합도 예측합니다(zero-shot link prediction). 약점은 그래프에 없는 신호(예: 최신 임상 결과, 미공개 사내 데이터)는 반영되지 않는다는 점이며, 따라서 회사는 자체 데이터로 그래프를 보강하는 방식으로 차별화합니다.

### 검증(validation)의 3단계

AI가 추천한 표적은 그대로 임상에 가지 않습니다. 3단계 wet-lab 검증이 표준입니다.

1. **분자 수준 검증**: CRISPR knockout, siRNA knockdown, 화학 probe(tool compound)로 표적 활성 조절 시 세포 표현형이 변하는지 확인.
2. **세포·조직 모델 검증**: 환자 유래 세포(iPSC), 오가노이드, ex vivo 조직에서 표적 조절 효과 재현.
3. **동물 모델 검증**: 질환 동물 모델에서 표적 조절(유전적 또는 약물)이 질환 표현형을 개선하는지 확인.

AI 발굴 표적이라도 이 검증을 통과해야 IND 패키지 진입이 가능하며, AI의 가치는 검증 자체를 제거하는 것이 아니라 **검증할 후보를 더 정확하게 좁히는 것**에 있습니다.

## 신약개발 적용

타겟 발굴에 AI를 본격 적용한 회사·사례입니다. 각 사례의 입력 데이터·핵심 알고리즘·산출물을 함께 보면 패턴이 보입니다.

- **BenevolentAI — Baricitinib의 COVID-19 재배치(repurposing, 2020)**: 자체 지식 그래프를 활용해 "SARS-CoV-2 세포 진입에 관여하는 AAK1 키나아제 억제"라는 가설을 NLP·그래프 추론으로 도출, 기존 류마티스 관절염 약물 baricitinib(JAK1/2 + AAK1 억제)이 후보로 제안되었습니다. 후속 임상시험(ACTT-2, *NEJM*, 2021)에서 회복 시간 단축이 입증되어 2020년 11월 FDA EUA(긴급사용승인)·2022년 5월 정식 승인을 받았습니다. AI 타겟 발굴 → 임상 검증 → 승인까지 약 2년에 도달한 초기 대표 사례입니다.
- **Insilico Medicine — PandaOmics + TNIK 표적 IPF**: 특발성 폐섬유증(IPF) 환자의 다중 오믹스 데이터에 PandaOmics를 적용해 **TNIK(TRAF2- and NCK-interacting kinase)** 을 표적 후보로 도출한 후, Chemistry42로 분자 설계까지 진행하여 후보물질 INS018_055를 18개월 만에 발굴한 것으로 보고됩니다(Ren et al., *Nature Biotechnology*, 2024). 2024년 Phase IIa 진입이 보도되었으며, TNIK이 IPF 표적으로 임상에 진입한 첫 사례로 인용됩니다.
- **Verge Genomics — ALS 표적 PIKfyve(VRG50635)**: 환자 사후 조직의 대규모 전사체·단백체 데이터를 자체 CONVERGE 플랫폼에 통합해 ALS(루게릭병) 표적 PIKfyve를 도출, 환자에서 직접 시작하는(patient-first) AI 발굴 모델의 사례입니다. 2022년 IND 승인, 2023~2024년 Phase I 진행이 보고되었습니다.
- **Recursion — Phenomics 기반 표적 가설 생성**: 세포 이미징(약 1조 픽셀 데이터셋)에서 유전자·화합물 perturbation의 표현형 fingerprint를 비교하여 신규 표적-질환 가설을 생성합니다. 2024년 Exscientia 인수 후 1단계(표적)~3단계(리드) 통합 파이프라인을 구축 중입니다.

전통 타겟 발굴과 AI 타겟 발굴의 비교:

| 항목 | 전통 타겟 발굴 | AI 타겟 발굴 |
|------|--------------|-------------|
| 출발점 | 단일 가설(특정 경로·유전자) | 다중 오믹스 통합 + 그래프 |
| 후보 수 | 수십 개 | 수천 ~ 수만 개 우선순위 랭킹 |
| 소요 기간 | 2~3년 | 6~12개월(가설 단계) |
| 신규 표적 발굴 | 기존 약물군 인접 영역 | undruggable·신규 modality 표적 가능 |
| 검증 부담 | 동일(wet-lab 필수) | 동일하지만 후보 정확도 향상 기대 |

> 다만 AI 발굴 표적이 임상에서 통계적으로 더 높은 성공률을 보였다는 산업 평균 증거는 아직 누적 중입니다. 위 단축 효과·정확도는 각 회사의 자기보고이며, 약학 전공자는 이 발표 수치를 약리학적 메커니즘 관점에서 비판적으로 해석할 수 있어야 합니다.

## 창업 관점

Phase 1 단계이므로 두 문장으로 짚습니다. 타겟 발굴은 진입장벽이 높은 것처럼 보이지만(공개된 Open Targets·DepMap·OmniPath 데이터로 시작 가능) 약학 전공 창업자에게는 **특정 질환군(예: 신경퇴행성·희귀 질환·면역대사질환) 한정으로 약리학 지식과 데이터 큐레이션을 결합한 "심층 표적 발굴 SaaS"** 가 합리적 출발점일 수 있으며, 본격적 BM과 경쟁사 분석은 Phase 5 Week 1에서 다룹니다.

## 오늘의 과제

1. **Open Targets로 한 질환의 표적 후보 분석 (40분)**: Open Targets 플랫폼(공개)에서 본인이 관심 있는 질환 하나(예: Alzheimer's disease, type 2 diabetes, NASH)를 선택하여 association score 상위 20개 표적을 추출합니다. 각 표적에 대해 (a) 약물 승인 여부, (b) druggable target class(GPCR/kinase/etc), (c) 본인이 판단하는 약물화 가능성(High/Medium/Low)을 1페이지 표로 작성합니다. "Open Targets가 추천하지만 약학적으로는 회의적인 표적"이 무엇인지 그 이유와 함께 기록합니다.
2. **AI 타겟 발굴 회사 2곳 비교 (40분)**: BenevolentAI, Insilico Medicine, Verge Genomics, Recursion 중 2개 회사를 선택해 그들의 (a) 입력 데이터 축(5축 중 어느 것을 강조하는지), (b) 핵심 알고리즘(그래프·NLP·phenomics 등), (c) 현재까지의 임상 진입 표적 수와 단계를 1페이지로 비교 정리합니다. "약학 전공자로서 어느 회사 접근이 더 신뢰가 가는가"를 한 문단으로 결론짓습니다.
3. **표적 검증 시나리오 설계 (30분)**: AI가 추천한 가상의 표적 X(예: 어떤 키나아제)를 검증한다고 가정하고, 분자→세포→동물 3단계 검증 실험을 각 단계별 (a) 실험 방법, (b) 예상 비용·기간, (c) 통과 기준(go/no-go criteria)을 표로 작성합니다. 약학 전공 지식이 이 실험 설계의 어느 부분에서 발휘되는지 표 아래 3줄로 정리합니다.

## 참고 자료

- Santos, R. *et al.* (2017). "A comprehensive map of molecular drug targets." *Nature Reviews Drug Discovery*, 16, 19-34. — 승인 약물 표적 ~890개 통계의 표준 출처.
- Cook, D. *et al.* (2014). "Lessons learned from the fate of AstraZeneca's drug pipeline: a five-dimensional framework." *Nature Reviews Drug Discovery*, 13, 419-431. — 임상 II상 실패 원인 분석(5R framework).
- Ren, F. *et al.* (2024). "A small-molecule TNIK inhibitor targets fibrosis in preclinical and clinical models." *Nature Biotechnology* (Insilico Medicine, IPF 후보 INS018_055 보고).
- Hopkins, A.L. & Groom, C.R. (2002). "The druggable genome." *Nature Reviews Drug Discovery*, 1, 727-730. — druggability 개념의 원전.
- Open Targets Platform (https://platform.opentargets.org) — 표적-질환 association score를 무료 제공하는 공개 플랫폼. 학습자의 첫 실습 도구로 권장.
- BenevolentAI, Insilico Medicine, Verge Genomics 공식 IR 자료 — 각 회사의 플랫폼 구조와 임상 진입 표적 현황.
