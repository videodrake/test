# Day 1: 신약개발 파이프라인 전체 조망 — 타겟에서 시판까지

> 이전 학습(Week 1 Day 5)에서 AI/ML의 4개 축(표현·과제·신호·도메인)을 단일 기술 지도로 통합했습니다. 오늘은 이 지도를 신약개발 파이프라인 5단계 위에 직접 배치하여, 어떤 단계에서 어떤 AI가 어떤 의사결정을 자동화하는지의 **전체 좌표계**를 그립니다.

## 개요

신약개발 파이프라인(drug discovery and development pipeline)은 표적 발굴부터 시판 후 감시(post-market surveillance)까지 평균 10~15년, 26억 달러(DiMasi et al., 2016)가 소요되는 R&D 프로세스입니다. 단계마다 데이터 형태·평가 기준·실패 비용이 완전히 다르며, AI는 단일 해법이 아니라 **각 단계별로 표현·과제·손실이 다른 모듈의 집합**으로 통합됩니다. 이번 Day는 5개 핵심 단계(타겟 → 히트 → 리드 → 전임상 → 임상)의 정의·산출물·실패율을 정리하고, Week 1의 기술 지도가 어디에 들어가는지 점을 찍습니다. 약학 전공자의 차별적 기여 지점도 단계별로 명시합니다. 이 좌표계는 Week 2의 나머지 Day(타겟 발굴·히트 발견·임상)와 Phase 2~3 전체의 기준틀이 됩니다.

## 핵심 개념

### 신약개발 파이프라인의 5단계

각 단계는 입력·산출물·핵심 의사결정·실패율이 다른 **독립 모듈**입니다. 산업 통계는 추정치이며 표적·질환·회사에 따라 변동이 큽니다.

| 단계 | 입력 | 산출물 | 기간 | 단계별 성공률 |
|------|------|-------|------|-------------|
| 1. 타겟 발굴·검증 | 질환 유전체, 문헌, 환자 데이터 | validated target (단백질/RNA) | 1~3년 | ~30~50% |
| 2. 히트 발견 | 표적, 화합물 라이브러리 | 활성 분자 hit (μM 수준) | 1~2년 | ~50% |
| 3. 리드 최적화 | hit, ADMET 데이터 | clinical candidate (nM, ADMET 통과) | 2~3년 | ~30% |
| 4. 전임상 (IND-enabling) | candidate, 동물 모델 | IND 패키지 (안전성·PK) | 1~2년 | ~50% |
| 5. 임상 I/II/III → 승인 | IND, 환자 | NDA/BLA, 시판 | 6~10년 | Phase I 64.5%, II 32.4%, III 60.1% (Hay et al., 2014) |

임상 진입 약물의 누적 성공률은 약 **9.6%** 수준입니다(Hay et al., *Nature Biotechnology*, 2014). 약학 전공자에게 친숙한 이 숫자가 AI 신약개발의 가치 제안(value proposition)의 출발점입니다 — **실패 단계를 앞당겨**(fail fast, fail cheap) 후기 단계의 누적 비용을 절감하는 것이 모든 AI 모듈의 공통 목표입니다.

### 단계별 핵심 의사결정 — go/no-go의 비용 비대칭

각 단계의 본질은 "go/no-go" 의사결정이며, 약사·의약화학자·임상의가 이미 다루는 결정입니다. AI는 각 결정을 데이터 기반으로 보강합니다.

- **타겟 발굴**: 이 표적이 질환과 인과관계가 있고 약물화(druggable)되는가
- **히트 발견**: 이 분자가 표적에 결합하는가 (IC50 < 10 μM 또는 단일자릿수 μM)
- **리드 최적화**: 동시에 활성·선택성·ADMET·합성가능성을 만족하는가 (Multi-Parameter Optimization, MPO)
- **전임상**: 안전성 마진(safety margin)이 충분한가 — 효능 용량 대비 NOAEL(No Observed Adverse Effect Level) 배수
- **임상**: 환자에서 효능과 안전성이 입증되는가 (placebo 대비 통계적 유의성, 임상적 유의성 모두)

**비용 비대칭(cost asymmetry)** 이 단계마다 극단적으로 다르다는 점이 핵심입니다. 임상 III상 실패 1건의 비용(~10억 달러, Paul et al., 2010)은 히트 단계 위양성 1건의 비용(~수만 달러)의 10만 배 이상입니다. 따라서 AI 모델의 손실함수도 단계마다 다르게 설계되어야 합니다 — Week 1 Day 4의 **비용 민감 손실(cost-sensitive loss)** 이 적용되는 실전 맥락입니다. 후기 단계로 갈수록 위양성을 줄이기 위해 보수적(conservative) 임계값을, 초기 단계는 위음성을 줄이기 위해 관대한(permissive) 임계값을 사용하는 것이 일반적입니다.

### 약학 전공자의 차별적 기여

비전문가가 놓치는 지점을 약학 전공자는 정확히 인지합니다.

| 단계 | 비전문가의 한계 | 약학 전공자의 차별 |
|------|---------------|-----------------|
| 타겟 발굴 | 유전자 발현·차등분석만으로 표적 선정 | 약리 기전·alternative pathway·tissue-specific expression 평가 |
| 히트 발견 | IC50 단일 지표 신뢰 | 어세이 조건(pH, 셀라인, ATP 농도, 효소 농도) 평가 |
| 리드 최적화 | 활성 단일 최적화 | F(생체이용률)·CL(클리어런스)·Vd(분포용적) 등 PK 파라미터 동시 고려 |
| 전임상 | NOAEL 값 그대로 사용 | allometric scaling, species difference, HED 환산 적용 |
| 임상 | RCT 디자인 무지식 | endpoint·바이오마커·환자 stratification·중도탈락 보정 이해 |

## 작동 원리와 아키텍처

### Week 1 기술 지도 × 신약개발 파이프라인 매핑

Week 1에서 정의한 4개 축(표현·과제·신호·도메인)에 5개 단계를 적용하면 각 단계가 요구하는 AI 시스템의 윤곽이 분명해집니다.

```
[단계 1: 타겟 발굴·검증]
  표현: 유전자 발현 매트릭스, knowledge graph 임베딩, 단백질 서열
  과제: 분류·랭킹 (질환 연관 유전자/단백질의 우선순위화)
  신호: GWAS 통계 + 문헌 supervision + 약물 부작용 신호
  도메인: 신규 표적군 외삽 어려움 (희귀 질환)

[단계 2: 히트 발견]
  표현: SMILES, fingerprint, 분자 graph
  과제: 분류·회귀 (active/inactive, pIC50)
  신호: HTS 라벨 (극도 불균형, 활성률 <1%)
  도메인: 신규 scaffold 외삽 — 학습 데이터 분포 밖

[단계 3: 리드 최적화]
  표현: 분자 graph + 3D conformer + 단백질 결합 자세
  과제: 다중 회귀 + 생성(RL/diffusion) + MPO
  신호: 활성·ADMET·합성가능성 다중 보상 (다목표 최적화)
  도메인: 동일 화학 series 내 vs 신규 series 외삽

[단계 4: 전임상 (IND-enabling)]
  표현: 분자 + 종(species) 메타데이터 + 어세이 표준화
  과제: 회귀 (PK parameter, hERG IC50, in vivo Cmax/AUC)
  신호: 동물 실험 데이터, ADMET 분류 라벨
  도메인: 동물 → 인간 외삽 (가장 큰 일반화 갭)

[단계 5: 임상 I/II/III]
  표현: 환자 표현형 + 약물 + 시계열 EHR + 의료 영상
  과제: 분류·생존분석·NLP·환자 매칭
  신호: 임상시험 결과, 실세계 데이터(RWD)
  도메인: 임상시험 인구 → 일반 환자 인구 외삽
```

이 매핑은 Week 2의 Day 2(타겟)·Day 3(히트·리드)·Day 4(임상)의 도입부가 됩니다. 동일한 4개 축이 단계마다 다른 답을 갖는다는 점이 핵심입니다.

### 단계 간 산출물 통합 — 선형 파이프라인이 아닌 피드백 루프

파이프라인은 그림상 선형이지만, 실제로는 **양방향 피드백 루프**입니다. 임상 실패(5단계)의 원인 분석은 리드 최적화(3단계)와 타겟 검증(1단계)으로 역류합니다. AI 시스템도 이 구조를 반영해야 합니다.

```
타겟 ──→ 히트 ──→ 리드 ──→ 전임상 ──→ 임상
  ↑       ↑        ↑         ↑          │
  └───────┴────────┴─────────┴──────────┘
         실패 데이터 피드백 (active learning, retrospective analysis)
```

좋은 AI 신약개발 플랫폼은 단계 모듈을 잘 만드는 것에 그치지 않고 **단계 간 데이터 흐름과 피드백 루프**를 설계합니다. 이것이 Recursion·Insilico·Schrödinger의 플랫폼 가치의 본질이며, Phase 4 Week 2에서 다룰 데이터 인프라(MLOps) 학습의 동기입니다.

## 신약개발 적용

실제 회사·연구가 이 5단계 파이프라인에 AI를 어떻게 배치하는지 구체적 사례입니다. 회사마다 강점 단계가 다르다는 점에 주목합니다.

- **Insilico Medicine — INS018_055 (특발성 폐섬유증, IPF)**: 표적 발굴(PandaOmics)부터 IND-enabling 후보 도출까지 약 18개월·약 270만 달러로 단축한 것으로 보고되었습니다(Ren et al., *Nature Biotechnology*, 2024 — TNIK 표적 IPF 후보 논문). 전통적 4~5년·수억 달러 대비 단축의 핵심은 단계마다 **별도의 AI 모듈**(타겟=PandaOmics, 분자=Chemistry42, 임상예측=InClinico)을 조합한 점입니다. 2024년 Phase IIa 진입이 보도되었습니다.
- **Schrödinger — DESRES 슈퍼컴퓨터 기반 통합 플랫폼**: 자유에너지 계산(FEP+)과 분자 동역학을 핵심으로 리드 최적화(3단계) 강점이 두드러집니다. Nimbus Therapeutics의 TYK2 억제제 NDI-034858이 2022년 12월 Takeda에 수십억 달러 규모(상위 라인 약 40억 달러 + 마일스톤)로 라이선스된 사례가 AI/계산화학 기반 리드 최적화의 상업적 검증으로 자주 인용됩니다.
- **Recursion Pharmaceuticals — Phenomics + 화학 통합**: 세포 이미징 표현형(phenotype) 데이터를 대규모로 수집해 1·2단계를 통합한 접근입니다. 2024년 Exscientia 인수 발표(전 종목 거래)로 화학 디스커버리(3단계) 역량을 흡수, 5단계 파이프라인을 단일 회사 내에서 수직 통합하는 전략을 보이고 있습니다.

전통 파이프라인과 AI 통합 파이프라인의 비교:

| 항목 | 전통 파이프라인 | AI 통합 파이프라인 (2023~) |
|------|--------------|-------------------------|
| 타겟 검증 시간 | 2~3년 | 6~12개월 (omics + 문헌 AI) |
| 히트 발견 화합물 수 | HTS 100만~1000만 실제 실험 | 가상 스크리닝 10억+ |
| 리드 최적화 사이클 | 50~100회 (DMTA, ~3년) | 10~20회 (~1년, MPO 자동) |
| ADMET 평가 | wet-lab 실험 중심 | in silico 우선, wet-lab 확인 |
| 임상 환자 선별 | 표준 inclusion/exclusion | AI 바이오마커 stratification |

> 다만 이 표의 단축 효과는 발표 회사의 자기보고 기준이며 산업 평균은 아닙니다. 2024~2025년 시점에서 AI 발굴 후보의 임상 성공률이 전통 후보 대비 통계적으로 유의미하게 높다는 증거는 아직 부족합니다(BCG, "AI in biopharma research: A time to focus and scale", 2024 — 확인 필요). 약학 전공자는 발표 수치를 단계별 의미로 해석할 수 있어야 합니다.

## 창업 관점

Phase 1 도입 단계이므로 두 문장으로 짚습니다. 약학 전공 창업자의 합리적 출발점은 "5단계 전체를 다 하지 않고, 약학적 판단이 가장 차별을 만드는 단계 한두 개에 집중"하는 것입니다. 한국 환경에서는 ADMET(3~4단계)·임상 디자인(5단계)이 wet-lab 인프라 부담이 상대적으로 작고 약학 전공의 우위가 가장 큰 영역으로 판단되며, 본격적 시장·BM·경쟁 분석은 Phase 5 Week 1~2에서 다룹니다.

## 오늘의 과제

1. **나만의 파이프라인 좌표계 만들기 (40분)**: 신약개발 5단계(타겟→히트→리드→전임상→임상)를 행, Week 1의 4개 축(표현·과제·신호·도메인)을 열로 한 5×4 표를 직접 작성합니다. 각 칸에 가장 적합한 기법 1~2개를 1줄씩 채워 넣습니다(예: "1단계 × 표현 = knowledge graph 임베딩 + 유전자 발현 벡터"). 비어 있거나 자신 없는 칸이 Phase 2~3의 학습 우선순위입니다.
2. **회사의 단계별 포지셔닝 분석 (40분)**: Insilico Medicine, Recursion, Schrödinger, Atomwise 중 2개 회사를 선택해 공식 사이트·IR 자료·최근 논문을 빠르게 훑은 후, 각 회사가 5단계 중 어디에 집중하는지·어떤 단계는 비어 있는지를 1페이지 표로 정리합니다. 단순 "AI 신약개발 회사"가 아니라 **단계 단위 포지셔닝**으로 보는 훈련이며, Phase 5 시장 분석의 사전 작업입니다.
3. **단계별 위양성·위음성 비용 비대칭 추정 (30분)**: 5단계 각각에 대해 "위양성(false positive) 1건의 비용"과 "위음성(false negative) 1건의 비용"을 자료를 찾아 추정·작성합니다. 추정 근거(논문·산업 보고서)도 함께 기재합니다. 이 비대칭이 Week 1 Day 4 비용 민감 손실 설계의 출발점입니다.

## 참고 자료

- DiMasi, J.A., Grabowski, H.G., Hansen, R.W. (2016). "Innovation in the pharmaceutical industry: New estimates of R&D costs." *Journal of Health Economics*, 47, 20-33. — 신약개발 평균 26억 달러·10년 추정의 원전.
- Hay, M., Thomas, D.W., Craighead, J.L., Economides, C., Rosenthal, J. (2014). "Clinical development success rates for investigational drugs." *Nature Biotechnology*, 32, 40-51. — Phase I~III 성공률 표준 데이터.
- Paul, S.M. *et al.* (2010). "How to improve R&D productivity: the pharmaceutical industry's grand challenge." *Nature Reviews Drug Discovery*, 9, 203-214. — 파이프라인 단계별 비용·실패율의 고전 분석.
- Insilico Medicine — Pharma.AI 플랫폼 공식 사이트 (PandaOmics·Chemistry42·InClinico 모듈 구성).
- Recursion Pharmaceuticals — Phenomics 플랫폼과 OS(Operating System) 공개 자료.
