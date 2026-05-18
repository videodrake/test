# Day 3: 히트 발견과 리드 최적화 — 가상 스크리닝의 세계

> 이전 학습(Week 2 Day 2)에서 타겟 발굴·검증의 5축 데이터와 4가지 AI 방법론, 검증 3단계를 다뤘습니다. 오늘은 그 다음 단계인 **히트 발견(hit identification)과 리드 최적화(lead optimization)** 로 내려가, "검증된 표적에 결합하는 분자를 어떻게 찾고, 어떻게 임상 후보로 다듬는가"를 가상 스크리닝(virtual screening, VS) 중심으로 학습합니다.

## 개요

히트 발견은 검증된 표적에 결합하는 활성 화합물을 식별하는 단계이고, 리드 최적화는 그 활성 화합물을 임상 후보(clinical candidate) 수준으로 약물성·선택성·ADMET을 동시에 다듬는 단계입니다. 두 단계가 합쳐 신약개발 전체 기간의 약 3~5년, R&D 비용의 약 30~40%를 차지하는 것으로 추정됩니다(Paul et al., *Nature Reviews Drug Discovery*, 2010 추정치 갱신본). 전통적으로는 **고속대량스크리닝(HTS, High-Throughput Screening)** 으로 수십만~수백만 화합물을 실험적으로 평가했으나, 현대 라이브러리는 **수십억~수백조 규모의 가상 화합물 공간**으로 확장되었고 이를 다루는 핵심 기술이 가상 스크리닝(VS)입니다. 이번 Day는 (1) HTS와 가상 스크리닝의 관계, (2) 리간드 기반 vs 구조 기반 VS의 원리, (3) AI가 가속·재정의한 부분, (4) hit → lead → candidate로 이어지는 약물성 최적화의 다중 목표 구조, (5) 약학 전공자의 차별적 기여 지점을 정리합니다.

## 핵심 개념

### 히트, 리드, 후보의 정의 — 활성 기준의 단계적 강화

같은 "활성 화합물"이라도 단계마다 통과 기준이 다릅니다. 약학 전공자가 임상화학 시간에 익숙한 용어이지만 산업적 정의를 정확히 정렬해 둡니다.

| 단계 | 정의 | 활성 기준(소분자 기준 예시) | 추가 통과 기준 |
|------|------|--------------------------|---------------|
| Hit | 표적에 결합하는 1차 활성 분자 | IC50 또는 Kd < 10 μM | 재현성, 표적 특이성 1차 확인 |
| Lead | 최적화 출발점이 되는 화합물 시리즈 | IC50 < 1 μM, LE ≥ 0.3 | SAR 구축 가능, 합성 접근성 |
| Clinical candidate | IND 진입을 위한 최종 후보 | IC50 < 10~100 nM | ADMET·안전성·PK·CMC 통과 |

여기서 **리간드 효율(Ligand Efficiency, LE)** 은 ΔG/heavy atom으로 정의되며 분자량이 커질수록 활성이 단순 증가하는 함정을 막는 표준 지표입니다(Hopkins et al., *Drug Discovery Today*, 2004). 약학 전공자에게 친숙한 **Lipinski's rule of five**(MW<500, logP<5, HBD≤5, HBA≤10)도 hit→lead 단계에서 1차 필터로 작동합니다 — 단, 이는 경구 흡수 small molecule에 한정된 경험칙이며 covalent inhibitor·PROTAC·peptide에서는 자주 위반됩니다.

### 가상 스크리닝(VS)의 두 패러다임

가상 스크리닝은 "화합물 라이브러리에서 표적에 결합할 가능성이 높은 분자를 컴퓨터로 우선순위화하는 작업"이며 두 패러다임으로 나뉩니다. 약학 전공자가 의약화학 강의에서 SAR(structure-activity relationship)과 SBDD(structure-based drug design)로 배운 두 방식의 컴퓨터 버전입니다.

- **리간드 기반 가상 스크리닝(Ligand-Based VS, LBVS)**: 표적의 알려진 활성 분자(active compounds)와 화학적으로 유사한 분자를 찾습니다. 입력은 분자 구조(SMILES, fingerprint, 그래프)만이며 표적의 3D 구조가 필요 없습니다. 유사도 검색(Tanimoto similarity on Morgan fingerprint), QSAR 모델, 약물군(pharmacophore) 매칭이 대표 방법입니다.
- **구조 기반 가상 스크리닝(Structure-Based VS, SBVS)**: 표적의 3D 구조(X-ray, cryo-EM, AlphaFold 예측)에 화합물을 도킹(docking)하여 결합 자유에너지를 추정합니다. AutoDock Vina, Glide, GOLD 등 전통 도킹 프로그램과 최근의 AI 도킹(DiffDock 등 — Phase 2에서 상세 학습)이 여기에 속합니다.

두 패러다임은 경쟁이 아니라 보완 관계입니다. 활성 데이터가 풍부한 표적(kinase 등)은 LBVS가 빠르고 정확하며, 신규 표적이나 알로스테릭 사이트는 SBVS가 적합합니다. 실무에서는 두 결과를 **합의(consensus scoring)** 로 결합하는 것이 표준입니다.

### 화학 공간(chemical space)의 폭발과 ultra-large screening

전통 HTS는 회사 보유 라이브러리 50만~200만 화합물을 실험했지만, 합성 가능 화합물 공간은 이론적으로 **10^60 수준**(Bohacek et al., 1996)으로 추정됩니다. 2020년대 들어 Enamine REAL Space(약 480억 분자, 2024년 기준), ZINC22(370억 이상의 make-on-demand 화합물)와 같이 "make-on-demand" 카탈로그가 수십억 규모로 확장되었고, AI/computing 가속으로 이 전체를 가상 스크리닝하는 ultra-large screening이 가능해졌습니다.

> Lyu et al.(*Nature*, 2019)이 1억 7천만 분자를 도킹해 D4 dopamine receptor와 AmpC β-lactamase의 신규 활성 분자를 발굴한 사례는 ultra-large screening의 효과를 보인 대표 논문입니다. 이후 Sadybekov et al.(*Nature*, 2022)은 동일 접근을 *11억* 분자로 확장하여 sigma-2 receptor 신규 화학형(chemotype)을 발굴했습니다.

### 약학 전공자의 차별적 기여 — hit의 "약물성"을 판단

VS는 활성 점수(score)를 줄 뿐이며, 그 점수가 **실제 약물로 발전 가능한가**는 다른 문제입니다. 약학 전공자가 즉시 적용할 판단 기준:

- **PAINS(Pan-Assay Interference Compounds)**: redox cycler·aggregator·reactive electrophile 등 다수 assay에서 거짓 양성을 만드는 구조 패턴(Baell & Holloway, *J Med Chem*, 2010). VS hit 중 PAINS 필터를 통과해야 합니다.
- **frequent hitter**: 여러 표적에 비특이적으로 결합하는 분자(예: rhodanine, curcumin 유도체).
- **chemical tractability**: 합성 단계 수, scaffold의 SAR 확장 가능성, IP 공간 여유.
- **약리학적 modality 적합성**: 표적 위치(세포 내/외, 막관통, 핵)에 따라 small molecule이 적합한지, peptide·biologic이 더 적합한지를 1차 판단.

이 판단은 모델 후처리 필터로 정량화되며, 약학 전공자가 **모델 출력의 임계값과 필터 설계에 직접 기여**할 수 있는 지점입니다.

## 작동 원리와 아키텍처

### 가상 스크리닝 파이프라인의 표준 구조

```
[입력]
- 표적: 단백질 3D 구조(PDB) + 결합 포켓 정보
- 라이브러리: 수십억~수백억 분자 (SMILES + 3D conformer)

[처리]
1. 사전 필터링
   - 분자량/logP/PAINS/reactive group 제거 → 통상 라이브러리의 30~60% 통과
2. 1차 스크리닝 (빠른 모델, 전체 라이브러리)
   - 옵션 A: 리간드 유사도(Tanimoto > 0.7) → 좁힘
   - 옵션 B: AI 사전 분류기(known active로 학습한 분류 모델) → top 0.1~1%
   - 옵션 C: 빠른 도킹(low precision) → top 0.1%
3. 2차 정밀 평가 (느린 모델, 좁혀진 후보)
   - 정밀 도킹(Glide XP, Vina high exhaustiveness)
   - MM-GBSA/MM-PBSA 자유에너지 보정 (분자역학 기반, Phase 3에서 학습)
4. 후처리 필터
   - 약물성 필터(Lipinski, QED)
   - 합성 가능성 점수(SA score, RAscore)
   - 신규성·IP 공간 분석
5. 클러스터링과 시각화 → 화학자가 검토할 top 100~1000

[출력] 실험 검증 대상 화합물 50~200개
```

핵심은 **점진적 좁힘(funnel)** 구조입니다. 라이브러리가 클수록 1차 모델은 빠르고 부정확해도 되며, 후속 단계에서 정밀도가 보상합니다. 약학 전공자는 각 단계의 임계값과 필터를 표적·질환 맥락에 맞게 조정하는 역할을 합니다.

### LBVS와 SBVS의 비교

| 항목 | 리간드 기반(LBVS) | 구조 기반(SBVS) |
|------|----------------|----------------|
| 필수 입력 | 알려진 활성 분자 셋 | 표적의 3D 구조 |
| 핵심 기법 | 유사도 검색, QSAR, pharmacophore | 도킹, MM/MD, AI 결합력 예측 |
| 강점 | 속도 빠름, 활성 데이터로 검증된 패턴 활용 | 신규 화학형(novel chemotype) 발굴 가능 |
| 약점 | scaffold hopping 어려움, 알려진 공간 내 한정 | 도킹 score의 정확도 한계, 결합 양식 불확실 |
| 적합한 상황 | 활성 분자 ≥ 수십 개, kinase 등 데이터 풍부 표적 | 신규 표적, 알로스테릭 사이트, 활성 분자 부족 |

> 가상 스크리닝의 핵심 한계: 도킹 점수와 실제 결합 친화력의 상관계수는 평균 r ≈ 0.3~0.5 수준에 머무르는 경우가 많아(Cheng et al., *J Chem Inf Model*, 2009; 후속 벤치마크에서 일관) "VS는 가설 생성기이지 결합력 측정기가 아니다"라는 인식이 표준입니다. AI 기반 결합력 예측의 진전은 Phase 2 Week 4(단백질·도킹)에서 별도로 다룹니다.

### 리드 최적화의 다중목표 최적화(MPO) 구조

hit → lead → candidate로 가는 동안 활성만 올리면 되는 것이 아니라 **여러 약물성 파라미터를 동시에** 만족해야 합니다. AI 시대 이전부터 의약화학자가 알고 있는 사실이지만, AI 모델이 이를 정량화·자동화합니다.

일반적인 다중 목표(예시):

| 파라미터 | 목표 범위 | 측정·예측 방법 |
|---------|---------|--------------|
| 표적 활성(potency) | IC50 < 100 nM | 효소·세포 assay, QSAR/GNN 예측 |
| 선택성(selectivity) | off-target IC50 > 10× | 패널 assay, multi-task 모델 |
| 용해도(solubility) | > 100 μM (kinetic) | 실측 + logS 예측 모델 |
| 투과성(permeability) | Papp > 10×10⁻⁶ cm/s | Caco-2 + 예측 모델 |
| 대사 안정성 | t1/2 > 30 min (간 microsome) | LC-MS 실측 + CYP 예측 모델 |
| hERG 회피 | IC50 > 10 μM | 패치 클램프 + hERG 모델 |

이 다차원 공간에서 한 파라미터를 올리면 다른 것이 떨어지는 trade-off가 흔합니다(예: logP를 낮춰 용해도를 올리면 막투과성이 떨어짐). 약학 전공자가 PK/PD·ADME 수업에서 배운 바로 그 균형 잡기를 AI가 다중 작업(multi-task) 모델·다목적 강화학습으로 자동화합니다 — 이 부분은 Phase 2 Week 2~3에서 상세하게 다룹니다.

## 신약개발 적용

### AI/가상 스크리닝이 만든 실제 후보 사례

- **Atomwise — AIMS 프로그램의 다중 표적 hit 발굴(2020~)**: Atomwise의 AtomNet은 합성곱 신경망으로 결합 친화력을 추정하며, 학술 파트너에게 무료로 hit list를 제공하는 AIMS(Artificial Intelligence Molecular Screen) 프로그램으로 250개 이상 표적에 대해 hit를 발굴한 것으로 보고됩니다(Wallach et al., *Nature Communications*, 2024 — AtomNet 회고 평가). 다만 발굴된 hit의 실제 진전·임상 진입은 표적별로 편차가 크며, 회사의 자체 파이프라인(예: ATX-01 결절성 경화증)은 2024년 시점 초기 단계입니다.
- **Anti-COVID-19 main protease 저해제 — Nirmatrelvir의 SBDD 사례**: Pfizer의 PF-07321332(nirmatrelvir, Paxlovid의 활성 성분)는 SARS 시기에 개발된 protease 저해제 lineage에서 X-ray 구조 기반 의약화학으로 빠르게 최적화되었습니다(Owen et al., *Science*, 2021). 순수 AI 발굴 사례는 아니나, **구조 기반 설계의 속도와 효과**를 보인 동시대 표준 사례로 인용됩니다.
- **Insilico Medicine — Chemistry42 + INS018_055(IPF)**: Day 2에서 다룬 TNIK 표적에 대해 Chemistry42 생성 모델로 후보 분자를 설계하고 18개월 만에 임상 후보까지 도달, 2024년 Phase IIa 진입이 보도되었습니다(Ren et al., *Nature Biotechnology*, 2024).
- **Exscientia — DSP-1181 등 임상 진입 분자**: 2020년 DSP-1181(OCD 적응증, 일본 Sumitomo와 협업)을 12개월 만에 후보 도출 → 임상 I상 진입이 발표되었으며, 산업이 인용하는 "AI 발굴 분자의 첫 임상 진입" 사례입니다. 다만 동 분자는 후속 임상에서 진전이 멈춘 것으로 보도되었으며(2022년경), 사례를 인용할 때 임상 결과의 한계를 함께 명시하는 것이 정확합니다.

### 전통 HTS와 AI/가상 스크리닝의 비교

| 항목 | 전통 HTS | AI 기반 가상 스크리닝 |
|------|---------|---------------------|
| 평가 화합물 수 | 10⁵~10⁶ | 10⁹~10¹¹ |
| 1차 비용(스크리닝 자체) | ~$1~5M | ~$10K~$100K(컴퓨팅) |
| Hit rate | 0.01~0.1% | 표적 의존, 통상 5~30%(실험 검증 hit) |
| 신규 chemotype | 라이브러리 한정 | make-on-demand로 신규 골격 가능 |
| 실험 검증 부담 | 보유 라이브러리 합성 불필요 | 100~500개 합성·assay 필요 |
| 약점 | 표적과 무관한 일반 라이브러리 | 도킹 score의 정확도 한계 |

> 위 비용·hit rate는 산업 보고 평균이며 표적·라이브러리에 따라 큰 변동이 있습니다. 약학 전공자는 회사의 마케팅 자료에서 인용되는 hit rate를 **표적 종류·검증 기준·라이브러리 출처**와 함께 해석해야 합니다.

## 창업 관점

Phase 1 단계이므로 두 문장으로 짚습니다. 가상 스크리닝은 진입장벽이 비교적 낮아(Enamine REAL·ZINC22 등 공개 라이브러리, AutoDock Vina·DiffDock 등 오픈소스 도킹) **특정 표적군(예: kinase, GPCR) 또는 특정 적응증(예: 항감염, 희귀병)을 좁혀 hit-finding-as-a-service** 형태로 진입하는 것이 약학 전공 창업자가 검토할 만한 출발점이며, 본격적 BM·경쟁사 비교는 Phase 5에서 다룹니다.

## 오늘의 과제

1. **공개 도킹 도구로 한 표적에 5개 분자 점수 비교 (45분)**: AutoDock Vina(또는 웹 기반 SwissDock, DockThor) 등 무료 도구로 자신이 관심 있는 표적 하나(PDB ID 결정 — 예: COX-2 1CX2, EGFR 1M17)와 그에 결합하는 알려진 약물 5종(예: celecoxib, rofecoxib + 무관 분자 2개)을 도킹한 결과의 점수를 비교합니다. 결과 표에 (a) 도킹 점수, (b) 알려진 IC50, (c) 점수 순위와 실제 활성 순위의 일치 여부, (d) 불일치한 이유 가설을 1페이지로 정리합니다. "도킹 점수는 가설 생성기이지 측정기가 아니다"가 직접 체감되어야 합니다.
2. **AI 가상 스크리닝 회사 2곳의 차별점 분석 (40분)**: Atomwise, Schrödinger, Iktos, Exscientia, Insilico 중 2곳을 선택해 그들의 (a) 라이브러리 출처(자체/Enamine/ZINC), (b) 핵심 모델 유형(CNN/도킹/생성/QSAR), (c) 공개된 임상 진입 분자 수, (d) BM(SaaS/협업/자체 파이프라인)을 1페이지 비교표로 작성합니다. "약학 전공 창업자라면 어느 쪽 접근을 모방하지 말아야 하는가"를 한 문단으로 결론짓습니다.
3. **Hit triage 체크리스트 설계 (30분)**: 가상 스크리닝 결과에서 화학자에게 넘기기 전 적용할 Triage 체크리스트를 (a) 약물성(Lipinski 5/QED), (b) PAINS·reactive group 필터, (c) 합성 가능성, (d) IP 신규성, (e) 표적/적응증 특이적 약학 판단(예: BBB 투과 필요 여부, 간 1차 통과 회피 등)으로 구분하여 작성합니다. 각 항목의 임계값과 그 근거(논문/규칙)를 명시하고, 약학 전공자가 비전공자보다 강한 판단 영역이 무엇인지 표 아래 3줄로 정리합니다.

## 참고 자료

- Lyu, J. *et al.* (2019). "Ultra-large library docking for discovering new chemotypes." *Nature*, 566, 224-229. — 1.7억 분자 도킹으로 D4·AmpC 신규 활성 발굴.
- Sadybekov, A.A. *et al.* (2022). "Synthon-based ligand discovery in virtual libraries of over 11 billion compounds." *Nature*, 601, 452-459. — 110억 규모 ultra-large screening의 sigma-2 사례.
- Hopkins, A.L. *et al.* (2004). "Ligand efficiency: a useful metric for lead selection." *Drug Discovery Today*, 9, 430-431. — LE 지표의 표준 출처.
- Baell, J.B. & Holloway, G.A. (2010). "New substructure filters for removal of pan assay interference compounds (PAINS) from screening libraries." *Journal of Medicinal Chemistry*, 53, 2719-2740. — PAINS 필터의 원전.
- Owen, D.R. *et al.* (2021). "An oral SARS-CoV-2 Mpro inhibitor clinical candidate for the treatment of COVID-19." *Science*, 374, 1586-1593. — Nirmatrelvir 구조 기반 설계 사례.
- Enamine REAL Space (https://enamine.net/compound-collections/real-compounds) — 수십억 규모 make-on-demand 라이브러리 카탈로그.
- ZINC22 database — UCSF Shoichet Lab 운영, ultra-large VS의 표준 라이브러리.
