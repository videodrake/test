# Day 5: Week 2 종합 — 생성 모델의 실용성과 한계

> 이전 학습(Phase 2 Week 2 Day 1~4)에서 De Novo 분자 설계의 패러다임 전환(검색→생성), 잠재 공간 기반 모델(VAE/GAN), 강화학습 기반 목표지향 생성, 그리고 3D 확산 모델(Diffusion·SBDD)을 차례로 학습했습니다. Week 2를 마무리하는 오늘은 — 세 계열을 **하나의 의사결정 프레임**으로 묶고, 산업 현장에서 반복적으로 부딪히는 **실용성과 한계**(합성성, 평가 신뢰도, 신규성 vs 약물 유사성, IP 적격성)를 정면으로 다룹니다. 이 프레임은 다음 주(Week 3: ADMET) 학습 시 — 생성된 분자에 어떤 약학적 게이트를 어떤 순서로 적용할 것인지 결정하는 데 그대로 사용됩니다.

## 개요

분자 생성 모델 선택은 — Week 1의 물성 예측 모델 선택과는 본질적으로 다른 의사결정입니다. 예측 모델은 "주어진 분자의 라벨을 맞추는 정확도"라는 단일 지표로 비교할 수 있지만, 생성 모델은 **validity·uniqueness·novelty·약물 유사성·합성성·IP 적격성**이라는 6개 축의 trade-off로 평가됩니다. VAE/GAN은 분포 모사에 강하지만 목표 지향이 약하고, RL은 보상 함수를 따라가지만 보상 해킹(reward hacking)에 취약하며, 확산 모델은 3D·SBDD에서 강력하지만 합성성이 낮은 분자를 생성하기 쉽습니다. 본 Day는 (1) 세 계열을 **여섯 축**에서 비교하고, (2) **표적 정보·데이터 규모·자원**을 입력으로 받는 통합 의사결정 트리를 정립하며, (3) 산업 현장의 4대 한계(평가의 환상, 합성성, MPO의 함정, IP 적격성)를 직시한 뒤, (4) 약학 전공 창업자가 이 단계에서 가지는 차별적 우위를 정리합니다.

## 핵심 개념

### 1) 분자 생성 모델 — 한 장의 비교표

지난 4일간 학습한 세 가지 모델 계열을 6개 축에서 비교합니다.

| 축 | VAE/GAN (Day 2) | RL/MPO (Day 3) | 확산 모델 (Day 4) |
|-----|------------|-----------|------------|
| 학습 패러다임 | 분포 학습 p(mol) | 보상 최대화 E[R(mol)] | 노이즈→복원 학습 |
| 입력 표현 | SMILES / 그래프 / 잠재 벡터 | SMILES / 그래프 (autoregressive) | 2D 그래프 / 3D 좌표 |
| 목표 지향성 | 약 (사후 latent 조작 필요) | 강 (보상 함수 직접 반영) | 중~강 (조건부 가이던스) |
| 3D·포켓 조건 | 거의 불가 | 어려움 (별도 reward 필요) | **표준 지원** (TargetDiff 등) |
| 학습 비용 | 중 (수 GB / 수 일) | 중~고 (보상 계산 병목) | 고 (T=1,000 step, GPU 5~14일) |
| 합성성 (SAscore) | 우수 (ChEMBL 분포 모사) | 변동적 (보상 해킹 시 악화) | 보통~낮음 (3D 강조 시 더 악화) |
| 신규성 (novelty) | 낮음 | 중~높음 | **매우 높음** |
| 대표 모델 (2023~2024) | JT-VAE, MoFlow, MolGAN | REINVENT 4, ChemRL, GFlowNet | TargetDiff, DecompDiff, EDM |

**핵심 관찰** — 합성성과 신규성은 거의 반비례 관계입니다. 학습 데이터(ChEMBL) 분포를 모사하면 합성은 쉽지만 IP 신규성은 낮고, 분포에서 멀어지면 IP는 강해지지만 wet lab에서 만들 수 없는 분자가 대량 생성됩니다. 이 trade-off가 — 본 Day 후반부에서 다룰 산업의 4대 한계의 근원입니다.

### 2) 통합 의사결정 트리 — 표적·데이터·자원에서 모델로

산업 현장에서 De Novo 생성 모델을 고를 때 답해야 할 4개 질문:

```
[질문 1] 표적 단백질의 3D 구조가 있는가?
  ├─ YES → 질문 2
  └─ NO  → VAE/RL (리간드 기반) 우선

[질문 2] 알려진 활성 리간드 데이터가 충분(N ≥ 100)한가?
  ├─ YES → RL + docking reward 또는 확산 + post-filter
  └─ NO  → 확산 모델 (SBDD, TargetDiff/DecompDiff)

[질문 3] 목적이 "신규 IP"인가, "빠른 hit 확보"인가?
  ├─ 신규 IP   → 확산 또는 RL (분포 이탈 허용)
  ├─ 빠른 hit  → VAE + 가상 라이브러리 도킹

[질문 4] 가용 자원(GPU·시간·인력)은?
  ├─ < 1 GPU·1주    → 사전학습 VAE wrapper (JT-VAE, REINVENT pretrained)
  ├─ 1~4 GPU·1개월  → RL fine-tuning + 자체 보상 함수
  └─ 8+ GPU·수개월  → 확산 모델 자체 사전학습 또는 DecompDiff 재학습
```

이 트리는 — Day 1에서 그린 "검색→생성" 패러다임 지도의 실전 활용판입니다. 약학 전공자는 **질문 3**에서 차별적 판단을 내릴 수 있습니다 — 빅파마와 라이선스 협상을 노린다면 신규성이 IP 가치를 결정하지만, 자체 wet lab 검증 비용을 줄이려면 합성성·유사 chemotype이 더 중요합니다.

### 3) 평가의 환상 — 4개 표준 지표가 동시에 거짓말하는 구조

생성 모델 평가의 표준 4지표:

| 지표 | 정의 | 환상의 원인 |
|------|------|---------|
| Validity | RDKit 파싱 성공률 | 100%여도 "약 같은" 분자라는 보장 없음 |
| Uniqueness | 중복 없는 분자 비율 | 의미 없는 변형(메틸기 위치만 다름)도 unique로 카운트 |
| Novelty | 학습셋과 겹치지 않는 비율 | scaffold가 같으면 사실상 동일 chemotype |
| 약물 유사성 (QED, Lipinski) | 약물 분포 통계 적합도 | 통계 통과 ≠ 실제 약물성 |

산업의 표준 대응 — **MOSES, GuacaMol** 벤치마크는 위 4개를 다층적으로 결합하지만, 여전히 **wet lab과의 상관계수는 0.4~0.6 수준**(MOSES 2024 분석)입니다. 즉 in silico 평가에서 SOTA를 달성해도 실험에서 활성을 보이지 않는 사례가 절반에 가깝습니다. 이는 — 생성 모델의 본질적 한계가 아니라, **현재 평가 체계가 약학적 의미를 충분히 반영하지 못한다**는 의미입니다.

### 4) MPO의 함정 — 보상 함수 설계 실패의 4가지 전형

Day 3에서 다룬 RL/MPO는 — 보상 함수 설계에서 실패하면 모델 전체가 의미 없는 출력을 냅니다. 산업에서 반복적으로 관찰되는 4가지 전형:

| 함정 | 증상 | 약학적 원인 |
|------|------|---------|
| **보상 해킹** | QED·SAscore는 만점인데 실제 활성 없음 | 보조 보상이 주 보상을 압도 |
| **dominator 문제** | LogP 보상이 너무 강해 분자가 모두 작아짐 | 보상 정규화 없음 |
| **multi-objective collapse** | 가중 평균이 한쪽 극단으로 쏠림 | Pareto front 무시 |
| **임상 무관 최적화** | hERG, CYP를 제대로 페널티하지 않음 | 임상 실패 기준 미반영 |

약학 전공자의 우위는 — 임상 가이드라인(FDA, ICH S7B 등)을 보상 함수에 어떻게 코드화할지 판단하는 데 있습니다. 비전공자가 모든 ADMET 지표를 균등 가중하는 동안, 약학 전공자는 — "이 표적에서 hERG IC50 > 10 μM는 hard constraint, CYP3A4 억제는 soft penalty"처럼 임상 단계의 go/no-go를 보상에 직접 반영할 수 있습니다.

## 작동 원리와 아키텍처

산업에서 실제로 운영되는 통합 De Novo 파이프라인 (2024년 표준):

```
[De Novo 분자 생성 — 산업 표준 파이프라인]

1. 표적·요구사항 정의 (약학 전공 영역)
   - 표적 단백질, binding pocket(있다면)
   - Target Product Profile (TPP): 적응증, 투여 경로, 경쟁 약물
   - go/no-go 기준: pIC50 > 8, hERG > 10μM, CYP3A4 < 50% inhibition 등

2. 생성 엔진 (Day 1~4의 통합)
   - 1차: 확산 모델 (TargetDiff) 또는 RL fine-tuned (REINVENT 4)
   - 1 round당 1,000~10,000 분자 sampling

3. 필터링 cascade (Week 1의 cascade 아키텍처 재사용)
   a. 화학 검증: RDKit valence, salt 제거, tautomer 표준화
   b. 약물 유사성: Lipinski Ro5, QED ≥ 0.5
   c. 합성성: SAscore ≤ 4, AiZynthFinder 역합성 성공
   d. ADMET 예측: hERG, CYP, BBB, hepatotoxicity (Week 3 영역)
   e. 도킹·MD: Glide / AutoDock Vina, top 1% MD refinement

4. 다양성·scaffold 클러스터링
   - Murcko scaffold 기준 클러스터링
   - 클러스터별 대표 분자 5~10개 선정
   - 합성 우선순위 결정 (약학 전공자의 판단)

5. Wet lab 검증
   - 1차: SPR 결합 친화도 (수십 분자)
   - 2차: enzymatic / cell-based assay (수 분자)
   - 3차: in vivo PK / efficacy (1~2 분자)

GPU 1장 기준
- 1 round (생성→필터→docking 1,000개): 약 6~12시간
- 1 cycle (생성→wet lab feedback→재학습): 약 4~8주
- 표적 1개당 hit→lead 전환: 약 6~12개월
```

핵심 설계 결정 — 약학 전공자의 차별 영역:

| 결정 | 선택지 | 약학적 판단 기준 |
|------|------|------------|
| 보상 함수 가중치 | 균등 / 임상 기반 / Pareto | 임상 단계 go/no-go를 반영 |
| 신규성 vs 합성성 균형 | scaffold 새로움 / 합성 즉시 | IP 전략과 wet lab 자원 |
| ADMET 게이트 임계값 | 일반 / 적응증 특화 | 표적 조직, 투여 경로 고려 |
| Wet lab 우선순위 | SPR / 세포 / 동물 | 표적의 검증 가능성과 비용 |

## 신약개발 적용

2023~2024년 De Novo 생성 모델의 산업 적용을 — Day 1~4에서 다룬 회사의 통합 관점에서 정리합니다.

| 회사 | 1차 패러다임 | 임상 진입 후보 (~2024) | 차별 포인트 |
|------|--------|-----------------|-------|
| **Insilico Medicine** | RL (Chemistry42) + 3D 확산 통합 | INS018_055 (IPF, Phase II), Rentosertib (USP1, Phase I) | PandaOmics(타겟) + Chemistry42(분자) 통합 |
| **Iambic Therapeutics** | 자체 3D 확산 (NeuralPlexer + SBDD) | IAM1363 (HER2, Phase 1, 2024) | 단백질-리간드 공진화 |
| **Genesis Therapeutics** | GEMS (RL + GNN + 확산 혼합) | GTAEXS617 (CDK7, Phase 1, 2024) | Eli Lilly·BMS 라이선스 |
| **Recursion** | LOWE (생성 + phenomics) | REC-994 (CCM, Phase 2), REC-2282 (NF2, Phase 2/3) | 표현형 스크리닝 결합 |
| **Isomorphic Labs (DeepMind)** | 비공개 (AlphaFold3 + 확산) | 임상 후보 미공개, Lilly/Novartis 협업 | 단백질 구조 예측-생성 통합 |

핵심 패턴 — **단일 모델로 임상 후보를 만든 사례는 사실상 없습니다.** 모두 RL + 확산 + 사전학습 + cascade ADMET을 조합한 멀티모델 파이프라인이며, 그 위에 약학적 의사결정 게이트가 정렬되어 있습니다.

성공 사례 vs 실패 사례 — 2024년 시점:

| 항목 | 성공 사례 (Insilico INS018_055) | 알려진 실패 사례 (다수 익명) |
|------|------------------------|----------------------|
| 생성 → 임상 진입 시간 | 약 30개월 | 60개월 이상 |
| 단계별 dropout | 표준 수준 (~50% Phase I) | Phase I 진입 전 좌초 |
| 핵심 성공 요인 | 멀티모델 + 약학적 게이트 정렬 | in silico 평가만 신뢰 |
| 핵심 실패 요인 | (성공) | wet lab 활성 불일치, 합성 불가 |

수치 기준 비교 (생성 모델 vs 전통 HTS):

| 비교 항목 | 전통 HTS | De Novo + cascade |
|------|--------|------------------|
| 1차 hit rate | 약 0.01~0.1% | 약 1~10% (in silico) |
| 1차 비용 | 수십만~수백만 달러 | 수천~수만 달러 |
| 신규 chemotype 확률 | 낮음 | 중~높음 |
| Wet lab 활성 일치율 | 거의 100% (실제 측정) | 약 30~60% (예측-실측) |

결론은 명확합니다 — **De Novo는 hit 후보 수를 폭발적으로 늘리지만, wet lab 검증의 부담은 그대로**입니다. 약학 전공자의 우위는 — 어떤 in silico hit을 wet lab으로 보낼지 결정하는 게이트 설계에 있습니다.

## 창업 관점

De Novo 분자 생성은 — Phase 2 Week 1(QSAR 예측)보다 진입 장벽과 가치가 모두 높은 영역입니다. 1인 약학 창업자의 합리적 진입 전략은 — Day 4에서 다룬 wrapper SaaS 모델을 **Week 3에서 다룰 ADMET 게이트와 결합한 통합 De Novo 플랫폼**으로 발전시키는 경로입니다. 직접 모델을 학습하지 않고 — REINVENT 4 / TargetDiff / DecompDiff 사전학습 가중치를 활용하되, 약학 전공자의 보상 함수 설계·ADMET 게이트 정렬·임상 적응증 특화를 차별 포인트로 가져갑니다. 가격은 표적당 PoC 약 5~10만 달러, 월 SaaS 약 5,000~10,000 달러가 현실적이며, 2024년 SBDD AI 시장(약 5억 달러, BCC Research)의 약 1~3% 점유가 첫 ARR 목표가 됩니다. 빅파마와의 정면 경쟁은 어렵지만 — **특정 표적 family + ADMET 통합 + 임상 단계 go/no-go의 약학적 코드화**라는 3중 차별점은 도메인 깊이로만 만들 수 있는 해자입니다.

## 오늘의 과제

1. **통합 비교 정리 — 세 패러다임의 의사결정 매트릭스 작성 (40분)**: 본 Day의 6축 비교표를 본인의 표적 1개(Week 2 Day 4 과제에서 선정한 표적)에 적용하세요. (a) VAE/RL/확산 중 1차 선택, (b) 그 선택의 근거 3가지(표적 구조 유무, 데이터 규모, 자원), (c) 1차 선택이 실패할 경우 2차 fallback 전략을 1페이지 표로 정리하세요. 마지막에 — "이 표적에서 보상 함수의 hard constraint 3개와 soft penalty 3개"를 약학적 근거와 함께 작성하세요.

2. **비즈니스 분석 — 산업 4대 한계의 현실적 영향 평가 (50분)**: Insilico Medicine, Iambic Therapeutics, Genesis Therapeutics 3사의 2024년 공개 발표·논문·인터뷰를 검색하여 — (a) 평가의 환상, (b) 합성성, (c) MPO의 함정, (d) IP 적격성 중 어떤 한계가 각 회사의 핵심 도전 과제로 언급되는지 표로 정리하세요. 약학 전공 1인 창업자가 wrapper SaaS로 진입한다면 — 이 4대 한계 중 가장 약학적 우위로 해결 가능한 것은 무엇이고, 어떤 기능을 제품에 우선 탑재할지 2~3문장으로 작성하세요.

3. **종합 리서치 — Week 2 통합 보고서 작성 (60분)**: "De Novo 분자 생성: 약학 전공 창업자를 위한 14일 MVP 설계" 제목의 2페이지 보고서를 작성하세요. 구성: (a) Day 1~4의 세 패러다임 1줄 요약과 본 Day의 통합 트리, (b) 본인이 선택한 표적·1차 모델·보상 함수 hard/soft constraint, (c) 필터링 cascade 5단계와 각 단계의 약학적 게이트, (d) Wet lab 검증 우선순위 트리, (e) 14일 안에 만들 MVP의 기능 목록 5~7개와 차별 포인트. 마지막에 — "다음 주(Week 3: ADMET)에서 어떤 게이트를 추가 학습해 본 파이프라인에 통합할지" 예고 1문단을 포함하세요.

## 참고 자료

- **벤치마크**: Polykovskiy, D. et al. "Molecular Sets (MOSES): A Benchmarking Platform for Molecular Generation Models." *Frontiers in Pharmacology*, 2020. — 생성 모델 평가의 사실상 표준.
- **벤치마크**: Brown, N. et al. "GuacaMol: Benchmarking Models for de Novo Molecular Design." *J. Chem. Inf. Model.*, 2019. — 목표지향 생성의 표준 평가.
- **논문**: Gao, W., Coley, C. W. "The Synthesizability of Molecules Proposed by Generative Models." *J. Chem. Inf. Model.*, 2020. — 합성성 한계의 정량적 분석.
- **논문**: Walters, W. P., Murcko, M. "Assessing the Impact of Generative AI on Medicinal Chemistry." *Nature Biotechnology*, 2020. — De Novo 생성의 산업적 한계와 기대.
- **오픈소스**: REINVENT 4 (AstraZeneca), TargetDiff (guanjq4/targetdiff), JT-VAE (wengong-jin/icml18-jtnn). — Day 1~4에서 다룬 세 계열의 표준 구현체.
- **분석 보고서**: BCC Research. "AI in Drug Discovery Market Report" (2024). — SBDD AI 시장 규모와 전망(확인 필요).

다음 주(Week 3: ADMET) 예고 — 본 Week에서 만든 De Novo 후보들을 **임상 실패의 60%를 막는 ADMET 게이트**에 통과시키는 5일을 학습합니다. 흡수·분포·대사·배설·독성의 각 영역에서 약학 전공자의 차별적 판단이 어떻게 모델 설계와 게이트 임계값에 반영되는지 — Week 2에서 만든 cascade 파이프라인 위에 직접 쌓아 올립니다.
