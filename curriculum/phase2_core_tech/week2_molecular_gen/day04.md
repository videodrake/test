# Day 4: 확산 모델과 3D 생성 — 최신 생성 패러다임

> 이전 학습(Day 3, P2W2D3)에서 강화학습 기반 목표지향 생성 — 보상 함수에 약학적 우선순위를 코드화하여 정책을 갱신하는 패러다임 — 을 다뤘습니다. 그러나 SMILES·그래프 기반 RL은 **2D 위상(topology) 수준**의 분자를 만드는 데 머무릅니다. 실제 약물-표적 결합은 **3차원 공간의 형상(geometry)과 정전기적·소수성적 상호작용**으로 결정됩니다. 오늘은 2022~2024년 분자 생성의 새 표준으로 자리 잡은 **확산 모델(Diffusion Model)과 3D 분자/리간드 생성**을 학습합니다. 약학 전공자에게 이 주제가 결정적인 이유는 — 확산 모델이 처음으로 **단백질 결합 포켓(binding pocket)을 직접 조건으로 받아 그 안에 들어갈 3D 분자를 생성**할 수 있게 되었고, 이는 SAR·docking·약물-수용체 상호작용에 대한 약학적 직관이 알고리즘의 출력에 직접 반영되는 영역이기 때문입니다.

## 개요

**확산 모델(Diffusion Model)**은 — 데이터에 가우시안 노이즈를 점진적으로 더하는 **순방향 과정(forward process)**과, 노이즈로부터 원본을 복원하도록 신경망을 학습시키는 **역방향 과정(reverse process)**으로 구성된 생성 모델입니다. 2020년 Ho et al.의 DDPM(Denoising Diffusion Probabilistic Models)이 이미지 영역에서 GAN을 추월한 이후, 2022~2024년 분자 영역으로 빠르게 이식되어 — 2D 그래프, 3D 좌표, 단백질 포켓 조건부, 단백질-리간드 공진화 등 다양한 변형이 산업·학술계의 표준이 되었습니다. 핵심 차별점은 — VAE/GAN/RL과 달리 **3D 좌표를 직접 생성 변수**로 다루며, **E(3)/SE(3) equivariance**(회전·평행이동 대칭) 같은 물리적 제약을 아키텍처에 내재화할 수 있다는 점입니다. 이는 약학적으로 — 단백질-리간드 결합 자세, pharmacophore의 3D 배치, 입체화학(stereochemistry)을 처음부터 다루는 패러다임이라는 의미입니다. 2024년 기준 구조 기반 약물 설계(structure-based drug design, SBDD) 분야 De Novo 신규 논문의 약 60% 이상이 확산 모델 기반입니다.

## 핵심 개념

### 1) 확산 모델의 기본 틀 — Forward / Reverse 과정

확산 모델의 학습 목표는 — 데이터 분포 p_data(x)에 점진적으로 가우시안 노이즈를 더하여 표준 정규분포 N(0, I)로 보내는 forward process를 정의하고, 그 역과정을 학습하는 것입니다.

**Forward (noising)**: T-step 동안 데이터에 noise를 더함.

```
x_0 ~ p_data
x_t = √(α_t) · x_0 + √(1 - α_t) · ε,   ε ~ N(0, I)
```

여기서 α_t는 시간에 따라 0으로 감소하는 스케줄(보통 cosine 또는 linear). T가 충분히 크면 x_T ≈ N(0, I).

**Reverse (denoising)**: 신경망 ε_θ(x_t, t)가 x_t에 포함된 노이즈를 예측하도록 학습.

```
loss = E_{t, x_0, ε} [ ‖ε - ε_θ(x_t, t)‖² ]
```

샘플링 시에는 — x_T ~ N(0, I)에서 시작하여 step별로 노이즈를 제거(reverse SDE/ODE 적분)하면서 분자를 복원합니다. 약학적 직관으로 — **medicinal chemist가 단순한 scaffold에서 출발해 한 step씩 R-그룹·치환기를 정교화하는 과정**의 수학적 일반화로 이해할 수 있습니다. 다만 확산 모델은 단계마다 전체 분자를 동시에 업데이트한다는 점이 다릅니다.

### 2) 분자에 적용하기 — 2D 그래프 vs 3D 좌표

확산 모델을 분자에 적용하는 두 갈래:

| 접근 | 노이즈를 더하는 변수 | 대표 모델 | 핵심 이슈 |
|------|------------------|---------|---------|
| **2D 그래프 확산** | 원자 종류(one-hot)와 결합 행렬 | DiGress (Vignac et al., 2023, ICLR) | 이산(discrete) 변수의 확산을 어떻게 정의할지 |
| **3D 좌표 확산** | 원자별 (x, y, z) 좌표 + 원자 종류 | EDM, GeoDiff, MolDiff, EQGAT-diff | E(3)/SE(3) 대칭성 보존 |

이산 변수의 확산은 — 원소(C/N/O/...)나 결합 종류(single/double/...) 같은 카테고리에 가우시안 노이즈를 단순 적용할 수 없습니다. DiGress는 — **categorical diffusion**(절대 확률을 균등 분포로 점차 흡수)으로 해결합니다. 3D 좌표는 연속 변수이므로 표준 가우시안 확산이 가능하지만 — 단순 적용은 회전·평행이동에 대해 불변(invariant)이 깨집니다.

### 3) E(3)/SE(3) Equivariance — 3D 분자 생성의 핵심 제약

분자의 물리적 성질(에너지·결합 친화력)은 — 회전·평행이동에 대해 불변이어야 합니다. 단순 MLP/Transformer에 (x, y, z) 좌표를 그대로 넣으면 — 학습된 모델은 좌표계에 의존하는 인공적 패턴을 학습할 수 있습니다. 이를 막기 위해 — **E(3)/SE(3) equivariant network**(예: EGNN, SE(3)-Transformer)를 사용합니다.

핵심 아이디어:
- **스칼라 특성**(원자 종류, 거리)에 대해서는 **불변(invariant)** 출력
- **벡터 특성**(좌표, 변위)에 대해서는 입력 회전 시 출력도 같은 회전이 적용되는 **공변(equivariant)** 출력

EGNN(Satorras et al., 2021)의 message passing:

```
m_ij = φ_e(h_i, h_j, ‖x_i - x_j‖²)
x_i ← x_i + Σ_j (x_i - x_j) · φ_x(m_ij)
h_i ← φ_h(h_i, Σ_j m_ij)
```

거리(‖x_i - x_j‖²)는 회전 불변이고, 좌표 갱신은 (x_i - x_j) 벡터를 통해 공변성을 유지합니다. 약학적 비유 — pharmacophore feature(H-bond donor/acceptor, aromatic ring 등) 간의 **상대적 3D 거리·각도**가 활성을 결정하지 좌표계가 결정하지 않는 것과 동일한 사고입니다.

### 4) 무조건 vs 조건부 생성 — 단백질 포켓 조건부의 결정적 가치

확산 모델의 진짜 산업적 의미는 — **조건부 생성(conditional generation)** 에서 나옵니다.

| 조건 유형 | 의미 | 대표 모델 (2023~2024) |
|---------|------|----------------|
| 무조건 | 약물 유사 3D 분자 생성 | EDM (Hoogeboom et al., 2022) |
| 물성 조건 | 특정 dipole moment·HOMO-LUMO·QED | EDM 조건부 변형 |
| **단백질 포켓 조건** | 주어진 binding pocket에 맞는 리간드 | **DiffSBDD, TargetDiff, Pocket2Mol, DecompDiff** |
| 단편 조건 | scaffold·linker 고정, 나머지 생성 | DiffLinker, FragDiff |
| 단백질-리간드 공진화 | 단백질 + 리간드를 동시 생성 | Chroma, RFdiffusion 변형 |

특히 **포켓 조건부 생성**(Structure-Based Drug Design, SBDD)은 약학적으로 가장 큰 의미를 가집니다:

- 입력: 표적 단백질의 binding pocket 3D 좌표(예: PDB의 ATP 결합 자리)
- 출력: 그 포켓 안에 자연스럽게 들어가는 신규 3D 리간드(원자 종류 + 좌표 + 결합)
- 효과: 도킹·MD 시뮬레이션 없이 — 한 번의 forward sampling으로 결합 가능성이 높은 리간드 후보 생성

이는 — 약학에서 fragment-based drug discovery(FBDD)·de novo SBDD의 자동화이며, 전통적으로 medicinal chemist가 포켓 cavity를 보며 손으로 그리던 작업의 알고리즘화입니다.

### 5) 대표 3D 확산 모델 — EDM에서 TargetDiff까지

2022~2024년 핵심 모델의 계보:

| 모델 | 연도 | 핵심 기여 | 주요 한계 |
|------|------|---------|---------|
| **EDM (Equivariant Diffusion Model)** | 2022 | 무조건 3D 분자 생성의 시발점, E(3) equivariance + DDPM 결합 | 약물 유사 분자 한정, 포켓 미고려 |
| **GeoDiff** | 2022 | 분자의 **conformer(3D 형태) 생성** 특화 | topology 자체는 입력으로 받음 |
| **DiGress** | 2023 | 2D 분자 그래프의 categorical diffusion | 3D 정보 부재 |
| **DiffSBDD** | 2023 | 단백질 포켓 조건부 3D 리간드 생성(Schneuing et al.) | 단백질 측은 고정 |
| **TargetDiff** | 2023 (ICLR) | 포켓 좌표를 condition으로 받는 SE(3)-equivariant diffusion | 합성성 별도 후처리 필요 |
| **DecompDiff** | 2024 | scaffold + arm을 분리하여 단계적 생성, SBDD 성능 향상 | 학습 복잡도 ↑ |
| **MolDiff** | 2023 | 원자·결합·좌표를 동시 확산, valid 분자 비율 ↑ | |
| **DiffDock** | 2023 | 분자-단백질 도킹을 확산으로 재정의(이 영역은 Phase 2 Week 4에서 상세 다룸) | 도킹 특화, 생성은 아님 |

산업에서는 — **TargetDiff / DecompDiff / DiffSBDD 계열**이 SBDD의 실용적 기준선으로 정착했습니다.

## 작동 원리와 아키텍처

TargetDiff/DecompDiff 류 포켓 조건부 3D 생성의 표준 파이프라인:

```
[포켓 조건부 3D 분자 확산 — 2024 산업 표준 아키텍처]

1. 입력 준비
   - 표적 단백질 PDB → binding pocket 추출
     · 알려진 리간드 주변 6~10Å 잔기, 또는
     · Fpocket/CASTp로 자동 검출된 cavity
   - 포켓 원자: (좌표, 원자 종류, residue ID)
   - 리간드 사전 정보: 원자 수 N(분포에서 샘플링, 보통 15~40)

2. Noise 초기화
   - 리간드 원자별 (x, y, z) ← N(0, I), 포켓 무게중심 부근으로 평행이동
   - 원자 종류 ← 균등 분포(C/N/O/S/F/Cl/...)

3. Reverse Diffusion (T = 1000 step 내외)
   매 step t에서:
   a. SE(3)-equivariant network(EGNN 또는 SE(3)-Transformer)에
      - 현재 리간드 (x_t, atom_type_t)
      - 포켓 좌표·원자 종류(고정 condition)
      를 입력
   b. ε_θ(x_t, atom_t, pocket, t) 예측 → x_{t-1}, atom_{t-1}로 한 step 역확산
   c. (선택) classifier-free guidance로 조건 강도 조절

4. 후처리
   - OpenBabel/RDKit으로 valence 정합성 체크, bond order 재구성
   - SAscore·QED·docking score(Glide/AutoDock Vina)로 재평가
   - 다양성 필터(scaffold 클러스터링)

GPU 1장(A100) 기준
- 학습 데이터: CrossDocked2020(약 100,000 단백질-리간드 쌍)
- 학습 시간: 약 5~14일
- 1 포켓당 분자 1,000개 샘플링: 1~4시간
- Validity: 약 90~95% (후처리 후)
- Vina docking score 평균 개선: pocket-specific baseline 대비 약 0.5~1.5 kcal/mol
```

핵심 설계 결정 — 약학 전공자의 의사결정 영역:

| 결정 항목 | 선택지 | 약학적 판단 기준 |
|---------|------|------------|
| 포켓 정의 범위 | 6Å / 8Å / 10Å | 핵심 상호작용(H-bond, π-stacking) 잔기 포함 여부 |
| 리간드 크기 분포 | 학습 데이터 평균 / 표적 특화 | 표적 family별 약물 분자 크기 통계 반영 |
| Validity 후처리 | RDKit / OpenBabel / 자체 룰 | tautomer·ionization state 통제 정확도 |
| 평가 기준 | docking score / MD / wet lab | 신뢰도-비용 trade-off |
| Guidance scale | 1.0 (무조건) ~ 3.0 (강조건) | 다양성 vs 포켓 적합도 균형 |

## 신약개발 적용

2023~2024년 산업·학술 적용 사례:

| 회사 / 기관 | 시스템 | 적용 / 결과 | 비고 |
|---------|------|---------|------|
| **Iambic Therapeutics** | NeuralPlexer + 자체 확산 SBDD | 표적 단백질 구조 + 리간드 동시 생성, 자사 파이프라인(다중 표적) | 2024년 시리즈 B 1억 달러 |
| **Genesis Therapeutics** | GEMS 플랫폼(생성 + ADMET) | Eli Lilly·BMS 협업, GTAEXS617(CDK7) Phase 1 진입(2024) | 다중 빅파마 라이선스 |
| **Insilico Medicine** | Chemistry42에 3D 확산 모듈 통합 | INS018_055(IPF) 등 파이프라인 가속 | 다중 패러다임 통합 |
| **MIT/Generate Biomedicines** | Chroma(2023) | 단백질 자체 확산 설계, 리간드와 공진화 가능성 | de novo 단백질-약물 동시 설계 |
| **DeepMind Isomorphic Labs** | 비공개 SBDD 확산 + AlphaFold 통합 | Eli Lilly(2024 약 17억 달러 계약), Novartis 협업 | |
| **DeepChem / 학계** | DiffSBDD, TargetDiff 오픈소스 | 학계 표준 비교군, 다수 코호트 재현 | MIT/Cambridge 협력 |

CrossDocked2020 벤치마크(2023~2024 보고치):

| 모델 | Vina docking score (kcal/mol, 낮을수록 좋음) | QED | SA score | Validity |
|------|---------|------|-----|-------|
| 무작위 분자 (baseline) | -5.5 | 0.45 | 5.0 | - |
| 학습 분포 평균 | -7.2 | 0.55 | 4.0 | - |
| AR(autoregressive SBDD, 2022) | -7.5 | 0.51 | 3.8 | 약 80% |
| Pocket2Mol | -7.8 | 0.57 | 3.5 | 약 88% |
| **TargetDiff (2023)** | **-8.1** | 0.48 | 3.6 | 약 90% |
| **DecompDiff (2024)** | **-8.4** | 0.50 | 3.4 | 약 93% |

해석:
- 확산 모델은 docking score 기준으로 autoregressive SBDD 대비 약 0.5~1.0 kcal/mol 우위(결합 친화력 약 3~10배에 해당).
- 그러나 — QED가 약간 하락하는 경향이 있어 **약물 유사성·합성성 후처리가 필수**입니다.
- 핵심 한계: docking score는 in silico 추정치이며, 실제 wet lab 활성과의 상관계수는 0.4~0.6 수준이라는 점은 RL과 동일합니다.

전통 SBDD(가상 스크리닝 + 도킹) 대비 비교:

| 비교 항목 | 전통 SBDD (도킹 기반 스크리닝) | 확산 모델 SBDD |
|------|-----------------|------------|
| 출발점 | 기존 라이브러리(ZINC, Enamine REAL 등) | de novo 신규 분자 |
| 화학 공간 탐색 | 약 10^9 ~ 10^11(라이브러리 한정) | 사실상 무한(생성 분포) |
| 1 포켓당 시간 | 며칠~수주(수십억 분자 도킹) | 1~4시간(GPU, 1,000 분자 생성) |
| 합성성 | 라이브러리는 합성 보장 | SAscore로 사후 필터링 필요 |
| 신규성(novelty) | 낮음(기존 분자) | 높음(IP 가치 ↑) |
| 검증 비용 | 동일(wet lab 단계 후) | 동일 |

따라서 — **확산 모델 SBDD는 "신규 IP를 빠르게 만들고 싶을 때" 최적**, **전통 가상 스크리닝은 "합성 즉시 가능한 알려진 분자"가 필요할 때 최적**입니다. 산업 현장은 — 두 접근을 병행하여 양쪽 결과를 cross-validate하는 흐름으로 정착하고 있습니다.

## 창업 관점

3D 확산 SBDD는 — 2024년 기준 AI 신약개발 영역에서 **기술 진입 장벽과 시장 가치가 모두 높은 영역**입니다. 이는 곧 — 진입은 어렵지만, 일단 신뢰할 수 있는 시스템을 구축하면 차별화 폭이 크다는 의미입니다.

**1) 진입 난이도 평가**: TargetDiff·DiffSBDD 같은 오픈소스 코드가 존재하지만 — (a) CrossDocked2020 같은 학습 데이터의 노이즈(다수가 실제 결합 자세가 아닌 docked pose), (b) E(3)-equivariant 학습의 안정성 이슈, (c) 평가가 docking score에 의존하는 본질적 한계 때문에 — production-grade 시스템 구축에는 6~12개월 + 인프라(A100 ×4 이상)가 현실적으로 필요합니다. 1인 약학 창업자가 처음부터 만들기는 어렵습니다.

**2) MVP 후보 — 포켓 친화적 wrapper SaaS**: 직접 모델을 학습하지 않고 — **공개된 TargetDiff/DecompDiff 사전학습 가중치를 활용한 SBDD wrapper SaaS**가 합리적 진입점입니다. 차별화는 — (a) 포켓 자동 검출 + 약학적 잔기 가중치 부여(예: 보존된 catalytic residue 강조), (b) 생성 분자에 대한 ADMET·CYP·hERG 예측 통합(Week 3에서 다룰 영역), (c) Vina/Glide 도킹 결과와 함께 medicinal chemist용 UI 제공. 가격은 프로젝트당 1~3만 달러 또는 월 SaaS 3,000~5,000달러 수준이 현실적입니다.

**3) 경쟁 구도와 차별화 기회**: Iambic·Genesis·Isomorphic Labs 등 자체 파이프라인을 가진 회사가 핵심 경쟁자입니다. 약학 전공 1인 창업이 정면 경쟁하기는 어렵지만 — (a) 특정 표적 family(GPCR, kinase, ion channel 중 하나)에 특화된 사전학습·평가 체계, (b) 합성 가능성·실제 wet lab 결합 친화력과의 alignment에 집중한 평가 모듈은 — 도메인 깊이로 차별화 가능합니다.

**4) BM과 시장 규모**: SBDD AI 시장은 2024년 약 5억 달러 규모, 2030년 약 25억 달러 전망(BCC Research 2024 인용)으로 — 분자 설계 SaaS 전체 시장의 약 30~40%를 차지합니다. 마진은 SaaS 표준 70~85% + 마일스톤 결합 모델 가능. 약학 전공 1인 창업의 첫 ARR 목표는 — 빅파마 1~2곳 PoC 계약 약 50~100만 달러 + 중소 바이오텍 SaaS 약 30만 달러 정도가 현실적 시나리오입니다.

## 오늘의 과제

1. **기초 학습 — 확산 모델의 분자 적용 핵심 파악(45분)**: Hoogeboom, E. et al. "Equivariant Diffusion for Molecule Generation in 3D" (*ICML*, 2022)의 Abstract와 Section 3(Method)를 읽고 — (a) E(3) equivariance를 어떻게 아키텍처에 내재화했는지, (b) 좌표 확산과 원자 종류 확산을 어떻게 결합했는지, (c) 약물 유사 데이터셋(QM9, GEOM-Drugs)에서 보고된 validity·uniqueness 수치를 1페이지로 정리하세요. 마지막에 — "이 무조건 생성 모델이 포켓 조건부로 확장될 때 어떤 추가 설계가 필요한가" 본인 의견 1문단을 덧붙이세요.

2. **비즈니스 분석 — 3D SBDD 플랫폼 3사 비교(40분)**: Iambic Therapeutics, Genesis Therapeutics, Insilico Medicine 3사의 (a) 공개된 3D 생성 모델 기반(확산 / autoregressive / 하이브리드), (b) 비즈니스 모델(자체 파이프라인 / 라이선스 / SaaS), (c) 최근 1~2년 마일스톤 또는 임상 진입 후보를 표로 정리하세요. 약학 전공 1인 창업자가 정면 경쟁이 아닌 **wrapper SaaS**로 진입한다면 — 어떤 표적 family(GPCR / kinase / ion channel / protein-protein interface 중 하나)를 첫 버티컬로 선택할지, 그 약학적 근거(임상적 중요도·기존 약물 시장·기술적 난이도)를 2~3문장으로 작성하세요.

3. **리서치 정리 — 포켓 조건부 생성의 약학적 검증 시나리오 설계(60분)**: 본인이 가장 잘 아는 표적 1개(예: EGFR T790M kinase domain, BACE1 active site, GLP-1R orthosteric pocket)를 선정하여 — (a) PDB ID와 핵심 결합 잔기 4~6개를 명시한 포켓 정의, (b) TargetDiff/DecompDiff로 100~1,000개 분자를 생성한다고 가정할 때 — Vina docking·QED·SAscore 외에 **약학적으로 추가해야 할 평가 지표 3~5개**(예: hERG IC50 예측, 표적 selectivity vs 동일 family 단백질, BBB 투과성 등), (c) 이 평가를 통과한 분자를 wet lab에서 검증할 때 어떤 assay 우선순위로 진행할지(SPR → enzymatic → cell-based) 단계별 의사결정 트리를 1.5~2페이지 보고서로 작성하세요. 마지막에 — "포켓 조건부 생성이 가장 잘 작동하지 않을 것 같은 표적 유형은 무엇이고, 그 이유는?" 1문단 분석을 포함하세요.

## 참고 자료

- **논문**: Hoogeboom, E., Satorras, V. G., Vignac, C., Welling, M. "Equivariant Diffusion for Molecule Generation in 3D." *ICML*, 2022. — 3D 분자 확산의 기준선, E(3) equivariance와 DDPM의 결합.
- **논문**: Guan, J. et al. "3D Equivariant Diffusion for Target-Aware Molecule Generation and Affinity Prediction (TargetDiff)." *ICLR*, 2023. — 단백질 포켓 조건부 SBDD의 대표 모델.
- **논문**: Guan, J. et al. "DecompDiff: Diffusion Models with Decomposed Priors for Structure-Based Drug Design." *ICML*, 2024. — scaffold/arm 분리 설계로 SBDD 성능 향상.
- **논문**: Schneuing, A. et al. "Structure-based Drug Design with Equivariant Diffusion Models (DiffSBDD)." (preprint 2023, Nature Computational Science 2024 발표). — SBDD 확산 모델의 또 다른 표준.
- **오픈소스**: TargetDiff GitHub(guanjq4/targetdiff), DiffSBDD(arneschneuing/DiffSBDD). — 학계 표준 비교군 코드.
- **벤치마크**: CrossDocked2020 (Francoeur et al., 2020). — SBDD 확산 모델 학습·평가의 사실상 표준 데이터셋.
