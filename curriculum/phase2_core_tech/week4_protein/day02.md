# Day 2: 단백질-리간드 도킹 — 열쇠와 자물쇠

> 이전 학습(Phase 2 Week 4 Day 1)에서 AlphaFold가 단백질 구조 예측의 판을 바꾸는 과정을 다뤘습니다. 오늘은 그 예측 구조 위에서 — 실제 약물 후보가 어떤 자세(pose)로, 얼마나 강하게 결합하는지 예측하는 **단백질-리간드 도킹(protein-ligand docking)** 기술을 학습합니다.

## 개요

단백질-리간드 도킹은 — 표적 단백질의 결합 부위(binding site)에 리간드(주로 소분자)를 계산적으로 배치해 *결합 자세(pose)*와 *결합 친화력(binding affinity)*을 예측하는 기술입니다. 1982년 Kuntz 등의 **DOCK** 프로그램으로 시작된 이 분야는 — 40년간 구조 기반 신약설계(Structure-Based Drug Design, SBDD)의 중심 도구였으며, 2019년 *Nature*에 발표된 *ultra-large library docking*(약 1.7억 화합물 대상)이 5-HT2A 수용체에서 신규 진통 후보를 발굴하며 재부흥기를 열었습니다. 도킹은 크게 **검색 알고리즘(pose 생성)**과 **스코어링 함수(affinity 순위)**로 구성되며, 두 요소 모두가 여전히 정확도의 병목입니다. 약학 전공자에게 본 Day의 의미는 — 도킹이 예측하는 값(binding pose·score)이 *실제 약효(IC50, EC50)와 왜 자주 어긋나는가*를 이해하고, 결과를 *약리학적으로 해석*하는 안목을 기르는 것입니다.

## 핵심 개념

### 1) 도킹의 정의와 역사

도킹은 두 가지 하위 문제로 나뉩니다.

- **Pose prediction (자세 예측)** — 리간드가 결합 부위에 어떤 3D 배치로 앉는가.
- **Scoring / affinity prediction (친화력 예측)** — 그 자세가 얼마나 안정한가, 결합 상수(Kd, Ki, IC50)는 얼마인가.

**Kuntz의 DOCK (1982, *J. Mol. Biol.*)** — 결합 부위를 구(sphere) 세트로 근사하고 리간드 원자를 기하학적으로 매칭한 최초의 도킹 프로그램입니다. 이후 40년간 아래 세대가 등장했습니다.

| 세대 | 대표 프로그램 | 특징 | 라이선스 |
|------|-------------|------|---------|
| 1세대 (1980~90년대) | DOCK, FlexX | 강체·조각 기반 | 학술/상용 혼재 |
| 2세대 (2000년대) | **AutoDock 4/Vina**, **Glide**, GOLD | Lamarckian GA·경험적 스코어링 | Vina: BSD, Glide: 상용 |
| 3세대 (2010년대) | Smina, rDock, Chemgauss | Vina 파생·QSAR 통합 | 대부분 오픈 |
| 4세대 (2020~) | **DiffDock**, Uni-Mol Docking, **Boltz-1/AF3** | 딥러닝·확산 모델 | 오픈/학술 |

이 중 **AutoDock Vina (Trott & Olson, 2010, *J. Comput. Chem.*)**는 — 오픈소스에 무료이면서 준수한 정확도로 학계·산업 표준이 되었고, 상용에서는 **Schrödinger Glide**가 정확도와 워크플로우로 사실상의 표준입니다.

### 2) 검색 알고리즘 — pose를 어떻게 찾는가

리간드의 conformation 공간은 회전 가능한 결합(rotatable bond) 수에 지수적으로 증가하므로 — 완전 탐색은 불가능합니다. 실제 도킹은 아래 알고리즘 조합을 사용합니다.

| 알고리즘 | 예시 도구 | 원리 | 강점 · 약점 |
|--------|--------|------|-----------|
| **유전 알고리즘(GA)** | AutoDock 4, GOLD | 자세 집단을 세대별로 교차·돌연변이 | 국소해 탈출 · 느림 |
| **몬테카를로(MC)** | AutoDock 4 (MC leg) | 무작위 이동 + Metropolis 수용 | 단순 · 수렴 느림 |
| **점진적 조립(Incremental)** | FlexX, DOCK 4 | 리간드를 조각 단위로 결합 부위에 성장 | 빠름 · 조각 순서 의존 |
| **경사 기반 최적화** | Vina | 반복 국소 탐색 + 무작위 재출발 | 빠르고 강건 · 스코어링 함수에 의존 |
| **딥러닝(확산·GNN)** | DiffDock, Uni-Mol Docking | 조건부 3D 좌표 생성 | 학습 데이터 편향 · 검증 필요 |

핵심은 — *어떤 알고리즘이든 스코어링 함수가 지시하는 지형(landscape)을 탐색*한다는 점입니다. 스코어링이 부정확하면 pose 검색은 잘못된 최소값으로 수렴합니다.

### 3) 스코어링 함수 — 친화력을 어떻게 매기는가

스코어링 함수는 크게 3가지로 나뉩니다.

| 유형 | 예시 | 원리 | 약학적 함의 |
|------|------|------|-----------|
| **역장(Force field) 기반** | DOCK, AutoDock | van der Waals + 정전기 + H-bond의 물리 항 합 | 물리적으로 해석 가능, entropy 무시 |
| **경험적(Empirical)** | Vina, Glide SP/XP, ChemPLP | 실험 Kd 데이터로 가중치 회귀 | 훈련셋 편향, 표적군 밖에서 저조 |
| **지식 기반(Knowledge-based)** | DrugScore, ITScore | PDB 통계 (원자쌍 거리 분포) | 데이터 풍부 표적에 강함 |
| **딥러닝 기반** | RTMScore, Gnina CNN, DeepDock | 3D CNN·GNN이 pose→affinity 학습 | 규모의 이점 · 해석성 낮음 |

**Vina scoring function**은 대표적으로 다음 항을 선형 결합합니다.
- Gauss1·Gauss2 (van der Waals 근사)
- Repulsion (충돌 페널티)
- Hydrophobic (소수성 접촉)
- Hydrogen bond (거리·각도 가중)
- 회전 가능 결합 수 페널티 (엔트로피 근사)

> **약학적 함의** — 스코어링 함수는 *결합 자유에너지 ΔG*를 근사하지만, 대개 *엔트로피·수용액 환경·conformational 재조정*을 단순화합니다. 그래서 도킹 스코어와 실제 IC50의 상관계수는 표적·데이터셋에 따라 대개 R² ≈ 0.3~0.5 수준입니다. 도킹은 *순위(ranking)*에는 유용해도 *절대 친화력*은 신뢰하기 어렵습니다.

### 4) 도킹의 3가지 표준 작업

| 작업 | 설명 | 성공 기준 |
|------|------|---------|
| **재도킹(Redocking)** | 실험 결정 구조의 리간드를 빼고 다시 도킹 | RMSD ≤ 2 Å |
| **교차 도킹(Cross-docking)** | 서로 다른 리간드-단백질 결정 쌍에서 pose 교환 예측 | RMSD ≤ 2 Å, 훨씬 어려움 |
| **가상 스크리닝(Virtual screening)** | 대규모 라이브러리에서 hit 후보 순위화 | Enrichment factor(EF), ROC AUC |

**PoseBusters (Buttenschoen et al., 2024, *Chemical Science*)** — 최근 도입된 표준 벤치마크로, 단순 RMSD를 넘어 *물리적 타당성*(결합 길이, 각도, 겹침, 카이랄성)까지 검증합니다. 딥러닝 도킹의 과대평가를 억제한 계기가 되었습니다.

### 5) 도킹의 근본 한계 — 왜 여전히 어려운가

| 한계 | 원인 | 대응 |
|------|------|-----|
| **단백질 유연성** | 결정 구조는 특정 conformer만 반영 | ensemble docking, induced-fit, MD 사전 준비 |
| **물 분자 처리** | 결합 부위 물 분자가 결합 매개·매체 역할 | WaterMap, 3D-RISM, 명시 물 모델 |
| **엔트로피 근사** | 결합 시 리간드·단백질·용매의 엔트로피 손실 | FEP, ligand strain penalty |
| **스코어링-활성 상관** | 표적별 편향, 훈련 데이터 한정 | consensus scoring, MM-GBSA/PBSA 재스코어 |
| **거대 시스템** | 막단백질·복합체·PROTAC | 하이브리드 MD-도킹, 로컬 도킹 |

> **약학적 판단 지점** — 도킹 결과를 실험 없이 그대로 신뢰하는 순간 실패합니다. *약학 전공자의 우위*는 — pose가 이미 알려진 co-crystal 리간드의 핵심 상호작용(예: hinge H-bond in kinase, ionic bond in aminergic GPCR)을 재현하는지, *pharmacophore*와 일치하는지 판단하는 능력에서 나옵니다.

## 작동 원리와 아키텍처

전형적인 도킹 파이프라인 — 1인 창업자가 바이브코딩으로 재현 가능한 최소 구성:

```
[입력 → 처리 → 출력]

1. 표적 준비 (Protein preparation)
   - PDB 또는 AlphaFold 구조 다운로드
   - 물·이온·co-solvent 제거, 결측 잔기 보정, 결측 side chain 재구성
   - 프로톤화(pH 7.4 표준), tautomer 결정
   - 결합 부위 상자(box) 정의 — 알려진 co-crystal ligand 중심 ±10 Å

2. 리간드 준비 (Ligand preparation)
   - SMILES → 3D conformer 생성 (RDKit, OMEGA)
   - 프로톤화 상태·tautomer·stereochem 열거
   - 부분전하 할당 (Gasteiger, AM1-BCC)

3. 도킹 실행
   - 예: Vina — box 지정 + exhaustiveness 8, num_modes 10
   - 상위 10 pose × 라이브러리 크기 개의 후보 생성

4. 재스코어링 (Rescoring)
   - MM-GBSA (Prime, gmx_MMPBSA)
   - CNN 재스코어링 (Gnina)
   - Consensus (Vina + Glide + ChemPLP 평균)

5. 필터링 · 약학적 검증
   - PoseBusters 물리 타당성 필터
   - 알려진 co-crystal pharmacophore 재현 여부
   - PAINS·독성·ADMET 필터 (Day 31~34에서 다룬 기준)

6. 출력
   - 순위화된 hit 후보 + pose PDB + 스코어 + 약학적 해석 메모
```

핵심 설계 결정 — 표적·데이터·예산에 따른 트레이드오프:

| 결정 | 선택지 | 추천 (1인 창업자 기준) | 이유 |
|------|--------|------|------|
| 도킹 엔진 | Vina / Glide / Gnina / DiffDock | **Vina + Gnina 재스코어** | 오픈·무료, 준수한 baseline |
| 단백질 유연성 | 강체 / side-chain 유연 / ensemble | side-chain 유연 시작 | 계산량과 정확도 균형 |
| 스코어링 | 단일 / consensus / ML 재스코어 | consensus + ML 재스코어 | Vina 단독의 편향 완화 |
| 라이브러리 규모 | 수천 (초점) / 수백만 (스크리닝) | 목적에 맞게 이원화 | hit ID는 대규모, 최적화는 초점 |
| 클라우드 | CPU / GPU / 하이브리드 | 하이브리드 (Vina는 CPU, ML은 GPU) | 비용 최적 |

## 신약개발 적용

도킹이 실제로 신규 후보를 발굴한 대표 사례 — 최근 5년의 흐름을 중심으로.

| 사례 | 표적 | 접근 | 결과 |
|------|------|------|------|
| **Ultra-large library docking (Lyu et al., 2019, *Nature*)** | AmpC β-lactamase, D4 도파민 수용체 | 약 1.7억 화합물 Vina 도킹 | 549개 합성, D4 부분작용제 hit rate 24% |
| **초대규모 확장 (Sadybekov et al., 2022, *Nature*)** | σ2·5-HT2A 수용체 | 약 3억 화합물 도킹 | 신규 진통 후보 발굴, 후속 임상 진행 |
| **Deep Docking (Ton et al., 2020, *Molecular Informatics*)** | SARS-CoV-2 Mpro 외 | 딥러닝 대리모델로 라이브러리 필터, 최종 도킹 | 약 10억 화합물 실효 스크리닝 시간 40배 단축 |
| **Insilico Medicine INS018_055 (IPF)** | Novel target(공개 후 확인) | AF2 예측 구조 + Chemistry42(도킹·생성 통합) | 18개월 만에 IND, Phase 2 임상 진행 |
| **Isomorphic Labs + AF3 (2024)** | 다중 표적 | AF3 통합 예측(단백질+리간드) | Eli Lilly·Novartis 파트너십 총 약 17억 달러 규모 |
| **Cyclica (Recursion 인수, 2023)** | 다중 표적 프로파일링(polypharmacology) | 프록시 리간드 기반 표적 예측 | Recursion 플랫폼에 통합 |

**기존 방법 대비 정량 비교** — 히트 발굴 단계:

| 항목 | 실험 HTS (High-Throughput Screening) | 도킹 기반 가상 스크리닝 | AI 도킹 (2024~) |
|------|-----------------|---------------------|---------------------|
| 라이브러리 규모 | 약 10⁵~10⁶ (실물 화합물) | 약 10⁷~10⁹ (가상 화합물) | 약 10⁸~10¹⁰ |
| 비용 (표적 1개) | 수십만~수백만 달러 | GPU 시간 수천~수만 달러 | GPU 시간 수천 달러 |
| 소요 시간 | 3~6개월 | 1~4주 | 며칠~1주 |
| Hit rate (평균) | 약 0.01~0.1% | 약 1~30% (연구별) | 약 20~76% (PoseBusters 기준) |
| 필요 인프라 | 로봇·화합물 라이브러리 | CPU/GPU 클러스터 | GPU 클러스터 |

**산업적 통찰** — 도킹은 단독 도구가 아니라 *AI 신약개발 파이프라인의 필터 레이어*로 재정의되고 있습니다. 최근 워크플로우는 — (a) **분자 생성**(Day 26~29)으로 후보 라이브러리 자동 확장, (b) **도킹**으로 표적 결합 pose·score 필터링, (c) **ADMET**(Day 31~34)로 개발성 필터링, (d) **MD·자유에너지 계산**(Day 41 예정)으로 상위 후보 재검증 — 순으로 결합됩니다. 도킹의 부활은 *AlphaFold의 구조 공급*과 *ultra-large library의 화학공간 확장*이 만든 재조합의 결과입니다.

## 창업 관점

도킹 SaaS는 이미 Schrödinger·OpenEye·CCG 등 30년 된 상용 기업이 존재하지만 — 1인 창업자의 진입 기회는 *특화 워크플로우*와 *약학적 후처리*에 있습니다. **약학 전공자의 차별 포지셔닝** — (a) *표적군별 도킹 프리셋*(kinase, GPCR, protease 등 각 표적군의 hinge·orthosteric·catalytic 사이트별 최적 파라미터·재스코어링 조합), (b) *약리학 해석 자동화*(pose가 알려진 pharmacophore를 재현하는지, off-target 유사 pocket과의 selectivity 리스크는 어느 정도인지 리포트), (c) *AF-도킹-ADMET 통합 파이프라인*(AlphaFold 구조 → 도킹 → ADMET → 후보 순위) — 세 방향이 유력합니다. MVP는 — Vina + Gnina 재스코어 + PoseBusters 물리 필터를 백엔드로, 표적 하나에 특화된 웹 UI(예: kinase panel 도킹 리포트)를 상단에 얹는 형태가 현실적이며, 학술 랩·중소 바이오의 월 500~5,000 달러 시장이 초기 진입점입니다. 기존 상용 대비 *가격·특화·해석 리포트*의 3축에서 차별화 가능하며, Boltz-1·DiffDock 등 오픈 딥러닝 도킹을 옵션으로 병렬 제공하면 *2세대(Vina)와 4세대(딥러닝) 도킹의 consensus*라는 차별점을 확보할 수 있습니다.

## 오늘의 과제

1. **본인 관심 표적 재도킹 실습 (60분)**: 본인 관심 표적의 co-crystal 구조 1개(PDB에서 선택, 가능하면 hinge H-bond가 명확한 kinase 또는 orthosteric 결합 GPCR)를 선정해 — (a) 리간드를 제거한 뒤 AutoDock Vina로 재도킹, (b) 상위 10 pose의 RMSD를 결정 구조 대비 계산, (c) 성공/실패 pose의 차이를 시각화해 원인을 3가지로 정리하세요. 결론으로 — 본인 SaaS에서 *어떤 표적·리간드 조합이 도킹으로 안전한가*의 사전 판정 기준을 명시하세요.

2. **가상 스크리닝 규모별 비용·성능 시뮬레이션 (40분)**: 라이브러리 규모별(10³, 10⁶, 10⁸, 10¹⁰) 도킹 스크리닝의 — (a) GPU/CPU 시간 예상, (b) 예상 hit 수(대략 hit rate 1% 가정), (c) 클라우드 비용(AWS·GCP 온디맨드 기준) 을 표로 작성하세요. Enamine REAL Space(약 60억 화합물)와 ZINC22를 참고하고 — Deep Docking 같은 대리모델 적용 시 예상 비용 절감을 함께 계산하세요.

3. **도킹-실제 활성 상관 리서치 (50분)**: 최근 2년 리뷰 또는 벤치마크 논문 1편(예: PoseBusters 논문 또는 CACHE Challenge 결과)을 읽고 — (a) 현재 도킹 스코어와 실제 IC50/Kd의 상관계수 (R², Spearman ρ) 의 표적군별 편차, (b) 딥러닝 도킹(DiffDock, Uni-Mol Docking, Boltz-1)이 물리 기반 도킹 대비 실제로 개선한 부분, (c) 여전히 실패하는 케이스 를 1페이지로 정리하세요. 결론으로 — 본인 SaaS에서 도킹 스코어를 *어떤 방식으로 사용자에게 표시*해야 오해를 줄일지(예: 순위만 노출, 절대값 숨김, 신뢰 대역 표시 등) 제안하세요.

## 참고 자료

- **AutoDock Vina**: Trott, O., Olson, A.J. "AutoDock Vina: Improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading." *Journal of Computational Chemistry*, 2010. — 오픈 도킹의 사실상 표준, 스코어링 함수 설계.
- **Ultra-large library docking**: Lyu, J. et al. "Ultra-large library docking for discovering new chemotypes." *Nature*, 2019. — 약 1.7억 화합물 도킹의 최초 대규모 성공 사례.
- **Sadybekov et al.** "Synthon-based ligand discovery in virtual libraries of over 11 billion compounds." *Nature*, 2022. — 초대규모 확장, σ2·5-HT2A 신규 진통 후보.
- **PoseBusters**: Buttenschoen, M., Morris, G.M., Deane, C.M. "PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences." *Chemical Science*, 2024. — 딥러닝 도킹의 물리 타당성 검증 표준.
- **Deep Docking**: Ton, A.T. et al. "Rapid Identification of Potential Inhibitors of SARS-CoV-2 Main Protease by Deep Docking of 1.3 Billion Compounds." *Molecular Informatics*, 2020. — 대리모델 기반 도킹 가속.
- **Kuntz DOCK**: Kuntz, I.D. et al. "A geometric approach to macromolecule-ligand interactions." *Journal of Molecular Biology*, 1982. — 도킹 분야의 원조 논문.
- **Schrödinger Glide**: www.schrodinger.com — 상용 도킹 표준.
- **Gnina**: github.com/gnina/gnina — CNN 재스코어링 오픈소스.
- **DiffDock**: Corso, G. et al. "DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking." *ICLR 2023*. — 확산 모델 기반 도킹의 시초.
