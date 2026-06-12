# Day 1: De Novo 분자 설계 — AI가 신약을 디자인한다

> 이전 학습(Phase 2 Week 1)에서 주어진 분자의 물성을 **예측**하는 QSAR 4세대 모델(ECFP+XGBoost → D-MPNN GNN → MolFormer-XL Foundation Model)을 다뤘습니다. 오늘부터 Week 2에서는 — **분자를 직접 생성**하는 패러다임으로 넘어갑니다. 첫 Day는 "AI가 분자를 처음부터 디자인한다"는 De Novo 설계의 정의·문제 정의·전체 지형도를 그리고, 이번 주에 다룰 VAE/GAN(Day 2), 강화학습(Day 3), 확산 모델(Day 4), 종합·한계(Day 5)의 학습 동선을 한 장에 묶습니다.

## 개요

De Novo 분자 설계(De Novo Molecular Design)란 — 기존 라이브러리에서 후보를 **고르는 것이 아니라**, AI가 원하는 물성을 만족하는 분자를 **새로 그려내는** 기술입니다. 이는 신약개발 패러다임을 "탐색(search)에서 생성(generation)으로" 전환시킨 핵심 변화입니다. 약학 전공자에게 이 주제는 단순한 새 기술이 아니라 — Hansch 회귀로 시작된 QSAR의 60년 역사가 "구조-활성 관계를 **읽기**"에서 "구조-활성 관계로부터 **쓰기**"로 진화한 결정적 분기점입니다. 화학 공간(chemical space)의 크기는 약 10^60개로 추정되며, 우주의 별보다 많은 이 공간을 라이브러리 스크리닝만으로는 결코 다 탐색할 수 없습니다. De Novo AI는 이 무한 공간에서 — **약물 유사 영역**(drug-like region, 추정 10^23~10^33개)만 골라 생성하도록 학습된 확률 모델입니다.

## 핵심 개념

### 1) 화학 공간과 탐색의 한계

**화학 공간(Chemical Space)**은 이론적으로 합성 가능한 모든 유기 분자의 집합입니다. 분자량 500 Da 이하의 **약물 유사 분자(drug-like molecule)** 공간만 보아도 약 10^33개로 추정되며(Reymond, 2010), 이는 전 우주 원자 수(약 10^80)에는 못 미치지만 — 인류가 합성한 모든 분자(약 10^8개) 대비 10^25배 큽니다. ZINC22(약 376억 개), Enamine REAL Space(약 480억 개) 같은 **make-on-demand 라이브러리**조차 이 공간의 10^-22 비율에 불과합니다.

전통적 신약개발의 한계는 명확합니다 — **수십억 개 가상 스크리닝(virtual screening)**으로 후보를 추리지만, 이는 결국 "주어진 라이브러리 안에서의 검색"입니다. 약학 전공자의 관점에서 보면 — 이는 **약전(Pharmacopoeia)에 등재된 약물만으로 새 적응증을 찾는 것**과 구조적으로 같은 한계입니다. 라이브러리에 없는 chemotype은 결코 발견되지 않습니다.

### 2) De Novo 설계 — 검색에서 생성으로의 전환

De Novo 설계의 본질은 — 화학 공간을 **이산적 후보의 집합**이 아니라 **연속적 확률 분포**로 모델링하는 것입니다. AI 모델은 "약물 같은 분자의 분포 p(molecule)"을 학습한 뒤, 그 분포에서 새로운 sample을 뽑아냅니다. 약학에서 익숙한 개념으로 옮기면 — 이는 **모집단 약동학(population PK)**에서 개별 환자의 PK 파라미터가 분포로부터 추정되는 것과 같은 구조입니다. 개별 환자(분자)는 다르지만, 모집단(약물 유사 공간)의 분포는 학습 가능합니다.

생성 모델이 답해야 하는 세 가지 질문:

| 질문 | 의미 | 평가 지표 |
|------|------|--------|
| **Validity** | 화학적으로 valid한 분자인가? | RDKit 파싱 성공률 (%) |
| **Uniqueness** | 중복 없이 다양하게 생성하는가? | 1만 개 중 고유 분자 비율 |
| **Novelty** | 학습 데이터에 없던 분자인가? | 학습셋과의 Tanimoto < 0.4 비율 |

여기에 더해 — **Synthesizability(SAscore, SCScore)**, **Drug-likeness(QED, Lipinski)**, **목표 물성 만족도**(pIC50, logS 등)가 평가됩니다. Validity 100% + Uniqueness 95% + Novelty 80%가 산업적 베이스라인입니다(2023~2024 generative benchmark).

### 3) 세 가지 생성 패러다임 — Week 2의 지도

이번 주에 다룰 생성 모델은 크게 세 계열입니다.

| 패러다임 | 대표 모델 | 핵심 아이디어 | 약학적 비유 |
|---------|---------|----------|----------|
| **잠재 공간 기반 (Day 2)** | VAE, JT-VAE, GAN | 분자를 연속 잠재 벡터로 인코딩 → 디코딩 | 약물 분류표(ATC)의 좌표화 |
| **목표 지향 RL (Day 3)** | REINVENT, GCPN | 보상 함수로 원하는 물성을 향해 분자 진화 | MPO(다중 파라미터 최적화)의 자동화 |
| **확산 모델 (Day 4)** | EDM, DiffSBDD, MolDiff | 노이즈에서 분자를 단계적 복원 | 결정 구조 X-선 회절상의 역재구성 |

세 계열은 경쟁이 아니라 **상호 보완**입니다. 실제 산업 파이프라인은 — VAE로 화학 공간을 탐색하고, RL로 목표 물성에 맞춰 강화하며, 확산 모델로 3D 입체화학까지 고려하는 다단계 조합이 일반적입니다(Insilico Chemistry42 플랫폼).

### 4) Week 1(예측)과 Week 2(생성)의 결합 — Generation × Evaluation Loop

De Novo 설계는 단독으로는 작동하지 않습니다. **생성 → 평가 → 선택 → 재생성**의 루프가 핵심입니다. 여기서 평가에 쓰이는 모델이 바로 — Week 1에서 학습한 ECFP+XGBoost, MolFormer-XL 등의 **물성 예측 모델**입니다.

```
[De Novo 분자 설계 루프 — 의사 아키텍처]

1. Prior 생성 모델: ChEMBL 1.5M 약물 유사 분자로 사전학습
   → "약물 같은" 분자의 분포 p(mol) 학습
2. 목표 정의: pIC50(target) > 7, hERG < 5, logP 1~4, SAscore < 4
3. 생성 루프:
   a. Prior에서 N=10,000 분자 샘플링
   b. 물성 예측기(Week 1 모델)로 각 분자 평가
   c. 보상 함수: R = w1·pIC50 + w2·(10-hERG) - w3·SAscore
   d. R 상위 분자로 생성 모델 fine-tuning (RL 또는 conditional)
   e. 반복
4. 출력: 합성 가능, 안전, 활성 분자 100~1,000개 후보 풀
```

이 루프에서 **약학 전공자의 우위가 명확하게 발생하는 지점**은 — (a) 보상 함수 설계 시 어떤 물성을 어떤 가중치로 넣을지(예: 중추신경계(CNS) 약물이면 BBB 투과성 가중치 ↑, 항암제면 PK 가중치 ↓), (b) 생성된 분자의 약리학적 의미 평가(특정 작용기가 hERG 채널과 결합 가능한 형태인지 직관 판단), (c) MPO에서 trade-off의 임상적 우선순위 결정입니다.

## 작동 원리와 아키텍처

De Novo 시스템의 5개 구성 요소:

| 모듈 | 역할 | 대표 구현 |
|------|------|--------|
| **분자 인코딩** | SMILES/Graph/3D → 모델 입력 | RDKit, OEChem |
| **생성 백본** | 확률 분포 p(mol) 학습 | VAE / RL / Diffusion |
| **물성 예측기** | 생성 분자의 평가 | Week 1 모델 재사용 |
| **합성 가능성 평가** | SAscore, RetroSynth | AiZynthFinder, ASKCOR |
| **다양성 필터** | Tanimoto, Bemis-Murcko scaffold | RDKit `DiversityFilter` |

이 5개 모듈이 **루프 하나의 약 6~24시간**(GPU 1장 기준) 안에 한 사이클을 돌고, 5~10 사이클을 반복하면 — 초기 무작위 prior로부터 목표 물성에 수렴한 후보 분자 풀을 얻습니다(REINVENT 4.0 벤치마크 기준, 2023~2024).

핵심 설계 결정:

| 결정 항목 | 선택지 | Phase 2 Week 2에서 다룰 위치 |
|---------|-------|-----------------------|
| 분자 표현 | SMILES 문자열 / Graph / 3D 좌표 | Day 2(SMILES VAE), Day 4(3D Diffusion) |
| 학습 패러다임 | 자가지도(VAE/Diffusion) / RL / Conditional | Day 2, Day 3, Day 4 각각 |
| 목표 통제 방식 | Conditional 입력 / 보상 함수 / Classifier-guidance | Day 3(RL), Day 4(guidance) |
| 합성 가능성 | 후처리 필터 / 학습 단계 통합 | Day 5에서 종합 |

## 신약개발 적용

산업 현장의 De Novo 사례 — 모두 2022~2024년 실제 보고된 데이터입니다.

| 회사 | 모델 / 플랫폼 | 적응증 / 결과 | 시간·비용 |
|------|------------|-----------|--------|
| **Insilico Medicine** | Chemistry42 (RL + GAN + 물성 예측 통합) | IPF(특발성 폐섬유증) 치료제 후보 INS018_055, Phase 2 진입 | 타겟 발굴~IND 18개월, 약 260만 달러 (전통 대비 1/10) |
| **Iktos** | Makya (Conditional generation) | 다수 제약사 협업, 1차 후보 발굴까지 평균 6~9개월 단축 | 미공개 |
| **BenevolentAI** | LLM + 분자 생성 결합 | BEN-2293(아토피 피부염, Phase 2) | 발굴~후보 지정 약 12개월 |
| **Recursion Pharmaceuticals** | 표현형 스크리닝 + 생성 | REC-994(뇌해면체 기형, Phase 2) | 발굴 단계 비용 약 60% 절감 보고 |
| **Exscientia (현 Recursion 자회사)** | Centaur Chemist (AL + 생성) | DSP-1181(강박장애, 임상 진입한 최초의 AI 설계 분자, 2020) — 이후 임상 1상 중단(2022) | 발굴~IND 12개월 |

DSP-1181 사례는 — De Novo AI의 한계를 보여주는 동시에 중요한 교훈입니다. **생성 단계의 성공이 임상 단계의 성공을 보장하지 않습니다**. 약학 전공자의 관점에서 — 이는 in vitro/in silico 활성과 in vivo PK·효능·안전성의 간극이 AI 시대에도 여전히 존재한다는 사실을 재확인시킨 사례로 평가됩니다.

기존 방법 대비 정량 비교:

| 단계 | 전통 HTS+의 medicinal chemistry | De Novo AI |
|------|----------------------|---------|
| 타겟~lead 발굴 시간 | 4~6년 | 12~18개월 |
| 합성·평가 분자 수 | 2,000~5,000 | 50~200 |
| 평균 비용 | 약 3,000만~5,000만 달러 | 약 200만~500만 달러 |
| 후속 임상 성공률 | Phase 1 진입 후 약 10% | 데이터 부족(2024 기준 IND 진입 분자 약 20여 개) |

## 창업 관점

De Novo 분자 설계는 — 2024년 기준 AI 신약개발 시장에서 **가장 자본이 집중된 영역**입니다(Bio-IT World 2024 보고서, "Generative chemistry" 카테고리에 약 13억 달러 누적 투자). 그러나 — 이는 동시에 가장 경쟁이 치열한 영역이기도 하며, Insilico·Recursion·Schrödinger·Exscientia 같은 풀스택 플레이어가 이미 자리 잡고 있습니다.

약학 전공 풀스택 창업자의 차별화 포인트:

1. **특화 적응증·특화 표적**: 일반 De Novo 플랫폼이 약한 영역 — 예: CNS 약물(BBB 투과성), peptide-like 분자(분자량 500~1500 Da), 공유 결합 저해제(covalent inhibitor). 약학 전공자가 이 영역의 평가 기준을 더 정교하게 설계 가능.
2. **루프의 평가 모듈에 약학적 깊이 추가**: 생성 모듈은 오픈소스(REINVENT 등) 활용, 평가 모듈에 자체 ADMET·PK 모델 통합. 이번 Phase 2 Week 3(ADMET)에서 다룰 내용이 핵심 자산이 됩니다.
3. **MVP 후보**: 특정 단백질 family(예: kinase, GPCR) 한정 De Novo 생성 SaaS — 약학적 도메인 지식이 보상 함수와 후처리 필터에 직접 코드화된 형태.

비즈니스 모델 선택지는 — (a) 자체 파이프라인(Insilico 모델, 자본 집약적), (b) 빅파마 협업(Iktos 모델, 마일스톤 수익), (c) SaaS/플랫폼(Schrödinger 모델, ARR 기반)으로 나뉘며, 약학 전공+바이브코딩 1인 창업 단계에서는 (c)의 좁고 깊은 버티컬이 현실적 진입점으로 평가됩니다.

## 오늘의 과제

1. **기초 학습 — REINVENT 4.0 논문 핵심 파악(40분)**: "REINVENT 4: Modern AI–driven generative molecule design" (Loeffler et al., 2024, J. Cheminformatics) Abstract와 Section 2(Architecture)를 읽고 — (a) Prior 모델이 어떤 데이터로 학습되는지, (b) 보상 함수 구성, (c) Validity/Uniqueness/Novelty 보고 수치 3가지를 1페이지로 정리하세요.

2. **비즈니스 분석 — De Novo 플랫폼 3사 비교(40분)**: Insilico Medicine(Chemistry42), Iktos(Makya), Schrödinger LiveDesign의 공식 웹사이트에서 (a) 타겟 산업, (b) 가격 모델(SaaS / 마일스톤 / 파이프라인), (c) 임상 진입 후보 수를 정리하여 표로 만드세요. 약학 전공자로서 어느 모델의 어느 모듈에 가장 큰 차별화 여지가 있는지 2~3문장 의견을 덧붙이세요.

3. **리서치 정리 — DSP-1181 사례 분석(60분)**: 2020년 임상 진입 → 2022년 중단된 DSP-1181(Exscientia × Sumitomo Dainippon)의 (a) 발굴 과정, (b) 임상 1상 결과, (c) 중단 사유를 가능한 출처에서 종합하여 1페이지로 정리하세요. "AI De Novo 설계의 한계는 어디서 발생했는가"에 대한 본인의 약학적 가설 1개를 결론에 포함하세요.

## 참고 자료

- **논문**: Loeffler, H. H. et al. "REINVENT 4: Modern AI–driven generative molecule design." *Journal of Cheminformatics*, 2024. — De Novo RL 생성의 산업 표준 오픈소스 프레임워크.
- **논문**: Reymond, J.-L. "The Chemical Space Project." *Accounts of Chemical Research*, 2015. — 약물 유사 화학 공간 크기 추정의 기반 논문.
- **회사**: Insilico Medicine — Chemistry42 플랫폼(공개 자료 및 2023~2024 사례). https://insilico.com
- **벤치마크**: GuacaMol, MOSES — 생성 모델 평가의 표준 벤치마크 (Brown et al., 2019; Polykovskiy et al., 2020).
