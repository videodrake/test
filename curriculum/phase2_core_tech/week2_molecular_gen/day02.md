# Day 2: VAE와 GAN — 분자 생성의 두 축

> 이전 학습(Day 1, P2W2D1)에서 De Novo 분자 설계의 정의 — 화학 공간을 "이산 집합 검색"이 아니라 "연속 확률 분포 p(mol) 학습"으로 전환한 패러다임 — 과 이번 주에 다룰 세 계열(잠재 공간 / RL / 확산)의 지도를 그렸습니다. 오늘은 그중 첫 번째 계열인 **잠재 공간 기반 생성(latent-space generative model)**의 두 축, 변분 오토인코더(VAE)와 생성적 적대 신경망(GAN)을 학습합니다. 두 모델은 "분자를 연속 벡터로 표현하고, 그 벡터 공간을 항행(navigate)하여 새 분자를 만든다"는 공통 아이디어를 다른 방식으로 구현하며 — 2018~2020년 De Novo 생성의 첫 황금기를 연 모델군입니다.

## 개요

**변분 오토인코더(Variational Autoencoder, VAE)**는 분자를 잠재 공간(latent space)의 연속 벡터로 압축한 뒤 다시 분자로 복원하는 모델이며, **생성적 적대 신경망(Generative Adversarial Network, GAN)**은 생성자(generator)와 판별자(discriminator)가 서로를 속이며 학습하는 이중 신경망입니다. 두 모델의 공통 목표는 — ChEMBL 같은 약물 유사 분자 집합의 분포 p(mol)을 학습하여, 그 분포에서 새로운 분자를 sampling하는 것입니다. 약학 전공자의 관점에서 이 잠재 공간 모델이 중요한 이유는 — 잠재 공간이 **물성 좌표화(property coordinatization)**를 가능하게 만들기 때문입니다. 즉, "logP가 높은 방향", "BBB 투과성이 높은 방향" 같은 약학적으로 의미 있는 축을 잠재 공간에서 발견할 수 있고, 이 축을 따라 분자를 이동시키는 것만으로도 — 마치 SAR 표를 따라 R-그룹을 치환하듯 — 분자 최적화가 가능합니다.

## 핵심 개념

### 1) 오토인코더와 변분 오토인코더(VAE)

**오토인코더(autoencoder)**는 입력 x를 저차원 표현 z로 압축(encoder)했다가 다시 x'로 복원(decoder)하는 신경망입니다. 분자에 적용하면 — SMILES 문자열을 256~512차원 벡터로 압축하고, 그 벡터로부터 다시 SMILES를 복원하는 구조입니다. 그러나 표준 오토인코더의 잠재 공간은 "점들의 집합"이며 — 학습되지 않은 영역의 z에서 디코딩하면 invalid한 분자가 나옵니다.

**VAE**는 이 문제를 — 잠재 공간을 **확률 분포**로 만들어 해결합니다. encoder는 각 분자에 대해 잠재 벡터 z가 아니라 **분포 q(z|x) = N(μ, σ²)**의 파라미터(평균 μ, 분산 σ²)를 출력합니다. 그리고 학습 시 — 이 분포가 표준 정규분포 N(0, I)에 가까워지도록 KL divergence 항으로 강제합니다. 결과적으로 잠재 공간이 **연속적이고 매끄러운(smooth) 다양체**가 되어, 임의의 z ~ N(0, I)에서 sampling해도 valid한 분자가 디코딩됩니다.

손실 함수는 두 항의 합입니다:

```
L = -E[log p(x|z)]  + β · KL(q(z|x) || N(0, I))
    재구성 항(reconstruction)  정규화 항(regularization)
```

약학적으로 익숙한 비유로 — 이는 **모집단 약동학(population PK)의 mixed-effects 모델**과 구조적으로 동일합니다. 개별 환자(분자)의 PK 파라미터(잠재 벡터)는 모집단 분포(prior N(0,I))로부터 추출되며, 모델은 그 분포의 파라미터(μ, σ)와 관측 데이터의 우도를 동시에 최적화합니다.

### 2) 분자 VAE의 진화 — ChemVAE에서 JT-VAE까지

| 모델 | 출시 | 분자 표현 | Validity | 핵심 기여 |
|------|------|---------|----------|----------|
| **ChemVAE** | 2018 (Gomez-Bombarelli) | SMILES (RNN) | 약 30~70% | 분자 잠재 공간의 첫 입증, 베이지안 최적화 결합 |
| **GrammarVAE** | 2017 (Kusner) | SMILES + CFG 문법 | 약 60% | 문법 규칙으로 invalid 출력 차단 |
| **JT-VAE (Junction Tree VAE)** | 2018 (Jin et al., MIT) | 분자 그래프 + 부분구조 트리 | **100%** | 분자를 "부분구조 어휘(vocabulary)"의 조합으로 재구성 |
| **MoFlow** | 2020 | 그래프 + Normalizing Flow | 100% | invertible 변환으로 정확한 likelihood 계산 |

**JT-VAE의 약학적 의미**는 — 분자를 **scaffold + functional group 어휘**의 조합으로 본다는 점입니다. 이는 약학 전공자가 SAR 분석에서 "Bemis-Murcko 골격 + R1, R2 치환기"로 분자를 분해하는 사고방식과 정확히 일치합니다. JT-VAE는 약 800개의 부분구조 어휘를 미리 학습한 뒤, 새 분자를 그 어휘의 트리 조합으로 생성하므로 — **invalid 분자가 원리적으로 발생할 수 없습니다**. 이것이 100% validity의 비결입니다.

### 3) 잠재 공간 항행(Latent Space Navigation) — 약학적 최적화의 자동화

VAE의 가장 큰 산업적 가치는 — 잠재 공간이 **연속적이고 미분 가능(differentiable)**하다는 점입니다. 따라서 다음과 같은 최적화가 가능합니다:

| 기법 | 작동 방식 | 약학적 비유 |
|------|---------|----------|
| **선형 보간(interpolation)** | 분자 A와 B의 z 벡터를 1:1로 섞으면 중간 분자 생성 | A와 B 사이의 hybrid scaffold 설계 |
| **베이지안 최적화(BO)** | 잠재 공간 위에서 Gaussian Process로 물성 최대화 | Free-Wilson 분석의 자동화 |
| **속성 방향 벡터(property direction)** | "logP +1 방향" 같은 축을 회귀로 발견 후 z를 그 방향으로 이동 | SAR matrix에서 한 축을 따라 R-그룹 변경 |
| **Conditional VAE (CVAE)** | encoder/decoder에 목표 물성을 입력으로 추가 | 처방 시 약물 선택 기준(target profile)을 입력으로 줌 |

Gomez-Bombarelli et al.(2018, ACS Central Science)는 — ChemVAE의 잠재 공간에서 베이지안 최적화로 QED·SAS·logP를 동시 최적화하여, ChEMBL에 없던 신규 약물 유사 분자를 생성했고 — 이는 **잠재 공간 자체가 medicinal chemist의 작업 평면(workbench)**이 될 수 있음을 처음 입증했습니다.

### 4) 생성적 적대 신경망(GAN)과 분자 적용

**GAN**은 두 신경망이 적대적으로 경쟁하며 학습합니다:
- **생성자(Generator) G(z)**: 랜덤 노이즈 z를 받아 "그럴듯한" 분자 생성
- **판별자(Discriminator) D(x)**: 입력 분자가 실제(real) vs 생성(fake)인지 판정

학습 목표는 — G가 D를 속이고, D가 G를 잡아내는 **min-max game**입니다. 학습이 수렴하면 G의 출력 분포가 실제 데이터 분포에 가까워집니다. 분자 적용은 다음과 같이 진화했습니다:

| 모델 | 출시 | 핵심 기법 | 한계 |
|------|------|---------|-------|
| **ORGAN** | 2017 (Guimaraes) | RL 보상(QED, SAS) + SeqGAN | mode collapse, 다양성 낮음 |
| **MolGAN** | 2018 (De Cao) | 그래프 GAN + RL | 작은 분자(9 atom 이하)만 안정 |
| **LatentGAN** | 2019 | 사전학습 VAE의 잠재 공간에서 GAN 학습 | VAE+GAN 결합, validity 개선 |
| **Mol-CycleGAN** | 2019 | unpaired translation: 활성 ↔ 비활성 분자 변환 | scaffold hopping 응용 |

GAN의 가장 큰 약점은 **mode collapse** — 생성자가 판별자를 속이는 몇 가지 분자 패턴에만 수렴하여, 다양성(uniqueness)이 급격히 떨어지는 현상입니다. 약학적으로 이는 — "활성 좋은 chemotype 하나만 반복 생성하는 모델"이 되어 — 신규 chemical series 탐색이라는 De Novo 본래 목적과 충돌합니다. 이 때문에 2021년 이후 산업에서는 — **순수 GAN보다는 VAE+GAN 결합(LatentGAN), 또는 GAN을 폐기하고 강화학습/확산 모델로 이동**하는 흐름이 우세합니다.

### 5) VAE vs GAN — 산업적 선택의 기준

| 평가 축 | VAE | GAN |
|--------|------|------|
| Validity (JT-VAE 기준) | 100% | 70~95% |
| 다양성(Uniqueness) | 높음 (정규화로 보장) | 낮음 (mode collapse 위험) |
| 학습 안정성 | 안정적 | 불안정, 하이퍼파라미터 민감 |
| 잠재 공간 해석 가능성 | 명확(연속/매끄러움) | 불명확 |
| Conditional 생성 | 자연스러움(CVAE) | 복잡(cGAN 별도 설계) |
| 산업 채택(2024) | 광범위 | 한정적 |

요약하면 — **분자 생성 영역에서 VAE는 표준, GAN은 보조**입니다. Schrödinger의 LiveDesign, Insilico의 Chemistry42, AstraZeneca의 REINVENT 모두 — VAE 또는 RNN 기반 prior를 핵심으로 쓰고, GAN은 보조 모듈 또는 학술 연구 단계에 머무는 양상입니다.

## 작동 원리와 아키텍처

분자 VAE 시스템의 표준 구성을 — JT-VAE 기준으로 정리하면:

```
[JT-VAE 시스템 — 의사 아키텍처]

입력: 분자 그래프 G = (V, E)
1. Junction Tree 추출
   - 분자를 ring + functional group 단위로 분해
   - 약 800개 사전 정의 어휘로 트리 T 구성
2. Tree Encoder: T → z_tree (트리 잠재 벡터, 56차원)
3. Graph Encoder: G → z_graph (그래프 잠재 벡터, 56차원)
4. 잠재 공간 z = [z_tree; z_graph] ~ N(μ, σ²)
5. Tree Decoder: z_tree → T' (트리 재구성)
6. Graph Decoder: z + T' → 부분구조 조합 → G' (분자 재구성)
출력: SMILES 문자열 (RDKit valid 100%)
```

**핵심 설계 결정 — JT-VAE 기준**:

| 결정 항목 | 선택 | 이유 |
|---------|------|------|
| 잠재 차원 | 28+28 = 56 | 너무 작으면 표현력 부족, 너무 크면 KL 정규화 약화 |
| β (KL 가중치) | 0.1~1.0 (annealing) | 학습 초기 재구성 우선, 후기 정규화 강화 |
| 어휘 크기 | 약 800 | ChEMBL 250k 분자의 부분구조 빈도 컷오프 |
| 디코딩 방식 | tree-then-graph | 거시 구조 → 미시 구조 순서로 일관성 확보 |
| 사전학습 데이터 | ChEMBL/ZINC15 약 250k | 약물 유사 분포 학습 |

학습은 — GPU 1장(A100 기준)에서 약 24~48시간, ChEMBL 250k 분자로 수렴합니다. 추론 시에는 z 한 개당 약 50ms로 분자 1개 생성 — 즉 GPU 1시간이면 약 7만 개 후보 생성이 가능합니다.

산업 파이프라인에서는 이 VAE 위에 — Day 1에서 본 **물성 예측기(Week 1의 ECFP+XGBoost, MolFormer 등)**를 결합하여 — 잠재 공간 위 베이지안 최적화로 목표 물성 분자를 향해 좁혀가는 루프가 표준입니다.

## 신약개발 적용

산업 현장의 실제 사례 — VAE 계열을 중심으로:

| 회사·기관 | 모델 / 활용 | 결과 | 시점 |
|----------|-----------|------|------|
| **Insilico Medicine** | Chemistry42 내 RNN-VAE prior + RL | INS018_055(IPF, Phase 2) 발굴 18개월 | 2021~2023 |
| **AstraZeneca** | REINVENT (RNN+RL, VAE 변형) 오픈소스 공개 | 자체 파이프라인의 lead 발굴 가속, 2023 Nature Reviews Drug Discovery에 사례 보고 | 2017~현재 |
| **Atomwise** | AtomNet + 잠재 공간 탐색 | 다수 신규 적응증 hit 발굴 | 2020~ |
| **MIT / Jin et al. (학술)** | JT-VAE → 활성-비활성 translation | DRD2 활성 분자 변환 성공률 약 60% | 2018 |
| **Sanofi × Iktos** | Makya(conditional VAE-like) | 1차 후보 발굴 6~9개월 단축, 4개 파이프라인 협업 | 2021~2023 |

기존 medicinal chemistry 대비 정량 비교 — VAE 기반 De Novo 사용 시점:

| 단계 | 전통 R-그룹 탐색 | VAE 잠재 공간 BO |
|------|---------------|-----------------|
| 1 사이클 평가 분자 수 | 약 50개 | 약 1,000~5,000개 (in silico) |
| 1 사이클 소요 시간 | 4~6주 | 1~3일 |
| 동시 최적화 가능 물성 | 1~2개 | 5~8개 (MPO) |
| Scaffold hopping 가능성 | 낮음(국소 탐색) | 중간(잠재 공간 항행) |

다만 — **VAE 생성 분자가 실제 합성·검증을 통과하는 비율은 보고된 사례에서 약 5~15% 수준**이며, 합성 가능성(SAscore)과 약리학적 유효성을 함께 평가하지 않으면 — Day 1에서 본 DSP-1181 같은 임상 단계 실패로 이어질 수 있습니다. 이 때문에 약학적 평가 모듈의 정교화가 — 생성 모델 자체의 성능 향상 못지않게 중요한 산업적 과제입니다.

## 창업 관점

VAE/GAN 영역은 — 핵심 기법이 2018~2020년 이미 공개되었고, REINVENT·JT-VAE·MOSES 같은 오픈소스 구현이 성숙해 있어 — 알고리즘 자체로는 차별화 여지가 작습니다. 그러나 약학 전공자의 차별화 포인트는 분명히 존재합니다.

**1) 도메인 특화 prior 학습**: 일반 ChEMBL 250k 대신, 특정 표적 family(예: kinase, GPCR) 또는 특정 약물군(예: CNS, 항생제)의 분자만으로 prior를 재학습하면 — 해당 영역에서 더 일관된 chemotype을 생성합니다. 이는 학습 데이터 큐레이션 단계에서 약학 지식이 핵심 자산이 되는 지점입니다.

**2) 잠재 공간의 약학적 축 발견 및 제품화**: 잠재 공간 위에 "BBB 투과 +", "hERG 회피 +", "CYP3A4 안정 +" 같은 축을 회귀로 학습하면 — 사용자(medicinal chemist)가 슬라이더로 약학 속성을 조정하며 분자를 항행하는 UX가 가능합니다. 이는 Schrödinger LiveDesign이 부분적으로 구현한 영역이며, 약학 전공자가 더 정교한 축 정의로 차별화할 수 있습니다.

**3) MVP 후보 — "Property-Slider Molecule Designer"**: REINVENT 또는 JT-VAE 오픈소스 위에, (a) 특정 표적 데이터로 fine-tuned prior, (b) Phase 2 Week 3에서 학습할 ADMET 모델, (c) 웹 UI(슬라이더 + 잠재 공간 시각화)를 결합한 SaaS. 바이브코딩으로 3~6개월 내 구축 가능하며, 가격은 사용자당 월 300~1,000달러 SaaS 모델이 현실적입니다.

순수 GAN 기반 창업은 — 산업의 채택률이 낮고 mode collapse 문제가 해결되지 않아 권장되지 않으며, VAE 또는 곧 다룰 RL/확산 모델 기반이 합리적 선택입니다.

## 오늘의 과제

1. **기초 학습 — JT-VAE 논문 핵심 파악(40분)**: Jin, W., Barzilay, R., Jaakkola, T. "Junction Tree Variational Autoencoder for Molecular Graph Generation" (ICML 2018) Abstract와 Section 3(Architecture)를 읽고 — (a) Junction tree 구성 알고리즘, (b) Tree decoder의 작동 방식, (c) ZINC 데이터셋에서 보고된 validity/uniqueness/novelty 수치를 1페이지로 정리하세요. 자신이 익숙한 약물 1개의 Bemis-Murcko scaffold를 손으로 그려보고, JT-VAE의 어휘로 분해된 형태를 추정해 보세요.

2. **비즈니스 분석 — VAE 기반 플랫폼 2사 + 오픈소스 1개 비교(40분)**: Insilico Medicine(Chemistry42 — VAE 사용 부분), Iktos(Makya), AstraZeneca REINVENT(오픈소스)의 (a) 공개된 아키텍처 구성, (b) 사용 가능한 적용 표적·약물군, (c) 라이선스/가격 모델을 표로 정리하세요. 약학 전공 1인 창업자가 이 중 하나의 위에 차별화 모듈을 올려 SaaS를 만든다면 — 어떤 베이스를 선택하겠는지 2~3문장 의견을 덧붙이세요.

3. **리서치 정리 — 잠재 공간 항행 실험 설계(60분)**: "내가 만든다면" 시나리오 — 특정 표적(예: EGFR T790M kinase)에 대해, JT-VAE의 잠재 공간에서 활성 분자 3개를 선형 보간(interpolation)하여 hybrid 분자를 만드는 실험을 설계하세요. (a) 입력 분자 3개의 선정 기준(예: gefitinib, osimertinib, mobocertinib), (b) 보간 비율(예: 0.5:0.3:0.2), (c) 생성된 분자를 어떤 평가 지표(QED, SAscore, predicted pIC50, BBB)로 검증할지 — 1페이지 실험계획서로 작성하세요.

## 참고 자료

- **논문**: Jin, W., Barzilay, R., Jaakkola, T. "Junction Tree Variational Autoencoder for Molecular Graph Generation." *ICML*, 2018. — 100% validity를 달성한 분자 VAE의 결정적 모델.
- **논문**: Gomez-Bombarelli, R. et al. "Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules." *ACS Central Science*, 2018. — 분자 VAE + 베이지안 최적화의 첫 입증.
- **벤치마크**: MOSES (Polykovskiy et al., 2020) — VAE/GAN 등 분자 생성 모델의 표준 평가 프레임워크, 다양한 metric 일관 비교.
- **오픈소스**: AstraZeneca REINVENT 4 — VAE/RNN prior + RL을 결합한 산업급 De Novo 프레임워크.
