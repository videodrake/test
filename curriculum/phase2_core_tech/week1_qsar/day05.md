# Day 5: Week 1 종합 — 물성 예측 모델 선택 가이드

> 이전 학습(Phase 2 Week 1 Day 1~4)에서 QSAR의 60년 역사·세 패러다임 전환, ECFP4 + Random Forest/XGBoost로 대표되는 2세대 베이스라인, Chemprop(D-MPNN)로 대표되는 3세대 GNN, MolFormer-XL·Uni-Mol 같은 4세대 Foundation Model을 차례로 학습했습니다. Week 1을 마무리하는 오늘은 — 네 세대의 모델을 **하나의 선택 의사결정 프레임**으로 묶고, "주어진 적응증·데이터 규모·자원에서 무엇을 어떻게 조합할 것인가"에 답할 수 있도록 정리합니다. 이 프레임은 Phase 2 Week 2(분자 생성), Week 3(ADMET 심화)에서 모델을 결정할 때마다 직접 재사용되는 핵심 도구입니다.

## 개요

물성 예측 모델 선택은 AI 신약개발 프로덕트의 **첫 번째이자 가장 비용 비대칭적인 의사결정**입니다. 같은 hERG 차단 예측 과제라도 ECFP4 + XGBoost를 고르면 1시간 안에 베이스라인이 나오고, MolFormer-XL fine-tuning을 고르면 GPU 한 장과 반나절이 필요하며, 자체 Foundation Model 사전학습을 시도하면 수만 달러의 비용이 발생합니다. 그러나 데이터 N이 500개인 희귀 표적에서는 — 가장 비싼 옵션이 가장 정확하지 않을 수 있습니다. 본 Day는 (1) 4세대 모델을 **여섯 축**(데이터 효율, 표현력, 계산 비용, 해석성, 배포 난이도, 도메인 적응성)에서 비교하고, (2) **데이터 N과 과제 유형**을 입력으로 받는 의사결정 트리를 정립하며, (3) 약학 전공자가 이 단계에서 어떤 판단으로 비전공자 대비 우위를 가지는지 명시화하고, (4) Phase 2 Week 1을 마치고 진입하는 Week 2(분자 생성)·Week 3(ADMET) 학습의 토대를 정리합니다.

## 핵심 개념

### 1) 4세대 QSAR — 한 장의 비교표

지난 4일간 학습한 네 가지 모델 계열을 6개 축에서 비교합니다.

| 세대 | 대표 모델 | 데이터 효율(N) | 표현력 | 계산 비용 | 해석성 | 배포 난이도 | 도메인 적응성 |
|------|---------|------------|------|---------|------|----------|----------|
| **1세대** | Hansch 회귀 | N=20~100 | 낮음 | 매우 낮음 | 매우 높음 | 매우 낮음 | 좁은 series만 |
| **2세대** | ECFP4 + XGBoost / RF | N=500~10,000 (sweet spot) | 중 | 낮음 (CPU 분 단위) | 중-높음 (SHAP) | 낮음 (pickle) | 도메인 지문 추가 용이 |
| **3세대** | Chemprop / D-MPNN GNN | N=5,000~100,000 | 높음 | 중 (GPU 시간 단위) | 중 (attention) | 중 (PyTorch 의존) | 추가 학습 가능 |
| **4세대** | MolFormer-XL / Uni-Mol | N=500~수만 (전이학습) | 매우 높음 | 사전학습 비싸나 fine-tuning은 낮음 | 낮음-중 | 중-높음 (HF 모델 의존) | 추가 사전학습 시 강함 |

이 표는 Week 1의 결론을 한 장에 응축한 것입니다. 어떤 축도 절대적 우위가 없으며, 선택은 항상 "내 과제에서 **어떤 축이 결정적인가**"의 함수입니다.

### 2) 데이터 N — 모델 선택의 1차 결정 변수

학습 데이터 크기 N은 모델 선택의 **가장 강한 1차 변수**입니다. Week 1을 통틀어 가장 자주 인용된 정량 관찰을 정리하면 다음과 같습니다.

| 데이터 N | 1차 선택 | 2차 선택(앙상블) | 피해야 할 선택 |
|---------|---------|----------------|-------------|
| **N < 300** | ECFP4 + 정규화된 RF | Foundation Model 동결 + linear head | 처음부터 학습하는 GNN/Transformer |
| **N = 300~1,000** | ECFP4 + XGBoost | MolFormer-XL fine-tuning (frozen encoder) | 큰 GNN, 대형 D-MPNN |
| **N = 1,000~10,000** | ECFP4 + XGBoost (베이스라인) + Chemprop | MolFormer-XL fine-tuning | 사전학습 없는 12-layer Transformer |
| **N = 10,000~100,000** | Chemprop / D-MPNN | MolFormer-XL / Uni-Mol fine-tuning | 1세대 선형 회귀 |
| **N > 100,000** | 사전학습 + fine-tuning 또는 자체 GNN scale-up | 도메인 특화 추가 사전학습 | 단순 RF (성능 천장 도달) |

핵심 패턴은 — **데이터가 적을수록 inductive bias가 강한 단순 모델이 강하고, 많을수록 자유도가 높은 모델이 강하다**는 것입니다. 약학적 직관으로는 — 어세이 한 종에서 보통 얻는 데이터(N=500~3,000)에서는 ECFP4 + XGBoost가 거의 항상 강력한 1차 선택이며, MolFormer-XL 같은 Foundation Model이 그 베이스라인을 의미 있게 뛰어넘는지를 정량 비교해야 합니다.

### 3) 과제 유형 — 2차 결정 변수

데이터 N이 비슷해도 과제 유형이 다르면 최적 모델이 달라집니다. Week 1에서 다룬 주요 과제 유형별 권장 조합은 다음과 같습니다.

| 과제 유형 | 1차 권장 | 약학적 이유 |
|---------|---------|----------|
| **회귀 (logS, logP, pIC50)** | ECFP4 + XGBoost / MolFormer-XL | 연속값 예측은 트리 기반이 안정, FM은 작은 N에서 우위 |
| **이진 분류 (hERG block, BBB+/-)** | ECFP4 + XGBoost + RDKit 기술자 | SHAP 해석 + 약학적 검증 가능 |
| **다중 task (Tox21, ToxCast)** | Multi-task GNN(Chemprop) | task 간 표현 공유로 정보 부족 task 보강 |
| **희귀 활성 (active rate < 1%)** | XGBoost + class weight + EF@1% 평가 | 불균형 데이터에서 트리 모델 robust |
| **3D 의존 (결합 친화도, conformer 민감)** | Uni-Mol / 3D GNN(SchNet, MACE) | 입체화학 없이는 천장에 부딪힘 |
| **분포 외 (신규 chemotype)** | XGBoost + 사전학습 FM 앙상블 | 한 모델로는 OOD 처리 한계 |

이 표는 — 한 가지 모델을 만능 솔루션으로 채택하지 않는 산업적 표준을 반영합니다. 첫 베이스라인은 항상 ECFP4 + XGBoost로 잡고, 과제별로 추가 모델을 비교 검증하는 절차가 합리적입니다.

### 4) 모델 결합 — Stacking, Ensembling, Cascade

실제 산업에서는 **단일 모델보다 모델 결합이 표준**입니다. Week 1의 4세대를 어떻게 조합할지 세 가지 패턴을 정리합니다.

| 결합 방식 | 구조 | 강점 | 약학 적용 예 |
|---------|------|------|----------|
| **Averaging Ensemble** | 여러 모델 예측 평균 | 분산 감소, 단순함 | hERG 차단 확률을 3 모델 평균으로 사용 |
| **Stacking** | 1차 모델 출력을 2차 모델 입력 | 보완적 예측 결합 | XGBoost 확률 + GNN 임베딩 → 메타 분류기 |
| **Cascade (필터링)** | 빠른 모델로 거른 뒤 비싼 모델 적용 | 대량 스크리닝 비용 절감 | 10억 분자 → ECFP+XGBoost로 1만 후보 선별 → MolFormer 정밀 평가 |

Cascade 구조는 — 약학 전공 창업자가 만들 수 있는 가장 실용적 SaaS 아키텍처입니다. 사용자가 10억 개 화합물 라이브러리를 업로드하면, 1단계로 가벼운 XGBoost가 99.9%를 걸러내고, 2단계로 Foundation Model이 남은 후보 1만 개를 정밀 평가하는 구조는 — Insilico·Atomwise·Recursion이 실제로 운영하는 산업 표준 파이프라인입니다.

### 5) 약학 전공자의 의사결정 우위 — 모델이 아니라 **무엇을 예측할지**

Week 1을 마치는 시점의 핵심 통찰 — **모델 선택보다 더 결정적인 것은 "무엇을 예측 대상으로 정의하느냐"이며, 여기서 약학 전공자의 도메인 지식이 가장 큰 차이를 만든다**는 것입니다.

비약학 전공자가 흔히 빠지는 함정:
- **임상적 의미 없는 endpoint 선택**: ChEMBL의 raw IC50을 그대로 회귀하지만, 실제 임상에서 의미 있는 것은 hepatocyte clearance 기반 in vivo 노출 예측이라는 점을 놓침.
- **다중 어세이 데이터의 단순 합치기**: 실험 조건(세포주, 농도 범위, 검출법)이 다른 IC50을 한 모델로 학습해 노이즈가 누적됨.
- **임계값의 자의적 설정**: hERG 활성 임계값을 10 μM로 잡으면 안전한 분자도 위양성, 1 μM로 잡으면 위음성. 약학적 맥락 없이는 결정 불가.

약학 전공자가 발휘하는 우위:
- **임상 연관 endpoint 정의**: AUC, Cmax, t½ 같은 PK 파라미터를 예측 대상으로 재정의해 모델 출력의 의사결정 가치를 높임.
- **데이터 큐레이션**: 같은 IC50이라도 세포주별·검출법별로 분리해 통합 학습. 약사 면허 수준의 어세이 이해가 직접 활용됨.
- **임계값 정당화**: hERG block IC50 < 10 μM가 임상에서 QT 연장과 강한 상관을 보인다는 ICH S7B 가이드라인 근거로 임계값을 정당화.

이 차이는 — 같은 모델, 같은 데이터에서도 **최종 제품의 신뢰성과 규제 수용성에서 결정적 격차**를 만듭니다. Phase 2 Week 3(ADMET)에서 이 우위가 본격적으로 발휘됩니다.

## 작동 원리와 아키텍처

### 통합 의사결정 트리 — 모델 선택 한 장 알고리즘

```
[입력]: 데이터 N, 과제 유형, 예산(GPU), 해석 요구
        ↓
[1단계: 데이터 N 확인]
   N < 300            → ECFP4 + RF (정규화 강화) + Cross-validation 5-fold
   N = 300~1,000      → ECFP4 + XGBoost (early stopping)
   N = 1,000~10,000   → 위 두 모델 + Chemprop + MolFormer-XL fine-tuning 병행
   N > 10,000         → Chemprop + Foundation Model fine-tuning, 앙상블
        ↓
[2단계: 과제 특수성 확인]
   3D 의존 (binding affinity, conformer 민감)?
       → Uni-Mol 또는 3D-aware GNN 우선
   Multi-task (Tox21 등)?
       → Multi-task Chemprop
   극도의 클래스 불균형?
       → XGBoost + class weight, 평가는 PR-AUC/EF@1%
        ↓
[3단계: 평가 프로토콜]
   scaffold split 5 seed
   metrics: 회귀=RMSE/Spearman ρ, 분류=ROC-AUC/PR-AUC/EF@1%
   reporting: mean ± std, 베이스라인(XGBoost) 대비 절대값 비교
        ↓
[4단계: 해석/배포 준비]
   해석 요구 강함(규제) → XGBoost + SHAP을 메인, GNN/FM 보조
   대량 스크리닝 필요   → Cascade 아키텍처(빠른 필터 → 정밀 평가)
   배포 환경 제한        → CPU 추론 가능 모델(XGBoost, MolE) 우선
        ↓
[출력]: 1차 모델 + 베이스라인 + 평가 프로토콜 + 배포 구조
```

이 알고리즘은 Claude Code에 첫 모델 파이프라인을 요청할 때 그대로 명세로 사용할 수 있습니다. "N=2,500 hERG 차단 분류, scaffold split 5 seed, XGBoost 베이스라인 + MolFormer-XL fine-tuning 병행, SHAP 해석 포함, FastAPI로 배포"라는 한 문장 프롬프트로 표준 파이프라인이 생성됩니다.

### 산업 표준 워크플로우 — 첫 모델 14일 일정

| 일자 | 단계 | 산출물 |
|------|------|------|
| Day 1-2 | 데이터 수집/큐레이션 (ChEMBL, in-house) | clean CSV, scaffold split |
| Day 3-4 | ECFP4 + XGBoost 베이스라인 | RMSE/ROC-AUC + SHAP 보고서 |
| Day 5-7 | Chemprop fine-tuning | D-MPNN 결과 + 베이스라인 비교 |
| Day 8-10 | MolFormer-XL fine-tuning | FM 결과 + 비교표 |
| Day 11-12 | 앙상블/Stacking 시도 | 최적 결합 결과 |
| Day 13-14 | 배포 (FastAPI + Docker) | 추론 API + 문서 |

이 14일 표는 — 한국 AI 신약개발 스타트업이 한 가지 ADMET endpoint에 대해 첫 프로덕션 모델을 만드는 표준 일정과 일치합니다. Phase 4 Week 1(바이브코딩)에서 이 일정을 Claude Code로 압축하는 실전 패턴을 다룹니다.

## 신약개발 적용

### 사례 1 — Chemprop의 Halicin 발굴 사례 재해석

MIT Collins lab은 2020년 Chemprop(D-MPNN)으로 2,335개 항생제 활성 데이터를 학습해, Drug Repurposing Hub의 6,111개 분자에서 새로운 항생물질 후보 **Halicin**을 발굴했습니다(Stokes *et al.*, *Cell*, 2020). 이 사례의 본질은 — N=2,335라는 작은 데이터에서 GNN이 성공한 이유가 **분자 표현보다 데이터 큐레이션과 임계값 정의의 정밀성**에 있었다는 점입니다. Collins lab은 *E. coli* 성장 억제율 80% 이상을 활성으로 정의하고, 시험 조건을 통일했으며, 시드 분자를 의약화학자가 검토해 모델 입력 분포를 정제했습니다. 약학 전공 창업자에게 이 사례의 시사점은 — **모델 세대(GNN인지 FM인지)보다 약학적 임계값과 데이터 큐레이션이 결과를 결정한다**는 것입니다.

### 사례 2 — Insilico Medicine INS018_055의 멀티모달 평가

**Insilico Medicine**의 IPF 치료제 후보 **INS018_055**는 2023년 6월 Phase II에 진입한 — AI 발굴 분자 중 가장 진전된 사례 중 하나입니다(IR 자료, 2024 기준). 이 분자가 통과한 in silico 평가는 단일 모델이 아닌 — Insilico의 사내 Foundation Model **Chemistry42**(분자 생성)와 **PandaOmics**(타겟 검증) + 외부 도구로 구성된 멀티모델 cascade였습니다(공개 IR + 2023 *Nature Biotechnology* 인터뷰 기준). 이 사례는 — 산업 최전선에서도 단일 모델 만능론이 아니라 **모델 조합과 약학적 의사결정 게이트의 정렬**이 성공 요인이라는 점을 보여줍니다.

### 사례 3 — Atomwise AtomNet의 Cascade 아키텍처

**Atomwise**의 **AtomNet**은 — 3D CNN 기반 단백질-리간드 결합 친화도 예측 모델로, **수십억 분자 라이브러리 가상 스크리닝**에 사용됩니다(Wallach *et al.*, 2015 이후 다수 후속 논문). Atomwise의 산업 워크플로우는 명확한 Cascade — (1) 가벼운 docking 또는 지문 기반 모델로 99% 제거, (2) AtomNet 3D CNN으로 정밀 평가, (3) 의약화학자 수동 검토. 2018년 이후 80여 개 표적에 적용해 다수의 hit 분자를 발굴했다고 보고합니다(공식 사이트, 2024 기준). 약학 전공 창업자에게 이 구조는 — Cascade 아키텍처가 산업의 표준이며, 단순한 모델도 잘 결합하면 강력하다는 점을 보여줍니다.

### 사례 4 — TDC ADMET Benchmark Group의 모델 선택 정답표

**Therapeutics Data Commons(TDC)의 ADMET Benchmark Group**(2022년 출시, Huang *et al.*)은 22개 ADMET task에서 표준 평가 프로토콜을 제공합니다. 2024년 기준 leaderboard를 보면 — 22개 task 중 약 14개에서 **MolFormer-XL 또는 Uni-Mol 등 Foundation Model**이 상위 3위 안에 들고, 약 8개에서는 **Chemprop 또는 XGBoost 기반 전통 모델**이 여전히 우위입니다(tdcommons.ai/benchmark/, 2024 확인). 패턴은 명확합니다 — N이 큰 task(Solubility AqSolDB N=10,000, Lipophilicity N=4,200)에서는 FM이 강하고, N이 작거나 분포 외 task(BBB Penetration N=2,030, hERG block N=655)에서는 XGBoost·RF가 여전히 경쟁력 있습니다. 이 leaderboard는 — 한국 스타트업이 자체 모델을 어떤 task에서 차별화할지 결정할 때 직접 참조해야 할 1차 자료입니다.

## 창업 관점

Phase 2 단계에서 창업 관점은 세 갈래로 짚습니다. 첫째, **AutoML-for-Chemistry SaaS**: 사용자가 자신의 어세이 데이터(엑셀 1장)를 업로드하면 Week 1에서 학습한 4세대 모델(XGBoost·Chemprop·MolFormer·Uni-Mol)을 자동으로 학습·비교·앙상블해 최적 모델을 제시하는 도구는 — 한국 중소 제약사와 학계 연구실이 즉시 필요로 하는 가장 명확한 제품 유형입니다. 둘째, **Cascade 가상 스크리닝 플랫폼**: Atomwise 모델을 한국 시장에 맞춰 — 사용자가 KFDA 허가 의약품·천연물·내부 라이브러리를 업로드하면 빠른 필터링(XGBoost) → 정밀 평가(MolFormer/Uni-Mol)를 자동 실행하는 클라우드 서비스. GPU 시간 단위 과금 모델로 SaaS 매출 구조를 만들 수 있습니다. 셋째, **Endpoint-First Consulting**: 모델보다 약학적 endpoint 정의 자체에 강점을 가진 약학 전공 창업자의 차별점을 — "당신의 약물 후보를 어떤 endpoint로 평가할지 정의해드립니다"는 서비스로 전환. PMS·CRO·임상 분야 약사 네트워크와 연결될 수 있는 한국 시장 특화 포지셔닝입니다. 모델 자체가 아니라 **모델이 답할 질문의 약학적 정확성**이 차별점이라는 점이 Week 1의 핵심 메시지입니다.

## Week 1 정리와 Week 2 예고

Week 1을 통해 — QSAR의 60년 역사(Day 1), 2세대 전통 ML(Day 2), 3세대 GNN(Day 3), 4세대 Foundation Model(Day 4), 통합 선택 가이드(Day 5)를 학습했습니다. 다음 Week 2는 **분자 생성(Molecular Generation)** — 단순히 주어진 분자를 평가하는 게 아니라 AI가 새로운 분자를 직접 설계하는 패러다임으로 넘어갑니다. VAE·GAN(Day 2), 강화학습(Day 3), 확산 모델(Day 4)을 차례로 다루며, Week 1에서 만든 QSAR 모델이 — 분자 생성의 보상 함수(reward function)로 어떻게 결합되는지가 핵심 연결점입니다. 약학 전공자에게는 — 합리적 약물 설계(rational drug design)의 컴퓨터 확장이라는 의미를 갖습니다.

## 오늘의 과제

1. **TDC ADMET Leaderboard 분석 (40분)**: tdcommons.ai/benchmark/admet_group/overview/에서 22개 ADMET task의 상위 3개 모델을 표로 정리합니다. 항목: task명·데이터 N·1위 모델·2위 모델·3위 모델·지배적 모델 계열(XGBoost/GNN/FM). 본인이 어떤 task에서 차별화할지 결정할 때 1차 근거 자료가 됩니다.

2. **모델 선택 의사결정 트리 본인 버전 작성 (40분)**: 본 Day의 통합 의사결정 트리를 — 본인이 가정한 첫 스타트업 제품(예: 한국 천연물 hERG 위험 평가 SaaS)의 맥락에 맞춰 재작성합니다. 데이터 N·과제 유형·예산 가정을 명시하고, 1차/2차 모델 선택 + 평가 프로토콜 + 배포 구조를 1장 다이어그램으로 정리합니다. 향후 VC 피칭에서 기술 차별화 슬라이드로 직접 활용 가능합니다.

3. **Endpoint-First 컨설팅 제안서 1쪽 작성 (40분)**: 약학 전공자의 차별점을 — "데이터 큐레이션 + endpoint 정의" 컨설팅 서비스로 정의한 1쪽 제안서를 작성합니다. 항목: 타깃 고객(중소 제약사/CRO)·서비스 범위·가격·약학 전공자만 제공 가능한 부분·1년 매출 추정. 한국 시장 내 5개 후보 고객사를 명시적으로 나열합니다. Phase 5 Week 1(시장 분석)에서 이 제안서를 정량 검증하는 작업으로 이어집니다.

## 참고 자료

- Huang, K. *et al.* (2022). "Artificial Intelligence Foundation for Therapeutic Science." *Nature Chemical Biology*, 18, 1033-1036. — TDC의 종합 벤치마크 논문. ADMET·약물 상호작용 등 22개 task의 표준화된 평가 프로토콜을 제시했으며, 한국 스타트업이 자체 모델을 비교 평가할 때 표준 참조 자료입니다.
- Stokes, J. M. *et al.* (2020). "A Deep Learning Approach to Antibiotic Discovery." *Cell*, 180, 688-702. — Chemprop으로 Halicin을 발굴한 사례. N=2,335의 작은 데이터로도 GNN이 산업 성과를 낸 결정적 레퍼런스이며, 데이터 큐레이션의 중요성을 보여줍니다.
- Yang, K. *et al.* (2019). "Analyzing Learned Molecular Representations for Property Prediction." *Journal of Chemical Information and Modeling*, 59, 3370-3388. — Chemprop(D-MPNN) 원논문. 산업 표준 GNN 베이스라인의 설계 근거이며, ECFP/XGBoost 대비 어떤 조건에서 우위인지를 정량적으로 보고합니다.
- Therapeutics Data Commons(tdcommons.ai), MoleculeNet(moleculenet.org), HuggingFace Molecular Models(huggingface.co/models?search=molecule) — 모델 선택 시 1차 비교 자료의 표준 출처. 본 Day의 의사결정 트리를 검증할 때 직접 사용할 자원입니다.
