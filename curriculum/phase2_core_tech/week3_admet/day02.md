# Day 2: 흡수와 분포 예측 — 용해도, 투과성, BBB, PPB

> 이전 학습(Phase 2 Week 3 Day 1)에서 ADMET 5축의 약학적 정의, 임상 실패 통계의 정확한 해석, AI 예측 시스템의 표준 아키텍처, 그리고 적응증·표적별 cutoff 조정의 차별점을 다뤘습니다. 오늘은 그 5축 중 **A(흡수)와 D(분포)** 두 축으로 들어가, AI가 예측해야 하는 4개 핵심 물성 — **용해도(logS), 투과성(Caco-2/PAMPA), BBB 통과, 혈장 단백질 결합(PPB)** — 의 약학적 의미와 모델 설계 결정을 정리합니다. 이 네 가지가 — 경구 약물 후보가 표적 조직에 도달할지의 일차 게이트입니다.

## 개요

흡수와 분포는 **약물이 투여 부위에서 표적 조직까지 도달하는 과정**을 결정합니다. 아무리 강력한 in vitro 활성을 가진 분자라도 — 용해되지 않으면 흡수되지 않고, 장 상피를 투과하지 못하면 전신 순환에 진입하지 못하며, 단백질에 100% 결합되면 자유 분율(free fraction)이 0이 되고, BBB를 통과하지 못하면 중추 적응증에서 무용합니다. AI 기반 흡수·분포 예측은 — 수십억 개의 가상 후보 중에서 이 네 게이트를 통과할 가능성이 높은 분자를 일차 선별하는 데 사용됩니다. 본 Day는 logS·Caco-2·BBB·PPB 네 물성의 **약학적 정의, 측정의 한계, AI 모델의 표준 데이터셋과 성능 수준, 약학 전공자가 적용해야 할 임상적 임계값**을 정리합니다.

## 핵심 개념

### 1) 용해도(Solubility, logS) — 모든 게이트의 첫 관문

**용해도(aqueous solubility)**는 분자가 수성 매질에 녹는 농도이며, 대수 변환된 값 **logS**(단위: log mol/L)로 보고됩니다. 약학적으로 일반적인 분류는 다음과 같습니다.

| logS 범위 | 분류 | 임상적 의미 |
|----------|------|----------|
| > −2 | 매우 가용성(highly soluble) | 경구 제형 설계 용이 |
| −4 ~ −2 | 가용성(soluble) | 표준 제형 가능 |
| −6 ~ −4 | 난용성(poorly soluble) | 가용화 기술 필요 (SEDDS, nano) |
| < −6 | 매우 난용성(very poor) | 경구 후보로 부적합 |

**Lipinski의 Rule of Five**의 핵심 제약 중 하나가 logS이며, BCS(Biopharmaceutical Classification System) Class I/II 구분의 핵심 변수이기도 합니다. **약학 전공자의 우위** — 의약화학자가 "logP를 낮추면 logS가 올라간다"는 단순 관계만 보는 동안, 약학 전공자는 — pH 의존 용해도(이온화 분획), 결정형(polymorph) 영향, 식후/공복 GI 환경의 차이까지 함께 해석합니다. AI 모델이 단일 logS 값을 출력해도, 약학적 후처리에서 위(pH 1.5)·소장(pH 6.5)·결장(pH 7.4) 조건별 가용 농도를 추정해 BCS 분류에 매핑하는 것이 차별 영역입니다.

**표준 데이터셋과 모델 성능** — AI logS 예측의 표준 학습 데이터는 **ESOL**(Delaney 2004, n≈1,128), **AqSolDB**(Sorkun et al. 2019, n≈9,982 정제), **TDC Solubility_AqSolDB**입니다. 2024년 기준 ChemProp D-MPNN과 Uni-Mol 사전학습 모델의 ESOL RMSE는 약 0.55~0.65 logS 단위로 보고됩니다(MoleculeNet 리더보드 기준). 측정 실험 자체의 재현성이 약 0.4~0.5 logS 단위인 점을 고려하면 — **모델 성능이 측정 노이즈 천장에 근접한 상태**이며, 추가 정확도 향상보다는 **새로운 chemotype에서의 일반화**가 차별 영역입니다.

### 2) 투과성(Permeability) — Caco-2와 PAMPA

전신 노출의 두 번째 게이트는 **장 상피 투과성**이며, in vitro 대리 지표로 **Caco-2 세포 단층 투과율**(logPapp, 단위: log cm/s)과 **PAMPA**(평행인공막)가 사용됩니다.

| 측정법 | 원리 | logPapp 임계값 | 임상 해석 |
|-------|------|------------|--------|
| **Caco-2** | 인간 결장암 유래 세포 단층, tight junction 형성 | > −5.15 (≈ 7×10⁻⁶ cm/s) | 고흡수(>70%) 후보 |
| **PAMPA** | 인공 인지질 막, 세포 무관 | > −5.5 (≈ 3×10⁻⁶ cm/s) | 수동 확산 위주 |
| **MDCK** | 개 신장 상피, BBB 대리 | > −5 | BBB 통과 가능성 |

**약학적 깊이** — Caco-2는 **수동 확산 + 능동 수송(P-gp, BCRP) + 외향 수송(efflux)** 효과를 모두 반영하며, MDCK-MDR1(human P-gp 과발현)로 P-gp 기질 여부를 분리 평가합니다. **PAMPA는 수동 확산만** 측정하므로 — Caco-2 − PAMPA 차이가 클수록 능동 수송 기여가 크다는 약학적 해석이 가능합니다. AI 모델이 logPapp만 출력해도, 약학 전공자는 P-gp efflux ratio(Papp_BA/Papp_AB)를 함께 보고 약물 상호작용 위험을 평가합니다.

**표준 데이터셋과 성능** — **TDC Caco2_Wang**(n≈910)과 **PAMPA_NCATS**(n≈2,034)가 표준 벤치마크입니다. 데이터 규모가 작아(<1,000) ChemProp·AttentiveFP 같은 GNN의 MAE는 약 0.34~0.40 logPapp 단위로 보고되며, 사전학습 Transformer(MolFormer, Uni-Mol) 도입으로 약 10~15% 추가 개선이 가능합니다. **데이터 소량성**이 가장 큰 제약이며 — 멀티태스크 학습(흡수 관련 22개 task 동시 학습)으로 데이터 효율을 높이는 것이 2024년 표준 접근입니다.

### 3) 혈액뇌장벽 투과(BBB Penetration) — CNS 약물의 결정적 게이트

**혈액뇌장벽(blood-brain barrier, BBB)**은 뇌 모세혈관 내피세포의 tight junction과 P-gp/BCRP 외향 수송체로 구성된 선택적 장벽입니다. CNS 적응증(알츠하이머, 파킨슨, 우울증, 정신분열, 뇌종양) 약물은 BBB를 통과해야 하며, 비CNS 적응증 약물은 CNS 부작용 회피를 위해 BBB를 통과하지 않는 것이 유리합니다 — **양방향 의사결정**이 필요한 드문 ADMET 축입니다.

| 지표 | 정의 | 분류 임계값 |
|------|------|----------|
| **logBB** | log(C_brain / C_plasma) | > 0.3 통과, < −1 불통과 |
| **logPS** | log(투과면적 곱) | > −2 능동 통과 |
| **K_p,uu** | 자유 분획 뇌/혈장 비 | > 0.3 의미있는 노출 |
| **BBB+/−** | 이진 분류 | 데이터셋별 정의 다양 |

**약학적 차별 영역** — logBB는 총 농도 비이지만, **자유 분획 보정 K_p,uu**가 임상 의사결정에 더 직접 연결됩니다. 단백질 결합률이 95%인 두 분자가 같은 logBB를 가져도 자유 분획 차이가 10배일 수 있습니다. AI BBB 모델이 logBB만 출력해도 — 약학 전공자는 PPB와 함께 K_p,uu를 추정합니다.

**표준 데이터셋** — **TDC BBB_Martins**(n≈1,975, 이진 분류)와 **B3DB**(Meng et al. 2021, n≈7,807, logBB 회귀)가 표준입니다. BBB_Martins에서 ChemProp D-MPNN의 AUROC는 약 0.89~0.91 수준으로 보고되지만 — **데이터 누출과 chemotype 편향** 문제가 반복 지적되었습니다. Polaris 벤치마크가 정제된 BBB 데이터셋을 재공급한 이후로는 — 같은 모델의 AUROC가 약 0.78~0.82로 떨어집니다. **약학 전공자의 우위** — chemotype 다양성 평가(scaffold split AUROC, 새 골격에서의 성능)를 반드시 함께 보고함으로써 "0.91"의 함정을 피합니다.

### 4) 혈장 단백질 결합(Plasma Protein Binding, PPB)

분자가 혈장에 들어가면 — 알부민(약 60%), α1-산성 당단백질(약 10%), 리포단백질에 결합하며, **자유 분획(fraction unbound, fu)**만이 표적 조직으로 분포·대사·배설됩니다.

| fu 범위 | 분류 | 임상적 의미 |
|--------|------|----------|
| > 0.3 | 낮은 결합 | 자유 농도 풍부, 표준 용량 |
| 0.05 ~ 0.3 | 중간 결합 | 일반적 약물 범위 |
| < 0.05 | 높은 결합 | 약물 상호작용·간기능 변화에 민감 |
| < 0.01 | 매우 높은 결합 | 임상 변동성 크고 위험 |

**약학적 깊이** — PPB는 그 자체로 효능을 결정하지 않지만, **약물 상호작용**(albumin 결합 부위 경쟁), **간/신 기능 저하 환자**의 자유 분획 급증, **임상 용량 계산**의 핵심 변수입니다. **자유 약물 가설(free drug hypothesis)** — 정상상태에서 자유 농도가 표적 노출을 결정한다 — 이 PPB의 임상적 의미를 정리한 핵심 원리입니다.

**표준 데이터셋** — **TDC PPBR_AZ**(AstraZeneca 자체 데이터, n≈2,790)가 표준 회귀 벤치마크입니다. 라벨이 % 결합(0~100%)으로 정규화되어 있어 logit 변환 후 회귀하는 것이 권장됩니다. 최신 D-MPNN + Uni-Mol 사전학습의 MAE는 약 7~9% 수준이며, 95~99% 구간(임상적으로 가장 민감한 영역)에서의 정확도는 별도 보고하는 것이 약학적 표준입니다.

### 5) 네 물성의 통합 — 흡수·분포 cascade

위 네 물성은 — 독립적으로 예측되지만 임상 의사결정에서는 cascade로 결합됩니다.

> **logS → Caco-2 → PPB → BBB(CNS 후보면)**

각 단계의 약학적 cutoff 적용 — logS > −5(가용), Caco-2 logPapp > −5.15(흡수), PPB fu > 0.05(자유 분획), BBB AUROC > 0.85의 P 통과(CNS 적응증). 비CNS 적응증은 BBB 통과율이 낮은 후보를 선호하는 정반대 cutoff. 이 cascade가 — Phase 2 Week 2에서 다룬 분자 생성의 후처리 필터로 직결됩니다.

## 작동 원리와 아키텍처

흡수·분포 통합 예측 시스템의 표준 구조 — 2024년 기준:

```
[흡수·분포 cascade 예측 시스템]

1. 입력 표준화
   - SMILES → RDKit 표준화(salt 제거, tautomer 결정)
   - pH 7.4 ionization 상태 예측 (microspecies)
   - 3D conformer 생성 (ETKDG, 10개)

2. 멀티태스크 표현 학습 (공유 encoder)
   - Chemprop D-MPNN (3-layer message passing)
   - 또는 MolFormer-XL 사전학습 임베딩
   - 출력: 256~512차원 분자 임베딩

3. 4-헤드 예측 (각 task별 FC layer)
   - logS 회귀 (ESOL + AqSolDB 통합 학습)
   - logPapp 회귀 (Caco2_Wang + 사내 데이터)
   - PPB 회귀 (PPBR_AZ, logit 변환)
   - BBB 분류 (BBB_Martins + Polaris 정제판)

4. 약학적 후처리 레이어
   - K_p,uu = logBB × (fu_brain / fu_plasma) 추정
   - BCS 분류 매핑 (logS + Caco-2 → Class I~IV)
   - P-gp efflux 위험 점수 (Caco-2 단방향 + 양방향 차이)

5. 적응증·환자군 cutoff 엔진 — 약학 전공자의 차별 영역
   - CNS 적응증 → BBB hard cutoff (logBB > 0.3)
   - 비CNS 적응증 → BBB soft penalty (CNS 부작용 회피)
   - 간기능 저하 환자 표적 → PPB 보수적 cutoff (fu > 0.1)
   - 소아 적응증 → 용량-체중 환산 후 가용 농도 재계산

6. 불확실성 정량화 + 임상 보고
   - MC Dropout 95% 신뢰구간
   - Tanimoto 적용성 도메인 점수
   - 사용자 정의 cutoff 통과율 + cascade 시각화
```

핵심 설계 결정 — 흡수·분포 cascade에서 차별화할 수 있는 지점:

| 결정 | 일반 접근 | 약학적 차별 |
|------|--------|--------|
| 표현 학습 | 단일 SMILES | pH 7.4 ionization 명시적 표현 |
| 멀티태스크 학습 | 22개 task 균등 | 흡수·분포 4개 task 강가중치 + 공유 encoder |
| BBB 출력 | logBB 단일값 | K_p,uu + chemotype 적용성 |
| PPB 출력 | % 결합 단일값 | 95~99% 구간 별도 신뢰구간 |
| 통합 점수 | 4개 task 평균 | 적응증 가중치 + BCS 분류 매핑 |

## 신약개발 적용

흡수·분포 AI 예측은 — 발견 단계에서 가장 광범위하게 정착된 영역입니다. 대표 적용 사례와 정량 비교:

| 사례·도구 | 적용 영역 | 핵심 결과 |
|---------|--------|--------|
| **Insilico Medicine, Pharma.AI ADMET** | INS018_055 발굴 시 흡수·분포 필터로 60만 → 1만 후보 1차 선별 | 발견~임상 진입 30개월, 비용 약 270만 달러 (전통 평균의 약 1/10) |
| **Recursion LOWE + MolPhenix** | 표현형 스크리닝 결과를 ADMET 예측과 통합 | 임상 후보 4개 동시 운영, 2024년 8월 Exscientia 합병 발표 |
| **Genentech의 사내 BBB 모델 (2023 *J. Med. Chem.*)** | CNS 후보 우선순위, logBB + K_p,uu 통합 | BBB 통과 후보 적중률 약 35% → 약 70%로 향상 |
| **AstraZeneca의 PPB 사내 모델** | 95~99% 구간 별도 모델 운영 | 임상 1상 PK 변동성 예측 정확도 약 1.6배 향상 |

**대표 정량 비교 — 발견 단계 흡수·분포 평가의 변화**:

| 항목 | 전통 in vitro 패널 | AI 4-물성 cascade |
|------|--------------|------------|
| 분자당 비용 | 약 1,500~5,000 달러 (4개 측정) | 약 0.04 달러 (계산) |
| 처리 속도 | 약 2~4주 | 수초 (분자당) |
| 평가 가능 분자 | 수십~수백 | 수백만~수십억 |
| 회귀 정확도 (R² 또는 MAE) | 1.0 / 0 (정의상) | logS R²≈0.85, Caco-2 MAE≈0.35, PPB MAE≈8%, BBB AUROC≈0.85 |
| 임상 외삽 | 직접 측정 | 종간·in vitro-in vivo 격차 보정 필요 |

**산업적 통찰 — Cao et al. 2024 (*J. Cheminform.*)**는 ADMET-AI 등 6개 오픈소스 ADMET 모델을 다중 사내 데이터셋에 적용한 메타분석에서, **흡수·분포 4개 물성의 종합 cascade 통과 후보의 임상 후보 진입률이 ADMET 미적용군 대비 약 2.3배 높았다**고 보고했습니다. 이는 — AI 흡수·분포 예측이 "정확도 향상"보다 "후보 풀의 약학적 품질 향상"이라는 다른 가치 축에서 작동한다는 점을 보여줍니다. 약학 전공자가 이 cascade의 cutoff를 적응증·환자군별로 조정하면 — 같은 모델로도 2.3배 → 3~4배까지 진입률을 끌어올릴 여지가 있습니다.

## 창업 관점

흡수·분포 4-물성 예측은 — 1인 약학 창업자의 가장 현실적 진입 영역입니다. TDC 표준 데이터셋(AqSolDB, Caco2_Wang, BBB_Martins, PPBR_AZ)이 모두 공개되어 있고 ChemProp v2 + MolFormer 임베딩의 멀티태스크 학습으로 약 1주일 내 베이스라인 구축이 가능합니다. 차별 포인트는 — (a) **적응증·투여경로별 cutoff 엔진** (CNS / 비CNS / 만성 vs 급성), (b) **BCS 분류 자동 매핑**과 제형 권고 (Class II → 가용화 기술 제안), (c) **K_p,uu 등 자유 분획 기반 임상 지표** 출력, (d) **chemotype 적용성 도메인 점수**의 별도 보고. 시장 진입점은 — 빅파마용 상용 도구(StarDrop, ADMET Predictor)의 연 5만~50만 달러 가격을 회피해야 하는 중소 바이오·CRO·학계 대상의 월 500~3,000 달러 SaaS이며, CRO 시장만 2024년 약 750억 달러 규모(BCC Research 2024, 확인 필요) 중 발견 CRO 약 80억 달러 부문에서 흡수·분포 평가 외주 비용을 직접 대체합니다. MVP 구성 — Chemprop v2 fine-tuning + 4-물성 멀티태스크 헤드 + Streamlit 대시보드 + 적응증 선택 UI를 바이브코딩으로 약 2주 내 구현 가능. 핵심 해자는 모델 정확도가 아니라 — **약학적 의사결정 레이어의 코드화 깊이**입니다.

## 오늘의 과제

1. **4-물성 적응증 cutoff 매트릭스 작성 (50분)**: 본인 관심 적응증 3개(예: CNS 알츠하이머, 항암 폐암, 만성 대사질환 당뇨)에 대해 — logS, Caco-2 logPapp, PPB fu, BBB(logBB 또는 BBB+) 각 물성의 **(a) hard cutoff, (b) soft penalty 임계값, (c) 적응증별 가중치(1~5), (d) 약학적 근거 1줄**을 표로 정리하세요. 예를 들어 CNS 적응증에서 BBB는 가중치 5(hard), 비CNS 항암에서는 가중치 2(soft penalty)로 설정. 표 마지막에 — 같은 후보 분자 풀(예: 1만 개)에 본인 매트릭스를 적용했을 때 적응증별로 살아남는 후보 비율이 어떻게 달라질지 가설을 1문단으로 작성하세요.

2. **TDC 흡수·분포 리더보드 정밀 분석 (40분)**: tdcommons.ai의 ADMET Benchmark Group에서 **AqSolDB, Caco2_Wang, BBB_Martins, PPBR_AZ** 4개 데이터셋의 상위 5개 모델 점수를 표로 정리하세요. 각 데이터셋에서 — (a) 1위 모델이 다른 3개 데이터셋에서 몇 위인지, (b) 4개 모두에서 안정적 상위권인 모델이 있는지, (c) 멀티태스크 vs 단일태스크 모델의 평균 성능 차이를 비교하세요. 결론으로 — 본인의 wrapper SaaS 백엔드로 어떤 단일/통합 모델을 채택할지 3가지 근거와 함께 결정하세요.

3. **흡수·분포 cascade 시뮬레이션 시나리오 (30분)**: 본인이 PROTAC(약 800~1,200 Da, 일반적으로 logS 낮음, BBB 통과 어려움) 또는 펩타이드 치료제(흡수성 매우 낮음) 개발 회사라고 가정하고 — 본 Day의 4-물성 표준 cascade가 이 모달리티에서 왜 적용 불가능한지(또는 어떻게 수정해야 하는지)를 1쪽으로 분석하세요. 이는 Day 1의 "drug-like 편향" 한계의 구체적 사례이며, 본인 사업의 모달리티 선택에 직접 영향을 줍니다.

## 참고 자료

- **흡수·분포 표준 교과서**: Smith, D.A., Allerton, C., Kalgutkar, A.S., van de Waterbeemd, H., Walker, D.K. *Pharmacokinetics and Metabolism in Drug Design*. Wiley, 3rd ed., 2012. — 약학 전공자가 AI 예측 출력을 해석할 때의 표준 참조서.
- **logS 벤치마크**: Delaney, J.S. "ESOL: Estimating Aqueous Solubility Directly from Molecular Structure." *J. Chem. Inf. Comput. Sci.*, 2004. — ESOL 데이터셋과 logS 회귀의 표준 baseline.
- **AqSolDB**: Sorkun, M.C., Khetan, A., Er, S. "AqSolDB, a curated reference set of aqueous solubility." *Scientific Data*, 2019. — 9,982개 정제 logS 데이터셋.
- **BBB 검증**: Meng, F. et al. "B3DB: a curated database of blood-brain barrier permeability." *Scientific Data*, 2021. — logBB와 BBB+/−의 표준 데이터셋.
- **K_p,uu와 자유 약물 가설**: Smith, D.A., Di, L., Kerns, E.H. "The effect of plasma protein binding on in vivo efficacy." *Nat. Rev. Drug Discov.*, 2010. — PPB와 자유 분획의 임상 의미.
- **AI ADMET 메타 분석**: Cao, D., et al. "Comprehensive evaluation of open-source ADMET prediction tools." *J. Cheminform.*, 2024. — 6개 오픈소스 ADMET 모델의 사내 데이터 외부 검증.
- **오픈소스 베이스라인**: Chemprop v2 (chemprop/chemprop), ADMET-AI (swansonk14/admet_ai), TDC PyTDC (mims-harvard/TDC) — 1인 창업자가 즉시 활용 가능한 표준 스택.
