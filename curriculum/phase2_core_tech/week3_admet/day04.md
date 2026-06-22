# Day 4: 독성 예측 — hERG, 간독성, 변이원성, 임상 실패

> 이전 학습(Phase 2 Week 3 Day 3)에서 대사·배설(CYP, CL, t1/2) AI 예측과 PK 1구획 자동 구성을 다뤘습니다. 오늘은 ADMET 5축의 마지막 축인 **T(Toxicity)**로 들어가 — **hERG 심독성, 약물유발 간독성(DILI), 변이원성·유전독성, 일반 독성(LD50, organ toxicity)** 네 영역의 AI 예측 표준과, 약학 전공자가 모델 출력을 임상·규제 맥락으로 번역하는 의사결정 레이어를 정리합니다. 이 영역은 — 임상 회수의 직접 원인이 되며, 약학 전공자의 독성학·규제 지식이 가장 강력한 차별점으로 작동하는 ADMET 축입니다.

## 개요

독성은 — **임상 단계 약물 실패의 단일 최대 원인**입니다. Sun et al. 2022(*Acta Pharm. Sin. B*) 분석에 따르면 — Phase II/III 실패의 약 30%가 안전성 문제이며, 시판 후 회수의 약 40%가 hERG 매개 QT 연장과 특발성 간독성(idiosyncratic DILI)입니다. AI 독성 예측은 — 발견 단계에서 (a) hERG IC50 < 10 μM 분자의 사전 제거, (b) DILI 위험 분자의 합성 우선순위 강등, (c) Ames 양성 분자의 SAR 회피, (d) organ toxicity 신호를 가진 scaffold 회피를 자동화합니다. 본 Day는 네 영역의 분자 수준 메커니즘, 표준 AI 모델·데이터셋과 2024년 성능 수준, 그리고 약학 전공자가 단독으로 구축할 수 있는 "규제·임상 해석 레이어"의 구체적 설계를 다룹니다.

## 핵심 개념

### 1) hERG 심독성 — 임상 회수의 1순위

**hERG(human Ether-à-go-go Related Gene) 채널**은 심실 재분극의 Ikr 전류를 담당하는 칼륨 채널이며, 차단 시 — QT 간격 연장 → torsades de pointes(TdP) → 돌연사로 이어집니다. 1990~2000년대 사이 **terfenadine, cisapride, astemizole, sertindole** 등 7개 약물이 hERG 매개 QT 연장으로 시판 후 회수되었습니다. 현재 ICH E14·S7B 가이드라인은 — 모든 신약에 대해 hERG IC50 측정과 in vivo QT 평가를 요구합니다.

**약학적 임계값과 안전 여유(safety margin)**:

| hERG IC50 | 평가 | 임상 의미 |
|-----------|------|---------|
| > 30 μM | 안전 | safety margin > 30배 (목표 C_max 약 1 μM 가정) |
| 10~30 μM | 주의 | in vivo QT 평가 필수 |
| 1~10 μM | 위험 | scaffold 수정 또는 폐기 검토 |
| < 1 μM | 차단 | 발견 단계 즉시 제거 |

> **safety margin = hERG IC50 / 자유 혈장 C_max** ≥ 30이 발견 단계의 표준 cut-off입니다. AI 예측 IC50만으로 결정하지 않고 — Day 3에서 다룬 fu(자유 분율)와 결합해 *자유* C_max 기준으로 평가해야 합니다.

**표준 데이터셋과 성능** — **TDC hERG_Karim**(n≈13,445, 이진 분류 1 μM cutoff)와 **TDC hERG_Wang**(n≈648, 회귀)이 표준입니다. 2024년 기준 — Chemprop D-MPNN + 사전학습 임베딩(MolFormer-XL) hERG_Karim AUROC 약 0.86~0.89, hERG_Wang Spearman R 약 0.55~0.65 수준입니다. **약학 전공자의 차별** — AI가 IC50만 출력해도 (a) 분자의 logP·pKa·basic amine 구조에서 hERG affinity의 약리화학적 근거를 추정하고, (b) 안전 여유 계산을 자동화하며, (c) Y652/F656 결합 부위와의 docking 신호를 추가하면 — 단일 IC50 회귀보다 임상 외삽도가 큰 폭으로 향상됩니다.

### 2) DILI — 간독성의 두 얼굴

**약물유발 간독성(Drug-Induced Liver Injury, DILI)**은 — 시판 후 회수의 두 번째 큰 원인이며, 두 가지 별개 메커니즘을 가집니다.

| 유형 | 메커니즘 | 예측 가능성 | 대표 사례 |
|------|--------|----------|--------|
| **Intrinsic (용량 의존)** | 직접 간세포 독성, 미토콘드리아 손상 | 비교적 가능 | acetaminophen 과량 |
| **Idiosyncratic (특발성)** | 면역 매개, HLA 다형성, 대사 활성화 | 매우 어려움 | troglitazone, bromfenac |

**약학적 깊이** — 특발성 DILI는 환자 100,000명당 약 1~10건 빈도로 발생하지만 — 한번 발견되면 시판 약물도 회수됩니다(troglitazone, ximelagatran). AI 예측의 한계는 — 면역학적 반응(HLA-B*5701 같은 유전형 의존)을 분자 구조만으로는 정확히 예측할 수 없다는 점이며, 약학 전공자는 — (a) **반응성 대사물(reactive metabolite) 형성 위험**(quinone, epoxide 등의 친전자성 중간체), (b) **담즙산 수송체 BSEP 억제**(담즙정체성 손상의 핵심 메커니즘), (c) **미토콘드리아 OXPHOS 억제**(troglitazone 사례)를 별도 모델 신호로 분리해야 합니다.

**표준 데이터셋과 성능** — **TDC DILI**(n≈475, 이진 분류)는 데이터가 매우 작아 단독 학습 시 AUROC 약 0.75~0.82에 머무릅니다. **DILIst**(FDA, n≈1,279), **DILIrank**(n≈944, 위험도 4단계)가 보조 데이터로 활용되며, 멀티태스크 학습(BSEP + 반응성 대사물 + DILI 결합)으로 약 5~8% AUROC 향상이 보고됩니다. **약학 전공자의 차별** — DILI 단일 예측이 아니라 — (a) BSEP 억제, (b) 미토콘드리아 독성(HepG2 ATP 감소), (c) 반응성 대사물 형성(GSH conjugation 예측)을 결합한 **multi-mechanism DILI panel**이 단일 모델보다 임상 외삽도가 높습니다.

### 3) 변이원성과 유전독성 — Ames test의 AI 대체

**변이원성(mutagenicity)**은 — 약물 또는 그 대사물이 DNA 변이를 유발하는 성질이며, ICH M7 가이드라인은 모든 신약 후보의 변이원성 평가를 요구합니다. 표준 in vitro 시험은 **Ames test**(*Salmonella typhimurium* TA98, TA100 등 5종 균주 ± S9 mix)이며, AI 예측은 이 시험을 발견 단계에서 사전 대체합니다.

**약학적 분류 — ICH M7의 5등급 시스템**:

| Class | 정의 | 처리 |
|-------|------|------|
| Class 1 | 알려진 발암물질·변이원 | 회피 또는 매우 엄격한 한도 |
| Class 2 | 알려진 변이원, 발암성 미상 | TTC(독성학적 우려 임계값) 적용 |
| Class 3 | 경계성 구조 (alerting structure) | 추가 평가 또는 정제 강화 |
| Class 4 | 경계성 구조 있으나 동일 scaffold 데이터로 안전 입증 | 일반 한도 |
| Class 5 | 변이원성 우려 없음 | 일반 ICH Q3A/B 한도 |

**표준 데이터셋과 성능** — **TDC AMES**(n≈7,278, 이진 분류)가 표준이며, 2024년 기준 ChemProp D-MPNN AUROC 약 0.84~0.86, MolFormer 사전학습 결합 시 약 0.86~0.88 수준입니다. **structural alert 기반 접근**(Toxtree, VEGA, Sarah Nexus)과 AI 모델의 **앙상블**이 단일 AI보다 ICH M7 요구사항에 더 적합합니다 — 규제 당국이 AI 단독 예측을 수용하지 않고, 전문가 검토와 결합된 expert-guided 시스템을 요구하기 때문입니다.

**약학 전공자의 차별** — Ames 양성 신호가 떴을 때 — (a) 구조적 alert(aromatic amine, nitroaromatic, epoxide 등)의 위치를 식별하고, (b) S9 활성화(+) vs (−) 결과를 분리 예측하며(대사 활성화 여부), (c) ICH M7 Class 분류를 자동 출력할 수 있는 시스템이 규제 단계 비용을 직접 절감합니다.

### 4) 일반 독성·organ toxicity — LD50과 Tox21

**LD50**과 **organ-specific toxicity**(신독성, 심독성, 신경독성 등)는 — Tox21 컨소시엄(EPA·NIH·FDA, 2008~)의 약 12,000개 화합물 × 70개 in vitro 어세이 데이터로 AI 학습이 활성화된 영역입니다.

**대표 어세이와 약학적 의미**:

| 어세이 | 대상 메커니즘 | 약학적 의미 |
|------|----------|----------|
| **AR/ER agonist** | 내분비 교란 | 호르몬 시스템 부작용 |
| **AhR activation** | xenobiotic 감지 | CYP 유도, hepatotoxicity 연관 |
| **mitochondrial membrane potential** | 미토콘드리아 독성 | DILI·근육독성 시그널 |
| **p53 induction** | DNA 손상 반응 | 유전독성·발암성 |
| **NRF2/ARE** | 산화 스트레스 | reactive metabolite 형성 |

**Tox21 챌린지(2014~2015)** 우승 모델 **DeepTox**(Mayr et al.)는 — 12개 어세이 평균 AUROC 약 0.84를 달성했으며, 이후 ChemProp·Uni-Mol 기반 멀티태스크 모델이 약 0.86~0.88로 개선되었습니다. **TDC LD50_Zhu**(n≈7,385, 마우스 경구 LD50 회귀)는 — 일반 독성의 정량 지표로 활용되며 RMSE 약 0.55~0.65 log unit 수준입니다.

**약학 전공자의 차별** — Tox21 70개 어세이 출력을 — 약학 전공자가 (a) **장기별 위험 매핑**(간/심장/신장/신경/생식), (b) **MoA(mechanism of action) 클러스터링**, (c) **scaffold-level off-target 패널**로 재구성하면, raw AUROC 출력보다 신약 후보 의사결정에 훨씬 유용한 형태가 됩니다.

## 작동 원리와 아키텍처

독성 예측 통합 시스템의 표준 구조 — 2024년 기준:

```
[독성 cascade 예측 시스템]

1. 입력 표준화
   - SMILES → RDKit 표준화
   - structural alerts 사전 스캔 (Toxtree·SMARTS 규칙)
   - 반응성 대사물 후보 예측 (BioTransformer·MetaSite)

2. 멀티태스크 공유 encoder
   - Chemprop D-MPNN + MolFormer-XL 임베딩
   - 256~512차원 표현

3. 멀티헤드 독성 예측
   a) hERG IC50 (회귀) + 분류 (1 μM cutoff)
   b) DILI panel (DILI·BSEP·미토콘드리아·반응성 대사물)
   c) Ames (S9± 분리) + 12-pathway Tox21
   d) LD50 회귀 + organ-specific 어세이 출력

4. 약학적 후처리 레이어 — 차별 영역
   - hERG safety margin = IC50 / 자유 C_max
   - ICH M7 Class 자동 분류
   - DILI 다중 메커니즘 통합 점수
   - 장기별 위험 매트릭스 자동 구성

5. 규제 외삽 레이어
   - ICH S7B / E14 hERG·QT 평가 권고
   - ICH M7 변이원성 등급 + 한도 계산
   - FDA DILIrank 기반 위험 등급 외삽
   - Structural alerts → SAR 회피 권고
```

핵심 설계 결정 — 독성 cascade에서의 차별 지점:

| 결정 | 일반 접근 | 약학·규제 차별 |
|------|--------|--------|
| hERG 출력 | 단일 IC50 | safety margin + 결합 부위 도킹 |
| DILI 출력 | 단일 이진 분류 | BSEP + 미토콘드리아 + 반응성 대사물 통합 |
| Ames 출력 | 단일 AUROC | S9± 분리 + ICH M7 Class 분류 |
| Tox21 출력 | 12개 raw AUROC | 장기별 위험 매트릭스 + MoA 클러스터 |
| 규제 연결 | 없음 | ICH 가이드라인 직접 매핑 + 권고문 자동 생성 |

## 신약개발 적용

독성 AI 예측은 — 발견 단계 합성 우선순위 결정과 후기 단계 회수 위험 사전 평가에 활용됩니다. 대표 사례:

| 사례·도구 | 적용 영역 | 핵심 결과 |
|---------|--------|--------|
| **DeepTox (Mayr 2016, *Front. Environ. Sci.*)** | Tox21 챌린지 우승 멀티태스크 DNN | 12 어세이 평균 AUROC ≈0.84 |
| **ProTox-3.0 (Banerjee 2024, *Nucleic Acids Res.*)** | 33개 독성 endpoint 무료 웹 도구 | 사용량 누적 약 200만 분자, organ toxicity 통합 |
| **AstraZeneca 사내 hERG 모델 (2023 *J. Chem. Inf. Model.*)** | 사내 약 30만 분자 hERG 데이터 학습 | 발견 단계 hERG 양성률 약 40% 감소 |
| **Pfizer DILI Score (Greene 2010, *Chem. Res. Toxicol.*, 갱신 2022)** | Daily dose × LogP 결합 규칙 | 임상 DILI 분리 정확도 약 75% |
| **Insilico Medicine, INS018_055 발굴 시 독성 cascade** | hERG·DILI·Ames 통합 사전 필터 | 발견 단계 합성 후보 약 80% 사전 제거 |
| **ADMET-AI (Stanford 2024)** | 41개 ADMET task 중 독성 약 15개 통합 | 무료 웹 도구, TDC 리더보드 상위 |

**대표 정량 비교 — 독성 평가의 변화**:

| 항목 | 전통 in vitro 패널 | AI 독성 cascade |
|------|--------------|------------|
| 분자당 비용 | 약 5,000~20,000 달러 (hERG patch clamp + DILI panel + Ames + Tox21) | 약 0.05 달러 (계산) |
| 처리 속도 | 약 4~8주 | 수초 (분자당) |
| 평가 가능 분자 | 수십~수백 | 수백만~수십억 |
| 임상 외삽도 | 직접 측정이지만 표적-임상 격차 큼 | 메커니즘 기반 통합 시 약 70~80% |
| 규제 직접 활용 | 측정 데이터 그대로 제출 | ICH M7 expert-guided 시스템에 점차 수용 |

**산업적 통찰 — 회수 통계**: Onakpoya et al. 2016 분석에서 1953~2013년 시판 후 회수 약 462개 약물 중 — 약 32%가 hERG·QT, 약 18%가 DILI, 약 11%가 발암성·변이원성이었습니다. AI 독성 cascade가 발견 단계에서 hERG·DILI 위험을 사전 차단할 수 있다면 — 시판 후 회수의 약 30~40%를 예방할 잠재력이 있다는 것이 ADMET-AI·DeepTox 누적 분석의 결론입니다. **규제 측면 — FDA는 2023년 *Innovative Science Initiative*를 통해 ICH M7 변이원성 평가에 AI/ML 도구의 활용을 점진적으로 수용하고 있으며**, 이는 약학·규제 지식을 갖춘 창업자에게 직접적 시장 기회를 의미합니다.

## 창업 관점

독성 예측은 — Phase 2 Week 3의 ADMET 통합 SaaS에서 **규제 외삽 레이어**가 가장 두꺼운 차별 영역입니다. 단일 hERG IC50 모델은 — TDC 데이터셋만으로 약 1주일 베이스라인 구축이 가능하고 무료 도구(ADMET-AI, ProTox-3.0)와 경쟁이 어렵습니다. 차별 포인트는 (a) **safety margin 자동 계산**(Day 2 PPB + Day 3 CL 결합으로 자유 C_max 추정), (b) **ICH M7 자동 분류**(Class 1~5 + TTC 한도 계산), (c) **DILI multi-mechanism panel**(BSEP + 미토콘드리아 + 반응성 대사물), (d) **규제 권고문 자동 생성**(ICH S7B/E14/M7 가이드라인 인용 포함)입니다. **시장 위치** — 중소 바이오·임상약리학 컨설팅·CRO 대상 미들 마켓에서 — 월 2,000~10,000 달러 구간이 현실적입니다. Simulations Plus ADMET Predictor·Schrödinger LiveDesign은 연 5만~50만 달러로 빅파마 표준이지만, 규제 외삽 자동화는 약하므로 — "약사가 만든 규제 친화 ADMET" 포지셔닝이 가능합니다. **핵심 해자는 규제 지식의 깊이** — 본인의 ICH 가이드라인 해석과 약학 전공 지식이 코드로 자동화되면, 비약학 경쟁자는 6~12개월 학습 격차를 따라잡기 어렵습니다. 단, **AI 단독 예측을 임상·규제 결정에 사용하지 말 것**이라는 명시적 면책 조항과 — 전문가 검토 워크플로우(expert-in-the-loop) 통합이 신뢰의 전제 조건입니다.

## 오늘의 과제

1. **시판 회수 약물 hERG·DILI 분석 (45분)**: WHO Pharmaceuticals Newsletter·FDA 회수 데이터베이스에서 — 2000~2024년 사이 hERG 매개 QT 연장 또는 DILI로 회수된 약물 5개를 선택하세요. 각 약물에 대해 (a) 분자 구조(SMILES), (b) C_max·hERG IC50·safety margin(공개 자료), (c) 회수 결정의 시점과 ICH 가이드라인 위치, (d) AI 모델이 발견 단계에서 사전 감지할 수 있었을 가능성을 1~5등급으로 평가해 표로 정리하세요. 결론으로 — 본인 SaaS의 hERG·DILI 모듈이 어떤 회수 사례를 사전 차단하는 데 효과적일지 3개 시나리오를 정의하세요.

2. **TDC 독성 리더보드 + ADMET-AI 비교 분석 (40분)**: tdcommons.ai ADMET Benchmark Group에서 **hERG_Karim, AMES, DILI, LD50_Zhu** 4개 데이터셋의 상위 5개 모델 점수를 표로 정리하세요. 그 다음 — admet.ai.greenstonebio.com에 본인 관심 적응증의 표준 처방 약물 3개(예: clopidogrel, atorvastatin, warfarin)를 입력해 — 41개 ADMET 출력 중 독성 관련 15개의 결과를 받아보세요. (a) 실제 임상에서 알려진 부작용과의 일치도, (b) 모델이 놓친 신호(특발성 DILI, off-target 등), (c) 본인 SaaS가 ADMET-AI 대비 추가해야 할 약학·규제 후처리를 정리하세요.

3. **ICH M7 자동 분류 시나리오 설계 (35분)**: 본인이 발견 단계 후보 분자 3개에 대해 AI 출력 — Ames(S9−) 0.6, Ames(S9+) 0.85, structural alert "aromatic amine" 검출, hERG IC50 12 μM, DILI 양성 확률 0.4 — 를 받았다고 가정하세요. 이 분자가 (a) ICH M7 Class 몇으로 분류되는지(SAR 데이터 가정 명시), (b) TTC 한도가 적용된다면 일일 허용량은(1.5 μg/day 기준), (c) hERG safety margin이 충분한지(목표 C_max 1 μM, fu 0.05 가정), (d) DILI 신호를 발견 단계에서 SAR 회피로 처리할지 후기 단계 in vitro로 검증할지 의사결정 흐름도를 작성하세요. 이 의사결정 자동화가 — 본인 SaaS의 규제 외삽 레이어의 핵심 자산입니다.

## 참고 자료

- **hERG 기초**: Sanguinetti, M.C., Tristani-Firouzi, M. "hERG potassium channels and cardiac arrhythmia." *Nature*, 2006. — hERG 채널과 임상 QT 연장의 분자적 연결의 표준 정리.
- **회수 통계**: Onakpoya, I.J., Heneghan, C.J., Aronson, J.K. "Post-marketing withdrawal of 462 medicinal products because of adverse drug reactions." *BMC Med.*, 2016. — 1953~2013년 회수 약물의 ADR 원인 정량 분석.
- **DILI 임상**: Chalasani, N. et al. "ACG Clinical Guideline: Diagnosis and Management of Idiosyncratic Drug-Induced Liver Injury." *Am. J. Gastroenterol.*, 2021. — 특발성 DILI의 임상 진단·관리 표준.
- **Tox21·DeepTox**: Mayr, A. et al. "DeepTox: Toxicity Prediction using Deep Learning." *Front. Environ. Sci.*, 2016. — Tox21 챌린지 우승 모델 표준 참조.
- **ProTox-3.0**: Banerjee, P. et al. "ProTox 3.0: a webserver for the prediction of toxicity of chemicals." *Nucleic Acids Res.*, 2024. — 33 endpoint 무료 웹 도구 (tox.charite.de/protox3).
- **임상 실패 통계**: Sun, D. et al. "Why 90% of clinical drug development fails and how to improve it?" *Acta Pharm. Sin. B*, 2022. — 임상 단계별 독성 실패 통계.
- **ICH 가이드라인**: ICH M7(R2) 2023 (변이원성), ICH S7B/E14 2015 갱신 (hERG·QT) — 모든 신약의 규제 표준.
- **오픈소스 베이스라인**: Chemprop v2, ADMET-AI(swansonk14/admet_ai), PyTDC, Toxtree (oss) — 1인 창업자가 즉시 활용 가능한 스택.
