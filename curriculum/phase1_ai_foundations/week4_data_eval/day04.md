# Day 4: 벤치마크와 재현성 — MoleculeNet과 TDC

> 이전 학습(Week 4 Day 3)에서 **하나의 숫자로 모델의 가치를 표현할 수 없다**는 점, 그리고 신약 맥락에서는 AUPRC·EF·Activity cliff·Calibration까지 함께 보아야 한다는 점을 다뤘습니다. 오늘은 이러한 평가 지표들이 **누가, 어떤 데이터로, 어떤 분할로 측정했는지**를 통일해야만 모델 간 공정한 비교가 가능해진다는 사실, 즉 **벤치마크와 재현성**을 다룹니다. MoleculeNet에서 시작해 TDC, Polaris에 이르는 흐름은 AI 신약개발 분야가 "각자 자기 데이터로 SOTA"라는 무정부 상태를 어떻게 벗어나려 했는지를 보여줍니다.

## 개요

벤치마크(benchmark)는 동일한 데이터, 동일한 분할, 동일한 평가 절차 아래에서 여러 모델의 성능을 비교 가능하게 만드는 표준 시험대입니다. AI 신약개발은 2018년 **MoleculeNet**의 등장으로 비로소 공통 시험대를 갖게 되었고, 2021년 **Therapeutic Data Commons(TDC)**가 ADMET·표적 활성·임상 예측 등 70여 개 과제로 그 범위를 확장했으며, 2024년 **Polaris**는 기존 벤치마크의 **데이터 누출(leakage)**과 평가 불공정 문제를 정조준하며 등장했습니다. 그러나 벤치마크가 있다는 사실과 그것이 **신뢰할 수 있는가**는 별개의 문제입니다. 오늘은 표준 벤치마크의 구조와 한계, 그리고 재현성(reproducibility) 위기를 약학 전공자의 관점에서 어떻게 해석해야 하는지를 정리합니다. 좋은 벤치마크 위에서만 좋은 모델 선택이 가능하며, 결국 좋은 신약 후보 선택도 가능합니다.

## 핵심 개념

### 1) 벤치마크가 필요한 이유 — "내 모델이 SOTA"의 함정

학술적 신약 AI 분야의 오랜 문제 중 하나는 **각 논문이 서로 다른 데이터·분할·지표로 SOTA를 주장**한다는 점입니다. 같은 BBB(blood-brain barrier) 분류라도 데이터셋 A에서 학습·평가하면 AUROC 0.93, 데이터셋 B에서는 0.79가 나옵니다. 두 논문이 같은 "BBB 모델"을 주장해도 사용자는 어느 쪽이 자기 라이브러리에서 더 잘 작동할지 알 수 없습니다. 이 문제를 해결하기 위해 벤치마크는 다음 네 가지를 고정합니다.

- **데이터(data)**: 명시적인 버전과 소스. ChEMBL release 번호, PDB 다운로드 날짜.
- **분할(split)**: train/valid/test의 정확한 인덱스 또는 분할 규칙(scaffold, temporal).
- **지표(metrics)**: 어떤 지표를, 어떤 방식으로 계산할지의 함수 구현까지 공유.
- **절차(protocol)**: 학습 시드, 하이퍼파라미터 탐색 범위, 모델 선택 기준.

이 네 가지가 동일해야 두 모델의 차이가 **모델 자체의 차이**임을 주장할 수 있습니다. 약학에서 익숙한 **GLP(Good Laboratory Practice)**의 원리와 본질이 같습니다 — 실험 조건을 통제해야 결과의 비교가 의미를 가집니다.

### 2) MoleculeNet — 분야 표준의 시작 (2018)

**MoleculeNet**(Wu et al. 2018 *Chem. Sci.*)은 분자 물성·생리활성 예측 17개 데이터셋을 표준 분할과 함께 묶어 공개한 첫 대규모 벤치마크입니다. 카테고리는 다음과 같습니다.

| 카테고리 | 데이터셋 예 | 과제 유형 |
|---------|------------|----------|
| 양자역학 | QM7, QM8, QM9 | 회귀(에너지, 쌍극자 모멘트) |
| 물리화학 | ESOL, FreeSolv, Lipophilicity | 회귀(용해도, logD) |
| 생리활성 | PCBA, MUV, HIV, BACE | 분류(활성/비활성) |
| 생체활성 (ADMET) | BBBP, Tox21, ToxCast, SIDER, ClinTox | 분류(투과/독성) |

**기여**: 분야가 비교할 공통 언어를 갖게 됨. ChemProp(Yang et al. 2019), Uni-Mol(Zhou et al. 2023) 등 거의 모든 분자 모델 논문이 MoleculeNet 점수를 보고합니다.

**한계 — 실전과의 괴리**: ① 다수 데이터셋의 라벨 노이즈가 높습니다(Tox21, SIDER 등은 인 비트로/문헌 기반 라벨이라 측정 재현성이 낮음). ② 권장된 random/scaffold split이 통일적으로 적용되지 않은 채 논문마다 다르게 해석됩니다. ③ 일부 데이터셋(BBBP, BACE)은 크기가 작아(~2,000개 이하) 분산이 커서 SOTA 차이의 통계적 유의성이 약합니다. ④ 데이터 자체에 **누출(leakage)**이 포함된 사례가 후속 연구로 드러났습니다(아래 5번 참조). 약학적으로 보면 — Day 2에서 다룬 측정 노이즈와 전처리 결정의 천장 효과가 그대로 노출된 상태입니다.

### 3) Therapeutic Data Commons(TDC) — 다중 과제·다중 지표 (2021)

**TDC**(Huang et al. 2021 *NeurIPS Datasets and Benchmarks*)는 신약개발 전체 파이프라인을 22개 학습 과제 군으로 정리하고, 약 70여 개 데이터셋을 단일 Python 패키지(`pip install PyTDC`)로 제공합니다. 약학 전공자 관점에서 유의미한 차이는 다음과 같습니다.

- **ADMET 그룹** 단일 인터페이스로 22개 ADMET 데이터셋 접근 — Caco2, BBB Martins, hERG Karim, CYP 2C9/3A4 inhibition/substrate, Half_Life, Clearance Hepatocyte 등.
- **표준 split**: 각 데이터셋마다 **scaffold split**이 기본으로 적용됨. 사용자가 매번 별도로 결정할 필요가 없음.
- **다중 지표**: 같은 데이터셋에 대해 AUROC·AUPRC·MAE·Spearman을 동시에 리더보드에 표시. 단일 SOTA 주장의 함정을 줄임.
- **DOcking, 합성 가능성, 임상 시험 결과 예측 등** 단일 과제 벤치마크 너머로 범위를 확장.

**산업적 의미**: 스타트업이 "우리 모델 AUROC 0.91"을 주장할 때 빅파마 사이언티스트가 가장 먼저 묻는 질문이 "TDC에서는 어떤가요?"입니다. TDC 리더보드에 점수가 등록되어 있고, 같은 평가 절차에서 경쟁 모델 대비 우월하다면 즉시 검증된 신뢰가 생깁니다. 약학 전공자는 표적·약물군에 따라 어느 데이터셋이 임상적으로 의미있는지 알기 때문에, "어느 TDC 데이터셋에서 강한가"를 정확히 짚어 마케팅·실사 대응을 할 수 있습니다.

### 4) Polaris — 데이터 누출과 재현성을 정조준 (2024)

**Polaris**(Valence Labs · Recursion 주도, 2024)는 기존 벤치마크의 누적된 문제를 다음과 같이 진단했습니다.

1. **데이터 누출(leakage)**: train과 test에 사실상 같은 분자(또는 거의 같은 분자)가 동시에 포함되어 모델이 "암기"로 점수를 받는 경우. 대표적으로 MoleculeNet의 BBBP에서 일부 stereoisomer가 train/test에 양쪽 출현하는 문제, Tox21에서 중복 화합물 처리 미흡 문제가 보고된 바 있습니다.
2. **분할의 비현실성**: random split은 학습·테스트 화학 공간이 거의 동일해 실전 일반화 평가에 부적합. scaffold split도 의약화학적 유사성을 완전히 차단하지 못함.
3. **재현성 부족**: 같은 모델·같은 데이터로 다른 저자가 돌리면 점수가 ±5%까지 흔들리는 경우가 다수.

Polaris는 ① 클라우드 기반 평가(테스트셋 라벨 비공개) ② 누출 제거된 정제 데이터셋 재공급 ③ 명시적 분할 함수·평가 함수 버전 관리로 이 문제를 풉니다. 2024–2025년 사이 Recursion, AstraZeneca, Novartis 등이 공식 파트너로 참여하며 **차세대 산업 표준 벤치마크**로 자리잡고 있다고 보도되었습니다(확인 필요).

### 5) 재현성 위기 — 약학에서 익숙한 이야기

AI 분자 모델의 재현성 위기는 약학 전공자에게는 낯설지 않은 문제입니다. **임상 시험의 재현성 위기**(예: Begley & Ellis 2012 *Nature*의 53개 전임상 연구 중 6개만 재현)와 본질적으로 같은 구조를 갖습니다.

| 임상/전임상의 재현성 위기 | AI 모델의 재현성 위기 |
|------------------------|-------------------|
| 측정 노이즈, 모델 동물 차이 | 측정 노이즈, 라벨 정의 차이 |
| p-hacking, 사후 분석 | 하이퍼파라미터 사후 조정, test set 누출 |
| 출판 편향(양성 결과만 보고) | SOTA 편향(최고 점수만 보고) |
| GLP 미준수 | 평가 절차 비공개 |
| 임상 등록제, CONSORT | TDC 리더보드, Polaris |

이 대응 관계를 인식하면, 약학 전공자는 AI 모델 평가의 결함을 **자기 분야의 친숙한 문제**로 진단하고 해결책을 설계할 수 있습니다 — 이는 단순 비유가 아니라 동일한 통계적·방법론적 문제이기 때문입니다.

### 6) 벤치마크 선택과 보고의 약학적 기준

| 평가 시나리오 | 추천 벤치마크 | 보조 |
|--------------|--------------|-----|
| ADMET 표준 보고 | TDC ADMET Group | MoleculeNet (역사적 맥락) |
| 단백질-리간드 결합 | PDBbind v2020, PoseBusters | DOCKSTRING |
| 가상 스크리닝 농축 | LIT-PCBA, DUD-E (단, 편향 보고됨) | EF/BEDROC 지표 |
| 분자 생성 품질 | MOSES, GuacaMol | distribution learning + goal-directed 분리 |
| 단백질 구조 예측 | CASP15/16, CAMEO | AlphaFold 비교 baseline |
| 신규/엄격 평가 | Polaris | 클라우드 평가, 누출 차단 |

## 작동 원리와 아키텍처

### 벤치마크 평가 파이프라인

```
[데이터 로딩]
   │  TDC: from tdc.benchmark_group import admet_group
   │       group = admet_group(path = 'data/')
   │  표준화된 train/valid/test 분할 자동 제공
   │
   ├─ [모델 학습 — 5 seed 반복]
   │     seeds = [1, 2, 3, 4, 5]
   │     for seed in seeds:
   │         train/valid에서 학습 → test에서 예측
   │
   ├─ [지표 계산]
   │     TDC는 데이터셋별 권장 지표를 자동 적용
   │     ─ 회귀: MAE 또는 Spearman
   │     ─ 분류: AUROC 또는 AUPRC
   │
   ├─ [통계적 보고]
   │     평균 ± 표준편차 (5 seed)
   │     주장: 두 모델 차이가 표준편차 합보다 클 때만 유의
   │
   └─→ [리더보드 제출]
        ─ 모델 코드 + 시드 + 환경 공개
        ─ Polaris는 클라우드 평가로 test set 비공개 유지
```

### 좋은 벤치마크 보고의 체크리스트

| 항목 | 좋은 보고 | 흔한 나쁜 보고 |
|------|---------|-----------|
| 데이터 버전 | "ChEMBL 33, 2023-05" | "ChEMBL" |
| 분할 방식 | "Scaffold split, seed=42, 80/10/10" | "임의 분할" |
| 시드 반복 | 평균 ± SD over 5 seeds | 단일 seed 최고점 |
| 비교 baseline | 같은 split의 ChemProp, Random Forest | 자기 모델만 |
| 통계적 유의성 | bootstrap CI 또는 paired t-test | SOTA 주장만 |
| 누출 점검 | Tanimoto > 0.85 쌍 제거 보고 | 미언급 |

## 신약개발 적용

### 사례 1 — MoleculeNet의 BBBP 데이터 누출 분석

**Pat Walters의 2022년 블로그 분석**(*Practical Cheminformatics*)은 MoleculeNet의 BBBP(혈뇌장벽 투과성) 데이터셋에서 동일 분자가 다른 SMILES 표기로 train과 test에 동시 존재하는 사례, 그리고 거의 동일한 stereoisomer가 분리되지 않은 사례를 다수 보고했습니다. 정제 후 같은 모델을 다시 평가하면 AUROC가 0.03~0.05 하락했습니다. 산업적으로 의미있는 시사점은 — 학술 SOTA 차이의 상당 부분이 **모델 우월성이 아니라 데이터 누출 정도의 차이**에서 발생할 수 있다는 점입니다. 빅파마 평가팀은 도입 검토 시 데이터 누출 재검사를 표준 절차로 포함하기 시작했습니다.

### 사례 2 — TDC의 ADMET 리더보드와 산업적 영향

**TDC ADMET Benchmark Group**은 22개 ADMET 데이터셋에 대해 ChemProp, AttentiveFP, GROVER, Uni-Mol, MolFormer 등 주요 모델의 점수를 동일 평가 절차로 공개합니다. 흥미로운 관찰은 — **단일 모델이 모든 ADMET 과제에서 우월한 경우는 없다**는 점입니다. 같은 모델이 BBB Martins에서는 1위, hERG Karim에서는 5위, CYP2D6 substrate에서는 중하위권인 패턴이 흔합니다. 이는 **단일 universal ADMET 모델은 환상**이며, 사업적으로는 ADMET 카테고리별 전문 모델 포트폴리오가 더 합리적이라는 결론을 뒷받침합니다. 약학 전공자는 표적·약물군별로 어느 ADMET 지표가 중요한지 알기 때문에, 포트폴리오의 우선순위를 정확히 설정할 수 있습니다.

### 사례 3 — Polaris와 산업 표준 전환

**Polaris의 2024년 출범 발표**(Valence Labs)에 따르면 Recursion, AstraZeneca, Novartis, Microsoft, NVIDIA 등 25개 이상의 학계·산업 파트너가 참여하며, 기존의 학술 벤치마크가 가진 누출·평가 불공정 문제를 정조준합니다. 핵심 차별점은 **테스트 라벨을 비공개로 두고 클라우드에서 평가**한다는 점으로, 학술 모델 개발자가 부지불식간에 test set에 과적합하는 것을 구조적으로 차단합니다. 이는 약학 전공자가 익숙한 **이중 맹검(double-blind)** 임상 시험의 원리와 본질이 같습니다 — 평가자가 정답을 모르게 함으로써 편향을 제거하는 방식입니다(확인 필요: 최신 참여 기관 명단은 polarishub.io에서 확인 권장).

### 사례 4 — 도킹 벤치마크의 위기와 PoseBusters

**Buttenschoen et al. 2024 *Chem. Sci.***의 PoseBusters는 DiffDock 등 AI 도킹 모델이 보고된 RMSD < 2 Å 성공률(40–50%)이 실제로는 **물리적으로 불가능한 포즈를 포함한 채 측정된 결과**임을 보고했습니다. 결합 구조의 물리화학적 타당성(bond length, planarity, steric clash)을 함께 평가하면 진짜 성공률은 절반 이하로 떨어집니다. 이는 약학자가 실험적 결합 모드 검증에서 자연스럽게 적용하는 **품질 점검**을 벤치마크에 명시적으로 포함시킨 사례이며, "지표 숫자만으로 모델을 선택하면 안 된다"는 합의를 다시 확인시켰습니다.

## 창업 관점

Phase 1 단계에서는 짧게만 짚습니다 — AI 신약개발 스타트업의 기술 마케팅에서 가장 신뢰받는 한 줄은 "TDC 리더보드 1위"나 "Polaris 검증" 같은 **제3자 표준에서의 우월성** 보고입니다. 자체 데이터로 측정한 SOTA는 빅파마 실사에서 0점에 가깝게 평가됩니다. 약학 전공자가 임상적으로 의미있는 ADMET 카테고리에 집중해 그 영역에서 TDC 상위 점수를 안정적으로 유지하면, 좁고 깊은 차별화가 마케팅·파트너십·투자 모두에서 강력한 근거가 됩니다. 시장 규모와 BM 차원의 분석은 Phase 5에서 다룹니다.

## 오늘의 과제

1. **TDC ADMET 리더보드 비교 분석 (50분)**: tdcommons.ai의 ADMET Benchmark Group에서 데이터셋 3개(예: BBB_Martins, hERG_Karim, CYP3A4_Veith)를 선택하고, 각 데이터셋의 **상위 5개 모델**의 평균 ± SD 점수를 표로 정리합니다. 같은 모델이 세 데이터셋에서 순위가 어떻게 바뀌는지 관찰하고, 약학적으로 어느 데이터셋의 임상적 중요도가 높은지(BBB는 CNS 약물, hERG는 심독성, CYP3A4는 DDI) 한 단락으로 분석합니다. A4 1쪽.
2. **데이터 누출 자가 점검 시나리오 작성 (40분)**: 본인이 사내 BBB 분류 모델 개발 책임자라고 가정하고, **train/test 분리 전에 반드시 확인해야 할 누출 점검 항목 7개**를 체크리스트로 작성합니다(예: SMILES 정규화 후 중복, stereoisomer 처리, Tanimoto > 0.85 쌍 제거, scaffold split 비율 점검, 버전·날짜 기록, seed 고정, 외부 데이터셋 분리 보관 등). 각 항목에 약학적으로 "왜 중요한가"를 한 줄씩 덧붙입니다.
3. **벤치마크 선택 정책 초안 (30분)**: 본인 스타트업이 ADMET 예측 SaaS를 출시한다고 가정하고, 마케팅·기술 백서에 사용할 **벤치마크 보고 정책**을 1쪽으로 작성합니다. 항목 — (a) 어느 벤치마크를 보고할지(TDC vs Polaris vs MoleculeNet), (b) seed 반복 횟수와 통계 보고 형식, (c) baseline 비교 대상, (d) 데이터 누출 점검 절차, (e) 자체 사내 데이터 평가 결과의 보고 방식. Day 3의 평가 정책 초안과 묶으면 사내 ML 정책의 핵심이 됩니다.

## 참고 자료

- Wu, Z. *et al.* (2018). "MoleculeNet: A Benchmark for Molecular Machine Learning." *Chemical Science*, 9(2), 513–530. — 분자 ML 벤치마크의 출발점. 17개 데이터셋의 표준 분할과 지표를 정의.
- Huang, K. *et al.* (2021). "Therapeutic Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development." *NeurIPS Datasets and Benchmarks Track*. — 신약개발 전체 파이프라인을 포괄하는 22개 학습 과제 군 정의. ADMET Benchmark Group의 표준.
- Buttenschoen, M., Morris, G. M., Deane, C. M. (2024). "PoseBusters: AI-Based Docking Methods Fail to Generate Physically Valid Poses or Generalise to Novel Sequences." *Chemical Science*, 15, 3130–3139. — 도킹 벤치마크의 물리화학적 타당성 점검 도입.
- Walters, P. (2022~2023). "Practical Cheminformatics" 블로그 — MoleculeNet 데이터 누출, BBBP·BACE 사례 분석. 산업 현장의 검증 시각.
- Therapeutic Data Commons(tdcommons.ai) · Polaris Hub(polarishub.io) · MoleculeNet(moleculenet.org) — 본인 모델을 직접 등록·평가하며 익히는 것이 가장 효과적.
