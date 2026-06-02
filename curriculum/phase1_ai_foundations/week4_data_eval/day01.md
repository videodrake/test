# Day 1: 화학/생물 데이터베이스 — ChEMBL, PDB, ZINC

> 이전 학습(Week 3)에서 분자를 SMILES·지문·그래프·3D 좌표로 표현하는 네 가지 방식과 그 선택 기준을 다뤘습니다. Week 4의 첫날인 오늘은 그 표현들이 **무엇으로부터 만들어지는가** — 즉 ChEMBL·PDB·ZINC를 비롯한 공공 데이터베이스의 구조·규모·한계를 정리합니다. 표현이 모델의 **입력 형식**을 결정한다면, 데이터베이스는 모델의 **상한선과 일반화 범위**를 결정합니다.

## 개요

AI 신약개발의 모든 모델은 결국 공공 데이터베이스에서 출발합니다. **ChEMBL**은 문헌에서 정제된 약 240만 개 화합물과 약 2,000만 개 활성 측정치로 약리학적 활성·물성 학습의 표준 데이터셋이고, **PDB**(Protein Data Bank)는 약 22만 개 이상의 실험적 단백질 구조로 AlphaFold·DiffDock 등 구조 기반 AI의 기반이며, **ZINC**(특히 ZINC22)는 수십억 개 구매 가능 화합물 라이브러리로 가상 스크리닝의 검색 공간을 정의합니다. 세 데이터베이스는 각각 "활성(Activity) · 구조(Structure) · 가용성(Availability)"의 세 축을 담당하며, 산업 현장의 표준 파이프라인은 이 셋을 조합합니다. 오늘은 (1) 각 DB의 구성·규모·접근 방법, (2) 데이터 품질에서 발생하는 약학적 함정, (3) 약학 전공자가 비전공자보다 더 정확히 해석할 수 있는 지점을 정리합니다.

## 핵심 개념

### ChEMBL — 문헌에서 정제된 활성 데이터의 표준

**ChEMBL**은 EMBL-EBI(European Molecular Biology Laboratory–European Bioinformatics Institute)가 운영하는 공공 약리학 데이터베이스입니다. 2024년 기준 ChEMBL 34의 규모는 다음과 같습니다.

| 항목 | 규모 (ChEMBL 34) | 의미 |
|------|---------------|------|
| 화합물(distinct compounds) | 약 240만 개 | 학습·스크리닝의 기본 풀 |
| 활성 측정치(activities) | 약 2,030만 개 | IC50·Ki·Kd·EC50·% inhibition 등 |
| 표적(targets) | 약 15,000개 | 단백질·세포·organism 수준 모두 |
| 문헌(documents) | 약 88,000건 | 동료심사 논문 + 일부 특허 |
| 약물(approved/clinical) | 약 14,000개 | 임상 진입 단계 메타데이터 포함 |

핵심 특징은 **모든 활성값이 출처(논문 DOI, 특허번호)와 함께 큐레이션**된다는 점입니다. 같은 화합물의 IC50이 여러 논문에서 다르면 ChEMBL은 각각을 별도 레코드로 저장하고 어세이 조건(세포주, 기질 농도, 인큐베이션 시간)을 함께 기록합니다. 이 메타데이터가 ChEMBL을 PubChem 같은 단순 보관소(repository)와 구분짓는 결정적 지점입니다.

**약학 지식과 직접 연결되는 ChEMBL의 6가지 필드**:

1. **standard_type**: IC50/Ki/Kd/EC50 — 각 지표의 약학적 의미가 다름. IC50은 어세이 조건 의존, Ki는 더 본질적
2. **standard_relation**: `=` `>` `<` `~` — `>10000 nM`(미달 활성)은 회귀 모델에 직접 넣으면 안 됨
3. **assay_type**: B(binding) · F(functional) · A(ADMET) · T(toxicity) · P(physicochemical) — 모델링 전 분리 필수
4. **target_organism**: 사람·쥐·랫 — *species cross-reactivity* 가정은 위험
5. **confidence_score** (0-9): 표적 동정 신뢰도. 9는 단일 분자 표적 확정
6. **pchembl_value**: −log10(IC50) 형태의 통일 척도, 회귀에 직접 사용 가능

학부 약리학에서 배운 Hill 방정식·Cheng-Prusoff 보정(Ki = IC50 / (1 + [S]/Km))을 알면, 같은 표적의 IC50들을 비교 가능한 Ki로 환산할 때 약학 전공자가 비전공자보다 훨씬 빠른 판단을 내립니다.

**접근**: REST API(`ebi.ac.uk/chembl/api/`), 전체 SQLite/Oracle 덤프(약 5 GB 압축), Python 클라이언트(`chembl_webresource_client`). 라이선스는 **CC BY-SA 3.0** — 상업적 활용 가능하되 파생물 공유 의무가 있습니다.

### PDB — 단백질 구조의 보고

**PDB(Protein Data Bank)** 는 RCSB(미국)·PDBe(유럽)·PDBj(일본)가 공동 운영하는 3차원 구조 보관소입니다. 2024년 기준 약 22만 개 이상이 등재되어 있고 매년 약 15,000~20,000개씩 증가합니다.

| 결정 방법 | 비중 | 특징 |
|---------|------|------|
| X-선 결정학 | 약 85% | 고해상도(1.5-3 Å), 결정화 가능 단백질 한정 |
| 극저온 전자현미경(cryo-EM) | 약 10%(급증) | 대형 복합체·막단백질, 2024년 1.5 Å 수준 도달 |
| NMR | 약 5% | 용액 상태, 작은 단백질·내재적 무질서 영역 |

파일 형식은 전통적 PDB에서 **mmCIF/PDBx**로 표준이 전환되었으며, 사슬·리간드(HET 그룹)·물 분자·B-factor·점유율이 함께 기록됩니다. 약물 개발에서 가장 가치 있는 부분집합은 **PDBbind**(약 23,000개 단백질-리간드 복합체에 측정 친화도 부착)와 2024년 발표된 **PoseBusters 벤치마크**(AI 도킹 결과의 물리적 타당성 검증)입니다.

**약학 지식이 PDB 해석에서 결정적 차이를 만드는 지점**:

- **해상도(resolution)**: 2.0 Å 이하만 신뢰할지, 3.5 Å 막단백질도 받을지 — 표적이 GPCR이면 후자도 받아야 데이터가 충분
- **B-factor와 결측 잔기**: 활성 부위 loop가 결정에 안 잡힌 경우 AlphaFold로 보완 후 도킹할지 판단
- **결정학적 인공물**: detergent·cryo-protectant 분자가 결합 부위에 자리잡은 사례 식별
- **공결정 리간드 비교**: 같은 표적의 다른 PDB ID 사이에서 리간드별 conformation 차이가 induced fit인지 인공물인지 평가

AlphaFold3 등장 이후에도 **AI 예측 구조와 실험 PDB 구조의 교차검증**은 인간의 약학적 판단이 여전히 필요한 영역입니다.

### ZINC — 구매 가능 분자 라이브러리

**ZINC**(ZINC Is Not Commercial — UCSF Shoichet 연구실 운영)는 가상 스크리닝을 위한 구매 가능 화합물 데이터베이스입니다. 진화 경로는 다음과 같습니다.

| 버전 | 연도 | 규모 | 특징 |
|------|------|------|------|
| ZINC15 | 2015 | 약 7억 5천만 | 가상 스크리닝의 산업 표준이 됨 |
| ZINC20 | 2020 | 약 14억(tranche 단위 수십억 검색) | "make-on-demand" 확장 시작 |
| ZINC22 | 2022 | 약 370억+ | Enamine REAL 등 합성 가능 공간 통합 |

ZINC의 핵심은 **"실제로 살 수 있는 분자만 모은다"** 는 점입니다. PubChem이나 ChEMBL의 분자는 합성이 어려울 수 있지만, ZINC의 분자는 Enamine·Mcule 등 공급사 카탈로그를 통해 2주~수개월 내 입수 가능합니다. **make-on-demand**(요청 시 합성) 라이브러리인 Enamine REAL Space는 2024년 기준 약 380억 개에 도달했으며 ZINC22가 이를 검색 가능 형태로 제공합니다.

**ZINC의 약학적 의미**:

- **합성 가능성 보장**: AI 생성 분자가 실제로는 합성 불가능한 경우가 흔한데, ZINC 출발 가상 스크리닝은 이를 원천 차단
- **카탈로그 메타데이터**: 가격·리드 타임·순도·공급사가 함께 제공되어 우선순위화에 즉시 사용
- **drug-like 필터**: Lipinski Rule of 5·PAINS 필터가 사전 적용된 부분집합 제공
- **단계별 부분집합**: frag-like / lead-like / fragment subset이 분자량·logP 범위로 미리 분할

가상 스크리닝의 산업 표준 워크플로는 "ZINC22 수십억 분자 → 도킹 또는 ML 스코어링 → 상위 수천 개 우선순위화 → 실험 검증"입니다. 약학 전공자는 이때 **lipophilic efficiency(LipE = pIC50 − logP)**, **ligand efficiency(LE = ΔG / heavy atom count)** 같은 지표를 함께 반영해 "분자량이 클수록 도킹 점수가 좋게 나오는" 편향을 회피할 수 있습니다.

### 그 외 알아둘 보조 데이터베이스

| DB | 역할 | 규모 |
|----|------|------|
| **PubChem** (NCBI) | 화합물 보관소 — 큐레이션 약함, 규모 큼 | 약 1억 2천만 화합물 |
| **DrugBank** | 승인·임상 약물의 기전·표적·DDI | 약 17,000+ 약물 |
| **BindingDB** | 결합 친화도 데이터(ChEMBL과 일부 중복) | 약 280만 측정치 |
| **UniProt** | 단백질 서열·기능 주석 | 약 2억 5천만 서열 |
| **Therapeutic Data Commons (TDC)** | AI 벤치마크 모음(Day 4에서 상세) | 100+ 데이터셋 |
| **CCDC/CSD** | 소분자 결정 구조(상업) | 약 130만 |

## 작동 원리와 아키텍처

### AI 신약개발 데이터 파이프라인의 표준 흐름

```
[ChEMBL]  특정 표적의 활성 측정치 추출
   │      ─ standard_type=IC50, assay_type=B, confidence_score≥7
   │      ─ standard_relation="=" 만 (right-censored 제외)
   │      ─ pchembl_value 계산, 중복 InChIKey 통합 (median)
   ▼
[정제된 학습셋: SMILES + 활성값 (~수천~수만 분자)]
   │
   ├──→ Week 3에서 다룬 표현으로 변환 (지문/그래프/3D)
   │
   ├──→ [PDB] 표적 단백질 구조 결합 (도킹·구조 기반 모델용)
   │       ─ 해상도 ≤ 2.5 Å, 활성 부위 잔기 결손 없음 필터
   │
   └──→ [ZINC22] 가상 스크리닝 검색 공간 정의
           ─ drug-like subset, MW 250-500, logP -1~5
           ─ 학습된 모델로 점수 계산 → 상위 N개 우선순위화

[표적별 AI 모델] + [후보 분자 우선순위 목록]
```

각 단계가 어디서 다뤄지는지 — 활성값 정제·split은 Week 4 Day 2~3, 평가 지표는 Day 3, 벤치마크는 Day 4, 표적별 모델링은 Phase 2 전반에서 본격화됩니다.

### 데이터 접근 방식과 라이선스 요약

| DB | API | 다운로드 | 라이선스 | 상업 활용 |
|----|------|---------|---------|---------|
| ChEMBL | REST + Python client | SQLite/Oracle dump | CC BY-SA 3.0 | 가능, 파생물 공유 의무 |
| PDB | REST + GraphQL | mmCIF/PDB 직접 | 공공 도메인 | 자유 |
| ZINC20/22 | Web · 일괄 다운로드 | SMILES/SDF tranche | 자유 사용, 인용 권장 | 가능 |
| PubChem | REST + PUG-View | FTP bulk | 공공 도메인 | 자유 |
| DrugBank | API(라이선스) | XML/CSV | 학술 무료/상업 유료 | 상업 라이선스 필요 |

상업 활용에서 가장 주의할 것은 **DrugBank의 상업 라이선스**와 **ChEMBL의 ShareAlike 조항**입니다. 스타트업의 모델 가중치가 ChEMBL 파생물에 해당하는지에 대한 해석 논쟁이 있고, 일부 회사는 ChEMBL은 학습에만 쓰고 공개 가중치는 자체 데이터로 fine-tune한 버전만 배포하는 전략을 씁니다(IP는 Phase 4 Week 4에서 상세).

## 신약개발 적용

### 실제 사례로 본 데이터베이스의 위력

- **AlphaFold (DeepMind, 2021·2024)**: AlphaFold 2의 학습은 본질적으로 **PDB의 약 170,000개 구조**가 라벨이었습니다. AlphaFold 3(2024년 *Nature*)는 PDB를 단백질뿐 아니라 핵산·리간드 복합체로 확장 학습해 단백질-리간드 결합 포즈를 종단간(end-to-end) 예측할 수 있게 되었습니다. PDB 큐레이션의 수십 년 축적이 없었다면 불가능한 도약입니다.
- **할리신 (Stokes et al. 2020, *Cell*)**: 학습 데이터 2,335개는 자체 *E. coli* 성장 억제 스크린이었지만, 검색 공간은 **ZINC15의 약 1억 700만 분자** + Broad Repurposing Hub였습니다. ZINC가 없었다면 신규 골격 항생제 발견 공간 자체가 존재하지 않았습니다.
- **V-SYNTHES (Sadybekov et al. 2022, *Nature*)**: Enamine REAL Space의 약 110억 분자를 합성자(synthon) 단위로 분해해 가상 스크리닝하는 V-SYNTHES는 **ZINC22 / make-on-demand의 폭발적 성장이 직접 가능케 한 접근**입니다. ROCK1 키나아제 억제제 발굴에서 수십억 검색 → 60개 합성 → 단일 자릿수 µM 활성을 보고했습니다.
- **Insilico Medicine INS018_055**: 2022년 임상 1상, 2023년 2a상 진입한 IPF 치료제 후보의 초기 가상 스크리닝은 ChEMBL·자체 텍스트 마이닝 + ZINC 조합으로 알려져 있습니다. "공공 DB는 출발점이고 자체 데이터·실험 사이클이 차별점"이라는 산업 공통 패턴을 보여줍니다.

### 데이터 품질의 한계 — 모델 성능의 진짜 천장

| DB | 주요 한계 | 약학적 해석 |
|----|---------|---------|
| ChEMBL | 어세이 조건 이질성, 음성 결과 보고 부족, 출판 편향 | 같은 표적의 IC50 분포가 두 자릿수 흩어지는 게 정상 |
| PDB | 결정화 가능 단백질 편향, 막단백질 부족 | 키나아제는 많고 GPCR·이온채널은 상대적으로 적음 |
| ZINC | "구매 가능"이 실제로는 합성 실패율 10-30% | make-on-demand는 약속이지 보장이 아님 |

이 한계는 Day 2(데이터 품질·전처리)에서 본격적으로 다루며 Phase 2 전반에서 반복적으로 마주칩니다.

## 창업 관점

공공 DB는 모두가 쓰는 자원이므로 그 자체가 차별점은 아니지만, 약학 전공 창업자에게는 두 의미가 있습니다 — **(1)** ChEMBL/PDB의 잘 알려진 함정(어세이 이질성·해상도 편향·종간 차이)을 정확히 해석해 자체 데이터 큐레이션을 정직하게 한 회사는 일관되게 좋은 모델을 만들며, **(2)** make-on-demand 라이브러리(Enamine REAL · ZINC22)의 폭발적 성장으로 가상 스크리닝 검색 공간이 5년 사이 100배 이상 커진 것이 V-SYNTHES 같은 신규 알고리즘과 신생 스크리닝 서비스 창업 기회를 만들고 있습니다. 시장·BM 설계는 Phase 5에서 다룹니다.

## 오늘의 과제

1. **ChEMBL 표적 한 개를 직접 탐색 (50분)**: 관심 표적(예: EGFR, hERG, CYP3A4, JAK2 중 하나)을 선택해 ChEMBL 웹 인터페이스 또는 REST API로 접근합니다. 1쪽으로 정리 — (a) ChEMBL ID와 활성 측정치 총 개수, (b) standard_type 분포(IC50/Ki/Kd 비율), (c) confidence_score 9만 남겼을 때 개수, (d) `=` 관계 + `pchembl_value` 존재 조건으로 필터링 시 최종 학습 가능 크기, (e) 본인의 약학 지식으로 본 임상적 신뢰도 평가 한 문단.
2. **PDB와 ZINC의 표적 연결 (40분)**: 동일 표적의 PDB 검색 — 해상도 2.5 Å 이하 구조 몇 개, 그중 공결정 리간드를 가진 구조는 몇 개인지 정리합니다. 이어서 ZINC15/20의 drug-like subset에서 해당 리간드의 substructure를 가진 화합물이 대략 몇 개인지 확인하고, "이 표적의 가상 스크리닝을 시작한다면 어떤 검색 공간(ZINC 부분집합 또는 ZINC22 make-on-demand)에서 시작할지" 결정 근거를 적습니다. A4 1쪽.
3. **데이터 라이선스 비교 (30분)**: ChEMBL · DrugBank · PubChem · ZINC · CCDC/CSD의 라이선스를 본인이 창업자라고 가정하고 비교하는 표를 작성합니다. (a) 학술 연구 가능 여부, (b) 상업 모델 학습 가능 여부, (c) 학습된 모델 가중치 공개 가능 여부, (d) 모델을 SaaS로 판매할 때 라이선스 요구사항. 한 단락으로 "내 회사가 첫 모델 출시 시 ChEMBL을 어떻게 활용·표시할지" 정책을 결론으로 적습니다.

## 참고 자료

- Zdrazil, B. *et al.* (2024). "The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods." *Nucleic Acids Research*, 52(D1), D1180–D1192. — ChEMBL 34 공식 업데이트 논문. 데이터 구조·필드·접근 방법의 표준 출처.
- Berman, H. M. *et al.* (2000). "The Protein Data Bank." *Nucleic Acids Research*, 28(1), 235–242. — PDB 창립 논문. 50년 누적의 출발점.
- Tingle, B. I. *et al.* (2023). "ZINC-22 — A Free Multi-Billion-Scale Database of Tangible Compounds for Ligand Discovery." *Journal of Chemical Information and Modeling*, 63(4), 1166–1176. — ZINC22 공식 논문. make-on-demand 시대의 가상 스크리닝 검색 공간 정의.
- Sadybekov, A. A. *et al.* (2022). "Synthon-based ligand discovery in virtual libraries of over 11 billion compounds." *Nature*, 601, 452–459. — V-SYNTHES. 수십억 분자 검색의 실전 사례.
- ChEMBL(ebi.ac.uk/chembl) · RCSB PDB(rcsb.org) · ZINC22(cartblanche22.docking.org) — 본인 표적으로 직접 검색해보며 익히는 것이 가장 빠른 학습.
