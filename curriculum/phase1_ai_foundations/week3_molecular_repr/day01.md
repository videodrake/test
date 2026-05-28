# Day 1: SMILES와 분자 문자열 표현

> 이전 학습(Week 2 Day 5)에서 신약개발 파이프라인 위에 Insilico·Recursion·Schrödinger·Isomorphic 같은 회사들을 배치한 "회사 지도"를 완성했습니다. 오늘부터 Week 3은 그 회사들이 다루는 **분자를 컴퓨터가 어떻게 표현하는가**로 한 단계 내려갑니다. 첫 도구는 가장 오래되고 여전히 가장 널리 쓰이는 표현인 **SMILES**입니다.

## 개요

분자를 컴퓨터에 입력하려면 3차원 화학 구조를 어떤 데이터 타입으로든 환원해야 합니다. **SMILES(Simplified Molecular Input Line Entry System)** 는 분자를 ASCII 문자열 한 줄로 적는 표기법으로, 1988년 David Weininger가 제안한 이래 ChEMBL·PubChem·ZINC 등 모든 주요 화학 데이터베이스의 사실상 표준이 되었습니다. SMILES가 중요한 이유는 단순히 사람이 읽기 쉬워서가 아닙니다. **분자를 "자연어 문장"처럼 다룰 수 있게 만들었기에**, RNN·Transformer·LLM 같은 시퀀스 모델이 분자 생성과 물성 예측에 그대로 투입될 수 있었습니다. 오늘은 SMILES의 문법, 정규화(canonicalization) 문제, 약점, 그리고 이를 보완하는 SELFIES·InChI까지 다룹니다. 이 표현 계층을 이해해야 Week 3 후반의 그래프·3D 표현, Phase 2의 분자 생성 모델이 작동하는 원리를 따라갈 수 있습니다.

## 핵심 개념

### SMILES 문법의 4가지 규칙

SMILES는 분자 그래프를 깊이 우선 탐색(depth-first traversal)으로 풀어 적은 결과물입니다. 핵심 규칙은 네 가지입니다.

| 규칙 | 표기 | 예시 |
|------|------|------|
| **원자** | 대문자 또는 [대괄호] | `C`(탄소), `N`(질소), `[Na+]`(나트륨 이온) |
| **결합** | `-` 단일, `=` 이중, `#` 삼중, `:` 방향족 | `C=O`(카르보닐), `C#N`(시안기) |
| **분기(branch)** | 괄호 | `CC(=O)O`(아세트산: 메틸-카보닐-OH) |
| **고리(ring)** | 같은 숫자로 시작·끝 | `c1ccccc1`(벤젠), `C1CCCCC1`(시클로헥산) |

수소는 보통 생략되어(implicit hydrogen) 원자가 규칙으로 채워집니다. 소문자(`c`, `n`, `o`)는 **방향족(aromatic) 원자**를 의미하며, 이 한 글자로 Kekulé 구조와 방향족성 정보를 동시에 인코딩합니다. 예를 들어 아스피린은 `CC(=O)Oc1ccccc1C(=O)O`로 표기되며 — 아세틸기, 에스터 산소, 벤젠 고리, 카르복실산이 한 줄에 그대로 담깁니다.

### Canonical SMILES — "같은 분자, 다른 문자열" 문제

깊이 우선 탐색의 시작점을 어느 원자로 잡느냐에 따라 **하나의 분자가 수십~수백 가지 SMILES로 표기될 수 있습니다**. 에탄올은 `CCO`, `OCC`, `C(O)C`로 모두 같은 분자입니다. 이를 해결하기 위해 **Canonical SMILES**(정규 SMILES)는 Morgan 알고리즘 변형으로 원자 순위를 매겨 시작점과 탐색 순서를 결정합니다. RDKit·OpenEye·DayLight이 각자 다른 정규화 규칙을 사용하므로 — **"RDKit canonical"과 "DayLight canonical"이 다를 수 있다**는 점이 산업 현장의 함정입니다. 데이터셋을 합칠 때 같은 도구로 한 번 더 정규화하지 않으면 같은 분자가 중복으로 학습되어 데이터 누수(data leakage)가 발생합니다.

### 약학 전공자에게 친숙한 비교 — IUPAC 명명법과의 대응

약학에서 익숙한 **IUPAC 명명법**은 사람이 읽기 위한 표준이고, SMILES는 기계가 읽기 위한 표준입니다. 같은 정보(원자·결합·입체화학)를 다른 어휘로 옮긴 것에 가깝습니다. IUPAC가 "어떤 작용기가 주사슬을 결정하는가"라는 규칙을 따른다면, SMILES는 "어떤 원자에서 출발해 그래프를 순회하는가"의 규칙을 따릅니다. **입체화학(stereochemistry)** 도 SMILES가 표기합니다 — `@`·`@@`로 키랄 중심을, `/`·`\`로 이중결합 cis/trans를 적습니다. (R)-이부프로펜과 (S)-이부프로펜이 PK·약효에서 다르듯, 이 기호 하나가 빠진 SMILES로 학습한 모델은 **두 거울상이성질체를 같은 분자로 취급해 활성 예측이 깨집니다**. 약학 전공자는 데이터셋의 stereo 정보 보존 여부를 가장 먼저 점검할 수 있다는 점이 모델링 단계의 실질 우위입니다.

### SMILES의 구조적 약점과 대안

| 약점 | 결과 | 대안 |
|------|------|------|
| **문법적으로 유효하지 않은 문자열을 쉽게 만든다** | 생성 모델이 무효 SMILES를 다수 출력 (RNN 기반 초기 모델에서 30~50% 무효율 보고) | **SELFIES**(Self-Referencing Embedded Strings, Krenn et al. 2020): 어떤 문자열도 유효 분자에 매핑 |
| **같은 분자가 여러 표기** | 데이터 누수, 모델 일관성 저하 | Canonical SMILES + InChI 키 매칭 |
| **분자 동일성 비교 어려움** | 중복 제거가 표기에 의존 | **InChI/InChIKey**(IUPAC 표준, 27자 해시): 분자 단위 동일성 비교 표준 |
| **거대 분자(단백질·고분자)에서 한 줄이 비현실적으로 길어진다** | 시퀀스 모델의 context length 한계 | FASTA(아미노산 1글자 코드), HELM(거대분자), 그래프 표현 |

**SELFIES**는 100% 유효성을 수학적으로 보장하므로 생성 모델 학습 시 보상 함수가 "문법 학습"이 아닌 "물성 최적화"에 집중할 수 있습니다. **InChI**는 분자 동일성 비교의 표준이며, InChIKey는 27자 고정 길이 해시이므로 데이터베이스 중복 제거에 사용됩니다. 실무에서는 보통 SMILES로 입력받아 InChIKey로 중복 제거, 모델 학습 시에는 다시 Canonical SMILES 또는 SELFIES로 변환합니다.

## 작동 원리와 아키텍처

### 입력→토큰화→임베딩→모델 흐름

SMILES가 "분자의 자연어"라 불리는 이유는 NLP 파이프라인을 거의 그대로 차용할 수 있기 때문입니다.

```
[원시 SMILES]
   ↓ 1. 정규화: RDKit Canonical SMILES로 통일
   ↓ 2. 토큰화: 문자 단위 / BPE / 정규식 기반(atom-wise)
[토큰 시퀀스]
   ↓ 3. 임베딩: 각 토큰 → 고정 차원 벡터 (예: 256d)
   ↓ 4. 시퀀스 모델: RNN / Transformer / Mamba
[분자 임베딩]
   ↓ 5a. 분류 헤드 → 물성/활성 예측 (logP, pIC50, hERG)
   ↓ 5b. 생성 헤드 → 다음 토큰 확률 → 새 SMILES 생성
```

### 토큰화 방식의 선택 결정

토큰화 방식이 모델 성능을 좌우합니다. SMILES 토큰화는 자연어와 달리 도메인 규칙이 필요합니다.

| 방식 | 단위 | 장점 | 단점 |
|------|------|------|------|
| **문자 단위(character-level)** | 한 글자씩 | 단순, 어휘 작음 | `Cl`(염소)을 `C`+`l`로 잘못 분리 |
| **정규식 기반(atom-wise)** | 원자/결합 단위(`Cl`, `Br`, `[nH]`) | 화학적으로 자연스러움 | 정규식 설계 필요 |
| **BPE(Byte-Pair Encoding)** | 자주 등장하는 부분문자열 | 작용기를 한 토큰으로 학습 | 사전 학습이 필요 |

ChemBERTa·MolFormer 같은 화학 Foundation Model은 정규식 기반 또는 BPE를 사용하며, 작용기 패턴이 한 토큰으로 묶이면서 다운스트림 물성 예측 성능이 개선되는 경향이 보고됩니다. 이 설계 결정은 Week 3 후반의 그래프·3D 표현과 비교하는 기준점이 됩니다.

## 신약개발 적용

### SMILES가 만든 두 가지 산업적 변화

첫째, **데이터베이스의 표준화**입니다. ChEMBL(약 240만 화합물·2000만 활성값, 2024년 ChEMBL 34 기준)과 PubChem(1억 1천만 화합물 이상)은 SMILES를 1차 식별자로 사용해 화학 데이터의 상호운용성을 확립했습니다. 약학 연구의 메타분석·QSAR 모델 학습이 가능해진 직접적 기반입니다.

둘째, **분자 생성의 시퀀스 모델화**입니다. Segler et al.(2018, *ACS Central Science*)은 ChEMBL의 SMILES로 RNN을 학습해 새로운 활성 분자를 생성했고, 이는 이후 Insilico의 Chemistry42, BenevolentAI 등 산업 시스템의 원형이 되었습니다. 2024~2025년의 화학 Foundation Model로 발전했습니다 — **MolFormer-XL**(IBM, 2022)은 약 11억 개 SMILES로 사전학습한 Transformer로 다운스트림 물성 11개 과제에서 그래프 모델을 다수 상회한 결과가 보고되었습니다(Day 3에서 다룬 내용과 연결).

### 한계와 실패 사례

SMILES 기반 생성 모델이 만든 분자의 상당수는 **합성 불가능**하거나 **문법적으로 무효**합니다. 이 약점은 Phase 2 Week 2(분자 생성)에서 SELFIES·그래프 기반 생성으로 보완되는 과정을 본격적으로 다룹니다. 또한 SMILES는 **3D 구조와 입체적 상호작용**(단백질-리간드 도킹에 필요한 정보)을 표현하지 못하므로, Phase 2 Week 4(단백질·구조)에서 다룰 도킹·MD에는 부적합합니다 — 같은 분자라도 활성 conformer가 다르면 결합 친화력이 달라지는데, 이 정보는 SMILES에 없습니다.

## 창업 관점

분자 표현 자체는 Phase 1의 기초 인프라이므로 창업 주제로 좁아지지만, 한 가지 짚을 점이 있습니다 — **고품질로 정규화·중복 제거된 화학 데이터는 그 자체로 모트(moat)** 가 됩니다(Recursion·Insilico의 데이터 모트와 같은 논리). 약학 전공자가 적응증 특화 데이터셋을 SMILES·InChIKey 기준으로 큐레이션하는 작업은 향후 Phase 4·5에서 다룰 데이터 인프라 사업의 출발점입니다.

## 오늘의 과제

1. **SMILES 직접 해독 (40분)**: 본인이 잘 아는 약물 5개(예: 이부프로펜·아세트아미노펜·아토르바스타틴·메트포르민·시메티딘)의 SMILES를 PubChem 또는 ChEMBL에서 찾아 적고, 각 문자열에서 (a) 핵심 작용기, (b) 입체화학 표기 유무, (c) 카르복실산·아민 등 PK에 영향을 주는 부분을 손으로 표시합니다. A4 1쪽.
2. **Canonical SMILES와 InChIKey의 차이 검증 (30분)**: 같은 분자(예: 카페인 `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`)를 RDKit과 PubChem에서 각각 Canonical SMILES로 변환해 동일한지 비교하고, InChIKey가 도구와 무관하게 일치함을 확인합니다. 차이가 났다면 그 원인(방향족화 처리·수소 처리)을 1문단으로 정리합니다. 실행은 Claude Code/Cursor로 RDKit 코드를 짜게 시켜도 됩니다 — 결과 해석에 집중합니다.
3. **SELFIES와의 비교 리서치 (30분)**: Krenn et al.(2020) "Self-Referencing Embedded Strings (SELFIES)" 논문 Abstract와 Introduction을 읽고, SELFIES가 SMILES의 어떤 약점을 어떻게 해결했는지 핵심 3줄로 정리합니다. 이어서 "왜 산업이 여전히 SMILES를 1차 표준으로 쓰는가"를 2-3줄로 추측해 적습니다.

## 참고 자료

- Weininger, D. (1988). "SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules." *Journal of Chemical Information and Computer Sciences*, 28(1), 31-36. — SMILES 원전. 36년 지난 지금도 산업 표준의 근거.
- Krenn, M. *et al.* (2020). "Self-Referencing Embedded Strings (SELFIES): A 100% robust molecular string representation." *Machine Learning: Science and Technology*, 1(4), 045024. — SMILES 약점을 수학적으로 해결한 대안 표현.
- RDKit (rdkit.org) — Canonical SMILES, InChI 변환, 분자 처리의 오픈소스 사실상 표준. Phase 4 Week 2에서 본격 활용.
- ChEMBL (ebi.ac.uk/chembl) — SMILES 기반 약물·활성 데이터베이스. 본인 관심 적응증의 데이터 양 점검에 활용.
