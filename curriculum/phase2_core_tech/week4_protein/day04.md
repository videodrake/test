# Day 4: 단백질 언어 모델 — ESM, ProtTrans

> 이전 학습(Phase 2 Week 4 Day 3)에서 AlphaFold 3·Boltz-1으로 대표되는 co-folding과 딥러닝 도킹의 진화·한계를 다뤘습니다. 오늘은 그 구조 예측·도킹의 아래에서 표현(representation)을 공급하는 층 — **단백질 언어 모델(Protein Language Model, PLM)** 을 학습합니다. ESM·ProtTrans가 어떻게 아미노산 서열을 "언어"로 학습해 구조·기능·변이 효과를 예측하는지, 그리고 이것이 왜 AlphaFold와 상보적인 별도의 무기인지가 오늘의 핵심입니다.

## 개요

단백질 언어 모델은 — 수억 개의 아미노산 서열을 자연어처럼 자기지도학습(self-supervised learning)으로 학습해, 각 잔기(residue)의 문맥적 표현을 만들어내는 대규모 신경망입니다. 핵심 통찰은 단순합니다: 자연 진화가 걸러낸 단백질 서열에는 구조와 기능의 제약이 통계적으로 각인되어 있으므로 — 서열만으로 그 제약을 학습하면 구조 정보(contact map), 기능 주석, 변이의 해로움을 *실험이나 정렬(MSA) 없이* 추정할 수 있다는 것입니다. Meta FAIR의 **ESM(Evolutionary Scale Modeling)** 계열과 Rostlab의 **ProtTrans** 계열이 이 분야를 열었고, 2022년 **ESMFold**는 MSA 없이 초당 수준의 구조 예측으로 6억 개 이상의 메타지놈 단백질 구조를 공개했습니다. 약학 전공자에게 PLM의 의미는 — *구조가 없는 신규 표적의 druggability 사전 평가*, *효소·항체의 변이 효과 예측*, *서열만으로의 기능 주석* 이라는, 기존 약리학이 실험에 의존하던 판단을 계산으로 앞당기는 도구를 얻는 데 있습니다.

## 핵심 개념

### 1) "단백질은 언어다"라는 전제

자연어 처리(NLP)의 언어 모델은 대규모 텍스트에서 다음 단어(또는 가려진 단어)를 예측하며 문법·의미를 학습합니다. 단백질 언어 모델은 이 틀을 그대로 옮깁니다.

- **어휘(vocabulary)**: 20종 표준 아미노산 + 특수 토큰(gap, mask, 시작/끝). 각 잔기가 하나의 토큰입니다.
- **말뭉치(corpus)**: UniProt·UniRef·BFD 등 공개 서열 DB. ESM-2는 UniRef의 약 6,500만 개 비중복 서열(약 수십억 잔기)로 학습했습니다.
- **학습 과제**: **마스크드 언어 모델링(Masked Language Modeling, MLM)** — 서열의 약 15% 잔기를 가리고 원래 아미노산을 맞히도록 학습합니다. ProtBERT는 BERT식 MLM을, ProtT5는 T5식 span-corruption을 사용합니다.

핵심은 — 모델이 "이 위치에 어떤 아미노산이 올 확률이 높은가"를 배우려면, 필연적으로 *접힘·활성부위·소수성 코어·이차구조* 같은 물리·진화적 제약을 내부 표현에 담아야 한다는 점입니다.

> **약학적 대응** — 이 전제는 약학의 **서열 보존성(conservation)** 개념과 정확히 대응합니다. 효소의 촉매 잔기나 GPCR의 리간드 결합 포켓처럼 기능적으로 필수인 위치는 진화적으로 강하게 보존됩니다. PLM이 "이 위치의 아미노산을 높은 확신으로 예측한다"는 것은 곧 *그 잔기가 보존되어 있다 = 기능적으로 중요하다*는 신호이며, 이는 변이 효과 예측의 물리적 근거가 됩니다.

### 2) 주요 모델 계보

| 모델 | 개발 | 규모(파라미터) | 아키텍처 | 특징 |
|------|------|------|--------|------|
| **ESM-1b** (Rives et al., 2021, *PNAS*) | Meta FAIR | 6.5억 | Transformer encoder, MLM | 최초의 대규모 PLM, 구조·기능 emergent |
| **ESM-1v** (Meiler 아님; Meier et al., 2021, *NeurIPS*) | Meta FAIR | 6.5억 | MLM | 변이 효과 zero-shot 예측 특화 |
| **ESM-2** (Lin et al., 2023, *Science*) | Meta FAIR | 8M~150억 | Transformer encoder | ESMFold의 백본, 규모 스케일링 |
| **ESMFold** (Lin et al., 2023, *Science*) | Meta FAIR | ESM-2 기반 | folding head 추가 | MSA 없이 단일 서열 구조 예측 |
| **ESM3** (Hayes et al., 2024, EvolutionaryScale) | EvolutionaryScale | 최대 980억 | 서열·구조·기능 멀티모달 생성 | 생성형 PLM, 신규 단백질 설계 |
| **ProtBERT / ProtT5** (Elnaggar et al., 2021, *IEEE TPAMI*) | Rostlab (TUM) | 4.2억 / 30억 | BERT / T5 | 임베딩 품질 우수, 다운스트림 전이 강점 |
| **ProtGPT2** (Ferruz et al., 2022, *Nat. Commun.*) | Rostlab | 7.4억 | GPT-2 decoder | 자기회귀 생성으로 신규 서열 생성 |

계보의 두 축을 구분해야 합니다. **인코더형(ESM-2, ProtBERT)** 은 서열을 이해·표현하는 데 강하고(임베딩·변이 예측·구조), **디코더형(ProtGPT2, ESM3)** 은 새 서열을 생성하는 데 강합니다. 이는 NLP에서 BERT(이해)와 GPT(생성)의 구분과 동일합니다.

### 3) 무엇이 "저절로" 학습되는가 (emergent properties)

PLM의 놀라운 점은 — 구조를 명시적으로 가르치지 않았는데도 구조 정보가 내부에 나타난다는 것입니다.

- **접촉 지도(contact map)**: Transformer의 어텐션(attention) 가중치가 서열상 멀리 떨어졌으나 3차원에서 접촉하는 잔기 쌍(long-range contact)에 집중됩니다. 어텐션 맵에서 직접 contact를 회귀할 수 있습니다.
- **이차구조·용매 접근성**: 임베딩에 선형 분류기(linear probe)만 얹어도 α-helix/β-sheet, buried/exposed를 높은 정확도로 분류합니다.
- **변이 효과**: 야생형 대비 변이 서열의 로그 가능도(log-likelihood) 차이가 실험적 적합도(fitness)·병원성과 상관합니다(ESM-1v).
- **기능 클러스터**: 서열 임베딩을 평균 풀링하면 효소 계열(EC number)·Pfam 도메인이 임베딩 공간에서 군집화됩니다.

> **약학적 함의** — 이는 곧 *실험 데이터가 거의 없는 신규 표적에서도* 서열만으로 (a) 접힘 신뢰도, (b) 기능 도메인, (c) 핵심 잔기의 보존성을 추정할 수 있음을 뜻합니다. 신약개발 초기 표적 검증(target validation) 단계에서 "이 표적이 실제로 접히고 기능하는 단백질인가, 어느 잔기가 결합에 핵심인가"를 wet lab 이전에 좁힐 수 있습니다.

### 4) ESMFold vs AlphaFold — 상보적 관계

Day 36에서 다룬 AlphaFold와의 차이를 명확히 해야 합니다.

| 항목 | AlphaFold2/3 | ESMFold |
|------|------|------|
| 입력 | 서열 + **MSA(다중서열정렬)** | **단일 서열만** |
| 속도 | 느림(MSA 검색이 병목, 분당) | 빠름(초 단위, MSA 불필요) |
| 정확도 | 최상위 (특히 MSA 풍부한 표적) | AF2보다 약간 낮으나 준수 |
| 강점 | 진화 정보 풍부한 well-studied 표적 | MSA 없는 고아 단백질·메타지놈·대규모 스캔 |

핵심은 — ESMFold가 AF의 대체가 아니라 *다른 작동 영역(operating regime)* 을 담당한다는 것입니다. MSA를 구성할 상동 서열이 부족한 신규·orphan 단백질, 혹은 수억 개를 훑어야 하는 대규모 스크리닝에서는 ESMFold의 속도·MSA 비의존성이 결정적입니다. 실제로 ESM Metagenomic Atlas는 이 속도 덕에 6억 개 이상 구조를 공개할 수 있었습니다.

## 작동 원리와 아키텍처

PLM 기반 표적 분석 파이프라인 — 1인 창업자가 바이브코딩으로 구성 가능한 참조 구조:

```
[입력 → 처리 → 출력]

1. 서열 입력
   - UniProt ID 또는 raw FASTA
   - (선택) 변이 목록: WT → Mutant (예: G12D)

2. 임베딩 추출 (Feature extraction)
   - ESM-2 (예: esm2_t33_650M) 로드, HuggingFace/fair-esm
   - per-residue 임베딩(예: 1280차원) + 서열 평균 임베딩
   - 어텐션 맵에서 contact map 회귀

3. 다운스트림 헤드 (Task-specific heads)
   - 구조: ESMFold folding head → PDB 좌표 + pLDDT
   - 변이 효과: WT vs Mutant log-likelihood 차이(ESM-1v 방식)
   - 기능 주석: 임베딩 → 분류기 → EC/GO/Pfam
   - 결합 부위: 보존성·pLDDT·pocket 검출 결합

4. 약학적 해석 레이어
   - pLDDT < 70 영역 = 저신뢰(disordered 가능) 태그
   - 보존 핵심 잔기 = druggable hotspot 후보
   - 변이 효과 점수 = 내성 변이·병원성 우선순위

5. 출력 리포트
   - 구조(PDB) + 신뢰도 히트맵
   - 변이별 영향 점수 순위표
   - 기능 주석 + 결합 포켓 후보
```

**핵심 설계 결정** — 어떤 조합이 1인 창업자에게 최적인가:

| 결정 | 선택지 | 추천 | 이유 |
|------|--------|-----|------|
| 백본 모델 | ESM-2(150M~15B) / ProtT5 / ESM3 | **ESM-2 650M** | 정확도·GPU 메모리 균형, 오픈 |
| 구조 예측 | AlphaFold(MSA) / ESMFold | **표적 특성에 따라 분기** | MSA 풍부→AF, orphan·대량→ESMFold |
| 변이 예측 | 지도학습 / zero-shot(ESM-1v) | **zero-shot 우선** | 라벨 데이터 없이 즉시 적용 |
| 임베딩 활용 | fine-tuning / linear probe | **linear probe 먼저** | 소량 데이터로 빠른 baseline |
| 라이선스 | ESM-2(MIT) / ESM3(일부 제약) | **ESM-2 상업 안전** | ESM3는 규모별 라이선스 확인 필요 |

> **바이브코딩 팁** — ESM-2 임베딩 추출은 `fair-esm` 또는 HuggingFace `transformers`의 `EsmModel`로 수십 줄이면 됩니다. 창업자는 "이 모델을 어떻게 짜는가"가 아니라 *"어떤 다운스트림 헤드를 얹어 어떤 약학적 질문에 답할 것인가"* 에 집중해야 합니다.

## 신약개발 적용

PLM은 이미 표적 발굴·단백질 설계·항체 엔지니어링 전반에 통합되고 있습니다. 최근 2년 위주 사례:

| 사례 | 표적 / 상황 | 접근 | 결과 |
|------|-----------|------|------|
| **ESM Metagenomic Atlas (Meta, 2022~2023)** | 메타지놈 미지 단백질 | ESMFold 대량 구조 예측 | 6억+ 구조 공개, 신규 fold 다수 발견 |
| **EvolutionaryScale ESM3 (2024)** | 신규 형광단백질 esmGFP | 생성형 PLM으로 서열 설계 | 자연과 서열 동일성 ~58%인 신규 GFP 실험 검증 |
| **Profluent — 유전자편집 효소 (2024)** | Cas 단백질(OpenCRISPR) | PLM으로 신규 Cas 서열 생성 | 자연에 없는 기능성 유전자 편집기 설계·공개 |
| **Cradle Bio (2023~)** | 산업·치료 단백질 최적화 | PLM 기반 서열 제안 + 실험 루프 | 다수 제약·바이오 파트너십, 단백질 안정성·발현 개선 |
| **항체 최적화(일반)** | 치료 항체 CDR | 항체 특화 PLM(AbLang, IgLM 등) | 인간화·친화도 성숙 후보 서열 우선순위화 |

**정량·개념 비교** — 기존 방법 대비 PLM의 이점:

| 지표 | 전통적 접근 | PLM 기반 접근 |
|------|------|------|
| 신규 표적 구조 확보 | 결정학·cryo-EM(수개월~수년) | ESMFold 초 단위 예측(신뢰도 태그 포함) |
| 변이 효과 판정 | 부위지정 돌연변이 실험 | zero-shot log-likelihood 점수(즉시) |
| 기능 미지 단백질 주석 | BLAST 상동성 검색 | 임베딩 기반 분류(원거리 상동체도 포착) |
| 단백질 설계 리드 | 이성적/방향적 진화 실험 | 생성형 PLM 후보 → 실험 축소 |

> **산업적 통찰** — PLM은 구조 예측(AlphaFold)이 열어놓은 "구조는 있는데 그래서 뭘 하나"의 다음 단계를 담당합니다. 구조가 정적 스냅샷이라면, PLM 임베딩은 *기능·변이·설계 가능성*이라는 동적 질문에 답하는 좌표계입니다. 특히 항체·효소·펩타이드처럼 서열 자체가 곧 제품인 바이오로직스(biologics)에서 PLM은 소분자에서의 SMILES 표현(Phase 2 Week 1)에 상응하는 핵심 표현 계층으로 자리잡고 있습니다.

## 창업 관점

PLM은 소분자 중심 AI 신약개발이 상대적으로 붐빈 반면 — **단백질·바이오로직스 영역**에서 아직 빈 포지션이 많다는 점에서 창업 기회가 있습니다. ESM-2가 MIT 라이선스로 완전 공개되어 있어 1인 창업자도 강력한 백본을 무료로 활용할 수 있고, 차별화는 *어떤 다운스트림 문제에 약학 지식을 결합하느냐*에서 나옵니다.

유력한 제품 방향은 세 가지입니다. 첫째, **표적 검증 리포트 SaaS** — 서열만 입력하면 ESMFold 구조 + 보존 핵심 잔기 + druggability 예비 평가 + 알려진 변이의 영향을 종합한 "신규 표적 스크리닝 리포트"를 자동 생성합니다. 약학 전공자의 강점은 결합 포켓의 약리학적 타당성과 pLDDT 저신뢰 영역의 해석을 정직하게 등급화하는 데 있습니다. 둘째, **변이 효과·내성 예측 도구** — 항암제·항생제·항바이러스제의 표적 변이가 약물 결합에 미치는 영향을 zero-shot으로 예측해, 내성 변이 우선순위와 차세대 후보 설계를 지원합니다. 이는 CYP 대사·표적 변이 등 약학적 맥락(Day 33)과 직접 연결됩니다. 셋째, **항체·효소 설계 최적화 플랫폼** — Cradle·Profluent가 개척했으나 특정 modality(예: 이중특이 항체, 특정 효소군)에 특화하면 틈새가 존재합니다.

시장 관점에서 PLM 기반 단백질 설계는 Recursion·Isomorphic 같은 소분자 중심 대형 플레이어와 경쟁 축이 다르며, 초기 고객은 항체·효소 개발 중소 바이오와 학술 랩입니다. MVP 구상 — ESM-2 임베딩 + ESMFold + 변이 log-likelihood 스코어를 묶어 표적 하나에 특화한 웹 리포트를 제공하고, 약학적 해석 레이어를 차별점으로 세우는 형태가 현실적입니다. 기술적 해자는 모델 자체(오픈)가 아니라 — *약학 지식이 반영된 해석·검증 레이어*와 *특정 표적군의 실험 데이터로 튜닝한 다운스트림 헤드*에 있습니다.

## 오늘의 과제

1. **ESM vs AlphaFold 작동 영역 비교 정리 (50분)**: ESM-2/ESMFold 논문(Lin et al., 2023, *Science*)의 Abstract·Figure 1을 읽고 — (a) ESMFold가 AlphaFold2 대비 정확도·속도에서 어떤 트레이드오프를 갖는지, (b) MSA 비의존성이 결정적 이점이 되는 표적 유형 3가지, (c) 반대로 AlphaFold를 써야 하는 상황 2가지를 1페이지로 정리하세요. 결론으로 — 본인 표적 분석 SaaS에서 *어떤 입력 조건일 때 ESMFold로 분기하고 언제 AF로 분기할지* 판단 규칙을 스케치하세요.

2. **변이 효과 zero-shot 예측 실습 계획 (50분)**: 관심 표적 단백질 1개(예: EGFR, KRAS 등)를 정하고 — (a) HuggingFace의 ESM-2(`facebook/esm2_t33_650M_UR50D`)로 임베딩·마스크 예측을 실행하는 최소 절차, (b) 알려진 임상적 변이 1~2개(예: EGFR T790M)의 log-likelihood 차이를 계산해 "해로움" 점수로 해석하는 방법, (c) 이 점수를 실제 약물 내성 정보와 대조하는 검증 계획을 1페이지로 정리하세요. 실행이 가능하면 실제로 임베딩을 추출해 보세요.

3. **단백질 설계 스타트업 3사 비교 분석 (60분)**: PLM 기반 단백질 설계 회사 3곳(EvolutionaryScale, Profluent, Cradle Bio, Basecamp Research, Chai Discovery 등에서 3사 선택)의 — (a) 사용하는 PLM/자체 모델, (b) 표적 modality(항체/효소/신규 단백질/유전자편집기), (c) 비즈니스 모델(파트너십/플랫폼 구독/IP), (d) 차별화 포지셔닝을 표로 비교하세요. 결론으로 — 본인이 진입한다면 *비어있는 modality·고객 세그먼트*가 무엇인지, 그 포지션을 잡기 위한 MVP를 1페이지로 정리하세요.

## 참고 자료

- **ESM-1b**: Rives, A. et al. "Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences." *PNAS*, 2021. — 대규모 PLM에서 구조·기능이 emergent함을 보인 기념비적 연구.
- **ESM-1v**: Meier, J. et al. "Language models enable zero-shot prediction of the effects of mutations on protein function." *NeurIPS*, 2021. — 변이 효과 zero-shot 예측의 표준 접근.
- **ESM-2 / ESMFold**: Lin, Z. et al. "Evolutionary-scale prediction of atomic-level protein structure with a language model." *Science*, 2023. — MSA 없는 단일 서열 구조 예측과 메타지놈 아틀라스.
- **ESM3**: Hayes, T. et al. "Simulating 500 million years of evolution with a language model." *bioRxiv/Science*, 2024 (확인 필요). — 서열·구조·기능 멀티모달 생성 PLM, esmGFP 설계.
- **ProtTrans**: Elnaggar, A. et al. "ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning." *IEEE TPAMI*, 2021. — ProtBERT·ProtT5 등 다양한 PLM의 체계적 비교.
- **ProtGPT2**: Ferruz, N. et al. "ProtGPT2 is a deep unsupervised language model for protein design." *Nature Communications*, 2022. — 자기회귀 생성형 PLM으로 신규 서열 설계.
- **EvolutionaryScale**: evolutionaryscale.ai — ESM3를 개발한 스타트업(2024 설립, 대규모 투자 유치).
- **fair-esm / HuggingFace**: github.com/facebookresearch/esm — ESM 계열 오픈소스 구현 및 사전학습 가중치(확인 필요).
- **Profluent**: profluent.bio — PLM 기반 유전자편집 효소(OpenCRISPR) 설계 스타트업.
