# Day 1: 단백질 구조 예측 — AlphaFold가 바꾼 세계

> 이전 학습(Phase 2 Week 3 Day 5)에서 ADMET 5축과 MPO·약학적 판단을 통합한 분자(ligand) 중심 의사결정을 다뤘습니다. 오늘은 Phase 2 Week 4의 첫 Day로, 시각을 **단백질 표적(target) 측면**으로 확장해 — AlphaFold가 바꾼 단백질 구조 예측의 세계와 이것이 신약개발에 가져온 구조적 변화를 학습합니다.

## 개요

단백질 구조 예측은 — 아미노산 서열(1차 구조)로부터 단백질의 3차원 입체 구조를 컴퓨터로 예측하는 기술입니다. 50년 묵은 *protein folding problem*은 — 2020년 **AlphaFold 2**가 CASP14에서 실험 결정 구조 수준의 정확도(중간값 GDT-TS ≈ 92.4)를 달성하면서 사실상 해결되었고, 2024년 5월 *Nature*에 발표된 **AlphaFold 3**는 단백질뿐 아니라 리간드·핵산·이온·번역 후 수식까지 한 모델로 예측하는 단계로 진화했습니다. 이 변화는 신약개발에서 — *구조 모르는 표적(undruggable target)*의 정의를 다시 쓰고, 구조 기반 신약설계(Structure-Based Drug Design, SBDD)의 진입 장벽을 데이터·시간·비용 모든 차원에서 무너뜨렸습니다. 2024년 10월 Demis Hassabis·John Jumper·David Baker의 노벨화학상 수상은 — 이 기술이 단순 도구가 아닌 *과학 인프라*가 되었음을 공인한 사건입니다. 약학 전공자에게 본 Day의 의미는 — 표적의 구조를 자유롭게 다루는 시대에 *어떤 표적이 약물화 가능한가(druggability)*를 판단하는 약리학적 안목이 더 중요해진다는 점입니다.

## 핵심 개념

### 1) protein folding problem과 50년 난제

**Anfinsen의 도그마(1972 노벨화학상)** — *아미노산 서열이 단백질의 3차원 구조를 결정한다*는 가설은 — 1960년대부터 구조 예측의 이론적 기반이었습니다. 그러나 **Levinthal's paradox**(가능한 conformation의 수가 천문학적)로 인해 — 순수 물리 시뮬레이션으로는 폴딩을 재현할 수 없었고, 50년간 *호몰로지 모델링·de novo 폴딩·물리 기반 에너지 최소화*가 단편적 성과만 냈습니다.

**CASP(Critical Assessment of protein Structure Prediction)**는 — 1994년 시작된 격년 단백질 구조 예측 대회로, **GDT-TS(Global Distance Test - Total Score, 0~100)**로 예측 정확도를 평가합니다. 60점이 *실험 구조에 근접한 수준*, 90점 이상이 *원자 분해능에 준하는 수준*의 기준입니다.

### 2) AlphaFold 2 — 패러다임 전환

**AlphaFold 2 (Jumper et al., 2021, *Nature*)**의 CASP14 평균 GDT-TS는 약 87.0, 중간값은 약 92.4로 — 이전 SOTA(약 60) 대비 거의 30점 도약했습니다. 핵심 혁신:

| 구성요소 | 역할 | 약학적 의미 |
|---------|------|----------|
| **Multiple Sequence Alignment (MSA)** | 진화적 상관관계로 contact map 추론 | 보존 잔기 = 기능 잔기 → 활성 부위 단서 |
| **Evoformer (48 block)** | MSA + 잔기 쌍 표현을 반복 정제 | attention 기반 공진화 학습 |
| **Structure module (8 block)** | 3D 좌표 직접 예측, IPA(invariant point attention) | end-to-end 미분 가능 폴딩 |
| **pLDDT (per-residue confidence)** | 0~100, ≥ 90 = 신뢰 가능 | 표적 활성 부위가 신뢰 영역인지 판단 |
| **PAE (Predicted Aligned Error)** | 잔기 쌍 거리 오차 | 도메인 간 배치 신뢰도 평가 |

**AlphaFold Protein Structure Database (EMBL-EBI 공동, 2021~)** — 2024년 기준 약 2억 1,400만 개 단백질의 예측 구조가 무료 공개되어 있으며, 인간 단백질체의 약 98%가 모델링되었습니다. 이 중 약 35~58%(연구별)가 pLDDT ≥ 70의 신뢰 영역입니다.

### 3) AlphaFold 3 — 다중 분자 통합 예측 (2024년 5월, *Nature*)

**AlphaFold 3 (Abramson et al., 2024)**는 — Evoformer를 **Pairformer**로 단순화하고, 구조 생성을 **확산 모델(diffusion model)**로 교체한 새 아키텍처입니다. 단일 단백질을 넘어 — *단백질-단백질·단백질-리간드·단백질-DNA/RNA·이온·당쇄·번역 후 수식(PTM)*까지 한 모델로 예측합니다.

| 항목 | AlphaFold 2 | AlphaFold 3 |
|------|----------|----------|
| 구조 생성 | end-to-end regression | diffusion(반복 denoise) |
| 입력 | 단백질 서열만 | 단백질 + 리간드 + 핵산 + 이온 |
| 단백질-리간드 도킹 | 불가 | PoseBusters RMSD ≤ 2 Å 약 76% (2024) |
| 코드 공개 | 즉시(GitHub) | 학술 비상업 라이선스(2024년 11월 일부) |

> **약학적 함의** — AlphaFold 3는 단백질 구조 예측의 도구에서 *단백질-리간드 결합 예측의 새 기준선*으로 진화했습니다. 그러나 PoseBusters RMSD ≤ 2 Å 성공률은 약 76%로 — 여전히 약 24%는 실패하며, 약물 후보의 일부 conformer만 신뢰할 수 있습니다.

### 4) 대안 모델 — ESMFold·RoseTTAFold·오픈 AF3급

**ESMFold (Meta, Lin et al., 2023, *Science*)** — MSA 없이 **단일 서열만으로** 단백질 언어 모델(ESM-2 15B)에서 구조를 예측합니다. AlphaFold 2 대비 — 정확도는 약 10~15% 낮지만 속도는 약 60배 빠릅니다. 메타게놈 약 7억 개 단백질(*ESM Atlas*)의 구조를 빠르게 예측한 사례가 대표적입니다.

**RoseTTAFold·RFdiffusion (Baker lab, 2021~)** — David Baker 그룹의 라인은 — *구조 예측*보다 **de novo 단백질 설계**(RFdiffusion, 2023)에 강점이 있으며, 2024년 노벨화학상의 한 축이 되었습니다.

**오픈 AF3급 모델 — Boltz-1·Chai-1 (2024년 9~10월)** — MIT의 **Boltz-1**과 Chai Discovery의 **Chai-1**이 AlphaFold 3 수준의 단백질-리간드 통합 예측 오픈 모델을 잇따라 공개해, *AF3 학술 라이선스 제약의 우회로*가 열렸습니다. 1인 창업자에게 즉시 활용 가능한 인프라입니다.

### 5) 한계 — 여전히 어려운 표적

| 표적 유형 | 난이도 | 이유 |
|---------|------|------|
| **막단백질(GPCR, 이온채널)** | 중~상 | 막 환경 부재, 활성 conformation 변동 |
| **본질적 무질서 영역(IDR)** | 매우 상 | 단일 구조 가정 자체가 부정확 |
| **알로스테릭 conformer** | 상 | 활성·비활성 평형 분포 미반영 |
| **변이체·돌연변이 효과** | 중 | 점 돌연변이의 미세 conformer 변화 |
| **거대 복합체(> 3,000 잔기)** | 상 | 메모리·시간 제약 |

> **약학적 의미** — AlphaFold 예측 구조를 *그대로 SBDD에 사용*하는 것은 위험합니다. 활성 부위 측쇄 배치, 결합 conformation, 막단백질의 lipid 환경 보정이 필요합니다. pLDDT·PAE를 *비판적으로 읽는 능력*이 — 구조 예측 시대 약학 전공자의 차별 역량입니다.

## 작동 원리와 아키텍처

AlphaFold 계열 구조 예측 시스템의 표준 워크플로우:

```
[입력 → 처리 → 출력]

1. 입력
   - 단백질 서열(FASTA) — UniProt ID 또는 단일 서열
   - (AF3) 추가 입력: 리간드 SMILES/CCD, 핵산 서열, 이온, PTM

2. MSA 검색 (AF2/AF3)
   - UniRef, BFD, MGnify에서 유사 서열 5,000~10,000개 수집
   - jackhmmer + HHblits로 정렬, 공진화 신호 추출

3. 표현 학습
   - Evoformer(AF2) 또는 Pairformer(AF3)로 MSA·쌍 표현 정제
   - 48 block × 잔기 attention

4. 구조 생성
   - AF2: structure module이 좌표 직접 출력
   - AF3: diffusion으로 noise → atom coordinate 반복 denoise (수십~수백 step)

5. 출력
   - PDB 파일 + pLDDT 잔기별 신뢰도 + PAE 매트릭스
   - (AF3) 리간드·핵산까지 한 PDB에 통합 출력

6. 약학적 후처리 — 차별 지점
   - pLDDT < 70 영역 표시 (믿지 말 것)
   - PAE 도메인 간 신뢰도 시각화
   - 결합 부위 측쇄 conformer를 MD 또는 도킹으로 재정렬
   - 실험 구조(PDB/cryo-EM)와 RMSD 비교, ensemble 작성
```

핵심 설계 결정 — 1인 창업자가 알아야 할 트레이드오프:

| 결정 | 선택지 | 추천 | 이유 |
|------|--------|------|------|
| 예측 엔진 | AF2 / AF3 / ESMFold / Boltz-1 | Boltz-1 또는 AF3 | 리간드 통합 + 오픈/학술 라이선스 |
| 입력 표현 | 단일 서열 / MSA | MSA | 정확도 우선, 시간 여유 있을 때 |
| 신뢰 판단 | pLDDT만 / pLDDT+PAE+ensemble | 셋 모두 | 활성 부위는 pLDDT, 도메인 배치는 PAE |
| 후처리 | 단일 모델 / MD 보정 / 도킹 보정 | MD 또는 도킹 보정 | 측쇄 conformer 신뢰도 향상 |

## 신약개발 적용

AlphaFold가 신약개발에 가져온 구조적 변화는 — 단순 정확도 향상을 넘어, *어떤 표적이 약물화 가능한가*의 정의를 다시 썼습니다.

| 사례·도구 | 적용 영역 | 결과 |
|---------|--------|------|
| **AlphaFold Database 활용 가상 스크리닝(Lyu et al., 2024)** | AF2 예측 GPCR 구조 기반 도킹 | 실험 구조 대비 hit rate 동등 수준 보고 |
| **Isomorphic Labs (Alphabet 자회사)** | AF3 기반 신약 발견 플랫폼 | Eli Lilly·Novartis와 2024년 1월 파트너십 발표(총 합의금 약 17억 달러 규모) |
| **AlphaMissense (Cheng et al., 2023, *Science*)** | 인간 단백질 모든 점 돌연변이의 병원성 예측 | 약 7,100만 missense 변이 평가, 약 32%를 병원성으로 분류 |
| **Insilico Medicine + AF2** | 새 표적 발굴 → INS018_055 (IPF) | AF2 구조 + Chemistry42 생성으로 18개월 만에 IND |
| **Boltz-1 (MIT, 2024년 10월)** | 오픈 AF3급 단백질-리간드 예측 | 1인 창업자에게 즉시 사용 가능한 인프라 |
| **노벨화학상 2024** | Hassabis·Jumper·Baker 공동 수상 | AF 계열 + RFdiffusion 과학 인프라화 |

**기존 방법 대비 정량 비교**:

| 항목 | 실험 구조 결정(X-ray/cryo-EM) | AlphaFold 예측 |
|------|---------------------|--------------|
| 시간 | 6개월~수년 | 수 분~수 시간 |
| 비용 | 수만~수십만 달러 | GPU 시간 ≤ 100 달러 |
| 분해능 | 1~3 Å (실험 의존) | 평균 GDT-TS ≈ 92 (AF2 CASP14), AF3 PoseBusters ≈ 76% |
| 적용 범위 | 결정화 가능 표적만 | 사실상 거의 모든 가용 서열 |
| 신뢰도 표시 | 결정학 R-factor·resolution | pLDDT + PAE(잔기별·쌍별) |

**산업적 통찰** — AlphaFold 등장 이후 *드러그러블 표적의 정의*가 확장되었습니다. 과거 결정화 불가로 *undruggable*로 분류되던 표적(예: 다수 GPCR, PROTAC 표적, 무질서 단백질의 일부 도메인)이 — AF 예측 구조 기반으로 가상 스크리닝 진입이 가능해졌습니다. 단, *예측 구조의 활성 부위 측쇄 conformer는 보정 필요*라는 한계는 — 약학·구조생물학적 판단이 결합해야 메워집니다.

## 창업 관점

AlphaFold 인프라 위에서의 창업 기회는 — 모델 개발(빅테크 영역)보다 *적용·후처리·해석 레이어*에 있습니다. **약학 전공자의 차별 포지셔닝** 예시 — (a) AF 예측 구조의 *druggability 평가 SaaS*(활성 부위 신뢰도·pocket 크기·약물 선례 검색), (b) *변이체-구조 약리학*(AlphaMissense + AF + ADMET 연계), (c) *희귀 표적 보고서 자동화*(특정 적응증 표적군의 AF 구조 + 문헌 + 약학적 해석)가 가능합니다. 기존 경쟁 — Isomorphic Labs(빅테크 통합), Schrödinger(상용 SBDD), Cradle·Iambic(스타트업)는 *플랫폼·모델·자동화*에 강점이 있지만 — *적응증·표적군 특화 약학적 해석*은 약한 영역입니다. 1인 창업자 MVP는 — Boltz-1 또는 AF3(학술 라이선스) 위에 — *약학적 후처리 파이프라인*(pLDDT 활성 부위 경고, 회수 약물 표적과의 구조 유사도, 측쇄 conformer ensemble 작성)을 얹는 형태가 현실적이며, 월 1,000~5,000 달러의 중소 바이오·학술 시장이 진입점입니다.

## 오늘의 과제

1. **본인 관심 표적의 AlphaFold 구조 직접 분석 (50분)**: UniProt에서 본인 관심 표적 1개(가능하면 결정화 어려운 GPCR 또는 효소)를 선택해 — (a) AlphaFold Protein Structure Database(alphafold.ebi.ac.uk)에서 AF2 예측 구조를 다운로드, (b) PyMOL 또는 ChimeraX로 pLDDT를 색상 매핑해 활성 부위 신뢰 영역을 확인, (c) 같은 표적의 PDB 실험 구조와 RMSD를 비교하세요. 결론으로 — 본인 SaaS에서 *예측 구조를 사용해도 좋은 표적*과 *위험한 표적*을 가르는 3가지 기준을 명시하세요.

2. **AlphaFold 3 vs Boltz-1 단백질-리간드 예측 비교 (40분)**: 본인 관심 표적-리간드 쌍 1개(공개 PDB 복합체)를 선택해 — (a) AlphaFold Server(alphafoldserver.com, 학술 무료)와 Boltz-1(GitHub 오픈)에서 동일 입력으로 예측, (b) 두 출력을 PDB 결정 구조와 RMSD·PoseBusters 기준으로 비교, (c) 각 도구의 강점·약점을 표로 정리하세요. 결과는 — 본인 SaaS의 백엔드 엔진 선택 근거 자료가 됩니다.

3. **드러그러블 표적 정의 재작성 리서치 (50분)**: 2020년(AF 이전)과 2024년(AF3 이후)의 *undruggable target* 정의 변화를 — Hopkins & Groom(2002, *Nat. Rev. Drug Discov.*) 고전 정의와 최신 2023~2024 리뷰 1편을 비교해 1페이지로 정리하세요. 특히 — (a) 무질서 단백질, (b) PROTAC 표적, (c) 알로스테릭 부위에 대한 정의 변화를 강조하고, 본인 표적군에서의 함의를 결론으로 명시하세요.

## 참고 자료

- **AlphaFold 2**: Jumper, J. et al. "Highly accurate protein structure prediction with AlphaFold." *Nature*, 2021. — CASP14 결과와 Evoformer·structure module 표준 정의.
- **AlphaFold 3**: Abramson, J. et al. "Accurate structure prediction of biomolecular interactions with AlphaFold 3." *Nature*, 2024. — diffusion 기반 단백질-리간드-핵산 통합 예측.
- **ESMFold**: Lin, Z. et al. "Evolutionary-scale prediction of atomic-level protein structure." *Science*, 2023. — 단일 서열 기반 빠른 예측, 메타게놈 적용.
- **AlphaMissense**: Cheng, J. et al. "Accurate proteome-wide missense variant effect prediction." *Science*, 2023. — 약 7,100만 변이의 병원성 예측.
- **AF-기반 가상 스크리닝**: Lyu, J. et al. "AlphaFold2 structures guide prospective ligand discovery." (논문/연도 확인 필요) — AF 예측 GPCR 구조의 hit rate 검증.
- **드러그러블 정의 표준**: Hopkins, A.L., Groom, C.R. "The druggable genome." *Nat. Rev. Drug Discov.*, 2002. — 드러그러블 표적 정의의 고전.
- **Isomorphic Labs**: isomorphiclabs.com — Alphabet의 AF3 상업화 자회사, 빅파마 파트너십 사례.
- **Boltz-1 / Chai-1**: github.com/jwohlwend/boltz, github.com/chaidiscovery/chai-lab — 오픈 AF3급 모델, 2024년 가을 공개.
- **노벨화학상 2024**: D. Hassabis, J. Jumper, D. Baker — AF 계열과 RFdiffusion의 공동 수상.
