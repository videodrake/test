# Day 3: AI 도킹의 진화 — DiffDock과 신경망 스코어링

> 이전 학습(Phase 2 Week 4 Day 2)에서 AutoDock Vina·Glide로 대표되는 물리 기반 도킹의 원리와 한계를 다뤘습니다. 오늘은 그 한계를 극복하려는 딥러닝 도킹의 흐름 — **신경망 스코어링(neural scoring)**, **엔드투엔드 pose 생성(end-to-end generative docking)**, 그리고 **단백질-리간드 공동 접힘(co-folding)** — 세 축으로 학습합니다.

## 개요

AI 도킹은 — 물리 기반 도킹의 두 병목(부정확한 스코어링 함수, 리간드 conformation 공간의 조합적 폭발)을 딥러닝으로 우회하려는 시도입니다. 2015년 3D-CNN을 스코어링에 적용한 연구를 기점으로, 2022년 **EquiBind**·**TankBind**의 등장, 2023년 **DiffDock**의 확산 모델 도입, 2024년 **AlphaFold3**·**Boltz-1**·**Chai-1**의 co-folding 패러다임 확장으로 이어졌습니다. 그러나 같은 해 **PoseBusters**(Day 37에서 언급)와 후속 벤치마크는 — 딥러닝 도킹이 원자 겹침·비물리적 결합각·훈련 표적 편향에서 실패한다는 사실을 드러냈습니다. 이 사실을 어떻게 해석할 것인가 — 그리고 SaaS 사용자에게 어떻게 정직하게 노출할 것인가 — 가 오늘 학습의 핵심입니다. 약학 전공자에게 본 Day의 의미는 — *co-crystal의 pharmacophore 재현 여부*, *결합 부위 flexibility*, *induced-fit 여부*를 기준으로 딥러닝 도킹의 신뢰 대역을 판정하는 안목을 갖추는 데 있습니다.

## 핵심 개념

### 1) AI 도킹의 3세대 분류

물리 기반 도킹(Day 37)이 검색과 스코어링을 명시적으로 분리한 반면, AI 도킹은 — 그 경계를 점차 허물며 진화해 왔습니다.

| 세대 | 접근 | 대표 방법 | 등장 시기 |
|------|------|--------|--------|
| 1세대: **신경망 재스코어링(rescoring)** | 물리 도킹 pose를 CNN·GNN이 재평가 | **Gnina CNN**, RTMScore, DeepDock | 2015~ |
| 2세대: **엔드투엔드 pose 생성** | 리간드 + 단백질 → 3D 좌표 직접 예측 | **EquiBind**, TankBind, **DiffDock**, Uni-Mol Docking | 2022~ |
| 3세대: **Co-folding (공동 접힘)** | 단백질 + 리간드 + 이온 + 핵산 통합 예측 | **AlphaFold 3**, **Boltz-1/2**, **Chai-1**, RoseTTAFold All-Atom, Protenix | 2024~ |

핵심 흐름은 — *검색·스코어링의 분리 → 통합 → 단백질·리간드·주변 모두의 통합 예측* 순으로 확장돼 왔다는 것입니다. 그러나 세대가 올라간다고 반드시 정확도가 오르지는 않습니다. 자주 등장하는 표적·리간드 조합에서는 모든 세대가 준수하지만 — 훈련 분포 밖(novel 표적, allosteric site, PROTAC, covalent inhibitor)에서는 여전히 실패가 흔합니다.

### 2) 1세대 — 신경망 스코어링

물리 도킹의 스코어링 함수는 대개 5~10개 항의 선형 결합입니다(Day 37 참조). 신경망 스코어링은 — 이 선형성을 비선형 함수 근사기로 대체합니다.

- **Gnina (Ragoza et al., 2017, *J. Chem. Inf. Model.*)** — 결합 부위 주변을 24 Å × 24 Å × 24 Å 3D 격자로 표현하고 3D CNN이 pose를 "실험 결합 자세인가"의 분류·회귀로 학습. Vina로 얻은 상위 pose를 CNN이 재순위화합니다. 오픈소스이며 Vina 커맨드라인과 유사해 실무 적용이 쉽습니다.
- **RTMScore (Shen et al., 2022, *J. Cheminformatics*)** — Residue-atom 거리 기반 그래프 표현으로 잔기 수준의 상호작용을 학습. Vina·Glide 대비 재도킹 정확도 우위 보고.
- **DeepDock (Méndez-Lucio et al., 2021, *Nature Machine Intelligence*)** — pose의 원자 위치를 SE(3)-equivariant 네트워크로 예측 후 GA 검색과 결합.

> **약학적 함의** — 신경망 스코어링은 *물리 항의 대체가 아니라 보완*으로 봐야 합니다. 훈련셋(대개 PDBbind, CASF-2016)에 특정 표적군(kinase·protease)이 과대 대표되므로 — 훈련 표적과 유사한 계열에서는 이득이 크지만, 신규 표적에서는 물리 스코어보다 오히려 나쁜 경우도 흔히 보고됩니다. 실무에서는 *Vina + Gnina + MM-GBSA*의 consensus가 안전한 baseline입니다.

### 3) 2세대 — 엔드투엔드 도킹

pose 검색과 스코어링을 하나의 신경망으로 통합해, 리간드와 단백질 구조를 입력하면 pose를 직접 생성합니다.

| 방법 | 핵심 아이디어 | 특징 |
|------|-----------|------|
| **EquiBind (Stärk et al., 2022, *ICML*)** | SE(3)-equivariant GNN이 리간드 원자 좌표를 회귀 | 매우 빠르나(수 초) 물리 타당성 낮음 |
| **TankBind (Lu et al., 2022, *NeurIPS*)** | 단백질을 blocks로 나눠 후보 site별 pose 생성 | Blind docking에 강점 |
| **DiffDock (Corso et al., 2023, *ICLR*)** | 리간드의 위치·회전·비틀림(torsion) 자유도에 확산 모델 적용 | 다중 pose 샘플링, 확률적 |
| **DiffDock-L (Corso et al., 2024, *ICLR Workshop*)** | DiffDock의 대용량·multi-modal 확장 | RMSD ≤ 2 Å 성공률 개선 |
| **Uni-Mol Docking (Zhou et al., 2023, *arXiv*)** | 사전학습된 분자 3D Transformer 기반 도킹 fine-tuning | 물성 예측과 통합 |
| **SurfDock (Cao et al., 2024, *Nature Methods*, 확인 필요)** | 단백질 표면 mesh 기반 pose 생성 | Cross-docking 벤치마크 상위 |
| **FlowDock (2024, *arXiv*)** | Flow matching 기반 pose·affinity 통합 예측 | 확산 대비 샘플링 효율 개선 |

**DiffDock의 아이디어**는 — 리간드를 강체 병진(3자유도) + 회전(3자유도) + 각 회전 가능 결합의 torsion(N자유도) 으로 분해하고, 각각의 자유도 공간(각 torsion은 원환)에서 확산 모델을 학습합니다. 노이즈에서 시작해 역확산 과정을 거치며 실제 pose 분포로 수렴시킵니다. 강점은 — 다중 pose를 확률 분포로 자연스럽게 샘플링하고 신뢰도(confidence model)를 함께 학습한다는 점입니다.

### 4) 3세대 — Co-folding: 접힘과 도킹의 통합

**AlphaFold 3 (Abramson et al., 2024, *Nature*)**의 등장은 — 단백질 접힘과 도킹의 개념적 구분을 흐렸습니다. 입력이 단백질 서열 + 리간드 SMILES + 이온·핵산 등이고, 출력이 *하나의 원자 좌표 예측*입니다. 이로써 — 리간드 결합 시 단백질의 induced-fit 재조정이 자연스럽게 모델링됩니다.

**주요 co-folding 모델 비교**:

| 모델 | 공개 시점 | 라이선스 | 특징 |
|------|--------|--------|------|
| **AlphaFold 3** | 2024, Google DeepMind + Isomorphic Labs | 비상업 서버 사용, 코드 부분 공개 | 최초의 통합 예측, MSA 의존 |
| **Boltz-1** | 2024, MIT + 협업 | Apache 2.0 (완전 오픈) | MSA 없이도 준수한 성능, 상업 사용 가능 |
| **Boltz-2** | 2025 (확인 필요) | 오픈 | Boltz-1의 후속 개선판 |
| **Chai-1** | 2024, Chai Discovery | 비상업 오픈 | 성능이 Boltz-1과 유사 |
| **RoseTTAFold All-Atom** | 2024, Baker Lab | 학술 | 단백질·핵산·소분자 통합 |
| **Protenix** | 2024, ByteDance | 오픈 | AF3 기반 재구현 |

**PoseBench (Morehead et al., 2024, *arXiv*)**는 — DiffDock-L, DynamicBind, NeuralPLexer, RoseTTAFold-All-Atom, Chai-1, Boltz-1, AlphaFold 3와 전통 물리 도킹을 통합 비교한 최초의 표준 벤치마크입니다. 핵심 결과는 — self-docking(같은 단백질·리간드 쌍의 결정 구조 재현)에서는 co-folding 모델들이 준수하나, cross-docking(다른 리간드-단백질 쌍) 성능이 약 60%에서 정체된다는 점입니다. 반면 SurfDock, Uni-Mol Docking처럼 도킹에 특화된 딥러닝 아키텍처가 cross-docking에서 우위를 보였습니다.

### 5) 딥러닝 도킹의 실패 유형

**PoseBusters (Day 37 참조)**는 딥러닝 도킹이 pose는 생성하지만 물리적으로 타당하지 않은 사례가 흔하다는 사실을 정량화했습니다. 대표 실패 유형:

| 실패 유형 | 설명 | 임상적 함의 |
|--------|------|-----------|
| **원자 겹침(steric clash)** | 리간드 원자가 단백질 원자와 반경 이내로 겹침 | 실제 결합 불가능 |
| **비물리적 결합각·비틀림** | 리간드 내부 결합 기하가 유효 conformation 밖 | 합성 불가능한 pose |
| **카이랄성 뒤집힘** | R/S 뒤바뀜 | 실제 활성과 무관한 예측 |
| **결합 부위 이탈** | binding pocket 밖에 pose 배치 | pharmacophore 재현 실패 |
| **훈련 표적 편향** | 학습 데이터에 많은 표적군은 정확, 신규 표적은 부정확 | novel target 개발성 저평가 |

> **약학적 판단** — 딥러닝 도킹의 결과는 반드시 (a) PoseBusters류 물리 필터, (b) 알려진 co-crystal의 pharmacophore 재현 검증, (c) MM-GBSA·FEP 재스코어링을 통과해야 실무 활용 가능합니다. 특히 신규 표적의 경우 — self-docking 성능 좋다고 cross-docking 성능이 자동 보장되지 않으므로, 유사 계열의 결정 구조 1~2개로 사전 검증이 필수입니다.

## 작동 원리와 아키텍처

전형적인 **하이브리드 AI 도킹 파이프라인** — 1인 창업자가 바이브코딩으로 구성 가능한 참조 아키텍처:

```
[입력 → 처리 → 출력]

1. 표적 준비 (Protein preparation)
   - AlphaFold DB / PDB에서 구조 수집
   - 결합 부위 검출 (Fpocket, DoGSiteScorer) 또는 사용자 지정
   - 명시 물·이온 처리 (WaterMap 대안)

2. 리간드 준비 (Ligand preparation)
   - SMILES → 3D conformer (RDKit ETKDG)
   - stereochem·tautomer 열거 (Enumerator)

3. 이중 트랙 도킹 실행
   Track A (물리 baseline):
     - AutoDock Vina — box 지정, exhaustiveness 8, num_modes 10
   Track B (딥러닝):
     - DiffDock-L 또는 Boltz-1 (오픈 라이선스 우선)
     - N samples (예: 40), confidence score 함께 출력

4. Consensus + 재스코어링
   - 두 트랙의 pose를 RMSD 기반 클러스터링
   - Gnina CNN 재스코어 (Vina 상위 pose에 적용)
   - MM-GBSA (Prime, gmx_MMPBSA) — 상위 20 pose

5. 물리·약학 검증 필터
   - PoseBusters 실행 — 겹침·비물리적 각·카이랄성 체크
   - Pharmacophore 필터 — 알려진 co-crystal의 핵심 상호작용 재현 확인
   - PAINS·독성·ADMET (Day 31~34 기준)

6. 출력 리포트
   - 순위 (rank), 신뢰 등급 (high/medium/low), pose PDB
   - 실패 사유 태그 (steric clash / off-site / low pharmacophore match 등)
   - 약학적 해석 메모 (자동 생성)
```

**핵심 설계 결정** — 어떤 조합이 1인 창업자에게 최적인가:

| 결정 | 선택지 | 추천 | 이유 |
|------|--------|-----|------|
| 딥러닝 엔진 | DiffDock-L / Boltz-1 / AF3 서버 / Uni-Mol | **Boltz-1** | Apache 2.0 상업 사용, MSA 불필요 |
| 물리 baseline | 없음 / Vina / Glide | **Vina 필수 병행** | 딥러닝 실패 대비 안전망 |
| 스코어링 조합 | 단일 / consensus / MM-GBSA | **Consensus + MM-GBSA 상위만** | 비용·정확도 균형 |
| 물리 필터 | 없음 / RDKit 기본 / PoseBusters | **PoseBusters 필수** | 딥러닝의 물리 오류 차단 |
| 클라우드 | On-prem / AWS/GCP / RunPod | **GPU spot 인스턴스** | GPU 시간 최적화 |

## 신약개발 적용

AI 도킹은 이미 여러 신약개발 파이프라인의 필터 레이어로 통합되어 있습니다. 대표 사례 — 최근 2년 위주로:

| 사례 | 표적 / 상황 | 접근 | 결과 |
|------|-----------|------|------|
| **Isomorphic Labs × Eli Lilly · Novartis (2024)** | 다중 표적 | AlphaFold 3 기반 통합 예측 | 계약 규모 총 약 17억 달러, AI 신약개발 최대 파트너십 중 하나 |
| **Recursion + Exscientia 합병 (2024)** | 다중 표적 플랫폼 | 통합 AI 도킹·표적 프로파일링 | 두 AI 신약 선두 통합, 상장사 기준 최대 규모 |
| **Insilico Chemistry42 진화** | INS018_055 등 다중 후보 | 생성 + 도킹 + ADMET 통합 | INS018_055 Phase 2 진행 (Day 37 참조) |
| **DiffDock의 상업적 채택 (2024~)** | Terray Therapeutics 등 | 오픈 DiffDock을 자체 파이프라인에 통합 | 사내 hit ID 워크플로우 가속 |
| **CACHE Challenge (2023~)** | LRRK2, TDP2 등 공개 표적 | 다양한 AI 도킹의 blind prediction 비교 | 상위 팀 hit rate 20~40%, 대부분 hybrid 접근 |

**정량 비교** — 물리 도킹(Day 37) 대비 AI 도킹의 실제 개선:

| 지표 | 물리 기반 (Vina/Glide) | 딥러닝 특화 (DiffDock-L/SurfDock) | Co-folding (AF3/Boltz-1) |
|------|------|------|------|
| Self-docking 성공률 (RMSD ≤ 2 Å, PoseBusters valid) | 약 40~55% | 약 50~76% | 약 60~76% |
| Cross-docking 성공률 | 약 20~35% | 약 40~60% | 약 50~60% (정체) |
| 신규 표적(novel scaffold) 성공률 | 약 15~25% | 약 20~35% | 약 20~40% |
| 처리 속도 (pose/초, GPU) | 약 0.1~1 | 약 5~50 | 약 0.5~5 (co-folding 무거움) |
| 물리 타당성 (PoseBusters 통과) | 약 90% 이상 | 약 50~70% | 약 60~85% |

> **산업적 통찰** — AI 도킹은 물리 도킹의 대체가 아니라 *확장*으로 자리잡고 있습니다. 표준 워크플로우는 — (a) 대규모 라이브러리는 대리모델(Deep Docking 등)로 필터, (b) 축소된 후보를 딥러닝 도킹으로 pose 생성, (c) 물리 필터·MM-GBSA로 검증, (d) 상위 후보만 wet lab 검증 — 순의 다중 필터가 정착 중입니다. Co-folding(AF3/Boltz-1)의 도래는 특히 — *결정 구조가 없는 신규 표적*에 대한 접근성을 근본적으로 넓혔습니다.

## 창업 관점

Day 37에서 다룬 도킹 SaaS 기회 위에 — AI 도킹은 *진입 장벽 완화*와 *차별화 지점 상승*을 동시에 가져왔습니다. Boltz-1이 Apache 2.0으로 공개되면서 상업 사용의 라이선스 장벽이 사라졌고 — 1인 창업자도 co-folding 기반 도킹 서비스를 몇 주 안에 프로토타이핑할 수 있게 됐습니다. 반면 순수 모델 성능만으로는 차별화가 어려워졌으므로 — **약학 전공자의 진입 각도**는 다음 세 방향이 유력합니다.

첫째, **정직한 신뢰도 UX** — PoseBusters류 물리 필터, cross-docking 성공률, 표적군별 벤치마크를 사용자에게 노출해 *"이 결과가 얼마나 믿을 만한가"*를 등급화합니다. Schrödinger·OpenEye는 기술 성숙도 때문에 결과를 "확정적"으로 보여주지만 — AI 도킹 세대의 고객은 오히려 *불확실성의 정직한 노출*을 원합니다. 둘째, **표적군 특화 프리셋** — kinase hinge, GPCR orthosteric, protease catalytic 등 각 표적군의 co-crystal 데이터로 fine-tuning한 특화 모델을 제공. 셋째, **AF3-대체 오픈 파이프라인 서비스** — AF3 서버가 비상업 제약을 가지므로, 상업 사용 가능한 Boltz-1/Chai-1 기반 co-folding SaaS는 명확한 시장 니즈가 존재합니다.

MVP 구상 — Boltz-1 + Vina 하이브리드 백엔드에 PoseBusters 자동 검증을 얹고, 표적 하나(예: kinase panel)에 특화된 웹 리포트를 상단 UI로 제공하는 형태가 현실적입니다. 초기 시장은 학술 랩·중소 바이오의 월 500~5,000 달러 구독, 확장은 pharma의 표적별 배치 처리 계약(월 5,000~50,000 달러). 경쟁 우위는 — *약학 지식이 반영된 자동 해석 리포트*와 *신뢰도의 정직한 등급화* 입니다.

## 오늘의 과제

1. **DiffDock vs Vina 정면 비교 리서치 (60분)**: PoseBusters 논문(2024, *Chemical Science*) 또는 PoseBench 논문(2024, arXiv) 중 하나를 읽고 — (a) DiffDock-L·Boltz-1·AF3의 self-docking / cross-docking / novel scaffold 성능을 Vina·Glide와 비교한 수치를 표로 정리, (b) 물리 타당성(steric clash, chirality)에서의 실패율 비교, (c) 두 논문의 결론이 서로 다르다면 그 원인을 3가지로 정리하세요. 결론으로 — 본인 SaaS에서 *어떤 표적·리간드 조합에 어떤 모델을 우선 적용*할지 판단 매트릭스를 스케치하세요.

2. **Boltz-1 실습 계획 수립 (40분)**: 본인 관심 표적 1개를 정하고 — (a) Boltz-1을 로컬 또는 서버(Hugging Face Spaces, Rowan 등)에서 실행하는 데 필요한 입력(단백질 서열, 리간드 SMILES, MSA 옵션)과 예상 GPU 시간, (b) 출력(원자 좌표, confidence, PAE)의 해석 방법, (c) 결과를 PyMOL 또는 ChimeraX로 시각화하는 절차를 1페이지로 정리하세요. 실행이 가능하면 실제로 pose 하나를 생성해 PoseBusters로 검증까지 진행하세요.

3. **AI 도킹 SaaS 3사 비교 분석 (50분)**: 최근 등장한 AI 도킹 서비스 3곳(예: Iambic Therapeutics, Terray Therapeutics, Iktos, Deep Origin, Rowan 등에서 3사 선택)의 — (a) 사용하는 AI 도킹 모델, (b) 가격 모델(구독/사용량/파트너십), (c) 타겟 고객(학술/중소 바이오/빅파마), (d) 차별화 포지셔닝을 표로 비교하세요. 결론으로 — 본인 SaaS가 시장에 진입한다면 *비어있는 포지션*이 무엇인지, 그 포지션을 잡기 위한 최소 기능(MVP)이 무엇인지 1페이지로 정리하세요.

## 참고 자료

- **DiffDock**: Corso, G. et al. "DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking." *ICLR 2023*. — 확산 모델을 도킹에 적용한 최초의 주요 연구.
- **DiffDock-L**: Corso, G. et al. "Deep Confident Steps to New Pockets: Strategies for Docking Generalization." *ICLR 2024 Workshop*. — DiffDock의 확장·일반화 개선.
- **AlphaFold 3**: Abramson, J. et al. "Accurate structure prediction of biomolecular interactions with AlphaFold 3." *Nature*, 2024. — 단백질·리간드·이온·핵산 통합 예측의 이정표.
- **Boltz-1**: Wohlwend, J. et al. "Boltz-1 Democratizing Biomolecular Interaction Modeling." *bioRxiv*, 2024. — Apache 2.0 오픈 co-folding 모델.
- **Chai-1**: Chai Discovery Team. "Chai-1: Decoding the molecular interactions of life." *Technical Report*, 2024. — Boltz-1과 유사 성능의 co-folding 모델.
- **PoseBench**: Morehead, A. et al. "Assessing the potential of deep learning for protein-ligand docking." *Nature Machine Intelligence*, 2025 (확인 필요, arXiv 2405.14108). — AI 도킹 통합 벤치마크.
- **PoseBusters**: Buttenschoen, M., Morris, G.M., Deane, C.M. "PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences." *Chemical Science*, 2024. — 딥러닝 도킹의 물리 타당성 검증 표준.
- **Gnina**: Ragoza, M. et al. "Protein–Ligand Scoring with Convolutional Neural Networks." *Journal of Chemical Information and Modeling*, 2017. — 신경망 재스코어링의 대표 오픈 도구.
- **EquiBind**: Stärk, H. et al. "EquiBind: Geometric Deep Learning for Drug Binding Structure Prediction." *ICML 2022*. — 최초의 SE(3)-equivariant 엔드투엔드 도킹.
- **CACHE Challenge**: cache-challenge.org — AI 도킹의 blind prediction 벤치마크 컴피티션.
- **Rowan (Boltz-1 실행 도구)**: rowansci.com/tools/boltz-1 — Boltz-1 웹 실행 UI (확인 필요).
