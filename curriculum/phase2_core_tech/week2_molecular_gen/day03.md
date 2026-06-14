# Day 3: 강화학습과 목표지향 생성 — 원하는 물성으로 최적화

> 이전 학습(Day 2, P2W2D2)에서 잠재 공간 기반 생성 모델(VAE/GAN) — 화학 공간을 연속 벡터로 압축한 뒤 그 공간을 항행하여 신규 분자를 만드는 패러다임 — 을 다뤘습니다. 그러나 VAE의 잠재 공간 항행만으로는 "pIC50 > 8, hERG < 5, BBB +"처럼 **여러 약학적 목표를 동시에 만족하는 분자**를 효율적으로 찾기 어렵습니다. 오늘은 이 한계를 해결하는 두 번째 계열 — **강화학습(Reinforcement Learning, RL)과 목표지향 생성(goal-directed generation)**을 학습합니다. RL은 생성 모델을 "분포 학습기"에서 "보상 최대화 에이전트(reward-maximizing agent)"로 전환시키며, 약학 전공자에게 익숙한 **다중 파라미터 최적화(Multi-Parameter Optimization, MPO)** 사고를 그대로 알고리즘으로 옮긴 영역입니다.

## 개요

**강화학습 기반 분자 생성**은 — 생성 모델(policy)이 매 step에서 분자의 다음 토큰/노드를 선택할 때, 약학적으로 정의된 **보상 함수(reward function)**의 누적 기댓값을 최대화하도록 정책 파라미터를 갱신하는 방식입니다. 핵심 차이는 명확합니다 — VAE/GAN이 "ChEMBL 같은 학습 데이터의 분포 p(mol)에 가까운 분자"를 만드는 데 그친다면, RL은 "내가 원하는 물성 프로파일에 가까운 분자"를 만들도록 분포 자체를 이동(shift)시킵니다. 약학 전공자의 관점에서 이 주제가 중요한 이유는 — 보상 함수 설계가 사실상 **target product profile(TPP)을 수식으로 번역하는 작업**이며, 임상 단계의 go/no-go 기준·MPO 가중치·치료 영역별 우선순위에 대한 도메인 판단이 모델 성능을 직접 좌우하기 때문입니다. 2024년 기준 산업 De Novo 파이프라인의 약 70%(MOSES·GuacaMol 인용 분석)가 RL 또는 RL-VAE 하이브리드를 핵심 엔진으로 사용합니다.

## 핵심 개념

### 1) 강화학습의 기본 틀 — 분자 생성에의 매핑

강화학습은 **에이전트(agent)**가 **환경(environment)** 안에서 **행동(action)**을 선택하고, **보상(reward)**을 받으며 **정책(policy)** π(a|s)를 개선하는 학습 패러다임입니다. 분자 생성으로 옮기면:

| RL 요소 | 분자 생성에서의 의미 | 예시(SMILES 기반) |
|--------|----------------|--------------|
| **State (s)** | 지금까지 생성된 부분 분자 | "Cc1ccc(" (불완전 SMILES) |
| **Action (a)** | 다음에 붙일 토큰/원자/결합 | "N", "c", "=O", "(", ")" |
| **Policy π(a\|s)** | 다음 토큰의 확률 분포 | RNN/Transformer 출력 |
| **Trajectory** | 한 분자 생성의 전체 경로 | start → ... → end-of-SMILES |
| **Reward R** | 완성된 분자의 평가 점수 | QED·pIC50·-hERG·-SAscore 가중합 |

핵심은 — **보상이 완성된 분자에 대해서만 한 번 주어진다는 점**(sparse, terminal reward)입니다. 약학적으로 비유하면 — 후보 화합물을 합성·in vitro 평가까지 마쳐야만 활성치를 알 수 있는 medicinal chemistry 워크플로우와 동일한 구조입니다. RL 알고리즘의 본질은 — "어떤 토큰 선택이 최종 보상에 얼마나 기여했는지" 역추적하여 그 토큰의 확률을 조정하는 것(**credit assignment**)입니다.

### 2) 정책 경사(Policy Gradient)와 REINFORCE — 분자 RL의 표준 알고리즘

분자 생성에서 가장 널리 쓰이는 알고리즘은 **REINFORCE**(Williams, 1992)와 그 변형입니다. 학습 목표는 — 기대 보상 J(θ) = E_π[R]를 최대화하도록 정책 파라미터 θ를 갱신하는 것:

```
θ ← θ + α · ∇_θ log π_θ(τ) · (R(τ) - b)
```

- τ: 한 분자의 생성 trajectory
- R(τ): 그 분자의 보상
- b: baseline(보통 이동 평균 보상) — 분산(variance) 감소 목적
- α: 학습률

직관적 해석은 — **보상이 baseline보다 높으면 그 trajectory의 토큰 확률을 올리고, 낮으면 내린다**는 것입니다. 약학적으로 옮기면 — Free-Wilson 분석에서 특정 R-그룹 치환이 활성을 향상시킨 데이터를 보고 그 치환을 더 자주 시도하는 medicinal chemist의 직관과 정확히 동일합니다.

REINFORCE의 한계 — 분산이 크고 수렴이 느림 — 를 보완하기 위해 등장한 변형:

| 알고리즘 | 핵심 개선 | 분자 적용 |
|---------|---------|---------|
| **PPO (Proximal Policy Optimization)** | 정책 변화 크기를 clipping하여 안정화 | REINVENT 4, MolPPO |
| **A2C (Advantage Actor-Critic)** | Value network로 advantage A(s,a) 추정 | GCPN (Graph PPO) |
| **DQN (Deep Q-Network)** | Action-value Q(s,a)를 직접 학습 | MolDQN (개별 원자 추가/제거를 action으로) |
| **HER (Hindsight Experience Replay)** | 실패 trajectory를 다른 목표로 재라벨링 | 희소 보상 환경에서 sample 효율 ↑ |

산업 표준은 — **PPO + RNN/Transformer prior**의 조합입니다(REINVENT 4 기본 설정, 2024).

### 3) Prior와 Fine-tuning — 약물 유사성을 잃지 않는 학습

순수 RL을 빈 분자에서 시작하면 — 보상이 높지만 화학적으로 비현실적인(예: 30개 질소가 연결된) 분자로 수렴할 수 있습니다(이를 **reward hacking** 또는 **adversarial molecule**이라 부릅니다). 이를 막기 위한 표준 절차는:

1. **Prior 사전학습**: ChEMBL/ZINC15 1.5M 약물 유사 분자로 RNN을 maximum likelihood로 학습. 결과는 "약물 같은" 분자를 생성하는 baseline 정책 π_prior.
2. **Agent 초기화**: π_agent ← π_prior (동일 가중치로 복사)
3. **RL fine-tuning**: π_agent를 보상으로 갱신하되, π_prior와의 **KL divergence**(또는 augmented likelihood)를 정규화 항으로 추가하여 약물 유사성을 유지.

REINVENT(Olivecrona et al., 2017)가 제안한 augmented likelihood 보상:

```
augmented_logP(SMILES) = prior_logP(SMILES) + σ · S(SMILES)
loss = (agent_logP - augmented_logP)²
```

여기서 S는 desirability score(0~1), σ는 보상 가중치(보통 60~120). 약학적 해석은 — **"약물 유사 분포에서 너무 멀어지지 않으면서 원하는 물성을 향해 이동"** 하는 안전한 탐색입니다. 이는 약학 전공자가 SAR 전개 시 — "골격(scaffold)은 유지하면서 R-그룹만 변화시킨다"는 보수적 전략과 구조적으로 동일합니다.

### 4) 보상 함수 설계 — MPO의 자동화

보상 함수는 RL 분자 생성의 **핵심이자 가장 큰 약학적 의사결정 지점**입니다. 단일 물성 최적화는 거의 무의미하며 — 실제 약물 후보는 활성·물성·안전성·합성성을 동시에 만족해야 합니다(MPO). 표준 보상 구조는 — 각 물성 예측치 p_i(mol)을 desirability 함수 s_i로 0~1 정규화한 뒤 가중치 w_i를 적용해 곱하는 형태(R = Π s_i(p_i)^w_i)입니다.

- p_i: 물성 예측기(Week 1의 ECFP+XGBoost, MolFormer 등)
- s_i: desirability 함수(0~1로 정규화, sigmoid·gaussian·step)
- w_i: 가중치

대표적 desirability 함수 설계:

| 물성 | 목표 형태 | desirability 곡선 |
|------|--------|---------------|
| **pIC50** | maximize | sigmoid(상한 없음, 7 이상 ≈ 1.0) |
| **logP** | range(1~4) | gaussian 또는 trapezoid |
| **hERG IC50** | minimize toxicity | reverse sigmoid(10μM 이상 ≈ 1.0) |
| **QED** | maximize drug-likeness | 그대로 사용(0~1) |
| **SAscore** | minimize | (10 - SAscore) / 10 |

약학 전공자의 경쟁력이 명확히 드러나는 지점:

- **가중치 w_i 설정**: 중추신경계(CNS) 약물이면 BBB 투과성·hERG 가중치 ↑, 항암제 경구약이면 PK·CYP 안정성 가중치 ↑.
- **desirability 함수 형태**: 단순 sigmoid가 아니라 — 임상적 의미 있는 임계값(예: hERG 10μM, IC50 100nM)에서 가파른 변화가 일어나는 piecewise 설계.
- **물성 간 trade-off의 임상적 우선순위**: 활성이 약간 떨어져도 hERG 회피가 더 중요한 영역(IKr 의존성 부정맥 우려) 같은 판단은 약학 지식이 직접 코드화되는 지점입니다.

### 5) 그래프 RL과 단계별 분자 구축 — GCPN, MolDQN

SMILES RL이 토큰 단위라면 — **GCPN(Graph Convolutional Policy Network, You et al., 2018, NeurIPS)**과 **MolDQN(Zhou et al., 2019)**은 그래프 단위로 작동합니다. Action 공간이 더 풍부합니다:

| Action 유형 | 의미 |
|-----------|------|
| add_atom | 새 원자를 그래프에 추가 |
| add_bond | 두 원자 사이 결합 형성 |
| remove_bond | 결합 제거 |
| modify_atom | 원자 종류 변경(예: C → N) |
| terminate | 분자 완성 |

매 step에서 valid한 화학 규칙(valence)을 강제하므로 — validity 100% 보장이 가능합니다. 약학적 의미는 — medicinal chemist가 R-그룹을 한 번에 하나씩 치환하며 SAR을 탐색하는 과정의 직접적 알고리즘화입니다. 다만 — 그래프 action 공간이 SMILES 토큰보다 크고, 한 분자당 step 수가 많아 학습 비용이 큽니다(SMILES RL 대비 약 5~10배). 산업에서는 — **빠른 prototype은 REINVENT(SMILES), 정교한 scaffold-constrained 설계는 GCPN 계열**로 분업이 이루어집니다.

## 작동 원리와 아키텍처

REINVENT 4 기준 표준 RL 분자 생성 파이프라인:

```
[REINVENT 4 — 의사 아키텍처, 2024 기준]

1. Prior 사전학습 (1회, 약 48시간/GPU)
   - ChEMBL 1.5M 약물 유사 분자
   - RNN(GRU 3-layer, hidden 512) maximum likelihood 학습
   - 결과: π_prior — Validity ≈ 97%, drug-like 분포 수렴

2. Scoring Function 정의 (사용자 약학 지식 입력)
   - 활성: docking score 또는 QSAR pIC50 예측(Week 1 모델)
   - ADMET: hERG, BBB, CYP3A4 inhibition 예측기
   - 합성성: SAscore
   - drug-likeness: QED, Lipinski
   - 각 컴포넌트별 desirability 함수 + 가중치

3. Agent RL 학습 루프 (반복, 약 500~2000 step)
   a. Agent π_agent(현재 가중치)로 batch=128 분자 sampling
   b. 각 분자를 scoring function으로 평가 → R 산출
   c. PPO/REINFORCE로 정책 갱신
      loss = (agent_logP - (prior_logP + σ·R))²
   d. KL(π_agent || π_prior) 모니터링(과도한 drift 방지)

4. 출력 분자 풀
   - 최종 1,000~10,000 분자
   - Tanimoto 다양성 필터 + scaffold 클러스터링
   - top-200 의약화학자 수동 검토 대상

GPU 1장(A100) 기준 학습 시간: 4~12시간/full run
1분자당 평가 비용: 5~50ms (scoring 컴포넌트 수에 따라)
```

핵심 설계 결정:

| 결정 항목 | 선택 | 이유 |
|---------|------|------|
| Prior 데이터 | ChEMBL vs ZINC15 vs 특화 라이브러리 | 표적 family 특화 시 fine-tuned prior가 더 효율적 |
| RL 알고리즘 | REINFORCE / PPO / A2C | PPO가 안정성·sample 효율 모두 양호, 2024 표준 |
| σ (보상 가중치) | 60~120 | 너무 작으면 학습 안 됨, 너무 크면 reward hacking |
| KL 정규화 | 동적 조절(KL > 임계값 시 페널티 ↑) | 약물 유사성과 보상 최대화의 균형 |
| Diversity filter | Bemis-Murcko scaffold bucket, 분자당 50개 cap | mode collapse 방지 |

## 신약개발 적용

2023~2024년 산업 사례 — RL 기반 De Novo 생성:

| 회사 / 기관 | 시스템 | 적용 / 결과 | 비고 |
|----------|------|---------|------|
| **AstraZeneca** | REINVENT 4 (오픈소스, J. Cheminformatics 2024) | 사내 다수 lead 발굴 가속, 산업 표준 De Novo RL 프레임워크로 확산 | 비영리·연구용 무료 |
| **Insilico Medicine** | Chemistry42 내 RL 모듈 | INS018_055(IPF, Phase 2), USP1 저해제 등 | RL + GAN + Active Learning 통합 |
| **Iktos** | Makya (RL 기반 조건부 생성) | Sanofi·Almirall 등과 협업, 평균 lead 발굴 6~9개월 단축 | 마일스톤 + 라이선스 모델 |
| **Cyclica (현 Recursion 산하)** | Ligand Design (RL + 다중 표적) | 다중 표적 polypharmacology 분자 설계 | 2023년 Recursion 인수 |
| **DeepMind / Isomorphic Labs** | (비공개) RL + 구조 기반 점수 | 자체 파이프라인, 다수 빅파마 협업(Eli Lilly, Novartis) | 2024년 약 30억 달러 평가 |

정량적 효과 — REINVENT 4 공개 벤치마크(Loeffler et al., 2024):

| 평가 항목 | Prior only | RL fine-tuned |
|---------|----------|-------------|
| 목표 물성(pIC50 > 7) 달성률 | 약 2% | **약 60~85%** |
| Validity | 97% | 95% (약간 하락) |
| Uniqueness | 99% | 80~90% |
| 약물 유사 영역 유지(QED > 0.6) | 70% | 50~65% (drift 발생) |

해석 — RL은 목표 물성 달성률을 **30~40배** 향상시키지만, 약물 유사성·다양성은 일정 하락합니다. 이 trade-off는 약학적 보상 함수 설계(특히 QED·SAscore 가중치)로 어느 정도 통제 가능하며 — 이 통제력이 약학 전공자의 차별화 영역입니다.

기존 medicinal chemistry MPO 대비:

| 단계 | 전통 MPO design-make-test cycle | RL De Novo |
|------|---------------------------|---------|
| 1 사이클 분자 수 | 50~200 | 1,000~10,000 (in silico) |
| 1 사이클 시간 | 6~12주 (합성·in vitro) | 4~24시간 (GPU) |
| 동시 최적화 물성 | 3~5개 | 6~12개(스코어링 함수에 추가) |
| 합성·실험 비용 | 분자당 약 1만 달러 | 무료(검증 단계 전까지) |
| 임상 실패 예방 가능성 | 경험 의존 | ADMET·toxicity 모듈 통합 시 ↑ |

다만 **함정**도 분명합니다 — RL이 생성한 분자의 in silico 보상 점수와 실제 wet lab 결과 간 상관계수는 보고된 사례에서 약 0.4~0.6 수준(2023~2024 보고)이며, scoring 모델의 약점(out-of-distribution 분자 예측 불안정)이 그대로 노출됩니다. 이 때문에 RL 출력은 — 그 자체가 "후보 분자"가 아니라 "**합성·평가 우선순위가 매겨진 후보 풀**"로 봐야 합니다.

## 창업 관점

RL 기반 De Novo는 — Day 2의 VAE/GAN보다 산업적 채택률이 높고(2024년 기준 약 70%), 약학 도메인 지식이 알고리즘의 결정적 모듈(보상 함수)에 직접 코드화되므로 — 약학 전공 풀스택 창업자에게 가장 진입 매력이 큰 영역으로 평가됩니다.

**1) 차별화 포인트 — Scoring Function as a Service**: REINVENT 자체는 오픈소스이므로 RL 엔진 자체로는 차별화가 어렵습니다. 그러나 — **scoring function 모듈은 도메인 지식이 직접 반영되는 곳**입니다. 특정 적응증(예: 알츠하이머)·특정 표적 family(예: GPCR)·특정 환자군(예: 소아 PK)을 위한 정교한 desirability·MPO 가중치 세트가 자산이 됩니다. Schrödinger LiveDesign이 이 영역에서 부분적으로 차별화를 만들었고 — 약학 전공자가 더 좁은 버티컬에서 깊이로 경쟁할 수 있습니다.

**2) MVP 후보 — "MPO Designer for [특정 영역]"**: REINVENT 4 위에, (a) 특정 영역(예: BBB 투과 CNS 약물)의 ADMET 예측기 패키지, (b) 약학적 우선순위가 사전 설정된 desirability 템플릿(5~10개), (c) 웹 UI(가중치 슬라이더 + 진행 모니터링 + 분자 풀 다운로드)를 결합한 SaaS. 바이브코딩으로 3~6개월, 가격은 사용자당 월 500~2,000달러 또는 프로젝트당 1~5만 달러 라이선스 모델이 현실적입니다.

**3) 경쟁 구도와 진입점**: AstraZeneca(REINVENT, 무료), Insilico(Chemistry42, 자체 파이프라인 중심), Iktos(Makya, 빅파마 협업)로 주요 플레이어가 분포해 있습니다. 약학 전공 1인 창업의 합리적 진입점은 — **REINVENT 위 도메인 특화 wrapper SaaS**입니다. 직접 RL 엔진을 처음부터 만들 필요는 없으며, 약학적 평가 모듈·UX·특정 영역 사전 설정에 자원을 집중해야 합니다.

**4) 시장 규모와 BM**: AI 분자 설계 SaaS 시장은 2024년 약 4~6억 달러로 추정되며(BCC Research, IDC 인용), 2030년 약 20억 달러 전망입니다. 마진은 SaaS 표준 70~85% 수준. 약사 1인 창업의 첫 ARR 목표는 — 1년 차 5~10개 고객(주로 중소 바이오텍·CRO), 약 30~50만 달러 ARR이 현실적 시나리오입니다.

## 오늘의 과제

1. **기초 학습 — REINVENT 알고리즘 핵심 파악(45분)**: Olivecrona, M. et al. "Molecular De-Novo Design through Deep Reinforcement Learning" (*J. Cheminformatics*, 2017)의 Abstract와 Section 2(Methodology)를 읽고 — (a) augmented likelihood 수식의 직관적 의미, (b) prior와 agent의 차이, (c) DRD2 활성 분자 생성 사례에서 보고된 success rate를 1페이지로 정리하세요. 자신이 익숙한 약물 1개(예: imatinib, gefitinib)의 활성·hERG·logP·CYP 프로파일을 표로 만들고, 이를 REINVENT의 scoring function으로 옮긴다면 어떤 desirability 함수와 가중치로 설정할지 함께 작성하세요.

2. **비즈니스 분석 — RL 기반 De Novo 플랫폼 3사 비교(40분)**: AstraZeneca REINVENT 4(오픈소스, 깃허브 + 논문), Iktos Makya, Insilico Chemistry42의 (a) 공개된 RL 알고리즘(REINFORCE/PPO/기타), (b) scoring function의 도메인 커스터마이즈 가능성, (c) 라이선스·가격 모델·타겟 고객을 표로 정리하세요. 약학 전공 1인 창업자가 REINVENT 위에 wrapper SaaS를 만든다면 — 어떤 도메인(적응증·표적 family)을 첫 버티컬로 선택하겠는지 2~3문장 의견을 덧붙이세요.

3. **리서치 정리 — 보상 함수 설계 실습(60분)**: 본인이 가장 관심 있는 표적 1개(예: EGFR T790M, BACE1, JAK1)에 대해 — (a) target product profile(TPP)을 5~7개 항목으로 정의, (b) 각 항목을 desirability 함수로 변환(곡선 형태·임계값 명시), (c) 가중치 설계와 그 약학적 근거(임상 실패 데이터·기존 약물 사례)를 1.5~2페이지 보고서로 작성하세요. 마지막에 — "이 보상 함수가 reward hacking에 취약한 지점은 어디인가" 1문단 분석을 포함하세요.

## 참고 자료

- **논문**: Olivecrona, M., Blaschke, T., Engkvist, O., Chen, H. "Molecular De-Novo Design through Deep Reinforcement Learning." *Journal of Cheminformatics*, 2017. — REINVENT의 원전, RL 분자 생성의 augmented likelihood 패러다임 정립.
- **논문**: Loeffler, H. H. et al. "REINVENT 4: Modern AI–driven generative molecule design." *Journal of Cheminformatics*, 2024. — 산업 표준 RL De Novo 프레임워크의 2024년 버전.
- **논문**: You, J., Liu, B., Ying, R., Pande, V., Leskovec, J. "Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation." *NeurIPS*, 2018. — 그래프 단위 RL의 결정적 연구.
- **오픈소스**: AstraZeneca REINVENT 4 (GitHub: MolecularAI/REINVENT4) — Apache 2.0 라이선스, 산업·학술 양쪽 표준.
- **벤치마크**: GuacaMol (Brown et al., 2019) — RL 분자 생성 목표지향 최적화 능력의 표준 평가 프레임워크.
