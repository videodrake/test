# 매일 학습 콘텐츠 업데이트 루틴

아래 절차에 따라 학습 콘텐츠를 1일치 업데이트합니다.

## 0단계: 환경 확인 + 진도 base 동기화 (가장 중요)

> **이 단계를 건너뛰면 Day 5만 무한 반복하는 사고가 발생합니다.**
> 매 세션은 새 브랜치(`claude/jolly-mccarthy-XXX`)에서 시작하지만, **출발점인 main은 PR이 머지될 때까지 갱신되지 않습니다.** 그래서 여러 세션이 모두 main만 보고 시작하면 같은 Day를 중복 작성하게 됩니다.
> 따라서 매 루틴 시작 시 **미머지 `claude/*` 브랜치까지 모두 스캔해서 가장 진도 앞선 상태를 base로 선택**해야 합니다.

```bash
cd ~/test
git fetch origin --prune

# 1) 후보: main + (현 main을 조상으로 가진 origin/claude/* 브랜치만)
#    → 옛 폐기 커리큘럼(예: phase1_ml_foundations/) 브랜치는 자동 제외됨
CANDIDATES=$(
  echo main
  git branch -r | grep -oE "origin/claude/[^ ]+" | sed 's|origin/||' | while read ref; do
    if git merge-base --is-ancestor origin/main "origin/$ref" 2>/dev/null; then
      echo "$ref"
    fi
  done
)

# 2) 후보 중 가장 많은 day*.md를 가진 base 선택 (동률은 최근 커밋 우선)
BEST_BASE=$(
  echo "$CANDIDATES" | while read ref; do
    count=$(git ls-tree -r "origin/$ref" --name-only 2>/dev/null \
      | grep -cE "curriculum/.*/day[0-9]+\.md$")
    ts=$(git log -1 --format=%ct "origin/$ref" 2>/dev/null || echo 0)
    printf "%05d %d %s\n" "$count" "$ts" "$ref"
  done | sort -rn | head -1 | awk '{print $3}'
)
echo "[INFO] 진도 base = origin/$BEST_BASE"

# 3) 현재 작업 브랜치를 BEST_BASE로 정렬 (현재 브랜치는 막 만들어진 빈 작업 브랜치이므로 안전)
git reset --hard "origin/$BEST_BASE"
```

**왜 `merge-base --is-ancestor` 필터가 필요한가**: 과거에 커리큘럼 구조를 재설계한 적이 있어 (`phase1_ml_foundations/` → `phase1_ai_foundations/`), 옛 구조에서 작성된 브랜치들이 원격에 그대로 남아있습니다. 그 브랜치들은 main의 후속이 아니라 main이 분기되기 전 시점에서 갈라졌으므로 위 필터로 자동 배제됩니다.

**검증**: `find curriculum -name "day*.md" | sort | tail -3` 으로 가장 마지막에 작성된 Day가 다른 어떤 origin 브랜치보다 작거나 같지 않은지 확인합니다. 작다면 base 선택이 잘못된 것이므로 중단하고 사용자에게 보고합니다.

## 1단계: 현재 위치 파악 및 다음 Day 결정

`curriculum/overview.yaml`을 읽어 전체 커리큘럼 구조를 파악합니다.

다음으로 업데이트할 Day를 결정하는 **유일한 기준**:
- `find curriculum -name "day*.md" | sort` 결과에서 **가장 마지막 파일의 다음 Day**가 작성 대상입니다.
- 예: 마지막이 `phase1.../week1.../day05.md` 이면 다음은 `phase1.../week2.../day01.md`.
- overview.yaml에서 해당 Day의 `title`과 `content_path`를 확인합니다.

**중복 방지 추가 검증** (반드시 수행):
```bash
# 결정한 작성 대상 Day가 이미 다른 origin 브랜치에 존재하는지 확인
TARGET_PATH="curriculum/phase.../weekN_.../dayNN.md"  # 결정한 경로
git branch -r | grep -oE "origin/claude/[^ ]+" | while read ref; do
  if git ls-tree -r "$ref" --name-only 2>/dev/null | grep -q "^${TARGET_PATH}$"; then
    echo "[WARN] $ref 에 이미 ${TARGET_PATH}가 존재합니다. base 선택을 재확인하세요."
  fi
done
```
경고가 출력되면 0단계의 base 선택이 그 브랜치를 놓쳤다는 뜻이므로 중단·재진단합니다.

## 2단계: 디렉토리 및 파일 준비

해당 Day의 `content_path` 디렉토리가 없으면 생성합니다:

```bash
mkdir -p curriculum/phase1_ai_foundations/week1_ai_basics/quiz
# content_path의 디렉토리 부분에 맞게 조정
```

이전 Day의 .md 파일을 읽어 연결점을 파악합니다:
- Day 2~5: 같은 Week의 이전 Day .md 파일
- Day 1: 이전 Week의 Day 5 .md 파일 (Phase 첫 Day면 생략)

## 3단계: CONTENT_GUIDELINES.md 읽기

`CONTENT_GUIDELINES.md`를 읽고 모든 규칙을 따릅니다. 특히:
- 섹션 구조 (개요/핵심개념/작동원리/신약적용/창업관점/과제/참고자료)
- Phase별 창업 관점 비중 조절
- 약학 전공자 관점 연결
- 4,000~6,000자 범위

## 4단계: 콘텐츠 작성

해당 Day의 .md 파일을 아래 구조로 **완전히 새로 작성**합니다.

```markdown
# Day N: [overview.yaml의 title]

> 이전 학습에서 [X]를 다뤘습니다. 오늘은 이를 바탕으로 [Y]를 학습합니다.

## 개요
(3-5문장. 정의 + 왜 중요한지.)

## 핵심 개념
(학술적 톤. 용어 정의, 원리, 메커니즘. 약학 지식 연결.)

## 작동 원리와 아키텍처
(입력→처리→출력. 바이브코딩 시 필요한 구조적 이해.)

## 신약개발 적용
(구체적 사례 — 회사명, 연도, 수치. 기존 방법 대비 비교.)

## 창업 관점
(Phase별 비중: P1→1-2문장, P2-3→제품+경쟁, P4-5→시장+BM+MVP)

## 오늘의 과제
(2-3개. 30-60분 내 완료 가능. 기초학습/비즈니스분석/리서치 조합.)

## 참고 자료
(논문 1-2개 + 회사/도구 1-2개)
```

### 작성 핵심 규칙
- 코드 블록 3개 이하 (의사코드/아키텍처만)
- 약학 전공 지식 → 실제 경쟁력으로 연결
- 구체적 사례/수치 2개 이상
- 경어체, 용어 한영 병기
- 최신 정보 필요 시 WebSearch로 검색하여 보강 (확인 불가 시 "확인 필요" 표기)
- URL은 존재 여부를 확신할 수 없으면 기재하지 않음 (제목+저자+연도만)

## 5단계: 피드백 응답

Gist 설정이 되어있다면 drugai_feedback.json을 확인합니다.
로컬에서는 `data/feedback.json`이 있으면 확인합니다.

미응답 피드백(`answer`가 null)이 있으면:
1. 각 질문에 대한 답변을 작성합니다 (2-5문장, 학술적 톤).
2. 오늘의 콘텐츠 .md 파일 맨 끝, "참고 자료" 앞에 추가합니다:

```markdown
## 이전 질문에 대한 답변

> **Q (P1W1D3):** [질문 내용]
>
> **A:** [답변. 관련 개념 설명 포함.]
```

3. 피드백 JSON의 해당 항목에 `answer` 필드를 채웁니다.

## 6단계: 퀴즈 (Week의 마지막 Day일 때만)

해당 Day가 Day 5이면 같은 Week의 quiz yaml을 작성합니다.
quiz 디렉토리도 없으면 생성합니다.

10문제, 비율: 개념 4 + 적용/판단 3 + 비즈니스 3

```yaml
title: "Week N: [주제] 퀴즈"
description: "설명"
questions:
  - id: "wNqM"
    question: "질문"
    type: "multiple_choice"
    options: ["A", "B", "C", "D"]
    answer: "0"
    explanation: "해설"
```

## 7단계: 빌드와 배포

```bash
python build.py
git add -A
git commit -m "Day [N] 콘텐츠: [Day 제목]"
git push -u origin "$(git branch --show-current)"
```

### 7-1단계: 사용자에게 PR 머지 요청 (필수)

푸시만 하고 끝내면 main이 갱신되지 않아 **다음 루틴이 또 같은 Day를 작성**할 위험이 있습니다.
0단계의 자동 base 동기화로 진도 누락은 막을 수 있지만, **장기적으로는 main에 머지되어야** 다른 협업자/세션과의 혼선이 사라집니다.

루틴 종료 메시지에 다음을 반드시 포함합니다:

```
✅ Day [N] 푸시 완료 — 브랜치: claude/jolly-mccarthy-XXX
⚠️  main에 머지가 필요합니다. PR을 만들어 머지하시거나 "PR 만들어줘"라고 말씀해주세요.
   머지하지 않으면 미머지 브랜치가 누적되며, 진도 base 자동 선택에 부담이 커집니다.
```

사용자가 "PR 만들어줘" 또는 동등한 명시적 요청을 했을 때만 GitHub MCP 도구로 PR을 생성합니다.

## 8단계: 품질 자체 검증

빌드 후 확인:
- [ ] **진도가 1 진전했는지**: 작성한 Day 번호가 0단계 base 시점보다 정확히 1 큼
- [ ] **중복 작성이 아닌지**: `git ls-tree -r origin/main --name-only`에 같은 경로가 없거나, 있다면 의도된 덮어쓰기인지 확인
- [ ] .md 파일에 7개 섹션(개요~참고자료) 모두 존재
- [ ] 4,000~6,000자 범위
- [ ] 구체적 사례/수치 2개 이상
- [ ] 코드 블록 3개 이하
- [ ] 이전 Day와 연결문 존재 (첫 Day 제외)
- [ ] 과제가 구체적이고 실행 가능
- [ ] build.py 성공, JSON 파일 생성 확인
- [ ] 퀴즈 정답 인덱스가 옵션 범위 내 (Day 5인 경우)

만약 첫 번째 두 항목 중 하나라도 실패하면 **커밋·푸시를 하지 않고** 사용자에게 보고합니다.
