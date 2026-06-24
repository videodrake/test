# 매일 학습 콘텐츠 업데이트 루틴

아래 절차에 따라 학습 콘텐츠를 1일치 업데이트합니다.

## 0단계: 환경 확인과 최신 동기화

프로젝트 루트에서 **main 브랜치**에서 직접 작업합니다.

```bash
cd ~/test
git checkout main
git pull origin main
```

## 1단계: 다음 Day 결정

`curriculum/overview.yaml`에서 모든 Day의 `content_path`를 순서대로 확인합니다.
해당 .md 파일이 존재하지 않는 가장 앞선 Day가 다음 작성 대상입니다.

```
확인 방법:
1. overview.yaml에서 Phase 1 → Week 1 → Day 1부터 순회
2. 각 Day의 content_path (예: phase1_ai_foundations/week1_ai_basics/day01.md)
3. curriculum/ 아래에 해당 파일이 존재하는지 확인
4. 존재하지 않는 첫 번째 Day = 오늘 작성할 Day
```

## 2단계: 디렉토리 준비 + 이전 Day 확인

해당 Day의 디렉토리가 없으면 생성합니다.
이전 Day의 .md 파일을 읽어 연결점을 파악합니다.

## 3단계: CONTENT_GUIDELINES.md 읽기

`CONTENT_GUIDELINES.md`를 읽고 모든 규칙을 따릅니다.

## 4단계: 콘텐츠 작성

해당 Day의 .md 파일을 아래 구조로 작성합니다 (4,000~6,000자):

```markdown
# Day N: [overview.yaml의 title]

> 이전 학습에서 [X]를 다뤘습니다. 오늘은 이를 바탕으로 [Y]를 학습합니다.

## 개요
## 핵심 개념
## 작동 원리와 아키텍처
## 신약개발 적용
## 창업 관점
## 오늘의 과제
## 참고 자료
```

작성 규칙:
- 코드 블록 3개 이하, 약학 지식 연결, 구체적 사례 2개+, 경어체, 한영 병기
- 최신 정보는 WebSearch로 보강 (확인 불가 시 "확인 필요" 표기)

## 5단계: 피드백 응답

`data/feedback.json`이 있으면 미응답(`answer`가 null) 피드백에 답변을 작성합니다.
콘텐츠 .md에 "이전 질문에 대한 답변" 섹션을 추가합니다.

## 6단계: 퀴즈 (Day 5일 때만)

해당 Day가 Day 5이면 같은 Week의 quiz yaml을 작성합니다 (10문제).

## 7단계: 빌드 → main에 직접 커밋 → 푸시

**중요: PR을 만들지 않습니다. main에 직접 커밋합니다.**

```bash
python build.py
git add -A
git commit -m "Day N 콘텐츠: [Day 제목]"
git push origin main
```

## 8단계: 검증

- [ ] .md 파일에 7개 섹션 존재
- [ ] 4,000~6,000자
- [ ] 구체적 사례 2개+
- [ ] 코드 블록 3개 이하
- [ ] build.py 성공
- [ ] git push 성공 확인
