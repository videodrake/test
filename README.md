# AI 신약개발 학습 프로그램

약학 전공자가 AI 신약개발 스타트업 창업을 준비하는 100일 학습 프로그램.
개념/아키텍처/비즈니스 중심, 바이브코딩 활용.

## 빠른 시작

**로컬**:
```bash
pip install -r requirements.txt
python app.py
```
→ `http://localhost:5000`

**모바일 (GitHub Pages)**:
레포 Settings → Pages → Branch: `main` / `/ (root)` → `https://<user>.github.io/<repo>/`

## 커리큘럼 (20주 / 100일)

| Phase | 주제 | Week |
|-------|------|------|
| 1 | AI 기초와 신약개발 전체 그림 | 1-4 |
| 2 | 핵심 기술 심화 (QSAR, 분자생성, ADMET, 단백질) | 5-8 |
| 3 | 확장 기술 (가상스크리닝, 바이오의약품, LLM/에이전트, MD) | 9-12 |
| 4 | 실무 도구 (바이브코딩, 인프라, 프로덕트, 규제) | 13-16 |
| 5 | 창업 비즈니스 (시장, BM, 실행, 캡스톤) | 17-20 |

## 기능

- 오늘의 학습 대시보드 (반응형, 모바일 호환)
- 진도 추적 (localStorage)
- 인터랙티브 퀴즈 (Week당 10문제)
- 질문/피드백 기능 → 루틴이 다음 날 답변
- PC-모바일 동기화 (GitHub Gist 자동 + 링크 공유 수동)
- Claude Code 튜터 연동

## 주요 문서

- `CONTENT_GUIDELINES.md` — 콘텐츠 작성 규칙 (약학 관점, 구조, 톤)
- `ROUTINE_PROMPT.md` — Claude 매일 자동 업데이트 루틴
- `CLAUDE.md` — Claude Code 튜터 설정
- `curriculum/overview.yaml` — 전체 커리큘럼 맵

## 개발

콘텐츠 수정 후:
```bash
python build.py      # YAML/MD → static/data/*.json
git commit && git push
```

## 동기화

진도 현황 페이지에서 GitHub Personal Access Token(gist 권한) 설정 →
모든 기기에서 자동 동기화. 2초 디바운스로 Gist에 자동 push.
