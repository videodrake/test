"""
커리큘럼 콘텐츠를 정적 JSON으로 빌드합니다.
Flask 없이 GitHub Pages에서도 동작하도록 모든 콘텐츠를 번들링합니다.

사용법: python build.py
출력: static/data/ 디렉토리에 JSON 파일 생성
"""

import json
import os

import markdown
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CURRICULUM_DIR = os.path.join(BASE_DIR, "curriculum")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "data")


def load_overview():
    with open(os.path.join(CURRICULUM_DIR, "overview.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def render_md(path):
    full = os.path.join(CURRICULUM_DIR, path)
    if not os.path.exists(full):
        return "<p>콘텐츠 준비 중...</p>"
    with open(full, "r", encoding="utf-8") as f:
        raw = f.read()
    return markdown.markdown(raw, extensions=["fenced_code", "tables", "codehilite"])


def build():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    overview = load_overview()

    # 1. overview.json
    with open(os.path.join(OUTPUT_DIR, "overview.json"), "w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)
    print(f"[OK] overview.json")

    # 2. 각 Day 콘텐츠 → days/phase_week_day.json
    days_dir = os.path.join(OUTPUT_DIR, "days")
    os.makedirs(days_dir, exist_ok=True)

    day_count = 0
    for phase in overview["phases"]:
        pn = phase["number"]
        for week in phase.get("weeks", []):
            wn = week["number"]
            for day in week.get("days", []):
                dn = day["number"]
                html = render_md(day["content_path"])
                payload = {
                    "title": day["title"],
                    "html": html,
                    "exercises": day.get("exercises", []),
                }
                fname = f"{pn}_{wn}_{dn}.json"
                with open(os.path.join(days_dir, fname), "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)
                day_count += 1

    print(f"[OK] {day_count}개 Day 콘텐츠 빌드")

    # 3. 퀴즈 → quizzes/phase_week.json
    quiz_dir = os.path.join(OUTPUT_DIR, "quizzes")
    os.makedirs(quiz_dir, exist_ok=True)

    quiz_count = 0
    for phase in overview["phases"]:
        pn = phase["number"]
        for week in phase.get("weeks", []):
            wn = week["number"]
            quiz_path = week.get("quiz_path")
            if not quiz_path:
                continue
            full = os.path.join(CURRICULUM_DIR, quiz_path)
            if not os.path.exists(full):
                continue
            with open(full, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            fname = f"{pn}_{wn}.json"
            with open(os.path.join(quiz_dir, fname), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            quiz_count += 1

    print(f"[OK] {quiz_count}개 퀴즈 빌드")

    # 4. progress template
    template_path = os.path.join(BASE_DIR, "data", "progress.template.json")
    with open(template_path, "r", encoding="utf-8") as f:
        template = json.load(f)
    with open(os.path.join(OUTPUT_DIR, "progress_template.json"), "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    print(f"[OK] progress_template.json")

    print(f"\n빌드 완료! static/data/ 에 정적 파일 생성됨")


if __name__ == "__main__":
    build()
