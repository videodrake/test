import os

import markdown
import yaml

from tracker.config import CURRICULUM_DIR


def load_overview():
    """Load the master curriculum manifest."""
    path = os.path.join(CURRICULUM_DIR, "overview.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_phase_info(overview, phase_num):
    """Get info for a specific phase."""
    for phase in overview["phases"]:
        if phase["number"] == phase_num:
            return phase
    return None


def get_week_info(phase_info, week_num):
    """Get info for a specific week within a phase."""
    for week in phase_info.get("weeks", []):
        if week["number"] == week_num:
            return week
    return None


def get_day_info(week_info, day_num):
    """Get info for a specific day within a week."""
    for day in week_info.get("days", []):
        if day["number"] == day_num:
            return day
    return None


def load_day_content(phase_num, week_num, day_num):
    """Load the markdown content for a specific day."""
    overview = load_overview()
    phase_info = get_phase_info(overview, phase_num)
    if not phase_info:
        return None

    week_info = get_week_info(phase_info, week_num)
    if not week_info:
        return None

    day_info = get_day_info(week_info, day_num)
    if not day_info:
        return None

    content_path = os.path.join(CURRICULUM_DIR, day_info["content_path"])
    if not os.path.exists(content_path):
        return {"title": day_info["title"], "html": "<p>콘텐츠 준비 중...</p>", "raw": ""}

    with open(content_path, "r", encoding="utf-8") as f:
        raw = f.read()

    html = markdown.markdown(raw, extensions=["fenced_code", "tables", "codehilite"])
    return {
        "title": day_info["title"],
        "html": html,
        "raw": raw,
        "exercises": day_info.get("exercises", []),
    }


def load_quiz(phase_num, week_num):
    """Load quiz questions for a specific week."""
    overview = load_overview()
    phase_info = get_phase_info(overview, phase_num)
    if not phase_info:
        return []

    week_info = get_week_info(phase_info, week_num)
    if not week_info:
        return []

    quiz_path = week_info.get("quiz_path")
    if not quiz_path:
        return []

    full_path = os.path.join(CURRICULUM_DIR, quiz_path)
    if not os.path.exists(full_path):
        return []

    with open(full_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data.get("questions", [])
