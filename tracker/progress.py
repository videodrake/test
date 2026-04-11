import json
import os
import shutil
from datetime import date, datetime

from tracker.config import PROGRESS_FILE, PROGRESS_TEMPLATE


def _ensure_progress_file():
    if not os.path.exists(PROGRESS_FILE):
        shutil.copy(PROGRESS_TEMPLATE, PROGRESS_FILE)


def load_progress():
    _ensure_progress_file()
    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_progress(data):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_current_position(data):
    user = data["user"]
    return user["current_phase"], user["current_week"], user["current_day"]


def complete_task(task_type):
    """Mark a task as completed for the current day. Returns updated progress."""
    data = load_progress()
    phase, week, day = get_current_position(data)
    p, w, d = str(phase), str(week), str(day)

    day_data = data["phases"][p]["weeks"][w]["days"][d]
    day_data[task_type] = "completed"

    # Update streak
    today_str = date.today().isoformat()
    streak = data["streak"]
    if streak["last_activity"] != today_str:
        last = streak.get("last_activity")
        if last:
            last_date = date.fromisoformat(last)
            diff = (date.today() - last_date).days
            if diff == 1:
                streak["current"] += 1
            elif diff > 1:
                streak["current"] = 1
        else:
            streak["current"] = 1
        streak["last_activity"] = today_str
        streak["longest"] = max(streak["longest"], streak["current"])

    # Check if all tasks for the day are completed
    all_done = all(
        day_data.get(t) == "completed"
        for t in ["theory", "exercises", "quiz"]
    )
    if all_done:
        day_data["completed_at"] = datetime.now().isoformat()

    save_progress(data)
    return data


def advance_day():
    """Move to the next day in the curriculum."""
    data = load_progress()
    phase, week, day = get_current_position(data)

    if day < 5:
        data["user"]["current_day"] = day + 1
    elif week < 4:
        data["user"]["current_week"] = week + 1
        data["user"]["current_day"] = 1
    elif phase < 5:
        data["user"]["current_phase"] = phase + 1
        data["user"]["current_week"] = 1
        data["user"]["current_day"] = 1

    # Initialize new day data if needed
    p = str(data["user"]["current_phase"])
    w = str(data["user"]["current_week"])
    d = str(data["user"]["current_day"])

    phases = data["phases"]
    if p not in phases:
        phases[p] = {"status": "in_progress", "weeks": {}}
    if w not in phases[p]["weeks"]:
        phases[p]["weeks"][w] = {"days": {}}
    if d not in phases[p]["weeks"][w]["days"]:
        phases[p]["weeks"][w]["days"][d] = {
            "theory": "not_started",
            "exercises": "not_started",
            "quiz": "not_started",
            "completed_at": None,
        }

    save_progress(data)
    return data


def record_quiz_score(score, total, phase, week):
    """Record a quiz attempt."""
    data = load_progress()
    data["quiz_history"].append({
        "phase": phase,
        "week": week,
        "score": score,
        "total": total,
        "percentage": round(score / total * 100, 1) if total > 0 else 0,
        "date": datetime.now().isoformat(),
    })
    save_progress(data)
    return data


def get_overall_stats(data):
    """Calculate overall learning statistics."""
    total_days = 0
    completed_days = 0

    for p_key, phase in data["phases"].items():
        for w_key, week in phase.get("weeks", {}).items():
            for d_key, day in week.get("days", {}).items():
                total_days += 1
                if day.get("completed_at"):
                    completed_days += 1

    quiz_scores = data.get("quiz_history", [])
    avg_quiz = 0
    if quiz_scores:
        avg_quiz = round(
            sum(q["percentage"] for q in quiz_scores) / len(quiz_scores), 1
        )

    return {
        "total_days": total_days,
        "completed_days": completed_days,
        "progress_percent": round(completed_days / 100 * 100, 1),  # 100 total days
        "streak_current": data["streak"]["current"],
        "streak_longest": data["streak"]["longest"],
        "avg_quiz_score": avg_quiz,
        "total_quizzes": len(quiz_scores),
    }
