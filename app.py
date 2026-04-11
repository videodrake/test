from datetime import date

from flask import Flask, jsonify, render_template, request

from tracker.curriculum import load_day_content, load_overview, load_quiz
from tracker.progress import (
    advance_day,
    complete_task,
    get_current_position,
    get_overall_stats,
    load_progress,
    record_quiz_score,
    save_progress,
)

app = Flask(__name__)


@app.route("/")
def dashboard():
    data = load_progress()
    # Set start date on first visit
    if data["user"]["start_date"] is None:
        data["user"]["start_date"] = date.today().isoformat()
        save_progress(data)
    return render_template("index.html")


# --- API Endpoints ---


@app.route("/api/today")
def api_today():
    """Get today's learning plan."""
    data = load_progress()
    phase, week, day = get_current_position(data)
    content = load_day_content(phase, week, day)

    p, w, d = str(phase), str(week), str(day)
    day_progress = (
        data["phases"]
        .get(p, {})
        .get("weeks", {})
        .get(w, {})
        .get("days", {})
        .get(d, {})
    )

    return jsonify({
        "phase": phase,
        "week": week,
        "day": day,
        "content": content,
        "progress": day_progress,
        "streak": data["streak"],
    })


@app.route("/api/complete", methods=["POST"])
def api_complete():
    """Mark a task as completed."""
    task_type = request.json.get("task_type")
    if task_type not in ("theory", "exercises", "quiz"):
        return jsonify({"error": "Invalid task type"}), 400

    data = complete_task(task_type)
    phase, week, day = get_current_position(data)
    p, w, d = str(phase), str(week), str(day)
    day_data = data["phases"][p]["weeks"][w]["days"][d]

    return jsonify({
        "success": True,
        "day_progress": day_data,
        "streak": data["streak"],
    })


@app.route("/api/advance", methods=["POST"])
def api_advance():
    """Advance to the next day."""
    data = advance_day()
    phase, week, day = get_current_position(data)
    return jsonify({"phase": phase, "week": week, "day": day})


@app.route("/api/progress")
def api_progress():
    """Get overall progress statistics."""
    data = load_progress()
    stats = get_overall_stats(data)
    return jsonify({
        "stats": stats,
        "user": data["user"],
        "phases": data["phases"],
        "streak": data["streak"],
    })


@app.route("/api/curriculum")
def api_curriculum():
    """Get the full curriculum overview."""
    overview = load_overview()
    return jsonify(overview)


@app.route("/api/curriculum/<int:phase>/<int:week>/<int:day>")
def api_day_content(phase, week, day):
    """Get content for a specific day."""
    content = load_day_content(phase, week, day)
    if content is None:
        return jsonify({"error": "Content not found"}), 404
    return jsonify(content)


@app.route("/api/quiz/<int:phase>/<int:week>")
def api_quiz(phase, week):
    """Get quiz questions for a week."""
    questions = load_quiz(phase, week)
    # Strip answers for client-side
    safe_questions = []
    for q in questions:
        safe_q = {k: v for k, v in q.items() if k != "answer"}
        safe_questions.append(safe_q)
    return jsonify({"questions": safe_questions})


@app.route("/api/quiz/submit", methods=["POST"])
def api_quiz_submit():
    """Submit quiz answers and get results."""
    body = request.json
    phase = body.get("phase")
    week = body.get("week")
    answers = body.get("answers", {})

    questions = load_quiz(phase, week)
    results = []
    score = 0

    for q in questions:
        qid = q["id"]
        user_answer = answers.get(qid, "")
        correct = str(user_answer) == str(q["answer"])
        if correct:
            score += 1
        results.append({
            "id": qid,
            "correct": correct,
            "correct_answer": q["answer"],
            "explanation": q.get("explanation", ""),
        })

    record_quiz_score(score, len(questions), phase, week)
    complete_task("quiz")

    return jsonify({
        "score": score,
        "total": len(questions),
        "percentage": round(score / len(questions) * 100, 1) if questions else 0,
        "results": results,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
