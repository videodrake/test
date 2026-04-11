// === State ===
let currentQuizAnswers = {};
let currentQuizPhase = 1;
let currentQuizWeek = 1;

// === View Navigation ===
function showView(viewName) {
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));

    document.getElementById('view-' + viewName).classList.add('active');
    document.querySelector(`[data-view="${viewName}"]`).classList.add('active');

    // Load data for the view
    switch (viewName) {
        case 'dashboard': loadDashboard(); break;
        case 'today': loadToday(); break;
        case 'curriculum': loadCurriculum(); break;
        case 'quiz': loadQuiz(); break;
        case 'progress': loadProgress(); break;
    }
}

// === API Helpers ===
async function api(url, options = {}) {
    const res = await fetch(url, {
        headers: { 'Content-Type': 'application/json' },
        ...options,
    });
    return res.json();
}

function showToast(message) {
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    document.body.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add('show'));
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 2500);
}

// === Dashboard ===
async function loadDashboard() {
    const [today, progress] = await Promise.all([
        api('/api/today'),
        api('/api/progress'),
    ]);

    // Stats
    document.getElementById('stat-phase').textContent = today.phase;
    document.getElementById('stat-week').textContent = today.week;
    document.getElementById('stat-day').textContent = today.day;
    document.getElementById('stat-streak').textContent = today.streak.current;
    document.getElementById('sidebar-streak').textContent = today.streak.current;

    // Today's tasks
    const tasks = ['theory', 'exercises', 'quiz'];
    tasks.forEach(t => {
        const el = document.getElementById('dash-' + t);
        const status = today.progress[t];
        if (status === 'completed') {
            el.classList.add('completed');
            el.querySelector('.task-status').innerHTML = '&#10003;';
        } else {
            el.classList.remove('completed');
            el.querySelector('.task-status').innerHTML = '&#9711;';
        }
    });

    // Progress bar
    const stats = progress.stats;
    const pct = Math.round(stats.completed_days / 100 * 100);
    document.getElementById('dashboard-progress').style.width = pct + '%';
    document.getElementById('dashboard-progress-text').textContent =
        `${stats.completed_days} / 100일 완료 (${pct}%)`;

    // Quiz stats
    document.getElementById('stat-quiz-avg').textContent =
        stats.avg_quiz_score > 0 ? stats.avg_quiz_score + '%' : '-';
    document.getElementById('stat-quiz-total').textContent = stats.total_quizzes + '회';

    // Roadmap
    for (let i = 1; i <= 5; i++) {
        const el = document.getElementById('roadmap-' + i);
        el.classList.remove('active', 'completed');
        if (i < today.phase) el.classList.add('completed');
        else if (i === today.phase) el.classList.add('active');
    }
}

// === Today ===
async function loadToday() {
    const data = await api('/api/today');
    const { phase, week, day, content, progress } = data;

    document.getElementById('today-title').textContent = content?.title || '오늘의 학습';
    document.getElementById('today-meta').innerHTML = `
        <span class="meta-tag">Phase ${phase}</span>
        <span class="meta-tag">Week ${week}</span>
        <span class="meta-tag">Day ${day}</span>
    `;

    // Theory content
    if (content?.html) {
        document.getElementById('theory-content').innerHTML = content.html;
    }

    // Exercises content
    if (content?.exercises && content.exercises.length > 0) {
        let html = '<ul>';
        content.exercises.forEach(ex => {
            html += `<li><strong>${ex.id}</strong>: ${ex.title}</li>`;
        });
        html += '</ul>';
        document.getElementById('exercises-content').innerHTML = html;
    } else {
        document.getElementById('exercises-content').innerHTML =
            '<p>이론 학습 내용에 포함된 실습을 수행하세요.</p>';
    }

    // Update task card states
    ['theory', 'exercises', 'quiz'].forEach(t => {
        const card = document.getElementById('task-' + t);
        const btn = document.getElementById('btn-' + t);
        if (progress[t] === 'completed') {
            card.classList.add('completed');
            if (btn) { btn.disabled = true; btn.textContent = '완료됨'; }
        } else {
            card.classList.remove('completed');
            if (btn) { btn.disabled = false; btn.textContent = '완료'; }
        }
    });

    // Advance button
    const allDone = ['theory', 'exercises', 'quiz'].every(t => progress[t] === 'completed');
    document.getElementById('btn-advance').disabled = !allDone;

    // Store for quiz
    currentQuizPhase = phase;
    currentQuizWeek = week;
}

async function completeTask(taskType) {
    const data = await api('/api/complete', {
        method: 'POST',
        body: JSON.stringify({ task_type: taskType }),
    });

    if (data.success) {
        showToast(taskType === 'theory' ? '이론 학습 완료!' :
                  taskType === 'exercises' ? '실습 완료!' : '퀴즈 완료!');
        loadToday();
        // Update sidebar streak
        document.getElementById('sidebar-streak').textContent = data.streak.current;
    }
}

async function advanceDay() {
    const data = await api('/api/advance', { method: 'POST' });
    showToast(`Phase ${data.phase} Week ${data.week} Day ${data.day}로 이동!`);
    loadToday();
}

// === Curriculum ===
async function loadCurriculum() {
    const overview = await api('/api/curriculum');
    const progress = await api('/api/progress');
    const { current_phase, current_week, current_day } = progress.user;

    const container = document.getElementById('curriculum-tree');
    container.innerHTML = '';

    overview.phases.forEach(phase => {
        const block = document.createElement('div');
        block.className = 'phase-block';

        const isCurrent = phase.number === current_phase;

        block.innerHTML = `
            <div class="phase-header ${isCurrent ? 'active-phase' : ''}"
                 onclick="togglePhase(this)">
                <div>
                    <div class="phase-title">Phase ${phase.number}: ${phase.title}</div>
                    <div class="phase-desc">${phase.description}</div>
                </div>
                <span class="phase-toggle">&rsaquo;</span>
            </div>
            <div class="phase-content ${isCurrent ? 'open' : ''}">
                ${(phase.weeks || []).map(week => `
                    <div class="week-block">
                        <div class="week-title">Week ${week.number}: ${week.title}</div>
                        <div class="day-list">
                            ${(week.days || []).map(day => {
                                const isCurrentDay = phase.number === current_phase &&
                                    week.number === current_week && day.number === current_day;
                                return `
                                    <div class="day-item ${isCurrentDay ? 'current' : ''}"
                                         onclick="viewDay(${phase.number}, ${week.number}, ${day.number})">
                                        <span>Day ${day.number}: ${day.title}</span>
                                        ${isCurrentDay ? '<span class="meta-tag">현재</span>' : ''}
                                    </div>
                                `;
                            }).join('')}
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
        container.appendChild(block);

        // Auto-open current phase toggle
        if (isCurrent) {
            block.querySelector('.phase-toggle').classList.add('open');
        }
    });
}

function togglePhase(header) {
    const content = header.nextElementSibling;
    const toggle = header.querySelector('.phase-toggle');
    content.classList.toggle('open');
    toggle.classList.toggle('open');
}

async function viewDay(phase, week, day) {
    const content = await api(`/api/curriculum/${phase}/${week}/${day}`);
    showView('today');
    document.getElementById('today-title').textContent = content.title;
    document.getElementById('today-meta').innerHTML = `
        <span class="meta-tag">Phase ${phase}</span>
        <span class="meta-tag">Week ${week}</span>
        <span class="meta-tag">Day ${day}</span>
    `;
    document.getElementById('theory-content').innerHTML = content.html;
}

// === Quiz ===
async function loadQuiz() {
    const progress = await api('/api/progress');
    currentQuizPhase = progress.user.current_phase;
    currentQuizWeek = progress.user.current_week;

    document.getElementById('quiz-info').innerHTML =
        `<span class="meta-tag">Phase ${currentQuizPhase} - Week ${currentQuizWeek}</span>`;

    const data = await api(`/api/quiz/${currentQuizPhase}/${currentQuizWeek}`);
    const container = document.getElementById('quiz-container');
    const resultsEl = document.getElementById('quiz-results');
    resultsEl.style.display = 'none';
    currentQuizAnswers = {};

    if (!data.questions || data.questions.length === 0) {
        container.innerHTML = '<div class="card"><p>현재 주차에 퀴즈가 없습니다.</p></div>';
        return;
    }

    let html = '';
    data.questions.forEach((q, idx) => {
        html += `
            <div class="quiz-question" id="qq-${q.id}">
                <div class="question-number">문제 ${idx + 1}</div>
                <div class="question-text">${q.question}</div>
                <div class="options">
                    ${q.options.map((opt, oi) => `
                        <button class="option-btn" data-qid="${q.id}" data-value="${oi}"
                                onclick="selectOption('${q.id}', '${oi}', this)">
                            ${opt}
                        </button>
                    `).join('')}
                </div>
            </div>
        `;
    });

    html += `
        <div class="quiz-submit">
            <button class="btn btn-primary btn-lg" onclick="submitQuiz()">제출하기</button>
        </div>
    `;
    container.innerHTML = html;
}

function selectOption(qid, value, btn) {
    // Deselect siblings
    btn.parentElement.querySelectorAll('.option-btn').forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');
    currentQuizAnswers[qid] = value;
}

async function submitQuiz() {
    if (Object.keys(currentQuizAnswers).length === 0) {
        showToast('최소 하나의 문제에 답해주세요.');
        return;
    }

    const data = await api('/api/quiz/submit', {
        method: 'POST',
        body: JSON.stringify({
            phase: currentQuizPhase,
            week: currentQuizWeek,
            answers: currentQuizAnswers,
        }),
    });

    // Show results on each question
    data.results.forEach(r => {
        const qEl = document.getElementById('qq-' + r.id);
        if (!qEl) return;

        qEl.querySelectorAll('.option-btn').forEach(btn => {
            btn.disabled = true;
            if (btn.dataset.value === String(r.correct_answer)) {
                btn.classList.add('correct');
            }
            if (btn.classList.contains('selected') && !r.correct) {
                btn.classList.add('wrong');
            }
        });

        if (r.explanation) {
            const expEl = document.createElement('div');
            expEl.className = 'explanation';
            expEl.textContent = r.explanation;
            qEl.appendChild(expEl);
        }
    });

    // Show overall results
    const resultsEl = document.getElementById('quiz-results');
    resultsEl.style.display = 'block';
    resultsEl.innerHTML = `
        <div class="result-score">${data.score} / ${data.total}</div>
        <div class="result-label">${data.percentage}% 정답</div>
        <button class="btn btn-primary" onclick="showView('today')">학습으로 돌아가기</button>
    `;

    // Disable submit button
    document.querySelector('.quiz-submit').style.display = 'none';

    showToast(`퀴즈 완료! ${data.score}/${data.total} 정답`);
}

// === Progress ===
async function loadProgress() {
    const data = await api('/api/progress');
    const stats = data.stats;

    document.getElementById('prog-completed').textContent = stats.completed_days;
    document.getElementById('prog-total').textContent = 100;
    document.getElementById('prog-percent').textContent =
        Math.round(stats.completed_days / 100 * 100) + '%';
    document.getElementById('prog-streak').textContent = stats.streak_longest;

    // Phase progress bars
    const phaseNames = ['ML/DL 기초', '화학정보학', 'AI 신약개발', '고급 주제', '파이프라인 구축'];
    const container = document.getElementById('phase-progress-bars');
    let html = '';

    for (let p = 1; p <= 5; p++) {
        const phaseData = data.phases[String(p)];
        let completed = 0;
        let total = 20; // 4 weeks * 5 days

        if (phaseData) {
            Object.values(phaseData.weeks || {}).forEach(week => {
                Object.values(week.days || {}).forEach(day => {
                    if (day.completed_at) completed++;
                });
            });
        }

        const pct = Math.round(completed / total * 100);
        html += `
            <div class="phase-progress">
                <div class="phase-progress-header">
                    <span>Phase ${p}: ${phaseNames[p-1]}</span>
                    <span>${completed}/${total} (${pct}%)</span>
                </div>
                <div class="progress-bar-container">
                    <div class="progress-bar" style="width: ${pct}%"></div>
                </div>
            </div>
        `;
    }
    container.innerHTML = html;

    // Quiz history
    const historyEl = document.getElementById('quiz-history-list');
    const history = data.stats.total_quizzes > 0 ?
        (await api('/api/progress')).phases : null;

    // Use quiz_history from progress data
    const progressData = await api('/api/progress');
    // We don't have direct access to quiz_history from the API, show stats instead
    if (stats.total_quizzes > 0) {
        historyEl.innerHTML = `
            <div class="quiz-history-item">
                <span>총 퀴즈 수</span><span>${stats.total_quizzes}회</span>
            </div>
            <div class="quiz-history-item">
                <span>평균 점수</span><span>${stats.avg_quiz_score}%</span>
            </div>
        `;
    } else {
        historyEl.innerHTML = '<p style="color: var(--text-muted); padding: 1rem 0;">아직 퀴즈 기록이 없습니다.</p>';
    }
}

// === Init ===
document.addEventListener('DOMContentLoaded', () => {
    loadDashboard();
});
