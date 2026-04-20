// === DrugAI Lab - 정적/로컬 겸용 학습 앱 ===
// Flask 서버 없이도 동작 (GitHub Pages 호환)
// 진도는 localStorage에 저장

const STORAGE_KEY = 'drugai_progress';
const GIST_CONFIG_KEY = 'drugai_gist_config';
let currentQuizAnswers = {};
let currentQuizPhase = 1;
let currentQuizWeek = 1;
let _overviewCache = null;
let _syncPending = false;

// === GitHub Gist 동기화 ===
function getGistConfig() {
    const raw = localStorage.getItem(GIST_CONFIG_KEY);
    return raw ? JSON.parse(raw) : null;
}

function saveGistConfig(config) {
    localStorage.setItem(GIST_CONFIG_KEY, JSON.stringify(config));
}

async function gistApi(path, options = {}) {
    const config = getGistConfig();
    if (!config?.token) return null;
    const res = await fetch(`https://api.github.com${path}`, {
        headers: {
            'Authorization': `token ${config.token}`,
            'Accept': 'application/vnd.github.v3+json',
            'Content-Type': 'application/json',
        },
        ...options,
    });
    if (!res.ok) return null;
    return res.json();
}

async function pullFromGist() {
    const config = getGistConfig();
    if (!config?.gist_id) return null;
    const gist = await gistApi(`/gists/${config.gist_id}`);
    if (!gist?.files?.['drugai_progress.json']?.content) return null;
    try {
        return JSON.parse(gist.files['drugai_progress.json'].content);
    } catch { return null; }
}

async function pushToGist(data) {
    const config = getGistConfig();
    if (!config?.token) return;

    const body = {
        description: 'DrugAI Lab 학습 진도',
        files: { 'drugai_progress.json': { content: JSON.stringify(data, null, 2) } },
    };

    if (config.gist_id) {
        await gistApi(`/gists/${config.gist_id}`, { method: 'PATCH', body: JSON.stringify(body) });
    } else {
        body.public = false;
        const gist = await gistApi('/gists', { method: 'POST', body: JSON.stringify(body) });
        if (gist?.id) {
            config.gist_id = gist.id;
            saveGistConfig(config);
        }
    }
}

async function syncFromCloud() {
    const remote = await pullFromGist();
    if (!remote) return false;
    const local = loadProgress();
    if (!local) {
        saveProgress(remote);
        return true;
    }
    const remoteTime = remote.streak?.last_activity || '';
    const localTime = local.streak?.last_activity || '';
    if (remoteTime > localTime) {
        saveProgress(remote);
        return true;
    }
    return false;
}

function schedulePush(data) {
    if (_syncPending) return;
    _syncPending = true;
    setTimeout(async () => {
        _syncPending = false;
        await pushToGist(data);
    }, 2000);
}

// === localStorage 기반 진도 관리 ===
async function loadProgressTemplate() {
    const res = await fetch('static/data/progress_template.json');
    return res.json();
}

function loadProgress() {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
}

function saveProgress(data) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    if (getGistConfig()?.token) schedulePush(data);
}

async function ensureProgress() {
    let data = loadProgress();
    if (!data) {
        data = await loadProgressTemplate();
        data.user.start_date = new Date().toISOString().split('T')[0];
        saveProgress(data);
    }
    return data;
}

function getPosition(data) {
    return [data.user.current_phase, data.user.current_week, data.user.current_day];
}

function ensureDayData(data, p, w, d) {
    const ps = String(p), ws = String(w), ds = String(d);
    if (!data.phases[ps]) data.phases[ps] = { status: 'in_progress', weeks: {} };
    if (!data.phases[ps].weeks[ws]) data.phases[ps].weeks[ws] = { days: {} };
    if (!data.phases[ps].weeks[ws].days[ds]) {
        data.phases[ps].weeks[ws].days[ds] = {
            theory: 'not_started', exercises: 'not_started',
            quiz: 'not_started', completed_at: null
        };
    }
    return data.phases[ps].weeks[ws].days[ds];
}

// === 정적 데이터 로더 ===
async function loadOverview() {
    if (_overviewCache) return _overviewCache;
    const res = await fetch('static/data/overview.json');
    _overviewCache = await res.json();
    return _overviewCache;
}

async function loadDayContent(phase, week, day) {
    const res = await fetch(`static/data/days/${phase}_${week}_${day}.json`);
    if (!res.ok) return { title: '콘텐츠 준비 중', html: '<p>콘텐츠 준비 중...</p>', exercises: [] };
    return res.json();
}

async function loadQuizData(phase, week) {
    try {
        const res = await fetch(`static/data/quizzes/${phase}_${week}.json`);
        if (!res.ok) return null;
        return res.json();
    } catch { return null; }
}

// === 통계 계산 ===
function computeStats(data) {
    let completedDays = 0;
    for (const phase of Object.values(data.phases)) {
        for (const week of Object.values(phase.weeks || {})) {
            for (const day of Object.values(week.days || {})) {
                if (day.completed_at) completedDays++;
            }
        }
    }
    const quizHistory = data.quiz_history || [];
    const avgQuiz = quizHistory.length > 0
        ? Math.round(quizHistory.reduce((s, q) => s + q.percentage, 0) / quizHistory.length * 10) / 10
        : 0;
    return {
        completed_days: completedDays,
        progress_percent: Math.round(completedDays / 100 * 100 * 10) / 10,
        streak_current: data.streak.current,
        streak_longest: data.streak.longest,
        avg_quiz_score: avgQuiz,
        total_quizzes: quizHistory.length,
    };
}

// === 스트릭 업데이트 ===
function updateStreak(data) {
    const todayStr = new Date().toISOString().split('T')[0];
    const streak = data.streak;
    if (streak.last_activity !== todayStr) {
        if (streak.last_activity) {
            const last = new Date(streak.last_activity);
            const today = new Date(todayStr);
            const diff = Math.round((today - last) / 86400000);
            streak.current = diff === 1 ? streak.current + 1 : 1;
        } else {
            streak.current = 1;
        }
        streak.last_activity = todayStr;
        streak.longest = Math.max(streak.longest, streak.current);
    }
}

// === View Navigation ===
function showView(viewName) {
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('view-' + viewName).classList.add('active');
    const btn = document.querySelector(`[data-view="${viewName}"]`);
    if (btn) btn.classList.add('active');

    switch (viewName) {
        case 'dashboard': loadDashboard(); break;
        case 'today': loadToday(); break;
        case 'curriculum': loadCurriculum(); break;
        case 'quiz': loadQuiz(); break;
        case 'progress': loadProgressView(); break;
    }
}

function showToast(message) {
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    document.body.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add('show'));
    setTimeout(() => { toast.classList.remove('show'); setTimeout(() => toast.remove(), 300); }, 2500);
}

// === Dashboard ===
async function loadDashboard() {
    const data = await ensureProgress();
    const [phase, week, day] = getPosition(data);
    const dayData = ensureDayData(data, phase, week, day);
    const stats = computeStats(data);

    document.getElementById('stat-phase').textContent = phase;
    document.getElementById('stat-week').textContent = week;
    document.getElementById('stat-day').textContent = day;
    document.getElementById('stat-streak').textContent = data.streak.current;
    document.getElementById('sidebar-streak').textContent = data.streak.current;

    ['theory', 'exercises', 'quiz'].forEach(t => {
        const el = document.getElementById('dash-' + t);
        if (dayData[t] === 'completed') {
            el.classList.add('completed');
            el.querySelector('.task-status').innerHTML = '&#10003;';
        } else {
            el.classList.remove('completed');
            el.querySelector('.task-status').innerHTML = '&#9711;';
        }
    });

    const pct = Math.round(stats.completed_days / 100 * 100);
    document.getElementById('dashboard-progress').style.width = pct + '%';
    document.getElementById('dashboard-progress-text').textContent = `${stats.completed_days} / 100일 완료 (${pct}%)`;
    document.getElementById('stat-quiz-avg').textContent = stats.avg_quiz_score > 0 ? stats.avg_quiz_score + '%' : '-';
    document.getElementById('stat-quiz-total').textContent = stats.total_quizzes + '회';

    for (let i = 1; i <= 5; i++) {
        const el = document.getElementById('roadmap-' + i);
        el.classList.remove('active', 'completed');
        if (i < phase) el.classList.add('completed');
        else if (i === phase) el.classList.add('active');
    }
}

// === Today ===
async function loadToday() {
    const data = await ensureProgress();
    const [phase, week, day] = getPosition(data);
    const content = await loadDayContent(phase, week, day);
    const dayData = ensureDayData(data, phase, week, day);

    document.getElementById('today-title').textContent = content.title || '오늘의 학습';
    document.getElementById('today-meta').innerHTML = `
        <span class="meta-tag">Phase ${phase}</span>
        <span class="meta-tag">Week ${week}</span>
        <span class="meta-tag">Day ${day}</span>`;

    document.getElementById('theory-content').innerHTML = content.html || '';

    if (content.exercises && content.exercises.length > 0) {
        document.getElementById('exercises-content').innerHTML =
            '<ul>' + content.exercises.map(ex => `<li><strong>${ex.id}</strong>: ${ex.title}</li>`).join('') + '</ul>';
    } else {
        document.getElementById('exercises-content').innerHTML = '<p>이론 학습 내용에 포함된 실습을 수행하세요.</p>';
    }

    ['theory', 'exercises', 'quiz'].forEach(t => {
        const card = document.getElementById('task-' + t);
        const btn = document.getElementById('btn-' + t);
        if (dayData[t] === 'completed') {
            card.classList.add('completed');
            if (btn) { btn.disabled = true; btn.textContent = '완료됨'; }
        } else {
            card.classList.remove('completed');
            if (btn) { btn.disabled = false; btn.textContent = '완료'; }
        }
    });

    const allDone = ['theory', 'exercises', 'quiz'].every(t => dayData[t] === 'completed');
    document.getElementById('btn-advance').disabled = !allDone;
    currentQuizPhase = phase;
    currentQuizWeek = week;
}

async function completeTask(taskType) {
    const data = await ensureProgress();
    const [p, w, d] = getPosition(data);
    const dayData = ensureDayData(data, p, w, d);
    dayData[taskType] = 'completed';
    updateStreak(data);

    if (['theory', 'exercises', 'quiz'].every(t => dayData[t] === 'completed')) {
        dayData.completed_at = new Date().toISOString();
    }
    saveProgress(data);

    const msg = { theory: '이론 학습 완료!', exercises: '실습 완료!', quiz: '퀴즈 완료!' };
    showToast(msg[taskType]);
    loadToday();
    document.getElementById('sidebar-streak').textContent = data.streak.current;
}

async function advanceDay() {
    const data = await ensureProgress();
    let { current_phase: p, current_week: w, current_day: d } = data.user;

    if (d < 5) { d++; }
    else if (w < 4) { w++; d = 1; }
    else if (p < 5) { p++; w = 1; d = 1; }

    data.user.current_phase = p;
    data.user.current_week = w;
    data.user.current_day = d;
    ensureDayData(data, p, w, d);
    saveProgress(data);

    showToast(`Phase ${p} Week ${w} Day ${d}로 이동!`);
    loadToday();
}

// === Curriculum ===
async function loadCurriculum() {
    const overview = await loadOverview();
    const data = await ensureProgress();
    const { current_phase, current_week, current_day } = data.user;

    const container = document.getElementById('curriculum-tree');
    container.innerHTML = '';

    overview.phases.forEach(phase => {
        const block = document.createElement('div');
        block.className = 'phase-block';
        const isCurrent = phase.number === current_phase;

        block.innerHTML = `
            <div class="phase-header ${isCurrent ? 'active-phase' : ''}" onclick="togglePhase(this)">
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
                                return `<div class="day-item ${isCurrentDay ? 'current' : ''}"
                                             onclick="viewDay(${phase.number}, ${week.number}, ${day.number})">
                                        <span>Day ${day.number}: ${day.title}</span>
                                        ${isCurrentDay ? '<span class="meta-tag">현재</span>' : ''}
                                    </div>`;
                            }).join('')}
                        </div>
                    </div>`).join('')}
            </div>`;
        container.appendChild(block);
        if (isCurrent) block.querySelector('.phase-toggle').classList.add('open');
    });
}

function togglePhase(header) {
    header.nextElementSibling.classList.toggle('open');
    header.querySelector('.phase-toggle').classList.toggle('open');
}

async function viewDay(phase, week, day) {
    const content = await loadDayContent(phase, week, day);
    showView('today');
    document.getElementById('today-title').textContent = content.title;
    document.getElementById('today-meta').innerHTML = `
        <span class="meta-tag">Phase ${phase}</span>
        <span class="meta-tag">Week ${week}</span>
        <span class="meta-tag">Day ${day}</span>`;
    document.getElementById('theory-content').innerHTML = content.html;
}

// === Quiz ===
let _quizQuestions = [];

async function loadQuiz() {
    const data = await ensureProgress();
    currentQuizPhase = data.user.current_phase;
    currentQuizWeek = data.user.current_week;

    document.getElementById('quiz-info').innerHTML =
        `<span class="meta-tag">Phase ${currentQuizPhase} - Week ${currentQuizWeek}</span>`;

    const quizData = await loadQuizData(currentQuizPhase, currentQuizWeek);
    const container = document.getElementById('quiz-container');
    document.getElementById('quiz-results').style.display = 'none';
    currentQuizAnswers = {};

    if (!quizData || !quizData.questions || quizData.questions.length === 0) {
        container.innerHTML = '<div class="card"><p>현재 주차에 퀴즈가 없습니다.</p></div>';
        _quizQuestions = [];
        return;
    }

    _quizQuestions = quizData.questions;
    let html = '';
    _quizQuestions.forEach((q, idx) => {
        html += `
            <div class="quiz-question" id="qq-${q.id}">
                <div class="question-number">문제 ${idx + 1}</div>
                <div class="question-text">${q.question}</div>
                <div class="options">
                    ${q.options.map((opt, oi) => `
                        <button class="option-btn" data-qid="${q.id}" data-value="${oi}"
                                onclick="selectOption('${q.id}', '${oi}', this)">
                            ${opt}
                        </button>`).join('')}
                </div>
            </div>`;
    });
    html += `<div class="quiz-submit"><button class="btn btn-primary btn-lg" onclick="submitQuiz()">제출하기</button></div>`;
    container.innerHTML = html;
}

function selectOption(qid, value, btn) {
    btn.parentElement.querySelectorAll('.option-btn').forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');
    currentQuizAnswers[qid] = value;
}

async function submitQuiz() {
    if (Object.keys(currentQuizAnswers).length === 0) {
        showToast('최소 하나의 문제에 답해주세요.');
        return;
    }

    let score = 0;
    const results = [];
    _quizQuestions.forEach(q => {
        const userAnswer = currentQuizAnswers[q.id] || '';
        const correct = String(userAnswer) === String(q.answer);
        if (correct) score++;
        results.push({ id: q.id, correct, correct_answer: q.answer, explanation: q.explanation || '' });
    });

    const total = _quizQuestions.length;
    const percentage = total > 0 ? Math.round(score / total * 1000) / 10 : 0;

    // 결과 표시
    results.forEach(r => {
        const qEl = document.getElementById('qq-' + r.id);
        if (!qEl) return;
        qEl.querySelectorAll('.option-btn').forEach(btn => {
            btn.disabled = true;
            if (btn.dataset.value === String(r.correct_answer)) btn.classList.add('correct');
            if (btn.classList.contains('selected') && !r.correct) btn.classList.add('wrong');
        });
        if (r.explanation) {
            const expEl = document.createElement('div');
            expEl.className = 'explanation';
            expEl.textContent = r.explanation;
            qEl.appendChild(expEl);
        }
    });

    document.getElementById('quiz-results').style.display = 'block';
    document.getElementById('quiz-results').innerHTML = `
        <div class="result-score">${score} / ${total}</div>
        <div class="result-label">${percentage}% 정답</div>
        <button class="btn btn-primary" onclick="showView('today')">학습으로 돌아가기</button>`;
    document.querySelector('.quiz-submit').style.display = 'none';

    // 진도 기록
    const data = await ensureProgress();
    data.quiz_history.push({
        phase: currentQuizPhase, week: currentQuizWeek,
        score, total, percentage,
        date: new Date().toISOString(),
    });
    const [p, w, d] = getPosition(data);
    ensureDayData(data, p, w, d).quiz = 'completed';
    updateStreak(data);
    saveProgress(data);

    showToast(`퀴즈 완료! ${score}/${total} 정답`);
}

// === Progress View ===
async function loadProgressView() {
    const data = await ensureProgress();
    const stats = computeStats(data);

    document.getElementById('prog-completed').textContent = stats.completed_days;
    document.getElementById('prog-total').textContent = 100;
    document.getElementById('prog-percent').textContent = Math.round(stats.completed_days / 100 * 100) + '%';
    document.getElementById('prog-streak').textContent = stats.streak_longest;

    const phaseNames = ['ML/DL 기초', '화학정보학', 'AI 신약개발', '고급 주제', '파이프라인 구축'];
    let html = '';
    for (let p = 1; p <= 5; p++) {
        const phaseData = data.phases[String(p)];
        let completed = 0;
        if (phaseData) {
            Object.values(phaseData.weeks || {}).forEach(week => {
                Object.values(week.days || {}).forEach(day => { if (day.completed_at) completed++; });
            });
        }
        const pct = Math.round(completed / 20 * 100);
        html += `<div class="phase-progress">
            <div class="phase-progress-header"><span>Phase ${p}: ${phaseNames[p-1]}</span><span>${completed}/20 (${pct}%)</span></div>
            <div class="progress-bar-container"><div class="progress-bar" style="width: ${pct}%"></div></div></div>`;
    }
    document.getElementById('phase-progress-bars').innerHTML = html;

    const quizHistory = data.quiz_history || [];
    const historyEl = document.getElementById('quiz-history-list');
    if (quizHistory.length > 0) {
        historyEl.innerHTML = quizHistory.slice(-10).reverse().map(q =>
            `<div class="quiz-history-item">
                <span>P${q.phase}W${q.week} - ${q.date.split('T')[0]}</span>
                <span>${q.score}/${q.total} (${q.percentage}%)</span>
            </div>`
        ).join('');
    } else {
        historyEl.innerHTML = '<p style="color: var(--text-muted); padding: 1rem 0;">아직 퀴즈 기록이 없습니다.</p>';
    }
}

// === 데이터 동기화 ===

function compressData(data) {
    const json = JSON.stringify(data);
    return btoa(unescape(encodeURIComponent(json)));
}

function decompressData(encoded) {
    const json = decodeURIComponent(escape(atob(encoded)));
    const data = JSON.parse(json);
    if (!data.user || !data.phases || !data.streak) throw new Error('invalid');
    return data;
}

function getSyncUrl() {
    const data = loadProgress();
    if (!data) return null;
    const base = location.href.split('#')[0].split('?')[0];
    return base + '?sync=' + compressData(data);
}

function shareProgress() {
    const data = loadProgress();
    if (!data) { showToast('저장된 진도가 없습니다.'); return; }

    const url = getSyncUrl();

    if (navigator.share) {
        navigator.share({
            title: 'DrugAI Lab 진도 동기화',
            text: '다른 기기에서 이 링크를 열면 진도가 자동으로 동기화됩니다.',
            url: url,
        }).catch(() => {});
        return;
    }

    navigator.clipboard.writeText(url).then(
        () => showToast('동기화 링크가 복사되었습니다! 다른 기기에서 열어보세요.'),
        () => {
            const textarea = document.getElementById('sync-data');
            textarea.value = url;
            textarea.select();
            showToast('링크를 직접 복사하세요.');
        }
    );
}

function exportProgress() {
    const data = loadProgress();
    if (!data) { showToast('저장된 진도가 없습니다.'); return; }
    const encoded = compressData(data);
    const textarea = document.getElementById('sync-data');
    textarea.value = encoded;
    textarea.select();
    navigator.clipboard.writeText(encoded).then(
        () => showToast('동기화 코드가 클립보드에 복사되었습니다!'),
        () => showToast('코드가 생성되었습니다. 직접 복사하세요.')
    );
}

async function importProgress() {
    const textarea = document.getElementById('sync-data');
    let raw = textarea.value.trim();
    if (!raw) {
        try { raw = await navigator.clipboard.readText(); textarea.value = raw; }
        catch { showToast('동기화 코드를 붙여넣기 해주세요.'); return; }
    }
    try {
        const data = decompressData(raw);
        saveProgress(data);
        showToast('진도를 가져왔습니다!');
        loadDashboard();
    } catch {
        showToast('잘못된 동기화 코드입니다.');
    }
}

async function resetProgress() {
    if (!confirm('모든 진도가 초기화됩니다. 계속할까요?')) return;
    localStorage.removeItem(STORAGE_KEY);
    _overviewCache = null;
    await ensureProgress();
    showToast('진도가 초기화되었습니다.');
    loadDashboard();
}

function checkSyncFromUrl() {
    const params = new URLSearchParams(location.search);
    const syncData = params.get('sync');
    if (!syncData) return false;

    try {
        const data = decompressData(syncData);
        const existing = loadProgress();

        if (existing && existing.user.start_date) {
            const pos = `Phase ${data.user.current_phase} Week ${data.user.current_week} Day ${data.user.current_day}`;
            if (!confirm(`진도를 가져올까요?\n\n가져올 진도: ${pos}\n\n기존 데이터가 덮어씌워집니다.`)) {
                history.replaceState(null, '', location.pathname);
                return false;
            }
        }

        saveProgress(data);
        history.replaceState(null, '', location.pathname);
        showToast('진도가 자동으로 동기화되었습니다!');
        return true;
    } catch {
        history.replaceState(null, '', location.pathname);
        return false;
    }
}

// === Gist 설정 UI ===
async function setupGist() {
    const token = document.getElementById('gist-token').value.trim();
    if (!token || !token.startsWith('ghp_')) {
        showToast('올바른 GitHub 토큰을 입력하세요 (ghp_로 시작)');
        return;
    }

    const res = await fetch('https://api.github.com/user', {
        headers: { 'Authorization': `token ${token}` },
    });
    if (!res.ok) {
        showToast('토큰이 유효하지 않습니다.');
        return;
    }
    const user = await res.json();

    saveGistConfig({ token, gist_id: null, username: user.login });

    const data = loadProgress();
    if (data) await pushToGist(data);

    showToast(`${user.login} 계정으로 연결되었습니다!`);
    updateGistUI();
}

async function manualSync() {
    showToast('동기화 중...');
    const pulled = await syncFromCloud();
    if (pulled) {
        showToast('클라우드에서 최신 진도를 가져왔습니다!');
        loadDashboard();
    } else {
        const data = loadProgress();
        if (data) {
            await pushToGist(data);
            showToast('현재 진도를 클라우드에 저장했습니다!');
        }
    }
}

function disconnectGist() {
    if (!confirm('클라우드 동기화 연결을 해제할까요? (로컬 진도는 유지됩니다)')) return;
    localStorage.removeItem(GIST_CONFIG_KEY);
    showToast('연결이 해제되었습니다.');
    updateGistUI();
}

function updateGistUI() {
    const config = getGistConfig();
    const setup = document.getElementById('gist-setup');
    const connected = document.getElementById('gist-connected');
    const status = document.getElementById('gist-status');

    if (!setup || !connected || !status) return;

    if (config?.token) {
        setup.style.display = 'none';
        connected.style.display = 'block';
        status.innerHTML = `<span style="color:var(--green);">&#9679;</span> ${config.username || 'GitHub'} 연결됨`;
        status.style.cssText = 'font-size:0.85rem; margin-bottom:0.75rem; color:var(--green);';
    } else {
        setup.style.display = 'block';
        connected.style.display = 'none';
        status.innerHTML = '<span style="color:var(--text-muted);">&#9675;</span> 연결 안 됨';
        status.style.cssText = 'font-size:0.85rem; margin-bottom:0.75rem; color:var(--text-muted);';
    }
}

// === Init ===
document.addEventListener('DOMContentLoaded', async () => {
    checkSyncFromUrl();
    if (getGistConfig()?.token) await syncFromCloud();
    loadDashboard();
    updateGistUI();
});
