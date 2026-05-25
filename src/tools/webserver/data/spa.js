// ========== SPA JavaScript ==========
let currentPage = 1;
let pageLabels = []; // Cached nav labels for showMyPage()

function $(id) { return document.getElementById(id); }

// Fetch JSON helper
async function api(url, opts) {
    const r = await fetch(url, opts);
    return r.json();
}

// Render navigation bar
function renderNav(labels, active) {
    let html = '';
    for (const p of labels) {
        const cls = p.n === active ? ' class="active"' : '';
        html += '<a' + cls + ' onclick="loadPage(' + p.n + ')">' + p.icon + ' ' + p.label + '</a>';
    }
    html += '<a' + (active === 0 ? ' class="active"' : '') + ' onclick="showMyPage()">&#x1f464; 我的</a>';
    $('navBar').innerHTML = html;
}

// Load a page via API
async function loadPage(n) {
    currentPage = n;
    $('myPage').style.display = 'none';
    $('mainPage').style.display = 'block';
    const data = await api('/api/page/' + n);

    pageLabels = data.labels; // Cache for nav bar
    renderNav(data.labels, n);
      // Title is rendered in innerHTML below, no separate element needed
    let html = '<h1>' + data.icon + ' ' + data.title + '</h1>';
    html += data.content;
    html += '<hr><div class="comment-section">';
    html += '<h2>&#x1f4ac; 评论留言</h2>';
    html += '<div class="comment-form">';
    html += '<textarea id="commentInput" placeholder="写下你的想法..." required></textarea>';
    html += '<div class="form-row">';
    html += '<span class="user-badge" id="userBadge">&#x1f464; ' + data.username + '</span>';
    html += '<button onclick="submitComment(' + n + ')">提交评论</button>';
    html += '</div></div>';
    html += renderComments(data.comments, data.count);
    html += '</div>';
    $('mainPage').innerHTML = html;
    window.scrollTo(0,0);
}

// Render comment list
function renderComments(comments, count) {
    if (!comments || comments.length === 0) {
        return '<div class="comment-list"><h3 style="margin-bottom:8px;color:#555;font-size:0.95rem">已有评论 (0 条)</h3>'
            + '<p class="no-comments">还没有评论，快来写第一条吧！</p></div>';
    }
    let html = '<div class="comment-list"><h3 style="margin-bottom:8px;color:#555;font-size:0.95rem">已有评论 (' + count + ' 条)</h3>';
    for (const c of comments) {
        html += '<div class="comment-item">';
        html += '<span class="comment-num">#' + c.num + '</span>';
        html += '<div class="comment-body">';
        html += '<span class="comment-user">' + esc(c.user) + '</span>';
        html += '<span class="comment-time">' + esc(c.time) + '</span>';
        html += '<div class="comment-text">' + esc(c.text) + '</div></div></div>';
    }
    html += '</div>';
    return html;
}

// Submit comment via AJAX
async function submitComment(n) {
    const input = $('commentInput');
    const text = input.value.trim();
    if (!text) return;
    input.disabled = true;
    const body = 'content=' + encodeURIComponent(text);
    await api('/api/comment/' + n, {
        method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: body
    });
    input.value = '';
    input.disabled = false;
    loadPage(n); // Reload page to show new comment
}

// Show My Page
async function showMyPage() {
    $('mainPage').style.display = 'none';
    $('myPage').style.display = 'block';

    renderNav(pageLabels, 0); // Use cached labels so nav buttons stay visible
    const data = await api("/api/user"); // Fetch user info
    let html = '<h1>&#x1f464; 我的</h1>';
    html += '<div class="profile-card">';
    html += '<label>当前用户名</label>';
    html += '<div class="username-display" id="unameDisplay">' + esc(data.username) + '</div>';
    html += '<div class="username-input">';
    html += '<input type="text" id="unameInput" placeholder="输入新用户名" maxlength="20">';
    html += '<button class="uname-btn" onclick="updateUsername()">修改用户名</button>';
    html += '</div>';
    html += '<div class="hint" style="font-size:0.8rem;color:#999;margin-top:4px">修改用户名后，之前的评论也会更新</div>';
    html += '</div>';
    html += '<h2>&#x1f4dd; 评论历史 (' + data.history.length + ' 条)</h2>';
    if (data.history.length === 0) {
        html += '<p class="no-comments">还没有评论历史。</p>';
    } else {
        for (let i = 0; i < data.history.length; i++) {
            const h = data.history[i];
            html += '<div class="history-item">';
            html += '<span class="history-num">#' + (i+1) + '</span>';
            html += '<span class="history-page"><a onclick="loadPage(' + h.page + ')">' + esc(h.label) + '</a></span>';
            html += '<span class="history-time">' + esc(h.time) + '</span>';
            html += '<div class="history-text">' + esc(h.text) + '</div></div>';
        }
    }
    $('myPage').innerHTML = html;
}

// Update username via AJAX
async function updateUsername() {
    const input = $('unameInput');
    const name = input.value.trim();
    if (!name || name.length > 20) return;
    await api('/api/username', {
        method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: 'username=' + encodeURIComponent(name)
    });
    showMyPage();
}

// Escape HTML
function esc(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// Initialize: load page 1
loadPage(1);