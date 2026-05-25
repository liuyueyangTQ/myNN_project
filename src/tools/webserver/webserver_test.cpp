#include "webserver.h"
#define HTML_FILE_DIR "C:/workspace/myNN_project/src/tools/webserver"

// ========== Main SPA HTML Page (served on GET /) ==========
std::string getSPAPage() {
    return R"SPA_END(<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>静思小站 - SPA</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);min-height:100vh;padding:40px 20px;color:#333}
.container{max-width:900px;margin:0 auto;background:rgba(255,255,255,0.95);border-radius:16px;box-shadow:0 8px 32px rgba(0,0,0,0.15);padding:30px 40px}
.nav-bar{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:24px;padding-bottom:16px;border-bottom:2px solid #eef2f7}
.nav-bar a{text-decoration:none;padding:8px 14px;border-radius:8px;font-size:0.85rem;color:#555;background:#f0f2f5;cursor:pointer;transition:all 0.2s}
.nav-bar a:hover{background:#e0e4eb;color:#333}
.nav-bar a.active{background:#4a6cf7;color:#fff;font-weight:600}
h1{font-size:1.8rem;color:#1a1a2e;border-left:5px solid #4a6cf7;padding-left:18px;margin-bottom:24px}
.article h2{font-size:1.3rem;color:#2d3436;margin:20px 0 10px 0}
.article p{font-size:0.95rem;line-height:1.8;color:#555;margin-bottom:14px;text-indent:2em}
.meta{color:#999;font-size:0.82rem;margin-bottom:6px}
hr{border:none;border-top:1px solid #e8e8e8;margin:24px 0}
.comment-section h2{font-size:1.2rem;margin-bottom:14px;color:#2d3436}
.comment-form textarea{width:100%;min-height:70px;padding:10px 14px;border:1px solid #d0d0d0;border-radius:10px;font-size:0.9rem;font-family:inherit;resize:vertical}
.comment-form textarea:focus{outline:none;border-color:#4a6cf7;box-shadow:0 0 0 3px rgba(74,108,247,0.15)}
.comment-form .form-row{display:flex;gap:10px;align-items:center;margin-top:10px;flex-wrap:wrap}
.comment-form .form-row .user-badge{padding:4px 12px;background:#eef2ff;border-radius:16px;font-size:0.85rem;color:#4a6cf7;font-weight:500}
.comment-form button{padding:8px 24px;background:#4a6cf7;color:#fff;border:none;border-radius:8px;font-size:0.95rem;cursor:pointer}
.comment-form button:hover{background:#3b5de7}
.comment-list{margin-top:18px}
.comment-item{display:flex;gap:10px;padding:10px 14px;background:#f8f9fc;border-radius:8px;margin-bottom:8px;border-left:3px solid #4a6cf7}
.comment-item .comment-num{font-weight:600;color:#4a6cf7;font-size:0.82rem;white-space:nowrap;flex-shrink:0;margin-top:2px}
.comment-item .comment-body{flex:1;min-width:0}
.comment-item .comment-user{font-weight:600;color:#2d3436;font-size:0.85rem;margin-right:8px}
.comment-item .comment-time{color:#bbb;font-size:0.75rem}
.comment-item .comment-text{font-size:0.9rem;color:#444;word-break:break-word;margin-top:3px}
.no-comments{color:#aaa;font-style:italic;padding:8px 0;font-size:0.9rem}
.footer{text-align:center;margin-top:24px;color:#bbb;font-size:0.78rem}
.learn-more{margin-top:20px;padding:12px 18px;background:#f0f4ff;border-radius:10px;text-align:center}
.learn-more p{text-indent:0;font-size:0.95rem}
.learn-more a{color:#4a6cf7;text-decoration:none;font-weight:600;cursor:pointer}
.learn-more a:hover{text-decoration:underline}
#myPage{display:none}
.my-header{margin-bottom:20px}
.profile-card{background:#f8f9fc;border-radius:12px;padding:20px;margin-bottom:24px}
.profile-card label{font-size:0.9rem;color:#555;display:block;margin-bottom:6px}
.profile-card .username-display{font-size:1.3rem;font-weight:600;color:#1a1a2e;margin-bottom:6px}
.profile-card .username-input{display:flex;gap:10px;align-items:center;margin-top:6px}
.profile-card input[type=text]{padding:8px 12px;border:1px solid #d0d0d0;border-radius:8px;font-size:0.95rem;width:200px}
.profile-card input[type=text]:focus{outline:none;border-color:#4a6cf7}
.profile-card .uname-btn{padding:8px 20px;background:#4a6cf7;color:#fff;border:none;border-radius:8px;font-size:0.9rem;cursor:pointer}
.profile-card .uname-btn:hover{background:#3b5de7}
.history-item{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px;padding:10px 14px;background:#f8f9fc;border-radius:8px;margin-bottom:8px;border-left:3px solid #4a6cf7}
.history-num{font-weight:600;color:#4a6cf7;font-size:0.82rem}
.history-page a{color:#4a6cf7;text-decoration:none;font-weight:500;font-size:0.9rem;cursor:pointer}
.history-page a:hover{text-decoration:underline}
.history-time{color:#999;font-size:0.78rem}
.history-text{width:100%;margin-top:4px;font-size:0.9rem;color:#444}
@media(max-width:600px){body{padding:16px 10px}.container{padding:16px}.nav-bar a{padding:6px 10px;font-size:0.78rem}h1{font-size:1.4rem}}
</style>
</head>
<body>
<div class="container">
  <div class="nav-bar" id="navBar"></div>
  <div id="mainPage"></div>
  <div id="myPage"></div>
  <div class="footer">Powered by MiniWebServer SPA</div>
</div>
<script>
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
</script>
</body>
</html>)SPA_END";
}




int main() {
    httpServer server(std::string(HTML_FILE_DIR));
    server.start_service();
    return 0;
}
