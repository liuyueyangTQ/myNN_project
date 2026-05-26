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

// ========== Neural Network Page (Page 7) ==========

const ACTIVATION_TYPES = ["relu", "sigmoid", "tanh", "linear", "softmax"];
const ACTIVATION_COLORS = {
    relu: "#f97316", sigmoid: "#8b5cf6", tanh: "#06b6d4",
    linear: "#64748b", softmax: "#ec4899", input: "#3b82f6", output: "#10b981"
};

function initNNPage() {
    updateNNPreview();
}

function updateNNPreview() {
    const layersSlider = document.getElementById("nnLayers");
    const layersVal = document.getElementById("nnLayersVal");
    const configsDiv = document.getElementById("nnLayerConfigs");
    if (!layersSlider || !configsDiv) return;

    const numLayers = parseInt(layersSlider.value);
    layersVal.textContent = numLayers;

    let html = "";
    for (let i = 0; i < numLayers; i++) {
        const isInput = (i === 0);
        const isOutput = (i === numLayers - 1);
        let layerLabel, defaultNeurons, typeLocked, defaultType;
        if (isInput) {
            layerLabel = "输入层";
            defaultNeurons = 4;
            typeLocked = true;
            defaultType = "input";
        } else if (isOutput) {
            layerLabel = "输出层";
            defaultNeurons = 2;
            typeLocked = true;
            defaultType = "output";
        } else {
            layerLabel = "隐藏层 " + i;
            defaultNeurons = 6;
            typeLocked = false;
            defaultType = "relu";
        }
        html += '<div class="nn-layer-config">';
        html += '<span class="nn-layer-label">' + layerLabel + '</span>';
        html += '<input type="number" class="nn-neuron-count" value="' + defaultNeurons +
            '" min="1" max="256" onchange="drawNN()" style="width:70px">';
        html += '<span style="font-size:0.8rem;color:#94a3b8">类型</span>';
        if (typeLocked) {
            html += '<span class="nn-layer-num" style="color:#64748b">' + defaultType + '</span>';
        } else {
            html += '<select class="nn-layer-type" onchange="drawNN()">';
            for (const t of ACTIVATION_TYPES) {
                const sel = (t === defaultType) ? " selected" : "";
                html += '<option value="' + t + '"' + sel + '>' + t + '</option>';
            }
            html += '</select>';
        }
        html += '</div>';
    }
    configsDiv.innerHTML = html;
    drawNN();
}

function getNNConfig() {
    const numLayers = parseInt(document.getElementById("nnLayers").value);
    const layers = [];
    const types = [];
    const neuronInputs = document.querySelectorAll(".nn-neuron-count");
    const typeSelects = document.querySelectorAll(".nn-layer-type");
    let typeIdx = 0;
    for (let i = 0; i < numLayers; i++) {
        layers.push(parseInt(neuronInputs[i].value) || 1);
        if (i === 0) {
            types.push("input");
        } else if (i === numLayers - 1) {
            types.push("output");
        } else {
            types.push(typeSelects[typeIdx] ? typeSelects[typeIdx].value : "relu");
            typeIdx++;
        }
    }
    return { layers, types };
}

function drawNN() {
    const canvas = document.getElementById("nnCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const config = getNNConfig();
    const numLayers = config.layers.length;
    if (numLayers === 0) return;

    // Layout parameters
    const marginX = 80, marginY = 50;
    const usableW = W - 2 * marginX;
    const usableH = H - 2 * marginY;
    const layerSpacing = usableW / (numLayers - 1 || 1);

    // Precompute neuron positions
    const neuronPositions = [];
    for (let l = 0; l < numLayers; l++) {
        const n = config.layers[l];
        const layerX = marginX + l * layerSpacing;
        const layerPositions = [];
        const vertSpacing = usableH / (n + 1);
        for (let j = 0; j < n; j++) {
            layerPositions.push({
                x: layerX,
                y: marginY + vertSpacing * (j + 1)
            });
        }
        neuronPositions.push(layerPositions);
    }

    // Draw connections (behind neurons)
    for (let l = 0; l < numLayers - 1; l++) {
        const fromNeurons = neuronPositions[l];
        const toNeurons = neuronPositions[l + 1];
        for (let i = 0; i < fromNeurons.length; i++) {
            for (let j = 0; j < toNeurons.length; j++) {
                ctx.beginPath();
                ctx.moveTo(fromNeurons[i].x, fromNeurons[i].y);
                ctx.lineTo(toNeurons[j].x, toNeurons[j].y);
                ctx.strokeStyle = "rgba(148,163,184,0.25)";
                ctx.lineWidth = 0.7;
                ctx.stroke();
            }
        }
    }

    // Draw neurons
    const neuronRadius = 14;
    for (let l = 0; l < numLayers; l++) {
        const type = config.types[l];
        const color = ACTIVATION_COLORS[type] || "#64748b";
        for (const pos of neuronPositions[l]) {
            // Glow
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, neuronRadius + 3, 0, Math.PI * 2);
            const grad = ctx.createRadialGradient(pos.x, pos.y, neuronRadius - 4, pos.x, pos.y, neuronRadius + 4);
            grad.addColorStop(0, color);
            grad.addColorStop(1, "rgba(30,41,59,0)");
            ctx.fillStyle = grad;
            ctx.fill();

            // Neuron body
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, neuronRadius, 0, Math.PI * 2);
            ctx.fillStyle = "#0f172a";
            ctx.fill();
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.stroke();

            // Inner highlight
            ctx.beginPath();
            ctx.arc(pos.x - 3, pos.y - 3, 4, 0, Math.PI * 2);
            ctx.fillStyle = "rgba(255,255,255,0.15)";
            ctx.fill();
        }
    }

    // Draw layer type labels below neurons
    ctx.font = "11px -apple-system, sans-serif";
    ctx.textAlign = "center";
    for (let l = 0; l < numLayers; l++) {
        const type = config.types[l];
        const x = marginX + l * layerSpacing;
        ctx.fillStyle = ACTIVATION_COLORS[type] || "#64748b";
        ctx.fillText(type, x, H - 14);
    }

    // Draw layer labels at top
    ctx.font = "bold 11px -apple-system, sans-serif";
    ctx.fillStyle = "#94a3b8";
    for (let l = 0; l < numLayers; l++) {
        const x = marginX + l * layerSpacing;
        const label = (l === 0) ? "Input" : (l === numLayers - 1) ? "Output" : "Hidden " + l;
        ctx.fillText(label, x, 22);
    }
}

async function trainNN() {
    const config = getNNConfig();
    const lr = parseFloat(document.getElementById("nnLR").value) || 0.01;
    const epochs = parseInt(document.getElementById("nnEpochs").value) || 100;
    const batch = parseInt(document.getElementById("nnBatchSize").value) || 32;

    const btn = document.querySelector(".nn-train-btn");
    btn.disabled = true;
    btn.textContent = "训练中...";

    const body = "layers=" + encodeURIComponent(config.layers.join(",")) +
        "&types=" + encodeURIComponent(config.types.join(",")) +
        "&lr=" + lr + "&epochs=" + epochs + "&batch=" + batch;

    try {
        const data = await api("/api/nn/train", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: body
        });
        showNNResult(data);
    } catch (e) {
        showNNResult({ success: false, message: "请求失败: " + e.message });
    }
    btn.disabled = false;
    btn.textContent = "开始训练";
}

function showNNResult(data) {
    const resultDiv = document.getElementById("nnResult");
    if (!resultDiv) return;
    resultDiv.style.display = "block";

    if (!data.success) {
        resultDiv.innerHTML = '<div class="nn-error">训练失败: ' + esc(data.message || "未知错误") + '</div>';
        return;
    }

    let html = '<h3>训练完成</h3>';
    html += '<p style="color:#64748b;font-size:0.88rem;margin-bottom:10px">' + esc(data.message) + '</p>';
    html += '<div class="nn-metrics">';
    html += '<div class="nn-metric">学习率: <strong>' + (data.lr || "-") + '</strong></div>';
    html += '<div class="nn-metric">轮数: <strong>' + (data.epochs || "-") + '</strong></div>';
    html += '<div class="nn-metric">批次: <strong>' + (data.batchSize || "-") + '</strong></div>';
    html += '</div>';
    if (data.layers && data.layers.length > 0) {
        html += '<div style="margin-top:10px;font-size:0.85rem;color:#475569">网络结构: ';
        const layerDescs = data.layers.map(l => l.neurons + " (" + l.type + ")");
        html += layerDescs.join(" → ");
        html += '</div>';
    }
    resultDiv.innerHTML = html;
}

// Initialize: load page 1
loadPage(1);