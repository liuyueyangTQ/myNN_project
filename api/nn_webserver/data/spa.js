"use strict";
// ========== SPA TypeScript - Neural Network Web Frontend ==========
// ---- Global State ----
let currentPage = 1;
let pageLabels = [];
// ---- Utilities ----
function $(id) {
    return document.getElementById(id);
}
async function api(url, opts) {
    const r = await fetch(url, opts);
    return r.json();
}
function esc(s) {
    return s.replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}
// ---- Navigation ----
function renderNav(labels, active) {
    let html = '';
    for (const p of labels) {
        const cls = p.n === active ? ' class="active"' : '';
        html += '<a' + cls + ' onclick="loadPage(' + p.n + ')">' + p.icon + ' ' + p.label + '</a>';
    }
    html += '<a' + (active === 0 ? ' class="active"' : '') + ' onclick="showMyPage()">&#x1f464; 我的</a>';
    const navBar = $('navBar');
    if (navBar)
        navBar.innerHTML = html;
}
// ---- Page Loading ----
async function loadPage(n) {
    currentPage = n;
    const myPageEl = $('myPage');
    const mainPageEl = $('mainPage');
    if (myPageEl)
        myPageEl.style.display = 'none';
    if (mainPageEl)
        mainPageEl.style.display = 'block';
    const data = await api('/api/page/' + n);
    pageLabels = data.labels;
    renderNav(data.labels, n);
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
    if (mainPageEl)
        mainPageEl.innerHTML = html;
    if (n === 7) {
        setTimeout(initNNPage, 150);
        console.log('[NN] initNNPage scheduled');
    }
    window.scrollTo(0, 0);
}
// ---- Comments ----
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
async function submitComment(n) {
    const input = $('commentInput');
    if (!input)
        return;
    const text = input.value.trim();
    if (!text)
        return;
    input.disabled = true;
    const body = 'content=' + encodeURIComponent(text);
    await api('/api/comment/' + n, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: body
    });
    input.value = '';
    input.disabled = false;
    loadPage(n);
}
// ---- User Page ----
async function showMyPage() {
    const mainPageEl = $('mainPage');
    const myPageEl = $('myPage');
    if (mainPageEl)
        mainPageEl.style.display = 'none';
    if (myPageEl)
        myPageEl.style.display = 'block';
    renderNav(pageLabels, 0);
    const data = await api('/api/user');
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
    }
    else {
        for (let i = 0; i < data.history.length; i++) {
            const h = data.history[i];
            html += '<div class="history-item">';
            html += '<span class="history-num">#' + (i + 1) + '</span>';
            html += '<span class="history-page"><a onclick="loadPage(' + h.page + ')">' + esc(h.label) + '</a></span>';
            html += '<span class="history-time">' + esc(h.time) + '</span>';
            html += '<div class="history-text">' + esc(h.text) + '</div></div>';
        }
    }
    if (myPageEl)
        myPageEl.innerHTML = html;
}
async function updateUsername() {
    const input = $('unameInput');
    if (!input)
        return;
    const name = input.value.trim();
    if (!name || name.length > 20)
        return;
    await api('/api/username', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: 'username=' + encodeURIComponent(name)
    });
    showMyPage();
}
// ========== Neural Network Page (Page 7) ==========
const ACTIVATION_TYPES = ["relu", "sigmoid", "softmax", "layer_norm", "none", "origin"];
const ACTIVATION_COLORS = {
    relu: "#f97316",
    sigmoid: "#8b5cf6",
    softmax: "#ec4899",
    layer_norm: "#06b6d4",
    none: "#64748b",
    origin: "#3b82f6",
    output: "#10b981"
};
// ---- Thread Toggle ----
function toggleThreadNum() {
    const mt = document.getElementById("nnUseMT");
    const tg = document.getElementById("nnThreadGroup");
    if (mt && tg)
        tg.style.display = mt.checked ? "block" : "none";
}
// ---- Initialization ----
function initNNPage() {
    try {
        console.log("[NN] initNNPage started");
        updateNNPreview();
        const trainBtn = document.querySelector(".nn-train-btn");
        if (trainBtn) {
            trainBtn.addEventListener("click", trainNN);
            console.log("[NN] train button bound, element:", trainBtn);
        }
        else {
            console.error("[NN] train button NOT FOUND in DOM");
        }
        const mtCheckbox = document.getElementById("nnUseMT");
        if (mtCheckbox) {
            mtCheckbox.addEventListener("change", toggleThreadNum);
        }
        console.log("[NN] page initialized");
    }
    catch (e) {
        console.error("[NN] initNNPage error:", e);
    }
}
// ---- Layer Config UI ----
function updateNNPreview() {
    const layersSlider = document.getElementById("nnLayers");
    const layersVal = document.getElementById("nnLayersVal");
    const configsDiv = document.getElementById("nnLayerConfigs");
    if (!layersSlider || !configsDiv)
        return;
    const numLayers = parseInt(layersSlider.value);
    if (layersVal)
        layersVal.textContent = String(numLayers);
    let html = "";
    for (let i = 0; i < numLayers; i++) {
        const isInput = (i === 0);
        const isOutput = (i === numLayers - 1);
        let layerLabel;
        let defaultNeurons;
        let typeLocked;
        let defaultType;
        if (isInput) {
            layerLabel = "输入层";
            defaultNeurons = 4;
            typeLocked = true;
            defaultType = "origin";
        }
        else if (isOutput) {
            layerLabel = "输出层";
            defaultNeurons = 2;
            typeLocked = true;
            defaultType = "softmax";
        }
        else {
            layerLabel = "隐藏层 " + i;
            defaultNeurons = 6;
            typeLocked = false;
            defaultType = "relu";
        }
        html += '<div class="nn-layer-config">';
        html += '<span class="nn-layer-label">' + layerLabel + '</span>';
        html += '<input type="number" class="nn-neuron-count" value="' + defaultNeurons +
            '" min="1" max="256" onchange="drawNN()" style="width:70px">';
        html += '<span style="font-size:0.8rem;color:#94a3b8">神经元</span>';
        if (typeLocked) {
            html += '<span class="nn-layer-num" style="color:#64748b">' + defaultType + '</span>';
        }
        else {
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
// ---- Read Config from DOM ----
function getNNConfig() {
    const layersSlider = document.getElementById("nnLayers");
    const numLayers = parseInt(layersSlider.value);
    const layers = [];
    const types = [];
    const neuronInputs = document.querySelectorAll(".nn-neuron-count");
    const typeSelects = document.querySelectorAll(".nn-layer-type");
    let typeIdx = 0;
    for (let i = 0; i < numLayers; i++) {
        layers.push(parseInt(neuronInputs[i].value) || 1);
        if (i === 0) {
            types.push("origin");
        }
        else if (i === numLayers - 1) {
            types.push("softmax");
        }
        else {
            types.push(typeSelects[typeIdx] ? typeSelects[typeIdx].value : "relu");
            typeIdx++;
        }
    }
    return { layers, types };
}
// ---- Canvas Drawing ----
function drawNN() {
    const canvas = document.getElementById("nnCanvas");
    if (!canvas)
        return;
    const ctx = canvas.getContext("2d");
    if (!ctx)
        return;
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const config = getNNConfig();
    const numLayers = config.layers.length;
    if (numLayers === 0)
        return;
    const marginX = 80;
    const marginY = 50;
    const usableW = W - 2 * marginX;
    const usableH = H - 2 * marginY;
    const layerSpacing = usableW / (numLayers - 1 || 1);
    const neuronPositions = [];
    for (let l = 0; l < numLayers; l++) {
        const n = config.layers[l];
        const layerX = marginX + l * layerSpacing;
        const layerPositions = [];
        const vertSpacing = usableH / (n + 1);
        for (let j = 0; j < n; j++) {
            layerPositions.push({ x: layerX, y: marginY + vertSpacing * (j + 1) });
        }
        neuronPositions.push(layerPositions);
    }
    // Draw connections
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
    // Draw layer type labels
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
// ---- Training ----
async function trainNN() {
    console.log("[NN] trainNN called");
    const config = getNNConfig();
    const nnLR = document.getElementById("nnLR");
    const nnEpochs = document.getElementById("nnEpochs");
    const nnBatchSize = document.getElementById("nnBatchSize");
    const nnModelType = document.getElementById("nnModelType");
    const nnUseMT = document.getElementById("nnUseMT");
    const nnThreadNum = document.getElementById("nnThreadNum");
    if (!nnLR || !nnEpochs || !nnBatchSize || !nnModelType) {
        console.error("[NN] Missing form elements");
        return;
    }
    const lr = parseFloat(nnLR.value) || 0.001;
    const epochs = parseInt(nnEpochs.value) || 1000;
    const batch = parseInt(nnBatchSize.value) || 4;
    const modelType = nnModelType.value;
    const useMT = nnUseMT ? nnUseMT.checked : false;
    const threadNum = nnThreadNum ? parseInt(nnThreadNum.value) || 4 : 4;
    const btn = document.querySelector(".nn-train-btn");
    if (!btn) {
        console.error("[NN] train button not found");
        return;
    }
    btn.disabled = true;
    btn.textContent = "训练中...";
    const body = "layers=" + encodeURIComponent(config.layers.join(",")) +
        "&types=" + encodeURIComponent(config.types.join(",")) +
        "&lr=" + lr + "&epochs=" + epochs + "&batch=" + batch +
        "&model_type=" + encodeURIComponent(modelType) +
        "&use_mt=" + (useMT ? "1" : "0") +
        "&thread_num=" + threadNum;
    console.log("[NN] sending request...");
    try {
        const data = await api("/api/nn/train", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: body
        });
        showNNResult(data);
    }
    catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        console.error("[NN] trainNN error:", e);
        showNNResult({ success: false, message: "请求失败: " + msg });
    }
    btn.disabled = false;
    btn.textContent = "开始训练";
}
// ---- Result Display ----
function showNNResult(data) {
    var _a, _b, _c, _d;
    console.log("[NN] showNNResult", data);
    const resultDiv = document.getElementById("nnResult");
    if (!resultDiv) {
        console.error("[NN] nnResult div not found");
        return;
    }
    resultDiv.style.display = "block";
    if (!data.success) {
        resultDiv.innerHTML = '<div class="nn-error">训练失败: ' + esc(data.message || "未知错误") + '</div>';
        return;
    }
    let html = '<h3>训练完成</h3>';
    html += '<p style="color:#64748b;font-size:0.88rem;margin-bottom:10px">' + esc(data.message) + '</p>';
    html += '<div class="nn-metrics">';
    html += '<div class="nn-metric">模型: <strong>' + esc(data.model || "-") + '</strong></div>';
    html += '<div class="nn-metric">学习率: <strong>' + ((_a = data.lr) !== null && _a !== void 0 ? _a : "-") + '</strong></div>';
    html += '<div class="nn-metric">轮数: <strong>' + ((_b = data.epochs) !== null && _b !== void 0 ? _b : "-") + '</strong></div>';
    html += '<div class="nn-metric">批次: <strong>' + ((_c = data.batchSize) !== null && _c !== void 0 ? _c : "-") + '</strong></div>';
    if (data.multithread)
        html += '<div class="nn-metric">多线程: <strong>' + ((_d = data.threadNum) !== null && _d !== void 0 ? _d : "-") + ' 线程</strong></div>';
    html += '</div>';
    if (data.layers && data.layers.length > 0) {
        html += '<div style="margin-top:10px;font-size:0.85rem;color:#475569">网络结构: ';
        const layerDescs = data.layers.map((l) => l.neurons + " (" + l.type + ")");
        html += layerDescs.join(" → ");
        html += '</div>';
    }
    resultDiv.innerHTML = html;
}
// ---- Bootstrap ----
loadPage(1);
