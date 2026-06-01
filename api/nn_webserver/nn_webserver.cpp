#include "nn_webserver.h"
#define DATA_DIR "/data/records"
const int PORT = 8080;
const int KEEPALIVE_TIMEOUT_MS = 3000;

std::string nnhttpServer::readLocalWebFile(std::string relativePath) { // 读取本地html文件
    // 如 relativePath = "/data/index.html", "/data/spa.js"
    std::string fullPath = this->basic_path + relativePath;
    std::cout << "The fullpath is: " << fullPath << "\n";
    std::ifstream file(fullPath, std::ios::binary);
    if (!file.is_open()) {
        // 如果文件不存在，返回一个友好的错误提示
        return "<h1>404 - Web Asset Missing</h1><p>Path: " + fullPath + "</p>";
    }
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

// ========== Keep-Alive Request Handler ==========
void nnhttpServer::handleClient(SOCKET s) {
    // Set receive timeout for keep-alive idle detection
    int tv = KEEPALIVE_TIMEOUT_MS;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
    char buf[32768];
    bool firstRequest = true;
    while (true) {
        int n = recv(s, buf, sizeof(buf) - 1, 0);
        if (n <= 0) break;  // Timeout or client disconnected
        buf[n] = 0;
        HttpRequest req = parseRequest(std::string(buf)); // 解析请求
        if (req.method.empty()) break;
        std::string uid, setCookie;
        std::string username = resolveUser(req.cookie, uid, setCookie);
        std::string resp;
        bool keepAlive = !firstRequest;  // Keep-alive after first request
        firstRequest = false;

        // ---- 🟢 Route 1: GET / -> serve the SPA HTML page ----
        if (req.method == "GET" && req.path == "/") {
            std::string htmlContent = readLocalWebFile("/data/index.html");
            std::cout << "Route 1: GET / -> serve the SPA HTML page\n";
            resp = httpResponse(200, "OK", htmlContent, "text/html; charset=utf-8", setCookie, keepAlive);
        }
        // ---- 🟢 Route 2: GET /static/spa.js -> 新增的外部静态脚本通道 ----
        else if (req.method == "GET" && req.path == "/static/spa.js") {
            std::string jsContent = readLocalWebFile("/data/spa.js");
            std::cout << "Route 2: GET /static/spa.js\n";
            // 关键点：对于 JS 文件，Content-Type 必须指定为 application/javascript 
            resp = httpResponse(200, "OK", jsContent, "application/javascript; charset=utf-8", setCookie, keepAlive);
        }
        // ---- Route: GET /api/page/N -> JSON ----
        else if (req.method == "GET" && req.path.find("/api/page/") == 0) {
            int n = extractTrailingNum(req.path);
            if (n < 1 || n > total_pages) n = 1;
            std::string json = apiPageJson(n);
            // Inject username into JSON
            json.insert(json.size() - 1, ",\"username\":\"" + jsonEscape(username) + "\"");
            resp = httpResponse(200, "OK", json, "application/json; charset=utf-8", setCookie, keepAlive);
        }
        // ---- Route: GET /api/user -> JSON ----
        else if (req.method == "GET" && req.path == "/api/user") {
            resp = httpResponse(200, "OK", apiUserJson(uid), "application/json; charset=utf-8", setCookie, keepAlive);
        }
        // ---- Route: POST /api/comment/N -> save comment ----
        else if (req.method == "POST" && req.path.find("/api/comment/") == 0) {
            int n = extractTrailingNum(req.path);
            if (n < 1 || n > total_pages) n = 1;
            std::string body = urlDecode(req.body);
            std::string content;
            size_t p = body.find("content=");
            if (p != std::string::npos) {
                content = body.substr(p + 8);
                while (!content.empty() && (content.back() == '\n' || content.back() == '\r' || content.back() == ' '))
                    content.pop_back();
            }
            if (!content.empty()) {
                ensureDataDir();
                std::string ts = timestamp();
                appendLine(commentFile(n), uid + "|" + username + "|" + ts + "|" + content);
                addUserHistory(uid, n, ts, content);
                invalidateCache(n);
                std::cout << "[Page" << n << " " << username << "] " << ts << " - " << content << std::endl;
            }
            resp = httpResponse(200, "OK", "{\"success\":true}", "application/json; charset=utf-8", setCookie, keepAlive);
        }
        // ---- Route: POST /api/username -> change name ----
        else if (req.method == "POST" && req.path == "/api/username") {
            std::string body = urlDecode(req.body);
            std::string newName;
            size_t p = body.find("username=");
            if (p != std::string::npos) {
                newName = body.substr(p + 9);
                while (!newName.empty() && (newName.back() == '\n' || newName.back() == '\r' || newName.back() == ' '))
                    newName.pop_back();
            }
            if (!newName.empty() && newName.size() <= 20) {
                setUserName(uid, newName);
                std::cout << "[User] " << uid << " renamed to: " << newName << std::endl;
            }
            resp = httpResponse(200, "OK", "{\"success\":true}", "application/json; charset=utf-8", setCookie, keepAlive);
        }
        // ---- Route: POST /api/nn/train -> neural network training ----
        else if (req.method == "POST" && req.path == "/api/nn/train") {
            std::string body = urlDecode(req.body);
            nn::NNParams params = get_model_params(body); // 模型参数
            // 检查模型参数合法性
            params.check();
            // 执行相应的 ResNet/LinearNN 训练函数
            nn::model_data res = nn::run_model(params);
            res.check_model(); // 检查模型输出的正确性

            // ---- Return success JSON ----
            std::string json = "{\"success\":true,\"message\":\"Training completed\",\"layers\":[";
            for (size_t i = 0; i < params.layer_sizes.size(); ++i) {
                if (i > 0) json += ",";
                std::string typeStr = (i < params.layer_types.size())
                    ? dtensor::subTypeToStr(params.layer_types[i]) : "origin";
                json += "{\"neurons\":" + std::to_string(params.layer_sizes[i])
                     + ",\"type\":\"" + jsonEscape(typeStr) + "\"}";
            }
            json += "],\"lr\":" + std::to_string(params.lr)
                 + ",\"epochs\":" + std::to_string(params.epochs)
                 + ",\"batchSize\":" + std::to_string(params.batch_size)
                 + ",\"multithread\":" + std::string(params.use_multithread ? "true" : "false")
                 + ",\"threadNum\":" + std::to_string(params.thread_num)
                 + ",\"model\":\"" + std::string(params.model_type == nn::nn_type::Linear_Resnet ? "LinearResnet" : "LinearNN") + "\"}";
            resp = httpResponse(200, "OK", json, "application/json; charset=utf-8", setCookie, keepAlive);
        }
        // ---- 404 ----
        else {
            resp = httpResponse(404, "Not Found", "{\"error\":\"not found\"}", "application/json; charset=utf-8", setCookie, false);
            send(s, resp.c_str(), resp.size(), 0);
            break;  // Close on 404
        }

        send(s, resp.c_str(), resp.size(), 0);
    }
    closesocket(s);
}

// Parse form fields and url.body: 
nn::NNParams nnhttpServer::get_model_params(const std::string& body) {
    nn::NNParams params;
    // Parse model_type
    std::string mt = extractField("model_type", body);
    if (mt == "LinearResnet") 
        params.model_type = nn::nn_type::Linear_Resnet;
    // Parse layers (comma-separated ints)
    std::string layersStr = extractField("layers", body);
    {
        std::istringstream iss(layersStr);
        std::string token;
        while (std::getline(iss, token, ','))
            if (!token.empty()) 
                params.layer_sizes.push_back(std::atoi(token.c_str()));
        params.layer_num = params.layer_sizes.size();
        params.input_output_dim = {params.layer_sizes.front(), params.layer_sizes.back()};
    }
    // Parse types (comma-separated strings -> dtensor::sub_type)
    std::string typesStr = extractField("types", body);
    {
        std::istringstream iss(typesStr);
        std::string token;
        while (std::getline(iss, token, ','))
            if (!token.empty()) 
                params.layer_types.push_back(dtensor::strToSubType(token));
    }

    std::string lrStr     = extractField("lr", body);      
    std::string epochsStr = extractField("epochs", body);   
    std::string batchStr  = extractField("batch", body);
    params.lr         = lrStr.empty()     ? 0.001 : std::atof(lrStr.c_str());
    params.epochs     = epochsStr.empty() ? 1000  : std::atoi(epochsStr.c_str());
    params.batch_size = batchStr.empty()  ? 4     : (size_t)std::atoi(batchStr.c_str());
    return params;
}

std::string nnhttpServer::httpResponse(int code, const std::string &text, const std::string &body,
    const std::string &ct, const std::string &extraHeader, bool keepAlive) 
{
    std::ostringstream resp;
    resp << "HTTP/1.1 " << code << " " << text << "\r\n"
        << "Content-Type: " << ct << "\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << (keepAlive ? "Connection: keep-alive\r\n" : "Connection: close\r\n")
        << "Access-Control-Allow-Origin: *\r\n"
        << extraHeader
        << "\r\n" << body;
    return resp.str();
}

// ========== API JSON Response Helpers ==========
// Build JSON for a page: title, icon, content, comments[]
std::string nnhttpServer::apiPageJson(int n) {
    if (n < 1 || n > total_pages) n = 1;
    auto lines = readLines(commentFile(n));
    std::string json = "{\"page\":" + std::to_string(n) + ",\"title\":\"" + jsonEscape(pageTitle(n)) + "\","
        "\"icon\":\"" + jsonEscape(pageIcon(n)) + "\","
        "\"content\":" + "\"" + jsonEscape(contentHTML(n)) + "\","
        "\"labels\":[";
    for (int i = 1; i <= total_pages; ++i) {
        if (i > 1) json += ",";
        json += "{\"n\":" + std::to_string(i) + ",\"label\":\"" + jsonEscape(pageShortLabel(i)) + "\","
            "\"icon\":\"" + jsonEscape(pageIcon(i)) + "\"}";
    }
    json += "],\"comments\":[";
    for (size_t i = 0; i < lines.size(); ++i) {
        if (i > 0) json += ",";
        // Parse: uid|username|timestamp|text
        std::string l = lines[i];
        size_t p1 = l.find('|');
        size_t p2 = l.find('|', p1 + 1);
        size_t p3 = l.find('|', p2 + 1);
        if (p1 != std::string::npos && p2 != std::string::npos && p3 != std::string::npos) {
            std::string user = l.substr(p1 + 1, p2 - p1 - 1);
            std::string time = l.substr(p2 + 1, p3 - p2 - 1);
            std::string text = l.substr(p3 + 1);
            json += "{\"num\":" + std::to_string(i + 1) + ",\"user\":\"" + jsonEscape(user) + "\","
                "\"time\":\"" + jsonEscape(time) + "\",\"text\":\"" + jsonEscape(text) + "\"}";
        }
    }
    json += "],\"count\":" + std::to_string(lines.size()) + "}";
    return json;
}

void nnhttpServer::start_service() {
    WSADATA wd;
    if (WSAStartup(MAKEWORD(2,2), &wd) != 0) { std::cerr << "WSAStartup failed\n"; return; }
    SOCKET ls = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (ls == INVALID_SOCKET) { std::cerr << "socket failed: " << WSAGetLastError() << "\n"; WSACleanup(); return; }
    int opt = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(PORT);
    if (bind(ls, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        std::cerr << "bind failed on " << PORT << ": " << WSAGetLastError() << "\n";
        closesocket(ls); WSACleanup(); return;
    }
    if (listen(ls, SOMAXCONN) == SOCKET_ERROR) {
        std::cerr << "listen failed: " << WSAGetLastError() << "\n";
        closesocket(ls); WSACleanup(); return;
    }
    ensureDataDir();
    std::cout << "==============================================" << std::endl;
    std::cout << "  MiniWebServer SPA + Keep-Alive + Cache" << std::endl;
    std::cout << "  Listening on http://0.0.0.0:" << PORT << std::endl;
    std::cout << "  Local: http://127.0.0.1:" << PORT << std::endl;
    std::cout << "  Keep-Alive timeout: " << KEEPALIVE_TIMEOUT_MS << "ms" << std::endl;
    std::cout << "  Page cache: enabled (" << total_pages << " pages)" << std::endl;
    std::cout << "  Press Ctrl+C to stop." << std::endl;
    std::cout << "==============================================" << std::endl;
    while (true) {
        sockaddr_in ca; int cal = sizeof(ca);
        SOCKET cs = accept(ls, (sockaddr*)&ca, &cal);
        if (cs == INVALID_SOCKET) {
            if (WSAGetLastError() == WSAEINTR) break;
            continue;
        }
        char ip[INET_ADDRSTRLEN] = {};
        inet_ntop(AF_INET, &ca.sin_addr, ip, sizeof(ip));
        std::cout << "[connect] " << ip << ":" << ntohs(ca.sin_port) << std::endl;
        handleClient(cs);
    }
    closesocket(ls);
    WSACleanup();
}

// ========== File I/O ==========
std::vector<std::string> nnhttpServer::readLines(const std::string &path) {
    std::vector<std::string> lines;
    std::ifstream fin(path);
    if (!fin.is_open()) return lines;
    std::string line;
    while (std::getline(fin, line))
        if (!line.empty()) lines.push_back(line);
    return lines;
}

bool nnhttpServer::appendLine(const std::string &path, const std::string &line) {
    std::ofstream fout(path, std::ios::app);
    if (!fout.is_open()) return false;
    fout << line << std::endl;
    return true;
}

bool nnhttpServer::writeFile(const std::string &path, const std::string &content) {
    std::ofstream fout(path);
    if (!fout.is_open()) return false;
    fout << content;
    return true;
}

void nnhttpServer::ensureDataDir() {
    _mkdir((this->basic_path + DATA_DIR).c_str());
    _mkdir((this->basic_path + DATA_DIR + "/comments").c_str());
    _mkdir((this->basic_path + DATA_DIR + "/users").c_str());
}
// ========== User System ==========
std::string nnhttpServer::generateUid() {
    srand(time(nullptr) + rand());
    const char hex[] = "0123456789abcdef";
    char buf[9];
    for (int i = 0; i < 8; ++i) buf[i] = hex[rand() % 16];
    buf[8] = 0;
    return std::string(buf);
}

std::string nnhttpServer::userFilePath(const std::string &uid) {
    return this->basic_path + DATA_DIR + "/users/user_" + uid + ".txt";
}

std::string nnhttpServer::getUserName(const std::string &uid) {
    auto lines = readLines(userFilePath(uid));
    if (!lines.empty()) return lines[0];
    return "\xe8\xae\xbf\xe5\xae\xa2";
}

void nnhttpServer::setUserName(const std::string &uid, const std::string &name) {
    auto lines = readLines(userFilePath(uid));
    std::string out = name + "\n";
    for (size_t i = 1; i < lines.size(); ++i) out += lines[i] + "\n";
    writeFile(userFilePath(uid), out);
}

std::vector<std::string> nnhttpServer::getUserHistory(const std::string &uid) {
    auto lines = readLines(userFilePath(uid));
    if (lines.size() <= 1) return {};
    return std::vector<std::string>(lines.begin() + 1, lines.end());
}

void nnhttpServer::addUserHistory(const std::string &uid, int p, const std::string &ts, const std::string &cm) {
    appendLine(userFilePath(uid), std::to_string(p) + "|" + ts + "|" + cm);
}

// ========== Utilities ==========
std::string nnhttpServer::urlDecode(const std::string &src) {
    std::string out;
    for (size_t i = 0; i < src.size(); ++i) {
        if (src[i] == '%' && i + 2 < src.size()) {
            int hi = 0, lo = 0;
            char c1 = src[i+1], c2 = src[i+2];
            if (c1 >= '0' && c1 <= '9') hi = c1 - '0';
            else if (c1 >= 'A' && c1 <= 'F') hi = c1 - 'A' + 10;
            else if (c1 >= 'a' && c1 <= 'f') hi = c1 - 'a' + 10;
            if (c2 >= '0' && c2 <= '9') lo = c2 - '0';
            else if (c2 >= 'A' && c2 <= 'F') lo = c2 - 'A' + 10;
            else if (c2 >= 'a' && c2 <= 'f') lo = c2 - 'a' + 10;
            out += static_cast<char>((hi << 4) | lo);
            i += 2;
        } else if (src[i] == '+') { out += ' '; }
        else { out += src[i]; }
    }
    return out;
}

std::string nnhttpServer::extractField(const std::string& field, const std::string& body) {
    size_t p = body.find(field + "=");
    if (p == std::string::npos) return "";
    size_t start = p + field.size() + 1;
    size_t end = body.find("&", start);
    if (end == std::string::npos) end = body.size();
    return body.substr(start, end - start);
}

std::string nnhttpServer::htmlEscape(const std::string &s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        switch (s[i]) {
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '&': out += "&amp;"; break;
            case '"': out += "&quot;"; break;
            default: out += s[i];
        }
    }
    return out;
}

std::string nnhttpServer::jsonEscape(const std::string &s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        switch (s[i]) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += s[i];
        }
    }
    return out;
}

std::string nnhttpServer::timestamp() {
    time_t now = time(nullptr);
    struct tm *t = localtime(&now);
    char buf[64];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", t);
    return std::string(buf);
}

// ========== Page Metadata ==========
std::string nnhttpServer::pageTitle(int n) {
    const char* t[] = {"\xe9\x9d\x99\xe6\x80\x9d\xe5\xb0\x8f\xe7\xab\x99","C++ \xe7\xbd\x91\xe7\xbb\x9c\xe5\x8d\x8f\xe8\xae\xae\xe6\xa0\x88\xe5\xbc\x80\xe5\x8f\x91","Java \xe8\xaf\xad\xe6\xb3\x95\xe8\xa7\x84\xe5\x88\x99\xe4\xb8\x8e\xe8\xaf\xad\xe8\xa8\x80\xe7\x89\xb9\xe6\x80\xa7","Android \xe5\xae\xa2\xe6\x88\xb7\xe7\xab\xaf\xe5\xbc\x80\xe5\x8f\x91","Swift / Kotlin \xe5\x9f\xba\xe6\x9c\xac\xe7\x89\xb9\xe6\x80\xa7\xe4\xb8\x8e\xe8\xaf\xad\xe6\xb3\x95","JavaScript \xe7\xbd\x91\xe9\xa1\xb5\xe5\x89\x8d\xe7\xab\xaf\xe5\xbc\x80\xe5\x8f\x91", "\xe5\xa4\x9a\xe5\xb1\x82\xe7\xa5\x9e\xe7\xbb\x8f\xe7\xbd\x91\xe7\xbb\x9c\xe8\xae\xad\xe7\xbb\x83"};
    if (n >= 1 && n <= total_pages) return t[n-1];
    return "\xe6\x9c\xaa\xe7\x9f\xa5\xe9\xa1\xb5\xe9\x9d\xa2";
}

std::string nnhttpServer::pageIcon(int n) {
    const char* ic[] = {"\xf0\x9f\x8c\xbf","\xf0\x9f\x94\xa7","\xe2\x98\x95","\xf0\x9f\x93\xb1","\xf0\x9f\x92\xbb","\xf0\x9f\x8c\x90", "\xf0\x9f\xa7\xa0"};
    if (n >= 1 && n <= total_pages) return ic[n-1];
    return "";
}

std::string nnhttpServer::pageShortLabel(int n) {
    const char* l[] = {"\xe9\x9d\x99\xe6\x80\x9d\xe5\xb0\x8f\xe7\xab\x99","C++ \xe5\x8d\x8f\xe8\xae\xae\xe6\xa0\x88","Java \xe8\xaf\xad\xe6\xb3\x95","Android","Swift/Kotlin","JS \xe5\x89\x8d\xe7\xab\xaf", "\xe7\xa5\x9e\xe7\xbb\x8f\xe7\xbd\x91\xe7\xbb\x9c"};
    if (n >= 1 && n <= total_pages) return l[n-1];
    return "";
}

std::string nnhttpServer::commentFile(int page) {
    return this->basic_path +  DATA_DIR + "/comments/comments_" + std::to_string(page) + ".txt";
}

// Ensure user identified via cookie, return uid and set-cookie header if new
std::string nnhttpServer::resolveUser(const std::string &cookieHeader, std::string &outUid, std::string &outSetCookie) {
    std::string cookieVal = getCookie(cookieHeader, "uid");
    if (cookieVal.empty()) {
        outUid = generateUid();
        outSetCookie = "Set-Cookie: uid=" + outUid + "; Path=/; Max-Age=31536000\r\n";
        ensureDataDir();
        writeFile(userFilePath(outUid), "\xe8\xae\xbf\xe5\xae\xa2\n");
    } else {
        outUid = cookieVal;
        ensureDataDir();
        std::ifstream test(userFilePath(outUid));
        if (!test.good())
            writeFile(userFilePath(outUid), "\xe8\xae\xbf\xe5\xae\xa2\n");
    }
    return getUserName(outUid);
}

// ========== Page Content ==========
std::string nnhttpServer::contentHTML(int n) {
    const char* c[] = {
        R"P1(<div class="article"><div class="meta">发表于 2026-05-23 · 技术随笔</div><h2>关于构建一个 Web 服务器</h2><p>在计算机网络的世界里，Web 服务器是最基础也最有趣的组件之一。它监听端口、解析 HTTP 请求、返回资源——整个过程从底层理解起来并不复杂。</p><p>一个基本的 Web 服务器需要完成以下几件事：首先创建一个 socket 并绑定到 80 或 8080 端口，然后调用 listen 进入监听状态。当有客户端连接时，accept 返回一个新的 socket 用于通信。服务器读取 HTTP 请求报文，解析出方法（GET/POST）、路径和头部字段，最后构造响应报文返回。</p><p>HTTP 协议是基于文本的请求-响应模型。请求行包含方法、URL 和版本号；响应行包含状态码和状态描述。常见的状态码有 200（成功）、404（未找到）、500（服务器内部错误）。响应头部中的 Content-Type 告诉浏览器如何渲染内容，Content-Length 指明正文长度。</p><p>为了提升性能，现代 Web 服务器还会加入 Keep-Alive 连接复用、缓存策略和异步 I/O。我们的迷你服务器使用了 Keep-Alive 来减少 TCP 握手的开销，同时实现了页面级缓存来避免重复渲染。</p><div class="meta" style="margin-top:24px">发表于 2026-05-22 · 生活感悟</div><h2>编程的乐趣</h2><p>很多人问，编程的乐趣在哪里？我想，最大的乐趣莫过于「创造」。你写下几行代码，计算机就会忠实地执行——从无到有，从抽象到具体。无论是搭建一个网站、写一个游戏，还是自动化一项重复工作，编程都赋予你「构建」的能力。</p><p>除此之外，编程也是解决问题的艺术。每一个 bug 都是一个谜题，每一次调试都是一场推理。当程序终于按预期运行时，那种成就感是无与伦比的。</p><div class="learn-more"><p>想学习前端开发？&#x2192; <a href="#" onclick="loadPage(6)"><strong>用 JavaScript 进行网页前端开发</strong></a></p></div></div>)P1",
        R"HTML(<div class="article"><div class="meta">发表于 2026-05-23 · 技术深度</div><h2>网络协议栈与 C++</h2><p>网络协议栈是操作系统中负责网络通信的核心组件，实现了从物理层到应用层的多层协议处理。理解协议栈对每一位后端开发者都至关重要。</p><p>典型的 TCP/IP 协议栈分为四层：<br><b>应用层</b>（HTTP、FTP、DNS）—— 面向用户的具体协议；<br><b>传输层</b>（TCP、UDP）—— 提供端到端的可靠或不可靠传输；<br><b>网络层</b>（IP、ICMP）—— 负责路由寻址和分组转发；<br><b>网络接口层</b>（以太网、Wi-Fi）—— 处理物理介质上的数据帧。</p><p>在 C++ 中开发网络程序，通常使用 RAII（资源获取即初始化）来管理 socket 资源。借助 RAII，socket 的创建和销毁被封装在对象的构造和析构函数中，即使发生异常也能确保资源被正确释放。</p><p>Winsock API 是 Windows 平台上的套接字编程接口。使用流程为：WSAStartup 初始化库 → socket 创建 → bind 绑定端口 → listen 监听 → accept 接受连接 → send/recv 收发数据 → closesocket 关闭 → WSACleanup 清理。Linux 上的 POSIX socket API 类似，但不需要初始化步骤。</p><p>更进阶的方向包括：非阻塞 I/O（select/poll/epoll）、异步 I/O（IOCP）、SSL/TLS 加密传输。这些技术构成了现代高并发服务器的基石。</p></div>)HTML",
        R"HTML(<div class="article"><div class="meta">发表于 2026-05-23 · 语言基础</div><h2>Java 语言的核心特性</h2><p>Java 是面向对象、跨平台的编程语言，由 Sun Microsystems 于 1995 年发布。核心理念是 "Write Once, Run Anywhere"——通过 JVM（Java 虚拟机）实现平台无关性。</p><p><b>基本语法：</b>Java 的语法继承自 C/C++，但做了大量简化。没有指针、没有多继承、没有手动内存管理。所有对象都在堆上分配，由垃圾回收器（GC）自动管理生命周期。</p><p><b>面向对象：</b>Java 支持封装、继承、多态三大特性。每个 Java 文件只能有一个 public 类，类名必须与文件名一致。接口（interface）用于定义契约，类通过 implements 关键字实现接口。</p><p><b>泛型：</b>JDK 5 引入的泛型允许在编译时进行类型检查，避免强制类型转换。例如 List<String> 只能存放字符串，取出的元素无需转型。</p><p><b>Lambda 与函数式编程：</b>JDK 8 引入了 Lambda 表达式和 Stream API，极大简化了集合操作。配合 Optional 类，可以有效避免空指针异常。</p><p><b>并发编程：</b>Java 提供了丰富的并发工具：synchronized 关键字、Lock 接口、线程池（ExecutorService）、以及 java.util.concurrent 包下的各种同步容器和工具类。</p><p><b>生态：</b>Spring Boot 是目前最流行的 Java 框架，简化了配置和部署。Maven/Gradle 是主流的构建工具。</p></div>)HTML",
        R"HTML(<div class="article"><div class="meta">发表于 2026-05-23 · 移动开发</div><h2>Android 开发必备技能</h2><p>Android 开发是一个庞大的领域，涵盖了从 UI 设计到性能优化的方方面面。以下是成为一名合格的 Android 开发者所需的核心技能清单：</p><p><b>1. Kotlin 语言：</b>Google 官方推荐的语言，已取代 Java 成为 Android 开发的首选。需要掌握协程（Coroutines）进行异步编程、数据类（data class）、密封类（sealed class）、扩展函数等特性。</p><p><b>2. Jetpack Compose：</b>声明式 UI 框架，用 Kotlin 代码直接编写界面，取代了传统的 XML 布局。核心概念包括 Composable 函数、状态管理（State/Hoisting）、副作用（LaunchedEffect）和主题定制。</p><p><b>3. 架构组件：</b>ViewModel 管理 UI 数据并在配置变更时保持状态；Room 数据库提供类型安全的 SQLite 访问；Navigation 组件处理页面跳转和传参；WorkManager 管理后台任务。</p><p><b>4. 网络请求：</b>Retrofit + OkHttp 是业界标准组合。Retrofit 通过注解定义 API 接口，OkHttp 提供高效的 HTTP 连接池和拦截器机制。</p><p><b>5. 性能优化：</b>内存泄漏检测（LeakCanary）、布局层级优化、图片加载（Coil/Glide）、冷启动加速和包体积缩减。</p><p><b>6. 发布与维护：</b>Google Play 上架流程、签名配置、版本管理、Crashlytics 崩溃监控和 Firebase Analytics 用户分析。</p></div>)HTML",
        R"HTML(<div class="article"><div class="meta">发表于 2026-05-23 · 语言对比</div><h2>Swift / Kotlin 基本特性</h2><p>Swift 和 Kotlin 分别是 Apple 和 Google 力推的现代编程语言，两者在许多设计理念上不谋而合，都强调安全性、表达力和开发者体验。</p><p><b>Swift 特性：</b>Optional 类型安全处理 nil，编译器强制解包检查；Struct 是值类型，避免不必要的堆分配；Protocol 支持面向协议编程，比类继承更灵活；Closure 是一等公民，支持尾随闭包语法；枚举可以关联值和方法。Swift 的 ARC 自动管理内存，不需要 GC。</p><p><b>Kotlin 特性：</b>空安全（Nullable/NonNull 类型系统在编译期消除 NullPointerException）；数据类（data class）自动生成 equals/hashCode/toString；扩展函数可以为已有类添加新方法；协程（Coroutines）用同步的方式写异步代码；密封类（sealed class）定义受限的类层次结构。</p><p><b>两者的共同点：</b>都运行在各自的虚拟机上（Swift 运行在 LLVM 之上，Kotlin 运行在 JVM），都支持与旧语言（ObjC/Java）的互操作，都有现代化的包管理工具（Swift Package Manager / Gradle），都强调不可变性（val/let 关键字）。</p><p><b>适用场景：</b>Swift 主要用于 Apple 生态（iOS/macOS/tvOS/watchOS），Kotlin 主要用于 Android 开发，但 Kotlin Multiplatform 正在让 Kotlin 走进后端和全栈领域。</p></div>)HTML",
        R"HTML(<div class="article"><div class="meta">发表于 2026-05-23 · 前端入门</div><h2>JavaScript 与网页前端开发</h2><p>JavaScript 是 Web 前端的基石，与 HTML、CSS 并称为前端三剑客。随着 Node.js 的出现，JavaScript 已经可以运行在服务器端，成为全栈语言。</p><p><b>语言基础：</b>JavaScript 是动态类型、基于原型的语言。ES6 引入了 let/const、箭头函数、模板字符串、解构赋值、模块化（import/export）等现代特性。理解闭包（Closure）、作用域链和事件循环（Event Loop）是进阶的关键。</p><p><b>异步编程：</b>从回调函数到 Promise，再到 async/await，JavaScript 的异步模型不断演进。async/await 让异步代码看起来像同步代码，极大提升了可读性。fetch API 是现代浏览器中发起 HTTP 请求的标准方式。</p><p><b>DOM 操作：</b>document.getElementById、querySelector 用于查找元素；addEventListener 绑定事件；createElement 和 appendChild 动态创建和插入节点。现代框架通常封装了这些操作，但理解原生 DOM API 仍然是基本功。</p><p><b>主流框架：</b>React（组件化、虚拟 DOM、Hooks）、Vue（响应式数据、模板语法、渐进式框架）、Angular（TypeScript 为基础、依赖注入、完整生态）。三者各有优劣，选择取决于项目规模和团队偏好。</p><p><b>工程化：</b>Webpack/Vite 模块打包、Babel 语法转译、ESLint 代码检查、Prettier 格式化、Jest 单元测试。构建工具链是现代前端开发不可或缺的一部分。</p></div>)HTML",
        R"NN7(<div class="nn-page">
        <div class="nn-controls">
            <div class="nn-param-row">
            <div class="nn-param-group">
                <label>模型类型</label>
                <select id="nnModelType">
                <option value="LinearNN">LinearNN</option>
                <option value="LinearResnet">LinearResnet</option>
                </select>
            </div>
            <div class="nn-param-group">
                <label>网络层数 <span id="nnLayersVal">3</span></label>
                <input type="range" id="nnLayers" min="2" max="8" value="3" oninput="updateNNPreview()">
            </div>
            </div>
            <div id="nnLayerConfigs"></div>
            <div class="nn-param-row">
            <div class="nn-param-group">
                <label>学习率</label>
                <input type="number" id="nnLR" value="0.001" step="0.0001" min="0.00001" max="1">
            </div>
            <div class="nn-param-group">
                <label>训练轮数</label>
                <input type="number" id="nnEpochs" value="1000" step="100" min="1" max="100000">
            </div>
            <div class="nn-param-group">
                <label>批次大小</label>
                <input type="number" id="nnBatchSize" value="4" step="1" min="1" max="1024">
            </div>
            </div>
            <div class="nn-param-row">
            <div class="nn-param-group">
                <label style="display:flex;align-items:center;gap:6px">
                <input type="checkbox" id="nnUseMT" onchange="toggleThreadNum()" style="width:auto;accent-color:#3b82f6">
                多线程训练
                </label>
            </div>
            <div class="nn-param-group" id="nnThreadGroup" style="display:none">
                <label>线程数</label>
                <input type="number" id="nnThreadNum" value="4" step="1" min="1" max="32">
            </div>
            </div>
            <button class="nn-train-btn">开始训练</button>   
        </div>
        <div class="nn-canvas-wrap">
            <canvas id="nnCanvas" width="760" height="420"></canvas>
        </div>
        <div id="nnResult" class="nn-result" style="display:none"></div>
        </div>)NN7"
    };  /* line 497 的 <button class="nn-train-btn">开始训练</button> onclick="trainNN()" 绑定训练应该去掉，转而在 JavaScript 中绑定 */
    if (n >= 1 && n <= total_pages) return c[n-1];
    return "";
}

void nnhttpServer::invalidateCache(int page) {
    if (page >= 1 && page <= total_pages) 
        pageCaches[page - 1].valid = false;
}

// Build JSON for user info: uid, username, history[]
std::string nnhttpServer::apiUserJson(const std::string &uid) {
    std::string username = getUserName(uid);
    auto history = getUserHistory(uid);
    std::string json = "{\"uid\":\"" + jsonEscape(uid) + "\",\"username\":\"" + jsonEscape(username) + "\",\"history\":[";
    for (size_t i = 0; i < history.size(); ++i) {
        if (i > 0) json += ",";
        std::string l = history[i];
        size_t p1 = l.find('|');
        size_t p2 = l.find('|', p1 + 1);
        if (p1 != std::string::npos && p2 != std::string::npos) {
            std::string pStr = l.substr(0, p1);
            std::string ts = l.substr(p1 + 1, p2 - p1 - 1);
            std::string text = l.substr(p2 + 1);
            int pNum = 1;
            if (!pStr.empty()) pNum = std::atoi(pStr.c_str());
            json += "{\"page\":" + std::to_string(pNum) + ",\"label\":\"" + jsonEscape(pageShortLabel(pNum)) + "\","
                "\"time\":\"" + jsonEscape(ts) + "\",\"text\":\"" + jsonEscape(text) + "\"}";
        }
    }
    json += "]}";
    return json;
}

HttpRequest nnhttpServer::parseRequest(const std::string &raw) {
    HttpRequest req;
    size_t e = raw.find("\r\n");
    if (e == std::string::npos) return req;
    std::istringstream(raw.substr(0, e)) >> req.method >> req.path;
    size_t pos = 0;
    while (true) {
        size_t nl = raw.find("\r\n", pos);
        if (nl == std::string::npos || nl == pos) break;
        std::string header = raw.substr(pos, nl - pos);
        if (header.find("Cookie:") == 0 || header.find("cookie:") == 0) {
            size_t vp = header.find(':');
            if (vp != std::string::npos) {
                req.cookie = header.substr(vp + 1);
                while (!req.cookie.empty() && req.cookie[0] == ' ') req.cookie.erase(0, 1);
            }
        }
        pos = nl + 2;
    }
    size_t hd = raw.find("\r\n\r\n");
    if (hd != std::string::npos && hd + 4 < raw.size())
        req.body = raw.substr(hd + 4);
    return req;
}

std::string nnhttpServer::getCookie(const std::string &h, const std::string &name) {
    std::string s = name + "=";
    size_t p = h.find(s);
    if (p == std::string::npos) return "";
    size_t start = p + s.size();
    size_t end = h.find(';', start);
    if (end == std::string::npos) end = h.size();
    return h.substr(start, end - start);
}

int nnhttpServer::extractTrailingNum(const std::string &path) {
    size_t lastSlash = path.rfind('/');
    if (lastSlash == std::string::npos) return 0;
    std::string numStr = path.substr(lastSlash + 1);
    if (numStr.empty()) return 0;
    for (size_t i = 0; i < numStr.size(); ++i)
        if (!isdigit(numStr[i])) return 0;
    return std::atoi(numStr.c_str());
}

