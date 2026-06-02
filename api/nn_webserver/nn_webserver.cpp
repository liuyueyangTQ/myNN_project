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
    // 返回页面内容的 HTML，在nnhttpServer初始化的时候已经存入localPageContents数组中
    if (n >= 1 && n <= total_pages) return localPageContents[n - 1];
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

