/* MiniWebServer - Keep-Alive + Cache + AJAX SPA */
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstring>
#include <ctime>
#include <cstdlib>
#include <direct.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include"nn.h"
#include "enum_type.h"
#include "json_tool.h"
#pragma comment(lib, "ws2_32.lib")

// ========== Cache System ==========
struct PageCache {
    std::string shell;   // Cached: nav + title + article HTML (everything except comments)
    bool valid;
};
// ========== Request Parsing ==========
struct HttpRequest { std::string method, path, body, cookie; };

class nnhttpServer {
private:
    PageCache* pageCaches;
    std::string basic_path;
    Json::HTML_json_handler html_agent;
    int total_pages;
    std::string readLocalWebFile(std::string relativePath);
    void handleClient(SOCKET s);
    std::string httpResponse(int code, const std::string &text, const std::string &body,
        const std::string &ct = "text/html; charset=utf-8",
        const std::string &extraHeader = "",
        bool keepAlive = false);
    // ========== API JSON Response Helpers ==========
    // Build JSON for a page: title, icon, content, comments[]
    std::string apiPageJson(int n);

    // ========== File I/O ==========
    std::vector<std::string> readLines(const std::string &path);
    bool appendLine(const std::string &path, const std::string &line);
    bool writeFile(const std::string &path, const std::string &content);
    void ensureDataDir();

    // ========== User System ==========
    std::string generateUid();
    std::string userFilePath(const std::string &uid);
    std::string getUserName(const std::string &uid);
    void setUserName(const std::string &uid, const std::string &name);
    std::vector<std::string> getUserHistory(const std::string &uid);
    void addUserHistory(const std::string &uid, int p, const std::string &ts, const std::string &cm);
    std::string apiUserJson(const std::string &uid);
    // ========== Utilities ==========
    std::string urlDecode(const std::string &src);
    std::string htmlEscape(const std::string &s);
    std::string jsonEscape(const std::string &s);
    std::string timestamp();
    std::string contentHTML(int n);
    // ========== Page Metadata ==========
    std::string pageTitle(int n);
    std::string pageIcon(int n);
    std::string pageShortLabel(int n);
    std::string commentFile(int page);
    void invalidateCache(int page);
    // Ensure user identified via cookie, return uid and set-cookie header if new
    std::string getCookie(const std::string &h, const std::string &name);
    std::string resolveUser(const std::string &cookieHeader, std::string &outUid, std::string &outSetCookie);
    int extractTrailingNum(const std::string &path);
    HttpRequest parseRequest(const std::string &raw);
    std::string extractField(const std::string& field, const std::string& body);
    // === nn Utils ==
    nn::NNParams get_model_params(const std::string& body); // 模型参数

public:
    nnhttpServer(std::string basic_path) : html_agent(basic_path) {
        this->basic_path = basic_path;
        this->total_pages = html_agent.get_page_nums();
        std::cout << "Page num is: " << this->total_pages << "\n";
        pageCaches = new PageCache[total_pages]();
    }
    void start_service();
};
