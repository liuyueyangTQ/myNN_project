#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>
#include <json/json.h>

const int TOTAL_PAGES = 6; 
// 如果 CMake 没有注入，则兜底使用相对路径
#ifndef HTML_FILE_DIR
#define HTML_FILE_DIR "."
#endif
// 宏字符串化工具
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

// ========== 工具函数：规范化读取本地 HTML 文件 ==========
std::string readHtmlPage(int pageNum) {
    std::string path = std::string(HTML_FILE_DIR) + "/data/page_content/page_" + std::to_string(pageNum) + ".html";
    std::cout << "path is: " << path << "\n";
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "[WARNING] cannot find standardized HTML file: " << path << std::endl;
        return "";
    }
    std::stringstream buf;
    buf << fin.rdbuf();
    return buf.str();
}

// ========== 工具函数：从配置 JSON 文件中读取指定页面的元数据 ==========
bool getPageMetadata(int pageNum, std::string& outTitle, std::string& outIcon, std::string& outLabel) {
    std::ifstream fin(std::string(HTML_FILE_DIR) + "/data/configs/pages_config.json", std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "[ERROR] cannot find config file pages_config.json" << std::endl;
        return false;
    }

    Json::Value root;
    Json::CharReaderBuilder reader;
    std::string errs;
    if (!Json::parseFromStream(reader, fin, &root, &errs)) {
        std::cerr << "[ERROR] config file with wrong JSON synax: " << errs << std::endl;
        return false;
    }

    // 遍历 JSON 数组，寻找对应页面 n 的数据
    const Json::Value pages = root["pages"];
    for (unsigned int i = 0; i < pages.size(); ++i) {
        if (pages[i]["n"].asInt() == pageNum) {
            outTitle = pages[i]["title"].asString();
            outIcon = pages[i]["icon"].asString();
            outLabel = pages[i]["label"].asString();
            return true;
        }
    }
    return false;
}

// ========== 业务函数：读取外部文件并使用 JsonCpp 组装接口 ==========
std::string buildPageResponse(int n, const std::string& username) {
    // 1. 从规范化文件中提取元数据
    std::string title, icon, label;
    if (!getPageMetadata(n, title, icon, label)) {
        title = "unKnown Page";
    }

    // 2. 从外部 .html 文件中读取渲染内容
    std::string htmlPageCont = readHtmlPage(n);

    // 3. 开始用 JsonCpp 组装送往前端 SPA 的数据流
    Json::Value responseObj;
    responseObj["page"] = n;
    responseObj["title"] = title;
    responseObj["icon"] = icon;
    // 大段从文件读取的 HTML 字符串作为页面内容直接塞入
    responseObj["content"] = htmlPageCont; 
    responseObj["username"] = username;

    // 组装菜单标签列表
    responseObj["labels"] = Json::arrayValue;
    for (int i = 1; i <= TOTAL_PAGES; ++i) {
        std::string t, ic, l;
        if (getPageMetadata(i, t, ic, l)) {
            Json::Value labelNode;
            labelNode["n"] = i;
            labelNode["label"] = l;
            labelNode["icon"] = ic;
            responseObj["labels"].append(labelNode);
        }
    }
    
    // 假设评论数据暂时为空数组
    responseObj["comments"] = Json::arrayValue;
    responseObj["count"] = 0;

    // 4. 序列化为无缝紧凑字符串
    Json::StreamWriterBuilder writer;
    writer["commentStyle"] = "None";
    writer["indentation"] = ""; 
    return Json::writeString(writer, responseObj);
}

// ========== 自动化测试管线 ==========
void runIntegrationTest() {
    // 打印出 HTML_FILE_DIR 在编译器眼里的原始样貌
    std::cout << "[DEBUG] HTML_FILE_DIR text is: " 
              << TOSTRING(HTML_FILE_DIR) << std::endl;

    std::cout << "[1/3] start reading files & JsonCpp combination test..." << std::endl;

    // 执行业务函数获取组装结果
    std::string jsonResult = buildPageResponse(1, "visitor_28068");

    std::cout << "[2/3] successfully generated JSON. Decoding and varifying the content..." << std::endl;

    // 使用 JsonCpp 解析生成的大字符串
    Json::Value verifyRoot;
    Json::CharReaderBuilder readerBuilder;
    std::unique_ptr<Json::CharReader> const reader(readerBuilder.newCharReader());
    std::string errs;
    
    bool isJsonValid = reader->parse(jsonResult.c_str(), jsonResult.c_str() + jsonResult.length(), &verifyRoot, &errs);

    // 断言防御
    assert(isJsonValid && "vad error: the JSON file is invalid!");
    assert(verifyRoot["page"].asInt() == 1);
    
    // 核心安全测试：检查从文件读取的大段 HTML 文本是否成功保留，且包含双引号
    std::string verifiedHtml = verifyRoot["content"].asString();
    assert(!verifiedHtml.empty() && "error: cannot read the content from external html file!");
    
    std::cout << "\n[=== successfully verified: extrated the first 100 code HTML ===]" << std::endl;
    std::cout << verifiedHtml.substr(0, 100) << "..." << std::endl;
    std::cout << "[==============================================]\n" << std::endl;

    std::cout << "[3/3] [SUCCESS] the tests has passed successfully. C++ successfully transformed the outside html content as localized data!" << std::endl;
}

int main() {
    runIntegrationTest();
    return 0;
}