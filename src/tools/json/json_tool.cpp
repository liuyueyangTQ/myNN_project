#include "json_tool.h"
namespace Json {
const int TOTAL_PAGES = 6;
void HTML_json_handler::set_basic_path(std::string basic_path) {
    this->basic_path = basic_path;
    set_base_path = true;
}

int HTML_json_handler::get_page_nums() {
    return root["pages"].size();
}

// ========== 工具函数：规范化读取本地 HTML 文件 ==========
std::string HTML_json_handler::readHtmlPage(int pageNum) {
    assert(set_base_path == true);
    std::string path = basic_path + "/data/page_content/page_" + std::to_string(pageNum) + ".html";
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
void HTML_json_handler::getPageMetadata() {
    std::ifstream fin(basic_path + "/data/configs/pages_config.json", std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "[ERROR] cannot find config file pages_config.json" << std::endl;
        return;
    }

    std::string errs;
    if (!Json::parseFromStream(reader, fin, &root, &errs)) {
        std::cerr << "[ERROR] config file with wrong JSON synax: " << errs << std::endl;
        return;
    }

    return;
}

// ========== 业务函数：读取外部文件并使用 JsonCpp 组装接口 ==========
std::string HTML_json_handler::buildPageResponse(int n, const std::string& username) {
    // 1. 从规范化文件中提取元数据
    // 用 JsonCpp 组装送往前端 SPA 的数据流
    const Json::Value &pages = root["pages"];
    assert(n <= pages.size() && n >= 1);
    Json::Value responseObj;
    responseObj["n"] = n;
    responseObj["label"] = pages[n - 1]["label"].asString();
    responseObj["icon"] = pages[n - 1]["icon"].asString();
    responseObj["title"] = pages[n - 1]["title"].asString();
    
    // 假设评论数据暂时为空数组
    responseObj["comments"] = Json::arrayValue;
    responseObj["count"] = 0;  
    
    // 2. 从外部 .html 文件中读取渲染内容
    std::string htmlPageCont = readHtmlPage(n);

    // 大段从文件读取的 HTML 字符串作为页面内容直接塞入
    responseObj["content"] = htmlPageCont; 
    responseObj["username"] = username;

    // 3. 序列化为无缝紧凑字符串
    Json::StreamWriterBuilder writer;
    writer["commentStyle"] = "None";
    writer["indentation"] = ""; 
    return Json::writeString(writer, responseObj);
}

} // namespace Json