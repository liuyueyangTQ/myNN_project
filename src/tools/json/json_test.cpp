#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>
#include "json_tool.h"

// 如果 CMake 没有注入，则兜底使用相对路径
#ifndef HTML_FILE_DIR
#define HTML_FILE_DIR "."
#endif
// 宏字符串化工具
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

// ========== 自动化测试管线 ==========
void runIntegrationTest() {
    // 打印出 HTML_FILE_DIR 在编译器眼里的原始样貌
    std::cout << "[DEBUG] HTML_FILE_DIR text is: " 
              << TOSTRING(HTML_FILE_DIR) << std::endl;
    std::string basic_path = HTML_FILE_DIR; // 不包含引号
    Json::HTML_json_handler html_agent(basic_path);
    
    std::cout << "[1/3] start reading files & JsonCpp combination test..." << std::endl;

    // 执行业务函数获取组装结果
    int page_num = html_agent.get_page_nums();
    for(int i = 1; i <= page_num; ++i) {
        std::cout << "[" << i << "/" << page_num << "] Generating JSON files. Decoding and varifying the content..." << std::endl;
        std::string jsonResult = html_agent.buildPageResponse(i, "visitor_28068");

        // 使用 JsonCpp 解析生成的大字符串
        Json::Value verifyRoot;
        Json::CharReaderBuilder readerBuilder;
        std::unique_ptr<Json::CharReader> const reader(readerBuilder.newCharReader());
        std::string errs;
        bool isJsonValid = reader->parse(jsonResult.c_str(), jsonResult.c_str() + jsonResult.length(), &verifyRoot, &errs);

        // 断言防御
        assert(isJsonValid && "vad error: the JSON file is invalid!");
        assert(verifyRoot["page"].asInt() == i);
        std::cout << "\n[=== successfully verified: extrated the first 100 code HTML ===]" << std::endl;

        // 核心安全测试：检查从文件读取的大段 HTML 文本是否成功保留，且包含双引号
        std::string verifiedidx = verifyRoot["n"].asString();
        std::cout << "The " << verifiedidx << "-th page content is: \n";
        std::string verifiedHtml = verifyRoot["content"].asString();
        assert(!verifiedHtml.empty() && "error: cannot read the content from external html file!");
        std::cout << verifiedHtml.substr(0, 100) << "..." << std::endl;
        std::cout << "[==============================================]\n";
    }


    


    std::cout << "[3/3] [SUCCESS] the tests has passed successfully. C++ successfully transformed the outside html content as localized data!" << std::endl;
}

int main() {
    runIntegrationTest();
    return 0;
}