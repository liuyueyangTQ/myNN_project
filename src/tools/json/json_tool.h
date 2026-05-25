#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>
#include <json/json.h>
namespace Json {
class HTML_json_handler {
private:
    Json::Value root;
    Json::CharReaderBuilder reader;
    std::string basic_path;
    bool set_base_path;
    std::string related_path;
    bool set_related_path;
public:
    HTML_json_handler(std::string basic_path) {
        this->basic_path = basic_path;
        set_base_path = true;
    }
    void set_basic_path(std::string basic_path);
    std::string readHtmlPage(int pageNum);
    std::string buildPageResponse(int n, const std::string& username);
    void getPageMetadata();
    int get_page_nums();
};

} // namespace Json