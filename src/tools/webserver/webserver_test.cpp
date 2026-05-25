#include "webserver.h"
#define HTML_FILE_DIR "C:/workspace/myNN_project/src/tools/webserver"

int main() {
    httpServer server(std::string(HTML_FILE_DIR));
    server.start_service();
    return 0;
}
