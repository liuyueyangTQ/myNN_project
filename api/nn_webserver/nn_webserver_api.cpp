#include "nn_webserver.h"
#define HTML_FILE_DIR "C:/workspace/myNN_project/api/nn_webserver"

int main() {
    nnhttpServer server(std::string(HTML_FILE_DIR));
    server.start_service();
    return 0;
}
