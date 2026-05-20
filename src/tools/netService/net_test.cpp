#include "server.h"
int main() {
    try {
        netService::__netServer_base* server = new netService::demoServer(true, PORT, 10);
        server->start_service();
    } catch (const std::exception& e) {
        std::cerr << "Server error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}