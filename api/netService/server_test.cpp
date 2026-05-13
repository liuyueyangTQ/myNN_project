#include "server.h"
int main() {
    try {
        myServer server(PORT, 10);
    } catch (const std::exception& e) {
        std::cerr << "Server error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}