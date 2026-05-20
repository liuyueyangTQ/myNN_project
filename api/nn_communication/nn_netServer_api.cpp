#include "nn_netServer.h"
int main() {
    netService::nn_netServer server(true);
    server.start_service();
    return 0;
}