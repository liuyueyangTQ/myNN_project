#include <iostream>
#include <string>
#include "nn_netServer.h"
int main() {
    netService::nn_netClient client;
    std::string ip;
    std::cout << "Please input the ip...\n";
    std::cin >> ip;
    client.connect(ip); // 192.168.3.30, 192.168.137.1
    char* c = "dsadafaa";
    client.send_data(c, 5);
    client.receiving_loop();
    return 0;
}