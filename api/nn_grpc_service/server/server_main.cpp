// ================================================================
// server_main.cpp — gRPC 训练服务端入口
//
// 用法：./nn_grpc_server [listen_addr]
//   默认监听 0.0.0.0:50051
// ================================================================

#include <iostream>
#include "nn_grpc_server.h"

int main(int argc, char* argv[]) {
    std::string addr = "0.0.0.0:50051";
    if (argc > 1) addr = argv[1];

    std::cout << "=== NN Trainer gRPC Server ===" << std::endl;
    RunGrpcServer(addr);

    return 0;
}
