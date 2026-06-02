// echo_server.cpp — gRPC echo 服务端
//
// 收到 EchoRequest → 打印 → 返回 EchoResponse
// 用法：./echo_server [listen_addr]  默认 0.0.0.0:50051
// ================================================================

#include <grpcpp/grpcpp.h>
#include "echo_test.grpc.pb.h"
#include <cstdio>

class EchoServiceImpl final : public echo_test::EchoService::Service {
    grpc::Status Echo(
        grpc::ServerContext*,
        const echo_test::EchoRequest* req,
        echo_test::EchoResponse* resp) override
    {
        printf("[server] received: message=\"%s\" count=%d\n",
               req->message().c_str(), req->count());
        resp->set_reply("Server echoed: " + req->message());
        resp->set_doubled_count(req->count() * 2);
        return grpc::Status::OK;
    }
};

int main(int argc, char* argv[]) {
    std::string addr = "0.0.0.0:50051";
    if (argc > 1) addr = argv[1];

    EchoServiceImpl svc;
    grpc::ServerBuilder builder;
    builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
    builder.RegisterService(&svc);

    auto server = builder.BuildAndStart();
    if (!server) { printf("[server] FAILED to start\n"); return 1; }
    printf("[server] listening on %s\n", addr.c_str());
    server->Wait();
    return 0;
}
