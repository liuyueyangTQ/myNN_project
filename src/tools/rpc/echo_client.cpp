// echo_client.cpp — gRPC echo 客户端
//
// 发送 EchoRequest → 等待 EchoResponse → 打印结果
// 所有诊断用 fprintf 写文件 + printf 屏幕同步
// 用法：./echo_client [server_addr]  默认 localhost:50051
// ================================================================

#include <grpcpp/grpcpp.h>
#include "echo_test.grpc.pb.h"
#include <cstdio>
#include <chrono>

static FILE* g_fp = nullptr;
#define LOG(fmt, ...) do { \
    if (!g_fp) g_fp = fopen("echo_client.log", "w"); \
    if (g_fp) { fprintf(g_fp, fmt "\n", ##__VA_ARGS__); fflush(g_fp); } \
    printf(fmt "\n", ##__VA_ARGS__); \
} while(0)

int main(int argc, char* argv[]) {
    std::string addr = "localhost:50051";
    if (argc > 1) addr = argv[1];
    grpc_init();
    LOG("=== Echo gRPC Client ===");
    LOG("Target: %s", addr.c_str());

    // ---- 1. 创建 channel ----
    LOG("[1] Creating channel...");
    auto channel = grpc::CreateChannel(
        addr, grpc::InsecureChannelCredentials());
    LOG("[1] Channel created");

    // ---- 2. 创建 stub ----
    LOG("[2] Creating stub...");
    auto stub = echo_test::EchoService::NewStub(channel);
    LOG("[2] Stub created");

    // ---- 3. 构建请求 ----
    LOG("[3] Building request...");
    echo_test::EchoRequest req;
    req.set_message("hello from client");
    req.set_count(42);
    LOG("[3] Request: msg=\"%s\" count=%d req_bytes=%d",
        req.message().c_str(), req.count(), (int)req.ByteSizeLong());

    // ---- 4. 调用 ----
    LOG("[4] Calling Echo...");
    echo_test::EchoResponse resp;
    grpc::ClientContext ctx;
    ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

    grpc::Status status = stub->Echo(&ctx, req, &resp);
    LOG("[4] Echo returned, ok=%d", (int)status.ok());

    if (status.ok()) {
        LOG("[4] Response: reply=\"%s\" doubled=%d",
            resp.reply().c_str(), resp.doubled_count());
    } else {
        LOG("[4] FAIL: code=%d msg=%s",
            (int)status.error_code(), status.error_message().c_str());
        printf("Press Enter...\n"); getchar();
        if (g_fp) fclose(g_fp);
        return 1;
    }

    LOG("SUCCESS");
    printf("Press Enter...\n"); getchar();
    if (g_fp) fclose(g_fp);
    grpc_shutdown();
    return 0;
}
