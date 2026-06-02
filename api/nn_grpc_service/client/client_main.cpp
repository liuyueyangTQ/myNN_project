// ================================================================
// client_main.cpp — gRPC 训练客户端入口
//
// 用 fprintf 直接写 client_debug.log，不依赖 iostream。
// ================================================================

#include "nn_grpc_client.h"
#include "proto_utils.h"
#include <cstdio>
#include <cstdlib>
#include <csignal>

static FILE* g_fp = nullptr;

#define LOG(fmt, ...) do { \
    if (!g_fp) g_fp = fopen("client_debug.log", "w"); \
    if (g_fp) { fprintf(g_fp, fmt "\n", ##__VA_ARGS__); fflush(g_fp); } \
    printf(fmt "\n", ##__VA_ARGS__); \
} while(0)

void crash_handler(int sig) {
    fprintf(stderr, "!!! CRASH signal=%d !!!\n", sig);
    if (g_fp) { fprintf(g_fp, "!!! CRASH signal=%d !!!\n", sig); fflush(g_fp); }
    std::abort();
}

int main(int argc, char* argv[]) {
    signal(SIGSEGV, crash_handler);
    signal(SIGABRT, crash_handler);

    LOG("=== NN Trainer gRPC Client ===");

    std::string addr = "localhost:50051";
    if (argc > 1) addr = argv[1];
    LOG("Target: %s", addr.c_str());

    // ---- 1. 连接 ----
    LOG("[1/5] Creating channel...");
    NNTrainerGrpcClient client(addr);
    LOG("[1/5] Channel OK");

    // ---- 2. 构建请求 ----
    LOG("[2/5] Building request...");
    nn_proto::TrainRequest req;
    req.set_model_type(nn_proto::MODEL_LINEAR_NN);
    req.add_layer_sizes(10);
    req.add_layer_sizes(20);
    req.add_layer_sizes(20);
    req.add_layer_sizes(5);
    req.add_layer_activations(nn_proto::ACT_ORIGIN);
    req.add_layer_activations(nn_proto::ACT_RELU);
    req.add_layer_activations(nn_proto::ACT_RELU);
    req.add_layer_activations(nn_proto::ACT_SOFTMAX);
    req.set_batch_size(4);
    req.set_use_multithread(false);
    req.set_thread_num(4);
    req.set_epochs(500);
    req.set_loss(nn_proto::LOSS_CROSS_ENTROPY);
    req.set_input_dim(10);
    req.set_output_dim(5);
    req.set_samples(200);
    req.set_lr(0.01);
    LOG("[2/5] Request OK, layers=%d", req.layer_sizes_size());

    // ---- 3. 远端训练 ----
    LOG("[3/5] Calling RunModel...");
    nn_proto::TrainResponse resp = client.RunModel(req);
    LOG("[3/5] RunModel returned, success=%d", (int)resp.success());

    // ---- 4. 检查响应 ----
    LOG("[4/5] msg=%s data_bytes=%d",
        resp.message().c_str(), (int)resp.network_data().size());

    if (!resp.success()) {
        LOG("FAIL: %s", resp.message().c_str());
        printf("Press Enter...\n");
        getchar();
        return 1;
    }

    // ---- 5. 反序列化 ----
    LOG("[5/5] Deserializing...");
    proto::proto_utils agent;
    agent.get_binary(std::string(resp.network_data()));
    agent.proto2network();
    LOG("[5/5] Deserialization OK");

    agent.print_network();
    LOG("Done.");

    printf("Press Enter...\n");
    getchar();
    if (g_fp) fclose(g_fp);
    return 0;
}