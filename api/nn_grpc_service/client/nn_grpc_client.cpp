// ================================================================
// grpc_client.cpp — gRPC 训练客户端实现
// ================================================================

#include <iostream>
#include <chrono>
#include "nn_grpc_client.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

// ================================================================
// 构造函数：建立 gRPC channel 和 stub
// ================================================================
NNTrainerGrpcClient::NNTrainerGrpcClient(const std::string& target) {
    // 通过 ChannelArguments 设置最大消息大小（默认 4 MB 不够存网络权重）
    grpc::ChannelArguments args;
    args.SetMaxReceiveMessageSize(256 * 1024 * 1024);  // 256 MB
    args.SetMaxSendMessageSize(256 * 1024 * 1024);
    channel_ = grpc::CreateCustomChannel(
        target, grpc::InsecureChannelCredentials(), args);
    stub_ = nn_proto::NNTrainer::NewStub(channel_);
}

// ================================================================
// RunModel — 同步调用远端训练
//
// 超时 600 秒，最大消息 256 MB（训练结果可能很大）。
// ================================================================
nn_proto::TrainResponse NNTrainerGrpcClient::RunModel(
    const nn_proto::TrainRequest& request) {

    nn_proto::TrainResponse response;
    ClientContext ctx;

    // 超时：10 分钟（训练大型网络/大数据集可能耗时较长）
    ctx.set_deadline(
        std::chrono::system_clock::now() + std::chrono::seconds(600));

    std::cout << "[gRPC Client] Calling RunModel (timeout=600s)..." << std::endl;

    Status status;
    try {
        status = stub_->RunModel(&ctx, request, &response);
        std::cout << "[gRPC Client] stub call returned" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[gRPC Client] stub threw: " << e.what() << std::endl;
        response.set_success(false);
        response.set_message(std::string("gRPC exception: ") + e.what());
        return response;
    } catch (...) {
        std::cerr << "[gRPC Client] stub threw unknown exception" << std::endl;
        response.set_success(false);
        response.set_message("gRPC unknown exception");
        return response;
    }

    if (status.ok()) {
        std::cout << "[gRPC Client] Success, binary size: "
                  << response.network_data().size() << " bytes" << std::endl;
    } else {
        std::cerr << "[gRPC Client] RPC failed (code="
                  << (int)status.error_code() << "): "
                  << status.error_message() << std::endl;
        response.set_success(false);
        response.set_message("RPC error: " + status.error_message());
    }

    return response;
}