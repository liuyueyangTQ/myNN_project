// ================================================================
// grpc_client.h — gRPC 训练客户端
//
// 连接到远端 NNTrainer 服务，发送训练请求，
// 接收序列化后的 Network 二进制数据。
// 配合 proto_utils::proto2network() 即可在本地还原完整模型。
// ================================================================

#pragma once

#include <grpcpp/grpcpp.h>
#include <memory>
#include <string>
#include "nn.h"
#include "enum_type.h"
#include "network.grpc.pb.h"

class NNTrainerGrpcClient {
public:
    // 连接到指定地址的服务端，如 "localhost:50051"
    explicit NNTrainerGrpcClient(const std::string& target);

    // 发送训练请求，返回服务端响应（含序列化的 Network）
    nn_proto::TrainResponse RunModel(const nn_proto::TrainRequest& request);

private:
    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<nn_proto::NNTrainer::Stub> stub_;
};
