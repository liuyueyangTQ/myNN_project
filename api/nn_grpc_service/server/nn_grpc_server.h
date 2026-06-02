// ================================================================
// grpc_server.h — gRPC 训练服务端
//
// 基于 proto_utils 实现 NNTrainer::Service 接口。
// 收到 TrainRequest → 转为 NNParams → proto_utils 训练+序列化 → 返回二进制。
// ================================================================

#pragma once

#include <grpcpp/grpcpp.h>
#include "network.grpc.pb.h"
#include "proto_utils.h"
#include "nn.h"
#include "enum_type.h"
// ================================================================
// NNTrainerServiceImpl — 实现 .proto 中定义的 NNTrainer::Service
// ================================================================
class NNTrainerServiceImpl final : public nn_proto::NNTrainer::Service {
public:
    grpc::Status RunModel(
        grpc::ServerContext* context,
        const nn_proto::TrainRequest* request,
        nn_proto::TrainResponse* response) override;
protected:

private:
    // TrainRequest → NNParams：将 proto 请求消息转为本地参数结构体
    static nn::NNParams requestToNNParams(const nn_proto::TrainRequest& req);

};

// ================================================================
// RunGrpcServer — 启动 gRPC 服务并阻塞等待
//   listen_addr : 监听地址，如 "0.0.0.0:50051"
// ================================================================
void RunGrpcServer(const std::string& listen_addr);
