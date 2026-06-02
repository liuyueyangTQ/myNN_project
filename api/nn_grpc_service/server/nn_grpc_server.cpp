// ================================================================
// grpc_server.cpp — gRPC 训练服务端实现
//
// 调用链：
//   gRPC RunModel → requestToNNParams → proto_utils.train_model
//                 → proto_utils.network2proto → response.set_network_data
// ================================================================

#include <iostream>
#include "nn_grpc_server.h"
using namespace proto;
using namespace nn;
using namespace dtensor;
// ================================================================
// 辅助：TrainRequest → NNParams
//
// 将 protobuf 请求中所有字段一一映射到 C++ 端的 NNParams 结构体。
// 枚举值序号已在 .proto / enum_type.h 中约定为同步。
// ================================================================
nn::NNParams NNTrainerServiceImpl::requestToNNParams(
    const nn_proto::TrainRequest& req) {

    nn::NNParams p;

    // ---- 模型类型 ----
    p.model_type = (req.model_type() == nn_proto::MODEL_LINEAR_RESNET)
        ? nn::nn_type::Linear_Resnet : nn::nn_type::Linear_NN;

    // ---- 层结构 ----
    p.layer_num = req.layer_sizes_size();
    p.layer_sizes.reserve(p.layer_num);
    for (int i = 0; i < p.layer_num; ++i)
        p.layer_sizes.push_back(req.layer_sizes(i));

    p.layer_types.reserve(req.layer_activations_size());
    for (int i = 0; i < req.layer_activations_size(); ++i)
        // proto ActivationType 枚举值与 dtensor::sub_type 序号一一对应
        p.layer_types.push_back(
            static_cast<dtensor::sub_type>(req.layer_activations(i)));

    // ---- 训练超参 ----
    p.batch_size      = req.batch_size();
    p.use_multithread = req.use_multithread();
    p.thread_num      = req.thread_num();
    p.epochs          = req.epochs();
    p.lr              = req.lr();
    p.samples         = req.samples();
    p.input_output_dim = {
        static_cast<size_t>(req.input_dim()),
        static_cast<size_t>(req.output_dim())
    };

    // ---- 损失函数 ----
    p.lstp = (req.loss() == nn_proto::LOSS_MSE)
        ? dtensor::loss_type::mse : dtensor::loss_type::cross_entropy;

    return p;
}

// ================================================================
// RunModel RPC 实现
//
// Step 1: proto 请求 → NNParams
// Step 2: 参数校验
// Step 3: proto_utils 训练模型
// Step 4: proto_utils 将模型序列化为 Network → binary
// Step 5: binary 填入 TrainResponse 返回
// ================================================================
grpc::Status NNTrainerServiceImpl::RunModel(
    grpc::ServerContext* context,
    const nn_proto::TrainRequest* request,
    nn_proto::TrainResponse* response) {

    std::cout << "[gRPC] RunModel received" << std::endl;

    try {
        // Step 1: 请求转 NNParams
        nn::NNParams params = requestToNNParams(*request);

        // Step 2: 校验参数合法性
        params.check();

        // Step 3: 用 proto_utils 执行训练
        proto::proto_utils agent;
        agent.train_model(params);           // 调用 nn::run_model

        // Step 4: 模型 → proto Network → 二进制序列化
        agent.network2proto();
        std::string binary = agent.get_binary();


        // 调试输出：显示网络结构和二进制数据
        std::cout << "net structure encoded as: \n";
        agent.print_network();
        // 接收端解析数据并还原网络
        proto_utils agent_test;
        agent_test.get_binary(binary);
        agent_test.proto2network();
        std::cout << "------ Validating decoded network ------\n";
        std::cout << "net structure decoded: \n";
        agent_test.print_network();
        std::cout << "The binary data string is:\n";
        agent_test.print_binary();


        // Step 5: 封装响应
        response->set_success(true);
        response->set_message("Training completed");
        response->set_network_data(std::move(binary));

        std::cout << "[gRPC] RunModel success !! : "
                  << "layers=" << params.layer_num
                  << " epochs=" << params.epochs
                  << " binary_size=" << response->network_data().size()
                  << std::endl;

        return grpc::Status::OK;

    } catch (const std::exception& e) {
        response->set_success(false);
        response->set_message(std::string("Training failed: ") + e.what());
        std::cerr << "[gRPC] RunModel error: " << e.what() << std::endl;
        return grpc::Status(grpc::INTERNAL, e.what());
    }
}

// ================================================================
// RunGrpcServer — 构建并启动服务端
// ================================================================
void RunGrpcServer(const std::string& listen_addr) {
    NNTrainerServiceImpl service;

    grpc::ServerBuilder builder;
    builder.AddListeningPort(listen_addr, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    // 提高最大消息大小（默认 4 MB 可能不够存网络权重）
    builder.SetMaxReceiveMessageSize(256 * 1024 * 1024);  // 256 MB
    builder.SetMaxSendMessageSize(256 * 1024 * 1024);

    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    if (!server) {
        std::cerr << "[gRPC] Failed to start server on " << listen_addr << std::endl;
        return;
    }
    std::cout << "[gRPC] Server listening on " << listen_addr << std::endl;

    server->Wait();
}
