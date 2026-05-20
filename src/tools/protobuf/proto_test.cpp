#include<iostream>
#include"proto_utils.h"
#include "enum_type.h"
using namespace proto;
using namespace nn;
using namespace dtensor;
int main() {
    nn::NNParams params;
    params.model_type = nn::nn_type::Linear_NN;
    params.layer_num = 4;
    params.layer_sizes = {10, 20, 20, 5};
    params.layer_types = {sub_type::origin, sub_type::relu, sub_type::relu, sub_type::softmax};
    params.batch_size = 4;
    params.epochs = 1000;
    params.lr = 0.001;
    proto_utils agent1;
    agent1.train_model(params);
    agent1.network2proto();
    std::cout << "net structure to be encoded: \n";
    agent1.print_network();
    // 模拟传输过程（这里直接在内存中传递二进制数据）
    std::string binary_data;
    binary_data = agent1.get_binary();
    // 接收端解析数据并还原网络
    proto_utils agent2;
    agent2.get_binary(std::move(binary_data));
    agent2.proto2network();
    std::cout << "net structure decoded: \n";
    agent2.print_network();
    std::cout << "The binary data string is:\n";
    agent2.print_binary();
    return 0;
}