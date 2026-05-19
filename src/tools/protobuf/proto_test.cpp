#include<iostream>
#include"proto_utils.h"
#include "enum_type.h"
using namespace proto;
using namespace nn;
using namespace dtensor;
int main() {
    // nn::module_base* net = new nn::Linear_NN(5); // batch_num = 5
    // static_cast<nn::Linear_NN*>(net)->add_layer(10, dtensor::sub_type::relu);
    // static_cast<nn::Linear_NN*>(net)->add_layer(20, dtensor::sub_type::sigmoid);
    // static_cast<nn::Linear_NN*>(net)->add_layer(5, dtensor::sub_type::softmax);
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
    nn_proto::Network net_proto1 = agent1.network2proto();
    std::cout << "net structure to be encoded: \n";
    agent1.print_network();
    // 模拟传输过程（这里直接在内存中传递二进制数据）
    std::string binary_data;
    net_proto1.SerializeToString(&binary_data);
    // 接收端解析数据并还原网络
    proto_utils agent2;
    nn_proto::Network net_proto2;
    agent2.get_data(std::move(binary_data));
    agent2.proto2network(net_proto2);
    std::cout << "net structure decoded: \n";
    agent2.print_network();
    return 0;
}