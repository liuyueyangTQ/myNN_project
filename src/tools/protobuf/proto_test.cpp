#include<iostream>
#include"proto_utils.h"
#include "enum_type.h"
int main() {
    nn::module_base* net = new nn::Linear_NN(5); // batch_num = 5
    static_cast<nn::Linear_NN*>(net)->add_layer(10, dtensor::sub_type::relu);
    static_cast<nn::Linear_NN*>(net)->add_layer(20, dtensor::sub_type::sigmoid);
    static_cast<nn::Linear_NN*>(net)->add_layer(5, dtensor::sub_type::softmax);

    nn_proto::Network net_proto;
    network2proto(net, net_proto);

    std::cout << "Serialized Network Proto:\n" << net_proto.DebugString() << std::endl;

    nn::module_base* deserialized_net = new nn::Linear_NN(5);
    proto2network(net_proto, deserialized_net);

    std::cout << "Deserialization successful!" << std::endl;
    return 0;
}