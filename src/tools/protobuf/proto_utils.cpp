#include"proto_utils.h"
void matrix2proto(const base::metrix_float& m, nn_proto::WeightMatrix& w_proto) {
    auto rows = m.shape.first;
    auto cols = m.shape.second;
    w_proto.set_rows(rows);
    w_proto.set_cols(cols);
    w_proto.set_data(m.data, sizeof(float) * rows * cols);
}

void bias2proto(const base::metrix_float& b_vec, nn_proto::BiasVector& b_proto) {
    auto size = b_vec.shape.first;
    b_proto.set_size(size);
    b_proto.set_data(b_vec.data, sizeof(float) * size);
}

void proto2matrix(const nn_proto::WeightMatrix& w_proto, base::metrix_float& m) {
    assert(m.shape.first * m.shape.second == w_proto.data().size() / sizeof(float));
    memcpy(m.data, w_proto.data().data(), w_proto.data().size());
}

void proto2bias(const nn_proto::BiasVector& b_proto, base::metrix_float& b) {
    assert(b.shape.first == b_proto.size());
    memcpy(b.data, b_proto.data().data(), b_proto.data().size());
}

void network2proto(nn::module_base* net, nn_proto::Network& net_proto) {
    // // 1. 创建 proto 对象
    // net_proto.set_layer_count(3);  // 3 层网络

    // // ========== 层 1 ==========
    // auto layer1 = net_proto.add_layers();
    // layer1->set_neuron_count(10);                // 10 个神经元
    // layer1->set_activation(nn_proto::ACT_RELU);  // 激活函数

    // // 权重 W
    // matrix2proto(weight_mat1, *layer1->mutable_w());

    // // 偏置 b
    // bias2proto(bias_vec1, *layer1->mutable_b());

    // // ========== 层 2 ==========
    // auto layer2 = net_proto.add_layers();
    // layer2->set_neuron_count(20);
    // layer2->set_activation(nn_proto::ACT_SIGMOID);
    // matrix2proto(weight_mat2, *layer2->mutable_w());
    // bias2proto(bias_vec2, *layer2->mutable_b());

    // // ========== 序列化（用于传输） ==========
    // std::string binary_data = net_proto.SerializeAsString();
}

void proto2network(const nn_proto::Network& net_proto, nn::module_base* net) {
    // net_proto.ParseFromString(binary_data);
    // int layer_num = net_proto.layer_count();
    // for (int i = 0; i < layer_num; ++i) {
    //     const auto& layer = net_proto.layers(i);

    //     int neurons = layer.neuron_count();
    //     auto act_type = layer.activation();

    //     base::metrix_float w, b;
    //     proto2matrix(layer.w(), w);
    //     proto2bias(layer.b(), b);

    //     // 加载到你的神经网络中
    // }
}