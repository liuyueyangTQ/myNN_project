#pragma once
#include <cassert>
#include <vector>
#include <string>
#include "network.pb.h"
#include "metrix.h"
#include "dtensor.h"
#include "nn.h"
#include "enum_type.h"

namespace proto {
class proto_utils {
private:
    nn::NNParams params;
    nn::model_data model;
    std::string binary_data; // 存储序列化后的二进制数据
public:
    void get_data(std::string&& data);
    void train_model(const nn::NNParams& params);
    void metrix2proto(const base::metrix_float& m, nn_proto::WeightMetrix& w_proto);
    void proto2metrix(const nn_proto::WeightMetrix& w_proto, base::metrix_float& m);
    void layer2proto(dtensor::layer* layer_ptr, nn_proto::Layer& layer_proto);
    void proto2layer(const nn_proto::Layer& layer_proto, dtensor::layer* layer_ptr);
    void wm2proto(dtensor::tensor2D_float* wm, nn_proto::WeightMetrix& w_proto);
    void proto2wm(const nn_proto::WeightMetrix& w_proto, dtensor::tensor2D_float* wm);
    nn_proto::Network network2proto();
    void proto2network(nn_proto::Network& net_proto);
    void print_network();
};

}

