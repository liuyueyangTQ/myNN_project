#pragma once
#include <cassert>
#include <vector>
#include <string>
#include <fstream>
#include "network.pb.h"
#include "metrix.h"
#include "dtensor.h"
#include "nn.h"
#include "enum_types.h"

namespace dtensor { // 声明多类 tensor
class tensor2D_float;
class layer;
}

namespace proto {
using namespace dtensor;

class proto_utils {
private:
    nn::NNParams params;
    nn::model_data model;
    std::string binary_data; // 存储序列化后的二进制数据
    nn_proto::Network net_proto; // 存储网络结构的protobuf对象
public:
    void get_binary(std::string&& data);
    void get_binary(const std::string& data);
    void get_binary(char* data, size_t len);
    void train_model(const nn::NNParams& params);
    void metrix2proto(const base::metrix_float& m, nn_proto::WeightMetrix& w_proto);
    void proto2metrix(const nn_proto::WeightMetrix& w_proto, base::metrix_float& m);
    void layer2proto(layer* layer_ptr, nn_proto::Layer& layer_proto);
    void proto2layer(const nn_proto::Layer& layer_proto, layer* layer_ptr);
    void wm2proto(tensor2D_float* wm, nn_proto::WeightMetrix& w_proto);
    void proto2wm(const nn_proto::WeightMetrix& w_proto, tensor2D_float* wm);
    void construct_netproto();
    void proto2network();
    void network2proto();
    std::string get_binary();
    void print_network();
    void print_binary();
    void save_model(const std::string& filepath);
    void load_model(const std::string& filepath, bool decode = true);
};

}

