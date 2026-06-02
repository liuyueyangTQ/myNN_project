#include"proto_utils.h"
namespace proto {
void proto_utils::get_binary(std::string&& data) {
    this->binary_data = std::move(data);
}
void proto_utils::get_binary(const std::string& data) {
    this->binary_data = data;
}
void proto_utils::get_binary(char* data, size_t len) {
    this->binary_data.assign(data, len);
}
void proto_utils::train_model(const nn::NNParams& params) {
    this->params = params;
    this->model = nn::run_model(params);
}
void proto_utils::metrix2proto(const base::metrix_float& m, nn_proto::WeightMetrix& w_proto) {
    auto rows = m.shape.first;
    auto cols = m.shape.second;
    w_proto.set_rows(rows);
    w_proto.set_cols(cols);
    w_proto.set_data(m.data, sizeof(float) * rows * cols);
}
void proto_utils::proto2metrix(const nn_proto::WeightMetrix& w_proto, base::metrix_float& m) {
    assert(m.shape.first * m.shape.second == w_proto.data().size() / sizeof(float));
    memcpy(m.data, w_proto.data().data(), w_proto.data().size());
}
void proto_utils::layer2proto(dtensor::layer* layer_ptr, nn_proto::Layer& layer_proto) {
    static_cast<dtensor::layer*>(layer_ptr); // 有肯能传入的是 dtensor_base* 类型，先进行类型转换
    layer_proto.set_neuron_count(layer_ptr->get_n());
    layer_proto.set_activation(static_cast<nn_proto::ActivationType>(layer_ptr->get_layer_type())); // 枚举类型顺序需要完全一致
    layer_proto.set_b(layer_ptr->get_bias_data(), sizeof(float) * layer_ptr->get_n());
}
void proto_utils::proto2layer(const nn_proto::Layer& layer_proto, dtensor::layer* layer_ptr) {
    static_cast<dtensor::layer*>(layer_ptr);
    assert(layer_ptr->get_n() == layer_proto.neuron_count());
    assert(static_cast<nn_proto::ActivationType>(layer_ptr->get_layer_type()) == layer_proto.activation());
    memcpy(layer_ptr->get_bias_data(), layer_proto.b().data(), layer_proto.b().size());
}   
void proto_utils::wm2proto(dtensor::tensor2D_float* wm, nn_proto::WeightMetrix& w_proto) {
    auto shape = wm->get_weight()->shape;
    auto rows = shape.first;
    auto cols = shape.second;
    w_proto.set_rows(rows);
    w_proto.set_cols(cols);
    w_proto.set_data(wm->get_weight()->data, sizeof(float) * rows * cols);
}
void proto_utils::proto2wm(const nn_proto::WeightMetrix& w_proto, dtensor::tensor2D_float* wm) {
    auto shape = wm->get_weight()->shape;
    assert(shape.first * shape.second == w_proto.data().size() / sizeof(float));
    memcpy(wm->get_weight()->data, w_proto.data().data(), w_proto.data().size());
}
void proto_utils::network2proto() { // 将当前模型转换为 proto 对象
    // 1. 创建 proto 对象
    this->net_proto.Clear(); // 清空之前的内容
    int layer_num = this->params.layer_num;
    net_proto.set_layer_count(layer_num);  // layer_num 层网络
    auto& layer_sizes = this->params.layer_sizes;
    auto& layer_types = this->params.layer_types;
    auto& metrixs = this->model.param_w;
    auto& biases = this->model.param_b;
    assert(layer_num == layer_sizes.size() && layer_num == layer_types.size() && layer_num - 1 == metrixs.size() && layer_num == biases.size());
    for(int i = 0; i < layer_num - 1; ++i) {
        auto layer_proto  = net_proto.add_layers();
        assert(layer_sizes[i] == biases[i].size());
        layer_proto->set_activation(static_cast<nn_proto::ActivationType>(layer_types[i]));
        layer_proto->set_neuron_count(layer_sizes[i]);
        layer_proto->set_b(biases[i].data(), sizeof(float) * layer_sizes[i]);
        auto wm_proto = net_proto.add_metrixs();
        assert(metrixs[i].size() == layer_sizes[i + 1] && metrixs[i][0].size() == layer_sizes[i]);
        int rows = metrixs[i].size();
        int cols = metrixs[i][0].size();
        wm_proto->set_rows(rows);
        wm_proto->set_cols(cols);
        std::vector<float> flat_metrix; // 存储连续数据
        for (const auto& row : metrixs[i]) {
            flat_metrix.insert(flat_metrix.end(), row.begin(), row.end());
        }
        assert(flat_metrix.size() == rows * cols);
        wm_proto->set_data(flat_metrix.data(), sizeof(float) * flat_metrix.size());
    }
    auto last_layer_proto = net_proto.add_layers();
    assert(layer_sizes[layer_num - 1] == biases[layer_num - 1].size());
    last_layer_proto->set_neuron_count(layer_sizes[layer_num - 1]);
    last_layer_proto->set_activation(static_cast<nn_proto::ActivationType>(layer_types[layer_num - 1]));
    last_layer_proto->set_b(biases[layer_num - 1].data(), sizeof(float) * layer_sizes[layer_num - 1]);
    // ========== 序列化（用于传输） ==========
    net_proto.SerializeToString(&this->binary_data);
    return;
}
std::string proto_utils::get_binary() {
    return this->binary_data;
}
void proto_utils::proto2network() { // 将二进制数据转换为模型
    net_proto.Clear(); // 清空之前的内容
    net_proto.ParseFromString(this->binary_data);
    int layer_num = net_proto.layer_count();
    std::vector<int> layer_sizes(layer_num);
    std::vector<dtensor::sub_type> layer_types(layer_num);
    std::vector<std::vector<float>> biases(layer_num);
    std::vector<std::vector<std::vector<float>>> metrixs(layer_num - 1);
    for(int i = 0; i < layer_num - 1; ++i) {
        auto layer_proto = net_proto.layers(i);
        int neurons = layer_proto.neuron_count();
        auto act_type = layer_proto.activation();
        layer_sizes[i] = neurons;
        layer_types[i] = static_cast<dtensor::sub_type>(act_type);
        auto b_data = layer_proto.b();
        std::vector<float> b(neurons);
        memcpy(b.data(), b_data.data(), b_data.size());
        biases[i] = std::move(b);
        auto wm_proto = net_proto.metrixs(i);
        int rows = wm_proto.rows();
        int cols = wm_proto.cols();
        auto wm_data = wm_proto.data();
        metrixs[i] = std::vector<std::vector<float>>(rows, std::vector<float>(cols));
        for(int j = 0; j < rows; ++j) {
            memcpy(metrixs[i][j].data(), wm_data.data() + j * cols * sizeof(float), cols * sizeof(float));
        }
    }
    auto last_layer_proto = net_proto.layers(layer_num - 1);
    int neurons = last_layer_proto.neuron_count();
    auto last_b_data = last_layer_proto.b();
    layer_sizes[layer_num - 1] = neurons;
    layer_types[layer_num - 1] = static_cast<dtensor::sub_type>(last_layer_proto.activation());
    std::vector<float> last_b(neurons);
    memcpy(last_b.data(), last_b_data.data(), last_b_data.size());
    biases[layer_num - 1] = std::move(last_b);
    // 2. 加载到模型中
    this->model.layer_sizes = std::move(layer_sizes);
    this->model.layer_types = std::move(layer_types);
    this->model.param_b = std::move(biases);
    this->model.param_w = std::move(metrixs);
    return;
}

void proto_utils::print_network() {
    std::cout << "Network Architecture:\n";
    std::cout << "Layer Sizes: ";
    for (const auto& size : this->model.layer_sizes) {
        std::cout << size << " ";
    }
    std::cout << "\nLayer Types: ";
    for (const auto& type : this->model.layer_types) {
        std::cout << static_cast<int>(type) << " "; // 输出枚举值的整数表示
    }
    std::cout << "\nBiases:\n";
    for (const auto& b_vec : this->model.param_b) {
        for (const auto& b : b_vec) {
            std::cout << b << " ";
        }
        std::cout << "\n";
    }
    std::cout << "Weight Matrices:\n";
    for (const auto& wm : this->model.param_w) {
        for (const auto& row : wm) {
            for (const auto& w : row) {
                std::cout << w << " ";
            }
            std::cout << "\n";
        }
        std::cout << "----\n";
    }
};
void proto_utils::print_binary() {
    std::cout << binary_data << "\n";
}
} // namespace proto