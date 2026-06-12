#include "dtensor_tools.h"

namespace dtensor {
    
[[nodiscard]] dtensor_base* dtensor_factory(tensor_type p, std::vector<size_t>& tensor_shape, sub_type q, size_t batch_num, op* temp_op) {
    dtensor_base* new_node = nullptr;

    switch (p)
    {
    case tensor_type::common:
        new_node = new multi_dim_tensor(tensor_shape, batch_num);
        new_node->set_type(tensor_type::common);
        break;

    case tensor_type::tensor2D: { // 此时不用管 sub_type
        assert(tensor_shape.size() == 2);
        if(tensor_shape.size() != 2) throw(1);
        new_node = new tensor2D_float({tensor_shape[0], tensor_shape[1]}, batch_num); // 二维矩阵张量
        new_node->set_type(tensor_type::tensor2D);
        break;
    }

    case tensor_type::layer:
        // 目前先支持输出为列向量的情况，后续再考虑更多输出形状的情况
        assert(tensor_shape.size() == 1);
        if(tensor_shape.size() != 1) throw(1);
        new_node = layer_tool(tensor_shape[0], batch_num, q); 
        new_node->set_type(tensor_type::layer);
        break;
    
    default:
        break;
    }
    if(!new_node) throw(1);
    new_node->set_op_last(temp_op);
    temp_op->output = new_node;
    return new_node;
}

layer* layer_tool(int n, size_t batch_num, sub_type stp) {
    switch (stp)
    {
    case sub_type::sigmoid:
        return new sigmoid(n, batch_num);
    case sub_type::relu:
        return new relu(n, batch_num);
    case sub_type::softmax:
        return new softmax(n, batch_num);
    case sub_type::origin:
        return new origin(n, batch_num);
    default:
        std::cerr << "Unsupported layer type! \n";
    }
    return nullptr;
}

} // namespace dtensor