#pragma once
namespace dtensor{


enum class layer_type{
    sigmoid,
    relu,
    layer_norm,
    softmax,
    origin
};
enum class loss_type{
    mse,
    cross_entropy
};

enum class tensor_type{
    common,
    layer,
    tensor2D,
    conv2D
};

enum class sub_type {
    sigmoid,
    relu,
    softmax,
    layer_norm,
    origin,
    none
};


enum class ops{
    matmul,
    add,
    sub,
    dot,
    concat
};
enum class tensor_pair{
    layer_layer,
    layer_tensor2D,
    tensor2D_tensor2D,
    common_common
};
}
namespace nn {
enum class nn_type {
    Linear_Resnet,
    Linear_NN
};

}