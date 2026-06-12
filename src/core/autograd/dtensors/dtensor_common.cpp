#include "dtensor_common.h"

namespace dtensor {
// get_shape 
std::vector<size_t> multi_dim_tensor::get_shape() {
    return this->shape;
}

metrix_float* multi_dim_tensor::_allocdata(size_t bias) {
    // 步骤 1: 分配原始内存
    // operator new[] 只分配内存，不调用构造函数
    try
    {
        *(pMemory + bias) = operator new[](this->batch_num * sizeof(metrix_float));
    }
    catch(const std::exception& e)
    {
        std::cerr << "failed to allocate tensorfloat memory! " << e.what() << '\n';
    }
    
    assert(*(pMemory + bias));
    // 步骤 2: 在分配的内存上构造对象
    metrix_float* myArray = static_cast<metrix_float*>(*(pMemory + bias));
    for (int i = 0; i < this->batch_num; ++i) {
        // placement new: 在指定地址 (myArray + i) 上构造一个 A 对象
        new (myArray + i) metrix_float(this->n, 1); 
    }
    return myArray;
}

/// @brief 得到内部矩阵引用
/// @return metrix_float&
metrix_float& multi_dim_tensor::get_input_metrix_ref(size_t batch_id) {
    return this->batch_val[batch_id];
}
metrix_float& multi_dim_tensor::get_grad_metrix_ref(size_t batch_id) {
    return this->batch_grad[batch_id];
}
metrix_float& multi_dim_tensor::get_output_metrix_ref(size_t batch_id) {
    return this->batch_val[batch_id];
}

/// @brief 得到内部矩阵指针
/// @return metrix_float*
metrix_float* multi_dim_tensor::get_input_metrix_ptr() { //一样，输入输出相同
    return this->batch_val;
}
metrix_float* multi_dim_tensor::get_grad_metrix_ptr() {
    return this->batch_grad;
}
metrix_float* multi_dim_tensor::get_output_metrix_ptr() {
    return this->batch_val;
}

/// @brief 得到内部矩阵数据指针
/// @return float*
float* multi_dim_tensor::get_input_data_ptr(size_t batch_id) { //一样，输入输出相同
    return this->batch_val[batch_id].data;
}
float* multi_dim_tensor::get_grad_data_ptr(size_t batch_id) {
    return this->batch_grad[batch_id].data;
}
float* multi_dim_tensor::get_output_data_ptr(size_t batch_id) {
    return this->batch_val[batch_id].data;
}

/// @todo @test 用于调试，设置梯度值
void multi_dim_tensor::__set_grads__(std::vector<std::vector<float>>& grads) { 
    assert(grads.size() == this->batch_num);
    assert(grads[0].size() == this->n);
    throw(1);
}

/// @todo @test 用于调试，设置输入
void multi_dim_tensor::__set_inputs__(std::vector<std::vector<float>>& inputs) { 
    assert(inputs.size() == this->batch_num);
    assert(inputs[0].size() == this->n);
    throw(1);
}

/// @todo @test 用于调试，设置输出
void multi_dim_tensor::__set_outputs__(std::vector<std::vector<float>>& outputs) { 
    assert(outputs.size() == this->batch_num);
    assert(outputs[0].size() == this->n);
    throw(1);
}
/// @todo 输入输出清零
void multi_dim_tensor::clear_value() {
    throw(1);
}

/// @todo 梯度清零 
void multi_dim_tensor::clear_grad() {
    throw(1);
}

/// @todo 更新梯度 
void multi_dim_tensor::update(double lr) {
    throw(1);
}

/// @brief 前向反向传播
void multi_dim_tensor::_forward() {
}
void multi_dim_tensor::_forward(size_t batch_id) {
}
void multi_dim_tensor::_backward() {
}
void multi_dim_tensor::_backward(size_t batch_id) {
}
void multi_dim_tensor::backward(float* next_grad) {
}
void multi_dim_tensor::backward(float* next_grad, size_t batch_id) {
}
void multi_dim_tensor::backward_grad() {
}
void multi_dim_tensor::backward_grad(size_t batch_id) {
}

#ifdef USE_DEBUG

/// @brief 前向反向传播
void multi_dim_tensor::_forward_D() {
}
void multi_dim_tensor::_forward_D(size_t batch_id) {
}
void multi_dim_tensor::_backward_D() {
}
void multi_dim_tensor::_backward_D(size_t batch_id) {
}

#endif

/// @brief @todo 打印输入输出、梯度
void multi_dim_tensor::_print_val(size_t batch_id) {

}
void multi_dim_tensor::_print_grad(size_t batch_id) {

}

/// @brief @test 设置输入值
void multi_dim_tensor::set_input_value(float* data, size_t batch_id) {
    assert(batch_id < this->batch_num);
    std::copy(data, data + this->n, (this->batch_val + batch_id)->data);

}
void multi_dim_tensor::set_input_value(std::vector<std::vector<float>>& data) {
    assert(data.size() == this->batch_num);
    throw(1);
}

/// @brief 反向传播计算损失
void multi_dim_tensor::count_loss_grad(std::vector<float>& label, loss_type loss_tp, size_t batch_id){
    throw(1);
}


} // namespace dtensor