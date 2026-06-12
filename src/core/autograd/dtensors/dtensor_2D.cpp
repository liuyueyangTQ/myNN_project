#include <cstring>
#include "dtensor_2D.h"

namespace dtensor {

#ifdef USE_DEBUG
    int tensor2D_float::wm_index = 0;
#endif

// get_shape 
std::vector<size_t> tensor2D_float::get_shape() {
    return {this->shape.first, this->shape.second};
}
/// @brief 得到内部矩阵引用
/// @return metrix_float&
// tensor2D_float
metrix_float& tensor2D_float::get_input_metrix_ref(size_t batch_id) {
    // return this->x->get_batch_input()[batch_id];  /////待修正
    throw(1);
}
metrix_float& tensor2D_float::get_grad_metrix_ref(size_t batch_id) {
    return this->batch_grad[batch_id];
}
metrix_float& tensor2D_float::get_output_metrix_ref(size_t batch_id) {
    // return this->next->get_batch_output()[batch_id];  /////待修正
    throw(1);
}

/// @brief 得到内部矩阵指针
/// @return metrix_float*
metrix_float* tensor2D_float::get_input_metrix_ptr() { //一样，输入输出相同
    return this->weight;
}
metrix_float* tensor2D_float::get_grad_metrix_ptr() {
    return this->batch_grad;
}
metrix_float* tensor2D_float::get_output_metrix_ptr() {
    return this->weight;
}

/// @brief 得到内部矩阵数据指针
/// @return float*
float* tensor2D_float::get_input_data_ptr(size_t batch_id) { //一样，输入输出相同
    return weight->data; // 这个函数在tensor2D_float中没有实际意义，因为它没有输入输出的区分，直接返回nullptr
}
float* tensor2D_float::get_grad_data_ptr(size_t batch_id) {
    return (batch_grad + batch_id)->data; // 这个函数在tensor2D_float中没有实际意义，因为它没有输入输出的区分，直接返回nullptr
}
float* tensor2D_float::get_output_data_ptr(size_t batch_id) {
    return weight->data; // 这个函数在tensor2D_float中没有实际意义，因为它没有输入输出的区分，直接返回nullptr
}

/// @test @brief 用于调试， 设置梯度值
void tensor2D_float::__set_grads__(std::vector<std::vector<float>>& grads) {
    assert(grads.size() == this->batch_num);
    assert(grads[0].size() == this->n);
    for(int batch_id = 0; batch_id < batch_num; ++batch_id) {
        float* grad_data = (this->batch_grad + batch_id)->data;
        for(int i = 0; i < this->n; ++i)
            grad_data[i] = grads[batch_id][i];
    }
}

/// @test @brief 用于调试， 设置输入
void tensor2D_float::__set_inputs__(std::vector<std::vector<float>>& inputs) {
    assert(inputs.size() == this->batch_num);
    assert(inputs[0].size() == this->n);
    float* w = (this->weight)->data;
    for(int i = 0; i < this->n; ++i) // 只使用一组
        w[i] = inputs[0][i];
    
}

/// @test @brief 用于调试， 设置输出值
void tensor2D_float::__set_outputs__(std::vector<std::vector<float>>& outputs) {
    assert(outputs.size() == this->batch_num);
    assert(outputs[0].size() == this->n);
    float* w = (this->weight)->data;
    for(int i = 0; i < this->n; ++i) // 只使用一组
        w[i] = outputs[0][i];
}

/// @brief 分配内存
metrix_float* tensor2D_float::_allocdata() {

#ifdef USE_DEBUG 
    wm_index++;
    this->index = wm_index;
#endif 

    // 分配原始内存 
    metrix_float* myArray = reinterpret_cast<metrix_float*>(malloc(this->batch_num * sizeof(metrix_float)));

    //在分配的内存上构造对象
    for (int i = 0; i < this->batch_num; ++i) {
        // placement new: 在指定地址 (myArray + i) 上构造一个 A 对象
        new (myArray + i) metrix_float(this->shape.first, this->shape.second, base::init_type::simple, false); // 要传入 false， 不能默认参数，否则会触发隐式转换
    }
    return myArray;
}

/// @brief 释放
void tensor2D_float::_release_data(metrix_float* myArray) {
    // --- 当不再需要数组时，必须手动销毁 ---
    try{
        // 步骤 A: 手动调用每个对象的析构函数
        for (int i = 0; i < this->batch_num; ++i) {
            myArray[i].~metrix_float();
        }
        // 步骤 B: 释放原始内存
        free(reinterpret_cast<void*>(myArray));
    } catch (const char* err) { // 捕获字符串类型异常
        std::cout << "exception when releasing weight metrix data: " << err << std::endl;
    } catch (...) { // 兜底捕获其他异常
        std::cout << "Unkown error! " << std::endl;
    }
}

/// @brief 梯度清零 
void tensor2D_float::clear_grad() {
    for(int batch_id = 0; batch_id < this->batch_num; ++batch_id) {
        float* wg = (batch_grad + batch_id)->data;
        int sz = (this->shape).first * (this->shape).second;
        memset(wg, 0, sizeof(float) * sz);
    }
}

/// @attention 无用
void tensor2D_float::clear_value() { // 无需操作
}

/// @brief 更新参数
void tensor2D_float::update(double lr) {
    if(this->lock_grad || have_updated) // 不更新梯度或已经更新过了
        return;
    size_t row = this->weight->shape.first, col = this->weight->shape.second;
    float dw; float* w;
    float temp;
    for(int i = 0; i < row * col; ++i) {
        dw = 0; 
        for(int batch_id = 0; batch_id < this->batch_num; ++batch_id){
            w = (this->batch_grad + batch_id)->data;
            temp = w[i];
            if(temp > 1) temp = 1;
            else if(temp < -1) temp = -1;
            dw += temp;
        }

        dw /= ((float)this->batch_num); //梯度取平均值
        this->weight->data[i] -= lr * dw;
    }
    have_updated = true;
    return;
}

/// @brief _forward() 和 _backward() 的重载版本，分别用于处理整个batch和单个样本的情况
/// @attention 不能 throw， 因为反向传播时会调用，只不过不进行任何计算
void tensor2D_float::_forward() { //无需操作
}
void tensor2D_float::_forward(size_t batch_id) { //无需操作
}
void tensor2D_float::_backward() { // 无需任何操作，grad即为传入值，在op的backward中已计算
}
void tensor2D_float::_backward(size_t batch_id) { // 无需任何操作，grad即为传入值，在op的backward中已计算
}
void tensor2D_float::backward(float* next_grad) {
}
void tensor2D_float::backward(float* next_grad, size_t batch_id) {
}
void tensor2D_float::backward_grad() {
}
void tensor2D_float::backward_grad(size_t batch_id) {
}

#ifdef USE_DEBUG

void tensor2D_float::_forward_D() { //无需操作
}
void tensor2D_float::_forward_D(size_t batch_id) { //无需操作
}
void tensor2D_float::_backward_D() { // 无需任何操作，grad即为传入值，在op的backward中已计算
}
void tensor2D_float::_backward_D(size_t batch_id) { // 无需任何操作，grad即为传入值，在op的backward中已计算
}

#endif

/// @brief 打印权重
void tensor2D_float::_print_val(size_t batch_id) {
    assert(batch_id <= batch_num);
    std::cout << "tensor2D_float value:\n";
    this->weight->print(); //只有一个weight值
}

/// @brief 打印第 batch_id 个样本的梯度
void tensor2D_float::_print_grad(size_t batch_id) {
    std::cout << "tensor2D_float grad of sample " << batch_id << ":\n";
    this->batch_grad[batch_id].print();
}

/// @brief 用于调试，设置输入值
void tensor2D_float::set_input_value(float* data, size_t batch_id) {
    assert(batch_id == 0); // tensor2D_float 没有batch的概念，直接断言 batch_id 必须为0
    std::cout << "set input value" <<std::endl;
    std::cout << "n is: " << this->n <<std::endl;
    // std::copy(data, data + this->n, this->weight->data);
    for(int i = 0; i < n; ++i) {
        this->weight->data[i] = data[i];
    }
}
void tensor2D_float::set_input_value(std::vector<std::vector<float>>& data) {
    throw("Could not set input value for Weight Metrix!"); 
}

void tensor2D_float::count_loss_grad(std::vector<float>& label, loss_type loss_tp, size_t batch_id){
    throw(1);
}

} // namespace dtensor