#pragma once
#include "dtensor.h"

namespace dtensor {
    
class multi_dim_tensor: public dtensor_base{
private:
    metrix_float* batch_grad; 
    metrix_float* batch_val; 
    std::vector<size_t> shape;
    void** pMemory; // 辅助变量
    metrix_float* _allocdata(size_t bias);
public:
    multi_dim_tensor(std::vector<size_t> shape, size_t batch_num) : 
        dtensor_base(false, false, tensor_type::common, batch_num),
        pMemory(new void*[2]),
        shape(shape)
    {
        batch_val = this->_allocdata(0);
        batch_grad = this->_allocdata(1);
        size_t total_size = 1;
        for (size_t dim : shape) {
            total_size *= dim;
        }
        this->n = total_size;
    }
    multi_dim_tensor(_size _shape, size_t batch_num) : 
        dtensor_base(false, false, tensor_type::common, batch_num),
        pMemory(new void*[2])
    {
        shape.push_back(_shape.first);
        shape.push_back(_shape.second);
        batch_val = this->_allocdata(0);
        batch_grad = this->_allocdata(1);
        size_t total_size = 1;
        for (size_t dim : shape) {
            total_size *= dim;
        }
        this->n = total_size;
    }
    void _print_val(size_t batch_id = 0) override;
    void _print_grad(size_t batch_id = 0) override;
    void set_input_value(float* data, size_t batch_id) override;
    void set_input_value(std::vector<std::vector<float>>& data) override;

    std::vector<size_t> get_shape() override;
    void _forward() override;
    void _forward(size_t batch_id) override;

    void _backward() override;  //用于中间层
    void _backward(size_t batch_id) override; //用于中间层

    //void backward_mul(); // 之后在考虑

    void backward(float* next_grad) override;  //用于尾部层 // 可额外添加
    void backward(float* next_grad, size_t batch_id) override; //用于尾部层 // 可额外添加

    void backward_grad() override;
    void backward_grad(size_t batch_id) override;

    void count_loss_grad(std::vector<float>& label, loss_type loss_tp, size_t batch_id) override;
    void clear_grad() override;
    void clear_value() override;
    void update(double lr) override;
#ifdef USE_DEBUG 
    void _forward_D() override;
    void _forward_D(size_t batch_id) override;

    void _backward_D() override;  //用于中间层
    void _backward_D(size_t batch_id) override; //用于中间层
#endif    

    metrix_float& get_input_metrix_ref(size_t batch_id) override;
    metrix_float& get_grad_metrix_ref(size_t batch_id) override;
    metrix_float& get_output_metrix_ref(size_t batch_id) override;

    metrix_float* get_output_metrix_ptr() override;
    metrix_float* get_grad_metrix_ptr() override;
    metrix_float* get_input_metrix_ptr() override;

    float* get_input_data_ptr(size_t batch_id) override;
    float* get_grad_data_ptr(size_t batch_id) override;
    float* get_output_data_ptr(size_t batch_id) override;

    // DEBUG funtions
    void __set_grads__(std::vector<std::vector<float>>& grads) override;
    void __set_inputs__(std::vector<std::vector<float>>& input) override;
    void __set_outputs__(std::vector<std::vector<float>>& output) override;
};

/// @todo
// class common_tensor: public dtensor_base{
// private:
//     friend class base::metrix_float;
//     metrix_float* w;
//     metrix_float* grad;
//     _size shape;
//     std::vector<dtensor_base*> next;
//     std::vector<dtensor_base*> last;
//     // std::vector<std::pair<tensor_base*,tensor::ops> > next1; //算子和矩阵的组合
//     // std::vector<tensor::ops> forward_ops;
//     // std::vector<tensor::ops> back_ops;
// public:
//     common_tensor(_size shape) : shape(shape), w(new metrix_float(shape)), grad(new metrix_float(shape)), dtensor_base(false, false, tensor_type::common) {}
// };

} // namespace dtensor