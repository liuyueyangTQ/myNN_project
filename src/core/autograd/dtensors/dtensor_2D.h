#pragma once
#include "dtensor.h"

namespace dtensor {

class tensor2D_float: public dtensor_base {
private:
    metrix_float* weight; //    size(z) * size(x) 的矩阵 (输入为 x， 输出为 z)
    // 用于记录前后的层关系，非必须
    // layer* x;  // 即 x ,  z = w * x + b
    // layer* next;// 即 z 
    metrix_float* batch_grad;
    bool is_identity;
#ifdef USE_DEBUG 
    int index;
    static int wm_index;
#endif 
    _size shape;
    void check(); //{  // 循环包含，要放到.cpp文件中实现
    //     assert(this->x && this->next);
    //     assert(this->x->n == this->weight->shape.second);
    //     assert(this->next->n == this->weight->shape.first);
    // }
    metrix_float* _allocdata();
    void _release_data(metrix_float* myArray);
public:
    // 声明外部友元 namespace tensor::
    friend class layer;
    friend class Linear_NN;
    // 声明外部友元 namespace base::
    friend float* base::_matmul(metrix_float &m1, metrix_float &m2);
    friend void base::_matmul(metrix_float &m1, metrix_float &m2, bool t1, bool t2, float* data);
    friend void base::_matmul(metrix_float &m1, metrix_float &m2,float* data);
    friend float* base::_alloc_data(metrix_float &m1, metrix_float &m2);
    friend _size base::_get_size(metrix_float &m1, metrix_float &m2);
    friend void base::_matmul_add(metrix_float& m1, metrix_float& m2, bool t1, bool t2, float* data);
    // inline void set_layer_x(layer* p) {
    //     this->x = p;
    // }
    // inline void set_layer_next(layer* p) {
    //     this->next = p;
    // }
    inline metrix_float* get_weight() {
        return this->weight;
    }
    tensor2D_float(_size shape, int batch_num) :  //全零初始化
        dtensor_base(true, false, tensor_type::tensor2D, batch_num, shape.first * shape.second), //不锁梯度
        weight(new metrix_float(shape)),
        shape(shape),
        // x(nullptr),
        // next(nullptr),
        is_identity(false) 
    {
        batch_grad = this->_allocdata();
    }
    tensor2D_float(_size shape, int batch_num, init_type init_type) : // 随机初始化
        dtensor_base(true, false, tensor_type::tensor2D, batch_num, shape.first * shape.second), //不锁梯度
        weight(new metrix_float(shape, init_type)),
        shape(shape),
        // next(nullptr),
        is_identity(false) 
    {
        batch_grad = this->_allocdata();
    }
    tensor2D_float(_size shape, int batch_num, std::vector<float>& nums):  //全零初始化
        dtensor_base(true, false, tensor_type::tensor2D, batch_num, shape.first * shape.second), //不锁梯度
        weight(new metrix_float(shape, nums)),
        shape(shape),
        // x(nullptr),
        // next(nullptr),
        is_identity(false) 
    {
        std::cout << "initializing tensor2D_float using vector ...\n";
        batch_grad = this->_allocdata();
    }
    tensor2D_float(tensor2D_float const &t) = delete; // 不提供实现
    // void _forward() override {}
    ~tensor2D_float() 
    {
        this->_release_data(this->batch_grad);
#ifdef USE_DEBUG 
        auto p = this;
        //std::cout << "successfully released the " << index << "-th weight metrix data!" << " (data address: " << p << ")" << std::endl;   
        wm_index--;
#endif  
    }
    void set_input_value(float* data, size_t batch_id) override;
    void set_input_value(std::vector<std::vector<float>>& data) override;
    void _print_val(size_t batch_id = 0) override;
    void _print_grad(size_t batch_id = 0) override;
    void print_grad();
    std::vector<size_t> get_shape() override;
private:

    void _forward() override;
    void _forward(size_t batch_id) override;

    void _backward() override;
    void _backward(size_t batch_id) override;
    void backward_grad() override;
    void backward_grad(size_t batch_id) override;
#ifdef USE_DEBUG 
    void _forward_D() override;
    void _forward_D(size_t batch_id) override;

    void _backward_D() override;  //用于中间层
    void _backward_D(size_t batch_id) override; //用于中间层
#endif   

    void backward(float* next_grad);  //用于尾部层 // 可额外添加
    void backward(float* next_grad, size_t batch_id); //用于尾部层 // 可额外添加
    void count_loss_grad(std::vector<float>& label, loss_type loss_tp, size_t batch_id) override;    
    // void do_ops(tensor_base *p, ops op) override;
    metrix_float& get_input_metrix_ref(size_t batch_id) override;
    metrix_float& get_grad_metrix_ref(size_t batch_id) override;
    metrix_float& get_output_metrix_ref(size_t batch_id) override;

    metrix_float* get_output_metrix_ptr() override;
    metrix_float* get_grad_metrix_ptr() override;
    metrix_float* get_input_metrix_ptr() override;

    float* get_input_data_ptr(size_t batch_id) override;
    float* get_grad_data_ptr(size_t batch_id) override;
    float* get_output_data_ptr(size_t batch_id) override;
public:
    // DEBUG funtions
    void __set_grads__(std::vector<std::vector<float>>& grads) override;
    void __set_inputs__(std::vector<std::vector<float>>& inputs) override;
    void __set_outputs__(std::vector<std::vector<float>>& outputs) override;

private:
    void count_grad();
    void count_grad(size_t batch_id);
    void count_grad(metrix_float* next_grad, int n);
    void count_grad(metrix_float* next_grad, size_t batch_id, int n);
    void clear_value() override;
    void clear_grad() override;
    void update(double lr) override;
    void update_mul(size_t group_index, size_t group_size);
    void print_param();
};

} // namespace dtensor