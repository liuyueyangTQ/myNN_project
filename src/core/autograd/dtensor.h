#pragma once
#include<iostream>
#include<vector>
#include<cmath>
#include<cassert>
#include<string>
#include<map>
#include"enum_types.h"
#include"ops.h"
#include"buffer_util.h"
#include"metrix.h"

using _size = std::pair<size_t,size_t>;

namespace dtensor{
    class dtensor_base;
    class op;
} // namespace dtensor

namespace base{ //先声明一下 metrix.h 里面提供的函数

class metrix_float;

float* _matmul(metrix_float &m1, metrix_float &m2);

void _matmul(metrix_float &m1, metrix_float &m2,float* data);

void _matmul(metrix_float &m1, metrix_float &m2, bool t1, bool t2, float* data);

float* _alloc_data(metrix_float &m1, metrix_float &m2);

_size _get_size(metrix_float &m1, metrix_float &m2);

size_t get_tensor_size(dtensor::dtensor_base* t);


} // namespace base


namespace dtensor{
using namespace base;

class dtensor_base{
    friend class op;
    friend size_t base::get_tensor_size(::dtensor::dtensor_base* t);
    friend _size _get_size(metrix_float &m1, metrix_float &m2);
protected:
    bool is_param;
    bool lock_grad;
    size_t n;
    size_t batch_num;
    std::vector<op*> op_next; // 指向下一个操作
    std::vector<op*> op_last; // 指向上一个操作
    size_t count_n, *temp_n; // dtensor指向的count_n个算子，这些算子剩temp_n个完成传递
    bool *have_forwarded;
    bool have_updated, have_printed;
#ifdef USE_DEBUG 
    //std::vector<std::pair<std::pair<dtensor*, dtensor*>, size_t>> layers;
    //std::vector<std::pair<op*, size_t>> opses;
#endif 
public:
    tensor_type tstp;
    dtensor_base(bool is_param = false, bool lock_grad = false, tensor_type type = tensor_type::common, size_t batch_num = 1, size_t n = 0) 
        : is_param(is_param),lock_grad(lock_grad), tstp(type), batch_num(batch_num), n(n),
        count_n(0), temp_n(new size_t[batch_num]()), have_forwarded(new bool[batch_num]()), have_updated(false), have_printed(false)
        {}
    ~dtensor_base() {
        delete[] temp_n;
        delete[] have_forwarded;
    }
    //tensor(metrix_float &m) : w(m),shape(m.shape),lock_grad(true),is_pram(false) {}
    virtual void set_input_value(float* data, size_t batch_id) = 0;
    virtual void set_input_value(std::vector<std::vector<float>>& data) = 0;
    inline void disgrad() { lock_grad = true; }

    void reset_count();

    // 调试性质的函数
    void __set_isparam_lockgrad__(bool is_param, bool lock_grad);

    void forward();
    void forward(size_t batch_id);
    void backward();  //用于中间层
    void backward(size_t batch_id); //用于中间层

    virtual void _forward() = 0;
    virtual void _forward(size_t batch_id) = 0;

    virtual void _backward() = 0;  //用于中间层
    virtual void _backward(size_t batch_id) = 0; //用于中间层

    //virtual void backward_mul() = 0; // 之后在考虑

    virtual void backward(float* next_grad) = 0;  //用于尾部层 // 可额外添加
    virtual void backward(float* next_grad, size_t batch_id) = 0; //用于尾部层 // 可额外添加

    virtual void backward_grad() = 0;
    virtual void backward_grad(size_t batch_id) = 0;
    
    void count_loss_grad(std::vector<std::vector<float>>& label, loss_type loss_tp);
    virtual void count_loss_grad(std::vector<float>& label, loss_type loss_tp, size_t batch_id) = 0;
#ifdef USE_DEBUG 

    void forward_D();
    void forward_D(size_t batch_id);
    void backward_D();  //用于中间层
    void backward_D(size_t batch_id); //用于中间层

    virtual void _forward_D() = 0;
    virtual void _forward_D(size_t batch_id) = 0;

    virtual void _backward_D() = 0;  //用于中间层
    virtual void _backward_D(size_t batch_id) = 0; //用于中间层

#endif 
    virtual std::vector<size_t> get_shape() = 0;

    virtual metrix_float& get_input_metrix_ref(size_t batch_id) = 0;
    virtual metrix_float& get_grad_metrix_ref(size_t batch_id) = 0;
    virtual metrix_float& get_output_metrix_ref(size_t batch_id) = 0;

    virtual metrix_float* get_input_metrix_ptr() = 0;
    virtual metrix_float* get_grad_metrix_ptr() = 0;
    virtual metrix_float* get_output_metrix_ptr() = 0;

    virtual float* get_input_data_ptr(size_t batch_id) = 0;
    virtual float* get_grad_data_ptr(size_t batch_id) = 0;
    virtual float* get_output_data_ptr(size_t batch_id) = 0;

    virtual void clear_grad() = 0;
    virtual void clear_value() = 0;
    virtual void update(double lr) = 0;

    // DEBUG functions
    virtual void __set_grads__(std::vector<std::vector<float>>& grads) = 0;
    virtual void __set_inputs__(std::vector<std::vector<float>>& inputs) = 0;
    virtual void __set_outputs__(std::vector<std::vector<float>>& outputs) = 0;

    inline void add_nopp(dtensor_base* p, op* op) {
        this->op_next.push_back(op);
    }
    inline void add_lopp(dtensor_base* p, op* op) {
        this->op_last.push_back(op);
    }
    inline bool is_parameter() {
        return this->is_param;
    }
    dtensor_base* get_next();
    inline op* get_op_next() {
        if(!op_next.size())
            return nullptr;
        return this->op_next[0];
    }
    inline op* get_op_last() {
        if(!op_last.size())
            return nullptr;
        return this->op_last[0];
    }
    inline size_t get_n() {
        return this->n;
    }
    inline size_t get_count_n() {
        return this->count_n;
    }
    void print(bool inc_grad = false, size_t batch_id = 0, bool rec = false);
    virtual void _print_val(size_t batch_id = 0) = 0;
    virtual void _print_grad(size_t batch_id = 0) = 0;
    // virtual void print_batches() {}

    void print_info();
    inline size_t get_batch_num() {
        return this->batch_num;
    }
    inline void set_type(tensor_type type) {
        this->tstp = type;
    }
    inline void set_op_next(op* p) {
        assert(!op_next.size());
        this->op_next.push_back(p);
    }
    inline void set_op_last(op* p) {
        assert(!op_last.size());
        this->op_last.push_back(p);
    }

};

// dtensor_base* tensor_tool(_size shape, size_t batch_num, tensor_type tstp);

// class concate_tensor{ //设计一个能将多个tensor连接起来的类，其前向传播和反向传播都会依次调用各个子tensor的前向传播和反向传播
// public:
//     std::vector<tensor_base*> parts;
// };

} // namespace dtensor

