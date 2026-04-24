#pragma once
#include<iostream>
#include<vector>
#include<cmath>
#include<cassert>
#include<string>
#include<map>
#include"enum_type.h"
#include"ops.h"
#include"buffer_util.h"
#include"metrix.h"
//#include "ops.h"

//#include"model.h"

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
class layer;
class Linear_NN;
using namespace base;

#ifdef USE_DEBUG 
void get_info(std::string input);
#endif

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
    //tensor(metrix_float &m) : w(m),shape(m.shape),lock_grad(true),is_pram(false) {}
    virtual void set_input_value(float* data, size_t batch_id) = 0;
    virtual void set_input_value(std::vector<std::vector<float>>& data) = 0;
    inline void disgrad() { lock_grad = true; }

    void reset_count();

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
};




class tensor2D_float: public dtensor_base {
private:
    metrix_float* weight; //    size(z) * size(x) 的矩阵 (输入为 x， 输出为 z)
    // 用于记录前后的层关系，非必须
    layer* x;  // 即 x ,  z = w * x + b
    layer* next;// 即 z 
    void** pMemory; // 辅助变量
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
    inline void set_layer_x(layer* p) {
        this->x = p;
    }
    inline void set_layer_next(layer* p) {
        this->next = p;
    }
    inline metrix_float* get_weight() {
        return this->weight;
    }
    tensor2D_float(_size shape, int batch_num) :  //全零初始化
        dtensor_base(true, false, tensor_type::tensor2D, batch_num, shape.first * shape.second), //不锁梯度
        weight(new metrix_float(shape)),
        shape(shape),
        pMemory(new void*), // 一定要分配初始指针！！！！！
        //last(nullptr),
        x(nullptr),
        next(nullptr),
        is_identity(false) 
    {
        std::cout << "     initializing tensor2D_float...\n";
        batch_grad = this->_allocdata();
        std::cout << "     successfully allocated data!\n";
    }
    tensor2D_float(_size shape, int batch_num, init_type init_type) : // 随机初始化
        dtensor_base(true, false, tensor_type::tensor2D, batch_num, shape.first * shape.second), //不锁梯度
        weight(new metrix_float(shape, init_type)),
        shape(shape),
        //last(nullptr),
        pMemory(new void*), // 一定要分配初始指针！！！！！
        next(nullptr),
        is_identity(false) 
    {
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


class layer: public dtensor_base{
protected:
    //   激活层输出值
    op* op_next; // 指向下一个操作
    op* op_last; // 指向上一个操作

    metrix_float* batch_input;
    metrix_float* batch_output;
    metrix_float* batch_grad; // grad 是这一层的 @brief 输入 对应的grad
    metrix_float* b;   // 偏置项  z = w * x + b
    void** pMemory; // 辅助变量
    std::vector<size_t> shape;



private:
    metrix_float* _alloc_m_data(size_t bias);

    void _release_data(metrix_float* myArray, size_t bias);

public:
    void set_input_value(float* data, size_t batch_id) override;
    void set_input_value(std::vector<std::vector<float>>& data) override;
    std::vector<size_t> get_shape() override;
    inline int get_batch_num() {
        return this->batch_num;
    }
    inline metrix_float* get_batch_output() {
        return this->batch_output;
    }
    inline metrix_float* get_batch_input() {
        return this->batch_input;
    }
    inline metrix_float* get_batch_grad() {
        return this->batch_grad;
    }
    void print_param();
    void print_grad();
    void clear_grad() override;
    void clear_value() override;
    float* get_bias_data();
    friend class tensor2D_float;
    friend class op;
#ifdef USE_DEBUG 
    int index;
    static int layer_index;
    inline int get_layer_index() {
        return this->index;
    }
#endif 
    layer(int n, size_t batch_num, layer* last = nullptr) : 
            dtensor_base(false, false, tensor_type::layer, batch_num, n), 
            //next(nullptr),  //梯度是前一层输入的梯度
            //last(last),
            b(new metrix_float({n, 1},"simple")),
            //w_next(nullptr),  // 初始化时，没有邻接权重矩阵
            pMemory(new void*[3])  // 分配初始指针
    {
#ifdef USE_DEBUG 
        layer_index++;
        this->index = layer_index;
#endif 
        this->shape.push_back(n);
        this->shape.push_back(1);
        assert(this->shape.size() == 2);
        // if(last)
        //     this->last.push_back(last); // 不太需要
        _set_batch();
        if(!last) {

        }
        else {
          //  input = last->output;
        }
    }
    virtual ~layer() {
        if(this->batch_num == 1)
            return;
        try{ // 尝试调用释放内存

            _release_data(this->batch_grad, 0);
            _release_data(this->batch_input, 1);
            _release_data(this->batch_output, 2); 
        } catch (const char* err) { // 捕获字符串类型异常
            std::cout << "exception when releasing layer!: " << err << std::endl;
        } catch (...) { // 兜底捕获其他异常
            std::cout << "Unkown error when releasing layer! " << std::endl;
        } 
#ifdef USE_DEBUG 
        auto p = this;
        //std::cout << "successfully released the " << index << "-th layer data!" << " (data address: " << p << ")" << std::endl;   
        layer_index--;
#endif 
    }

    void _print_val(size_t batch_id = 0) override;
    void _print_grad(size_t batch_id = 0) override;
    virtual void print_layer(bool inc_grad = false) = 0;
    virtual void print_batches() = 0;
private: 
    void _set_batch();
    void _forward() override;
    void _forward(size_t batch_id) override;

    void get_input(std::vector<std::vector<float>>& samples);
    void get_input(std::vector<float>& sample, size_t batch_id);
    // void _backward() override; // 不需要
    void _backward() override; // 通过下一层的梯度计算传递损失 
    void _backward(size_t batch_id) override; // 通过下一层的 第 batch_id 个样本 计算梯度传递损失
#ifdef USE_DEBUG 
    void _forward_D() override;
    void _forward_D(size_t batch_id) override;

    void _backward_D() override;  //用于中间层
    void _backward_D(size_t batch_id) override; //用于中间层
#endif   
    void backward(float* next_grad) override; // 用于尾部层  通过损失函数传递的梯度
    void backward(float* next_grad, size_t batch_id) override; // 用于尾部层  通过损失函数传递的 第 batch_id 个样本 的梯度

    void backward_grad() override; // 计算所有样本 的梯度并传递至其之前的层！！！
    void backward_grad(size_t batch_id) override; // 计算 第 batch_id 个样本 的梯度并传递至其之前的层！！！（重要，扩展至resnet的关键！）

    // void count_loss_grad(std::vector<std::vector<float>>& label, loss_type loss_tp) override; //用于通过标签传递损失函数
    void count_loss_grad(std::vector<float>& label, loss_type loss_tp, size_t batch_id) override; //用于通过标签传递损失函数

    void update(double lr) override; // 更新参数 b
    void update_mul(size_t group_index, size_t group_size); // 并行 更新参数 b ！！

    metrix_float& get_input_metrix_ref(size_t batch_id) override;
    metrix_float& get_grad_metrix_ref(size_t batch_id) override;
    metrix_float& get_output_metrix_ref(size_t batch_id) override;

    metrix_float* get_output_metrix_ptr() override;
    metrix_float* get_grad_metrix_ptr() override;
    metrix_float* get_input_metrix_ptr() override;

    float* get_input_data_ptr(size_t batch_id) override;
    float* get_grad_data_ptr(size_t batch_id) override;
    float* get_output_data_ptr(size_t batch_id) override;

    virtual void count_output() = 0;
    virtual void count_output(size_t batch_id) = 0;

    virtual void count_grad() = 0;
    virtual void count_grad(size_t batch_id) = 0;
};


class sigmoid: public layer {
public:
    sigmoid(int n, size_t batch_num) : layer(n, batch_num) {}
    void print_layer(bool inc_grad = false) override;
    void print_batches() override;

private:


    inline int get_batch_num() {
        return this->batch_num;
    }
    inline metrix_float* get_batch_output() {
        return this->batch_output;
    }
    inline metrix_float* get_batch_input() {
        return this->batch_input;
    }
    inline metrix_float* get_batch_grad() {
        return this->batch_grad;
    }


    void count_output() override;
    void count_output(size_t batch_id) override;
    void count_grad() override;
    void count_grad(size_t batch_id) override;
};

class layer_norm : public layer {
public:
    layer_norm(int n, size_t batch_num) : layer(n, batch_num), 
        min_dim(new float[batch_num]()), diff(new float[batch_num]()) {}
    void print_layer(bool inc_grad = false) override;
    void print_batches() override;
private:
    float *min_dim, *diff;
    void count_output() override;
    void count_output(size_t batch_id) override;
    void count_grad() override;
    void count_grad(size_t batch_id) override;
};

class relu: public layer{
public:
    relu(int n, size_t batch_num) : layer(n, batch_num) {}
    void print_layer(bool inc_grad = false) override;
    void print_batches() override;
private:
    void count_output() override;
    void count_output(size_t batch_id) override;
    void count_grad() override;
    void count_grad(size_t batch_id) override;
};

class softmax: public layer{
public:
    softmax(int n, size_t batch_num) : layer(n, batch_num) {}
    void print_layer(bool inc_grad = false) override;
    void print_batches() override;
private:
    float T = 1;
    void count_output() override;
    void count_output(size_t batch_id) override;
    void count_grad() override;
    void count_grad(size_t batch_id) override;
};


class origin: public layer {
public:
    origin(int n, size_t batch_num) : layer(n, batch_num) {}
    void print_layer(bool inc_grad = false) override;
    void print_batches() override;
private:
    void count_output() override;
    void count_output(size_t batch_id) override;
    void count_grad() override;
    void count_grad(size_t batch_id) override;
};

layer* layer_tool(int n, size_t batch_num, sub_type stp);

// dtensor_base* tensor_tool(_size shape, size_t batch_num, tensor_type tstp);

// class concate_tensor{ //设计一个能将多个tensor连接起来的类，其前向传播和反向传播都会依次调用各个子tensor的前向传播和反向传播
// public:
//     std::vector<tensor_base*> parts;
// };

} // namespace tensor

