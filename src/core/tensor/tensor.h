#pragma once
#include<iostream>
#include<vector>
#include<cmath>
#include<cassert>
#include<string>
#include "buffer_util.h"
#include"metrix.h"
//#include "ops.h"

//#include"model.h"

using _size = std::pair<size_t,size_t>;

namespace base{ //先声明一下 metrix.h 里面提供的函数

class metrix_float;

float* _matmul(metrix_float &m1, metrix_float &m2);

void _matmul(metrix_float &m1, metrix_float &m2,float* data);

void _matmul(metrix_float &m1, metrix_float &m2, bool t1, bool t2, float* data);

float* _alloc_data(metrix_float &m1, metrix_float &m2);

_size _get_size(metrix_float &m1, metrix_float &m2);

} // namespace base

namespace tensor{
class op;

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

class layer;
class op;
class Linear_NN;
using namespace base;


class tensor_base{
protected:
    bool is_param;
    bool lock_grad;
    float lr;
    tensor_type tstp;
    size_t batch_num;
    op* op_next; // 指向下一个操作
    op* op_last; // 指向上一个操作
public:
    tensor_base(bool is_param = false, bool lock_grad = false, tensor_type type = tensor_type::common, size_t batch_num = 1) 
        : is_param(is_param), lock_grad(lock_grad), lr(0.0001), tstp(type),
          op_next(nullptr), op_last(nullptr), batch_num(batch_num)
        {}
    //tensor(metrix_float &m) : w(m),shape(m.shape),lock_grad(true),is_pram(false) {}
    inline void disgrad() { lock_grad = true; }
    virtual void forward() = 0;
    virtual void forward(size_t batch_id) = 0;

    virtual void backward() = 0;  //用于中间层
    virtual void backward(size_t batch_id) = 0; //用于中间层

    //virtual void backward_mul() = 0; // 之后在考虑

    virtual void backward(float* next_grad) = 0;  //用于尾部层 // 可额外添加
    virtual void backward(float* next_grad, size_t batch_id) = 0; //用于尾部层 // 可额外添加

    virtual void backward_grad() = 0;
    virtual void backward_grad(size_t batch_id) = 0;
    
    // 张量运算
    // virtual void do_ops(tensor_base *p, ops op) = 0;


    virtual void add(tensor_base *p) {}
    virtual void sub(tensor_base *p) {}
    virtual void matmul(tensor_base *p) {}
    virtual void print(bool inc_grad = false) {}
    // virtual void print_batches() {}

    void print_info();

    friend class op;

};




class common_tensor: public tensor_base{
private:
    friend class base::metrix_float;
    metrix_float* w;
    metrix_float* grad;
    _size shape;
    std::vector<tensor_base*> next;
    std::vector<tensor_base*> last;
    // std::vector<std::pair<tensor_base*,tensor::ops> > next1; //算子和矩阵的组合
    // std::vector<tensor::ops> forward_ops;
    // std::vector<tensor::ops> back_ops;
public:
    common_tensor(_size shape) : shape(shape), w(new metrix_float(shape)), grad(new metrix_float(shape)), tensor_base(false, false, tensor_type::common) {}
    void add(tensor_base *p)override; //不管
        // 可能导致死循环
        // tensor_base* temp = new tensor_base(this->is_param,this->lock_grad);
        // this->next->push_back(temp);
        
        // p->next->push_back(temp); // p为基类指针是否拥有next成员？是否要强转
        

        // temp->last->push_back(this);
        // temp->last->push_back(p);
        // temp->back_ops.push_back(ops::add);
        // temp->back_ops.push_back(ops::add);

    
    void sub(tensor_base *p)override;
    void matmul(tensor_base *p)override;

};


class simple_tensor: private tensor_base{

};




class tensor2D_float: private tensor_base{
private:
    metrix_float* weight; //    size(z) * size(x) 的矩阵 (输入为 x， 输出为 z)
    metrix_float* grad; 
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
        tensor_base(true, false, tensor_type::tensor2D, batch_num), //不锁梯度
        weight(new metrix_float(shape)),
        grad(new metrix_float(shape)),
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
        tensor_base(true, false, tensor_type::tensor2D, batch_num), //不锁梯度
        weight(new metrix_float(shape, init_type)),
        grad(new metrix_float(shape)),
        shape(shape),
        //last(nullptr),
        pMemory(new void*), // 一定要分配初始指针！！！！！
        next(nullptr),
        is_identity(false) 
    {
        batch_grad = this->_allocdata();
    }

    tensor2D_float(tensor2D_float const &t) = delete; // 不提供实现
    // void forward() override {}
    ~tensor2D_float() 
    {
        this->_release_data(this->batch_grad);
#ifdef USE_DEBUG 
        auto p = this;
        std::cout << "successfully released the " << index << "-th weight metrix data!" << " (data address: " << p << ")" << std::endl;   
        wm_index--;
#endif  
    }
    void print(bool inc_grad = false) override;
    void print_grad();
private:

    void forward() override;
    void forward(size_t batch_id) override;
    void backward() override;
    // void backward_mul() override;
    void backward(size_t batch_id) override;
    void backward_grad() override;
    void backward_grad(size_t batch_id) override;

    void backward(float* next_grad);  //用于尾部层 // 可额外添加
    void backward(float* next_grad, size_t batch_id); //用于尾部层 // 可额外添加

    // void do_ops(tensor_base *p, ops op) override;


    void count_grad();
    void count_grad(size_t batch_id);
    void count_grad(metrix_float* next_grad, int n);
    void count_grad(metrix_float* next_grad, size_t batch_id, int n);
    void update();
    void update_mul(size_t group_index, size_t group_size);
    void print_param();
};


class layer: public tensor_base{
protected:
    metrix_float* input;
    //   input  = w * last->output  + b
    metrix_float* output;  
    //   激活层输出值
    metrix_float* grad;  // grad 是这一层的输入对应的grad
    std::vector<tensor2D_float*> w_next; // 旁边的权重矩阵
    std::vector<tensor2D_float*> w_last; // 旁边的权重矩阵
    metrix_float* b;   // 偏置项  z = w * x + b
    metrix_float* batch_input;
    metrix_float* batch_output;
    metrix_float* batch_grad; // grad 是这一层的 @brief 输入 对应的grad
    
    void** pMemory; // 辅助变量
    std::vector<layer*> next;  //指向所有接下来的层
    std::vector<layer*> last;  //指向所有前面的层
    bool layer_norm;
    bool batch_norm;
    int n;
    inline void add_linear(tensor2D_float* out_w) {
        //assert(this->w_next == nullptr);
        this->w_next.push_back(out_w);
    }

private:
    metrix_float* _alloc_m_data(size_t bias);

    void _release_data(metrix_float* myArray, size_t bias);

public:
    inline void add_wnext(tensor2D_float* p) {
        this->w_next.push_back(p);
    }
    inline void add_wlast(tensor2D_float* p) {
        this->w_last.push_back(p);
    }
    inline void set_layer_next(layer* p) {
        this->next.push_back(p);
    }
    inline int get_shape() {
        return this->n;
    }
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
    inline layer* get_next() {
        return this->next[0];
    }
    void print_param();
    void print_grad();
    friend class tensor2D_float;
    friend class Linear_NN;
    friend class op;
#ifdef USE_DEBUG 
    int index;
    static int layer_index;
    inline int get_layer_index() {
        return this->index;
    }
#endif 
    layer(int n, size_t batch_num, layer* last = nullptr) : 
            tensor_base(false, false, tensor_type::layer, batch_num), 
            n(n),
            grad(new metrix_float(n,1)), 
            //next(nullptr),  //梯度是前一层输入的梯度
            //last(last),
            b(new metrix_float({n, 1},"simple")),
            //w_next(nullptr),  // 初始化时，没有邻接权重矩阵
            layer_norm(false),  // 对 @ 输入 进行归一化
            batch_norm(false),  // 对 @ 输入 进行归一化
            pMemory(new void*[3])  // 分配初始指针
    {
#ifdef USE_DEBUG 
        layer_index++;
        this->index = layer_index;
#endif 
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
        std::cout << "successfully released the " << index << "-th layer data!" << " (data address: " << p << ")" << std::endl;   
        layer_index--;
#endif 
    }

    void print(bool inc_grad = false) override;
    virtual void print_layer(bool inc_grad = false) = 0;
    virtual void print_batches() = 0;
private: 
    void _set_batch();
    void forward() override;
    void forward(size_t batch_id) override;

    void get_input(std::vector<std::vector<float>>& samples);
    void get_input(std::vector<float>& sample, size_t batch_id);
    // void backward() override; // 不需要
    void backward() override; // 通过下一层的梯度计算传递损失 
    void backward(size_t batch_id) override; // 通过下一层的 第 batch_id 个样本 计算梯度传递损失

    void backward(float* next_grad) override; // 用于尾部层  通过损失函数传递的梯度
    void backward(float* next_grad, size_t batch_id) override; // 用于尾部层  通过损失函数传递的 第 batch_id 个样本 的梯度

    void backward_grad() override; // 计算所有样本 的梯度并传递至其之前的层！！！
    void backward_grad(size_t batch_id) override; // 计算 第 batch_id 个样本 的梯度并传递至其之前的层！！！（重要，扩展至resnet的关键！）


    void count_loss_grad(std::vector<float>& label, loss_type loss_tp); //用于通过标签传递损失函数
    void count_loss_grad(std::vector<float>& label, loss_type loss_tp, size_t batch_id); //用于通过标签传递损失函数
    void update(); // 更新参数 b
    void update_mul(size_t group_index, size_t group_size); // 并行 更新参数 b ！！


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

layer* layer_tool(int n, size_t batch_num, layer_type ltp);


// class concate_tensor{ //设计一个能将多个tensor连接起来的类，其前向传播和反向传播都会依次调用各个子tensor的前向传播和反向传播
// public:
//     std::vector<tensor_base*> parts;
// };

} // namespace tensor

