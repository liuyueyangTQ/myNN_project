#pragma once
#include "dtensor.h"

namespace dtensor {

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
    virtual sub_type get_layer_type() = 0;
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
            delete b;
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
public: 
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

public:
    // DEBUG funtions
    void __set_grads__(std::vector<std::vector<float>>& grads) override;
    void __set_inputs__(std::vector<std::vector<float>>& inputs) override;
    void __set_outputs__(std::vector<std::vector<float>>& outputs) override;
};

} // namespace dtensor
