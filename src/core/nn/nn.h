#pragma once
#include<string>
#include<iostream>
#include<vector>
#include<cmath>
#include<cassert>
#include<chrono>
#include<thread>
#include"threadpool.h"
#include"enum_type.h"
#include"metrix.h"
#include"dtensor.h"
#include"ops.h"
namespace dtensor{
class tensor_base;
layer* layer_tool(int n, size_t batch_num, sub_type stp);
class ThreadPool;
}
namespace nn{
using namespace dtensor;
// 神经网络参数结构体（前后台共用）
struct NNParams {  
    nn_type model_type;  // 模型类型："LinearNN" / "LinearResnet"
    std::vector<int> layer_sizes; // 各层神经元数（如 [10, 20, 20, 5]）
    std::vector<dtensor::sub_type> layer_types; // 各层神经元数 (如 origin, relu, relu, softmax)
    size_t batch_size;
    size_t thread_num;
    int epochs;
    loss_type lstp;
    double lr;
    NNParams() : lstp(loss_type::cross_entropy) , epochs(1000), lr(0.001), thread_num(4), batch_size(4) {}
};

class module_base{
protected:
    std::vector<dtensor_base*> inputs; // 模块的参数列表
    std::vector<std::vector<float>> data, labels;
    std::vector<std::vector<std::vector<float>>> train_data, train_labels;
    int batch_num, samples, groups;
    double lr;
    ThreadPool* thread_pool;
public:
    module_base(int batch_num) : 
        batch_num(batch_num) , lr(0.001), samples(0), groups(0), thread_pool(nullptr)
        {}
    virtual void forward(size_t batch_id) = 0;
    virtual void forward() = 0;
    virtual void backward(std::vector<float>& label, size_t batch_id, loss_type tp) = 0;
    virtual void backward(std::vector<std::vector<float>>& labels, loss_type tp) = 0;
    virtual void update_parameters(double learning_rate) = 0;
    virtual void clear_grad() = 0;
    virtual void clear_samples() = 0;
    virtual void check_net() = 0;
    virtual void validate() = 0;
    virtual void set_input_value(std::vector<std::vector<float>>& data) = 0;
    void set_learning_rate(double lr);
    void get_train_data(std::vector<std::vector<float>>& data, std::vector<std::vector<float>>& labels);
    void reshuffle_data();
    void train_one_epoch(double lr);
    void train_model(int epochs, double lr);
    void set_thread_pool(ThreadPool* pool);
    virtual void reset_count() = 0; 
    virtual void print_all_layers(size_t batch_id, bool inc_grad = false) = 0;
};
class Linear_NN : public module_base {
private:
    dtensor_base *first_layer, *last_layer; //*cur_layer;
public:
    Linear_NN(int batch_num) : 
        first_layer(nullptr), last_layer(nullptr), module_base(batch_num)
        {}
    void forward(size_t batch_id) override;
    void forward() override;
    void backward(std::vector<float>& label, size_t batch_id, loss_type tp) override;
    void backward(std::vector<std::vector<float>>& labels, loss_type tp) override;
    void set_input_value(std::vector<std::vector<float>>& data) override;
    void update_parameters(double learning_rate) override;
    void clear_grad() override;
    void clear_samples() override;
    void add_layer(int num, sub_type tp);
    void validate() override;
    void check_net() override;
    void reset_count() override;
    void print_count_n();
    void print_all_layers(size_t batch_id, bool inc_grad = false);
#ifdef USE_DEBUG
    void print_cur_layer() ;
#endif
};

class Linear_Resnet : public module_base {
private:
    dtensor_base *first_layer, *last_layer; //*cur_layer;
public:
    Linear_Resnet(int batch_num) :
        first_layer(nullptr), last_layer(nullptr), module_base(batch_num) 
        {}
    void add_layer(int num, sub_type tp);
    void add_res_layer(int num, sub_type tp);
    void forward(size_t batch_id) override;
    void forward() override;
    void backward(std::vector<float>& label, size_t batch_id, loss_type tp) override;
    void backward(std::vector<std::vector<float>>& labels, loss_type tp) override;
    void set_input_value(std::vector<std::vector<float>>& data) override;
    void update_parameters(double learning_rate) override;
    void clear_grad() override;
    void clear_samples() override;
    void validate() override;
    void check_net() override;
    void reset_count() override;
    void print_count_n();
    void print_all_layers(size_t batch_id, bool inc_grad = false);
    void train_model_multi_thread(int epochs, double lr, int thread_num);
    void train_model_multi_thread_with_pool(int epochs, double lr, int thread_num);
    
#ifdef USE_DEBUG
    void print_cur_layer() ;
#endif
};

void run_model(const NNParams& params);

}