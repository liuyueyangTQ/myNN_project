#pragma once
#include<string>
#include<iostream>
#include<vector>
#include<cmath>
#include<cassert>
#include<chrono>
#include"enum_type.h"
#include"metrix.h"
#include"dtensor.h"
#include"ops.h"
namespace dtensor{
class tensor_base;
layer* layer_tool(int n, size_t batch_num, sub_type stp);
}
namespace nn{
using namespace dtensor;
class module_base{
protected:
    std::vector<dtensor_base*> inputs; // 模块的参数列表
    int batch_num, samples, groups;
    double lr;
public:
    module_base(int batch_num) : 
        batch_num(batch_num) , lr(0.001), samples(0), groups(0)
        {}
    virtual void forward(size_t batch_id) = 0;
    virtual void forward() = 0;
    virtual void backward(std::vector<float>& label, size_t batch_id, loss_type tp) = 0;
    virtual void backward(std::vector<std::vector<float>>& labels, loss_type tp) = 0;
    virtual void update_parameters(double learning_rate) = 0;
    virtual void clear_grad() = 0;
    virtual void clear_samples() = 0;
    void set_learning_rate(double lr);
};
class Linear_NN : public module_base{
private:
    dtensor_base *first_layer, *last_layer;
    std::vector<std::vector<float>> data, labels;
    std::vector<std::vector<std::vector<float>>> train_data, train_labels;
public:
    Linear_NN(int batch_num) : 
        first_layer(nullptr), last_layer(nullptr), module_base(batch_num)
        // train_data(new std::vector<std::vector<std::vector<float>>>), 
        // train_labels(new std::vector<std::vector<std::vector<float>>>), 
        {}
    void forward(size_t batch_id) override;
    void forward() override;
    void backward(std::vector<float>& label, size_t batch_id, loss_type tp) override;
    void backward(std::vector<std::vector<float>>& labels, loss_type tp) override;
    void update_parameters(double learning_rate) override;
    void clear_grad() override;
    void clear_samples() override;
    void add_layer(int num, sub_type tp);
    void get_train_data(std::vector<std::vector<float>>& data, std::vector<std::vector<float>>& labels);
    void reshuffle_data();
    void train_model(int epochs, double lr);
    void train_one_epoch(double lr);
    void check_net();
    void validate();
};



}