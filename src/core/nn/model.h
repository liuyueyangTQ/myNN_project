#pragma once
#include<iostream>
#include<string>
#include<vector>
#include<map>
#include<chrono>
#include"metrix.h"
#include"tensor.h"

//#define USE_THREAD
#ifdef USE_THREAD
    #include"threadpool.h"
#endif

using _size = std::pair<size_t,size_t>;
namespace base{
class metrix_float;
void _matmul(metrix_float &m1, metrix_float &m2, bool t1, bool t2, float* data);

float* _matmul(metrix_float &m1, metrix_float &m2);

void _matmul(metrix_float &m1, metrix_float &m2, float* data);

void _matmul_add(metrix_float& m1, metrix_float& m2, bool t1, bool t2, float* data);

float* _alloc_data(metrix_float &m1, metrix_float &m2);

_size _get_size(metrix_float &m1, metrix_float &m2);

}
using namespace base;

namespace tensor{

// std::map<std::string, >

class layer;
class tensor2D_float;
class sigmoid;
class layer_norm;
class relu;
#ifdef USE_THREAD
    class ThreadPool;
#endif

class Linear_NN{
private:
    layer* first_layer;
    layer* last_layer;
    layer* cur_layer;
    std::vector<layer*> layers;
    int batch_num;
    loss_type loss_tp;
    std::vector<std::vector<float>>* data;
    std::vector<std::vector<float>>* labels;
    std::vector<std::vector<std::vector<float>>>* train_data;
    std::vector<std::vector<std::vector<float>>>* train_labels;
    std::vector<int> layer_shape;
    float lr;
    int layer_num;
    long long cost_ms;
    void _add_layer(int n, layer_type tp); // 仅添加层，而不添加 线性矩阵
    layer* layer_tool(int n, layer_type tp = layer_type::sigmoid);
    void _add_linear(int n, layer_type tp); // n 是添加层的神经元数
public:
    Linear_NN(int batch_num = 1) : 
        first_layer(nullptr), cur_layer(nullptr), last_layer(nullptr), batch_num(batch_num), layer_num(0), 
        data(nullptr), labels(nullptr),
        // 要预先分配内存！！不然会发生段错误
        train_data(new std::vector<std::vector<std::vector<float>>>), train_labels(new std::vector<std::vector<std::vector<float>>>),
        cost_ms(0),
        lr(0.001) {}
    Linear_NN(std::vector<std::vector<float>>& data, int batch_num = 1) : 
        first_layer(nullptr), cur_layer(nullptr), last_layer(nullptr), batch_num(batch_num), layer_num(0),
        data(&data), labels(nullptr),
        train_data(new std::vector<std::vector<std::vector<float>>>), train_labels(new std::vector<std::vector<std::vector<float>>>),
        cost_ms(0),
        lr(0.001) {}

    void add_layer(int n, layer_type tp, bool req_wm = false);

    void add_block(int layer1, int layer2); 
    void add_residual_block(int layer1, int layer2);

    void forward(std::vector<std::vector<float>>& samples);
    void forward(std::vector<float>& sample, size_t batch_id);
#ifdef USE_THREAD
    void forward(std::vector<std::vector<float>>& samples, ThreadPool& pool);
    void backward(std::vector<std::vector<float>>& labels, ThreadPool& pool);
#endif
    void backward(std::vector<std::vector<float>>& label);
    void update();
    void backward(std::vector<float>& label, size_t batch_id);
    void clear_grad();
    void clear_samples();
    std::vector<std::vector<float>> get_layer_output();
    std::vector<std::vector<std::vector<float>>> get_weight_metrix();

    // void forward(){ // 弃用
    //     assert(this->cur_layer == this->first_layer);
    //     while( this->cur_layer!=this->last_layer ){
    //         this->cur_layer->forward();
    //         this->cur_layer = this->cur_layer->next;
    //     }
    // }



   // void initialize(){ // 弃用
        // assert(this->cur_layer == this->first_layer);
        // float* dt = this->first_layer->input->data;
        // assert(this->cur_layer->get_shape() == (*(this->data))[0].size());
        // for(int i=0;i<this->cur_layer->get_shape(); ++i){
        //     *(dt + i) = (*(this->data))[0][i];
        // }

        // return;
 //   }
    void set_train_data(std::vector<std::vector<float>>& data, std::vector<std::vector<float>>& labels, int epochs, loss_type loss_tp, int batches, double lr);  
    void train(int epochs = 2000, loss_type tp = loss_type::cross_entropy, int batches = 4, double lr = 0.001);
    void validate();
    void reshuffle_data();
    void split_data();
    void split_labels();

    void check_net();
    void print_parameters();
    float count_loss(float* otpt, std::vector<float>& label);
    float count_batch_loss(size_t batch_index);
    void print_grad();
    long long get_model_cost_time();
#ifdef USE_THREAD
    void multithread_train(int epochs = 2000, loss_type tp = loss_type::cross_entropy, int batches = 4, double lr = 0.001);
#endif

};








}
