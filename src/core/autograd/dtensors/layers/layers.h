#pragma once

#include "dtensor_1D.h"

namespace dtensor {
    
class sigmoid: public layer {
public:
    sigmoid(int n, size_t batch_num) : layer(n, batch_num) {}
    void print_layer(bool inc_grad = false) override;
    void print_batches() override;
    sub_type get_layer_type() override;

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
    sub_type get_layer_type() override;
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
    sub_type get_layer_type() override;
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
    sub_type get_layer_type() override;
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
    sub_type get_layer_type() override;
private:
    void count_output() override;
    void count_output(size_t batch_id) override;
    void count_grad() override;
    void count_grad(size_t batch_id) override;
};

} // namespace dtensor