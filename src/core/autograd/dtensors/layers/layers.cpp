#include "layers.h"

namespace dtensor { 

// 返回layer类型
sub_type origin::get_layer_type() {
    return sub_type::origin;
}
sub_type sigmoid::get_layer_type() {
    return sub_type::sigmoid;
}
sub_type relu::get_layer_type() {
    return sub_type::relu;
}
sub_type softmax::get_layer_type() {
    return sub_type::softmax;
}
// count_output() 和 count_grad() 的重载版本，分别用于处理整个batch和单个样本的情况
void origin::count_output() {  
    for(int i = 0; i < this->batch_num; ++i) {
        this->count_output(i);
    }
}
void origin::count_output(size_t batch_id) {  
    float* otpt = (*(this->batch_output + batch_id)).data;
    float* ipt = (*(this->batch_input + batch_id)).data;
    // float* bpt = this->b->data; //origin不需要b
    for(int i = 0; i < this->n; ++i) {
        otpt[i] = ipt[i]; //+ bpt[i];
    }
}
void origin::count_grad() {  
    // 直接省略，batch_grad中存储的即为grad值，无需再计算
}
void origin::count_grad(size_t batch_id) {  
    // 直接省略，batch_grad中存储的即为grad值，无需再计算
}
void sigmoid::count_output() {  
    for(int i = 0; i < this->batch_num; ++i) {
        this->count_output(i);
    }
}
void sigmoid::count_output(size_t batch_id) {  
#ifdef USE_DEBUG 
    assert(this->batch_output + batch_id);
    assert(this->batch_input + batch_id);
#endif 
    float* otpt = (this->batch_output + batch_id)->data;  // 计算 batch 里面第 batch_id 个样本对应的输出
    float* ipt = (this->batch_input + batch_id)->data;
    float* bpt = this->b->data;
    for(int i = 0; i < this->n; ++i) {
        otpt[i] = 1 / (1 + exp(- ipt[i] - bpt[i]));
    }
}
void sigmoid::count_grad() {  
    for(int i = 0; i < this->batch_num; ++i) {
        this->count_grad(i);
    }
}
void sigmoid::count_grad(size_t batch_id) {
#ifdef USE_DEBUG 
    assert(this->batch_grad + batch_id);
    assert(this->batch_output + batch_id);
    assert(this->batch_input + batch_id);
#endif     
    //  output = 1 / (1 + e^(-x))  ;  input + b = 
            // grad = f(x)(1-f(x))
    float* g = (this->batch_grad + batch_id)->data;
    float* otpt = (this->batch_output + batch_id)->data;
    for(int i = 0; i < this->n; ++i) {
        g[i] = otpt[i] * (1 - otpt[i]) * g[i];
    }  
}

void relu::count_output() {
    for(int i = 0; i < this->batch_num; ++i) {
        this->count_output(i);
    }
}
void relu::count_output(size_t batch_id) {
    float* otpt = (*(this->batch_output + batch_id)).data;
    float* ipt = (*(this->batch_input + batch_id)).data;
    float* bpt = this->b->data;
    for(int i = 0; i < this->n; ++i) {
        otpt[i] = std::max(ipt[i] + bpt[i], (float)0);
    }
}
void relu::count_grad() {  
    for(int i = 0; i < this->batch_num; ++i) {
        this->count_grad(i);
    }
}
void relu::count_grad(size_t batch_id) {
#ifdef USE_DEBUG 
    assert(this->batch_grad + batch_id);
    assert(this->batch_output + batch_id);
    assert(this->batch_input + batch_id);
#endif  
    float* g = (this->batch_grad + batch_id)->data;
    float* otpt = (this->batch_output + batch_id)->data;
    for(int i = 0; i < this->n; ++i) {
        g[i] = (otpt[i] > 0) ? g[i] : 0; // 要乘上 next_grad
    }  
}

void softmax::count_output() {  
    for(int i = 0; i < this->batch_num; ++i) {
        this->count_output(i);
    }
}
void softmax::count_output(size_t batch_id) {  
    std::vector<float> otpt_exp(this->n, 0);
    float* otpt = (*(this->batch_output + batch_id)).data;
    float* ipt = (*(this->batch_input + batch_id)).data;
    float* bpt = this->b->data;
    float sum = 0;
    for(int i = 0; i < this->n; ++i) {
        otpt_exp[i] = std::exp((ipt[i] + bpt[i]) / this->T);
        sum += otpt_exp[i];
    }
    for(int i = 0; i < this->n; ++i)
       // otpt[i] += otpt_exp[i] / sum; // 不能这样！！！！！！
       //这是归一化计算，不能在原先的基础上作加法，而应当重新赋值
        otpt[i] = otpt_exp[i] / sum; 
}
void softmax::count_grad() { 
    for(int i = 0; i < this->batch_num; ++i) {
        this->count_grad(i);
    }
}
void softmax::count_grad(size_t batch_id) {
#ifdef USE_DEBUG 
    assert(this->batch_grad + batch_id);
    assert(this->batch_output + batch_id);
    assert(this->batch_input + batch_id);
#endif  
    float* g = (this->batch_grad + batch_id)->data;
    float* otpt = (this->batch_output + batch_id)->data;
    float dai_dzj;
    for(int i = 0; i < this->n; ++i) {
        // dai_dzj = 0;
        // for(int j = 0; j < this->n; ++j){
        //     dai_dzj += ( (i == j) ? ( *(otpt + j) * (1 - *(otpt + j)) ) : ( -(*(otpt + j) * otpt[i]) )); // 要乘上 next_grad
        // }
        g[i] = otpt[i] - g[i];  //优化过的交叉熵和softmax梯度
        // 是直接赋值而非+=
    }  
}

void origin::print_layer(bool inc_grad) {

}
void origin::print_batches() {
    std::cout << "the input value is:\n";
    this->batch_input[0].print();
}

void sigmoid::print_layer(bool inc_grad) {
}
void sigmoid::print_batches() {
}

void softmax::print_layer(bool inc_grad) {
}
void softmax::print_batches() {
}

void relu::print_layer(bool inc_grad) {
}
void relu::print_batches() {

}

} // namespace dtensor