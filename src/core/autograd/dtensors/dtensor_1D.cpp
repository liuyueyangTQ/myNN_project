#include <cstring>
#include "dtensor_1D.h"

namespace dtensor {

#ifdef USE_DEBUG
    int layer::layer_index = 0;
#endif

/// @brief 得到 vector 类型的 size  @attention 是 二维 向量！！
std::vector<size_t> layer::get_shape() {
    return this->shape;
}

/// @brief 得到内部矩阵引用
/// @return metrix_float&
metrix_float& layer::get_input_metrix_ref(size_t batch_id) {
    return this->batch_input[batch_id];
}
metrix_float& layer::get_grad_metrix_ref(size_t batch_id) {
    return this->batch_grad[batch_id];
}
metrix_float& layer::get_output_metrix_ref(size_t batch_id) {
    return this->batch_output[batch_id];
}

/// @brief 得到内部矩阵指针
/// @return metrix_float*
// layer
metrix_float* layer::get_input_metrix_ptr() { //一样，输入输出相同
    return this->batch_input;
}
metrix_float* layer::get_grad_metrix_ptr() {
    return this->batch_grad;
}
metrix_float* layer::get_output_metrix_ptr() {
    return this->batch_output;
}

/// @brief 得到内部矩阵数据指针
/// @return float*
float* layer::get_input_data_ptr(size_t batch_id) { //一样，输入输出相同
    return this->batch_input[batch_id].data;
}
float* layer::get_grad_data_ptr(size_t batch_id) {
    return this->batch_grad[batch_id].data;
}
float* layer::get_output_data_ptr(size_t batch_id) {
    return this->batch_output[batch_id].data;
}

/// @test @brief 用于调试， 设置梯度值
void layer::__set_grads__(std::vector<std::vector<float>>& grads) {
    assert(grads.size() == this->batch_num);
    assert(grads[0].size() == this->n);
    for(int batch_id = 0; batch_id < batch_num; ++batch_id) {
        float* grad_data = (this->batch_grad + batch_id)->data;
        for(int i = 0; i < this->n; ++i)
            grad_data[i] = grads[batch_id][i];
    }
}

/// @test @brief 用于调试， 设置输入
void layer::__set_inputs__(std::vector<std::vector<float>>& inputs) {
    assert(inputs.size() == this->batch_num);
    assert(inputs[0].size() == this->n);
    for(int batch_id = 0; batch_id < batch_num; ++batch_id) {
        float* input_data = (this->batch_input + batch_id)->data;
        for(int i = 0; i < this->n; ++i)
            input_data[i] = inputs[batch_id][i];
    }
}

/// @test @brief 用于调试， 设置输出值
void layer::__set_outputs__(std::vector<std::vector<float>>& outputs) {
    assert(outputs.size() == this->batch_num);
    assert(outputs[0].size() == this->n);
    for(int batch_id = 0; batch_id < batch_num; ++batch_id) {
        float* output_data = (this->batch_output + batch_id)->data;
        for(int i = 0; i < this->n; ++i)
            output_data[i] = outputs[batch_id][i];
    }
}

/// @brief layer 分配释放内存
void layer::_release_data(metrix_float* myArray, size_t bias) {
    // --- 当不再需要数组时，必须手动销毁 ---
    try{
        // 步骤 A: 手动调用每个对象的析构函数
        for (int i = 0; i < this->batch_num; ++i) {
            myArray[i].~metrix_float();
        }

        // 步骤 B: 释放原始内存
        operator delete[](*(this->pMemory + bias));
    } catch (const char* err) { // 捕获字符串类型异常
        std::cout << "exception when releasing layer data: " << err << std::endl;
    } catch (...) { // 兜底捕获其他异常
        std::cout << "Unkown error! " << std::endl;
    }
}
metrix_float* layer::_alloc_m_data(size_t bias) {
    // 步骤 1: 分配原始内存
    // operator new[] 只分配内存，不调用构造函数
    try
    {
        *(pMemory + bias) = operator new[](this->batch_num * sizeof(metrix_float));
    }
    catch(const std::exception& e)
    {
        std::cerr << "failed to allocate tensorfloat memory! " << e.what() << '\n';
    }
    
    assert(*(pMemory + bias));
    // 步骤 2: 在分配的内存上构造对象
    metrix_float* myArray = static_cast<metrix_float*>(*(pMemory + bias));
    for (int i = 0; i < this->batch_num; ++i) {
        // placement new: 在指定地址 (myArray + i) 上构造一个 A 对象
        new (myArray + i) metrix_float(this->n, 1); 
    }
    return myArray;
}
void layer::_set_batch() {
    // if(batch_num == 1) //去掉！！会导致不分配batch内存，后续访问时会出现野指针问题
    //     return;
    // 为每个batch 分配内存
    this->batch_grad = _alloc_m_data(0);
    this->batch_input = _alloc_m_data(1);
    this->batch_output = _alloc_m_data(2);
}

/// @brief 梯度清零 
void layer::clear_grad() {
    for(int batch_id = 0; batch_id < this->batch_num; ++batch_id) {
        float* g = (this->batch_grad + batch_id)->data;
        memset(g, 0, sizeof(float) * this->n);
    }
}

/// @brief 输入输出清零 
void layer::clear_value() {
    for(int batch_id = 0; batch_id < this->batch_num; ++batch_id) {
        float *g1, *g2;
        g1 = (this->batch_input + batch_id)->data;
        g2 = (this->batch_output + batch_id)->data;     
        memset(g1, 0, sizeof(float) * this->n);
        memset(g2, 0, sizeof(float) * this->n);
    }
}

/// @brief 得到偏差值（独有）
float* layer::get_bias_data() {
    return this->b->data;
}

/// @brief 更新参数
void layer::update(double lr) { // 更新参数 （b）
    if(have_updated)
        return;
    float db;
    float* grad_b;
    float temp;
    for(int i = 0; i < this->n; ++i) {
        db = 0;
        for(int batch_id = 0; batch_id < this->batch_num; ++batch_id){
            grad_b = (this->batch_grad + batch_id)->data;
            temp = grad_b[i];

            if(temp < -1)temp = -1; //梯度裁剪
            else if(temp > 1)temp = 1;

            db += temp;
        }
        db /= ((float)this->batch_num); //梯度取平均值
        this->b->data[i] -= lr * db;
    }
    have_updated = true;
}


/// @brief _forward() 和 _backward() 的重载版本，分别用于处理整个batch和单个样本的情况
void layer::_forward() {
    for(int i = 0; i < this->batch_num; ++i) {
        this->count_output(i);
    }
}
void layer::_forward(size_t batch_id) {
    this->count_output(batch_id);
}
void layer::_backward() {
    // std::cout << "layer backward start!\n";
    // dynamic_cast<layer*>(this);
    for(int i = 0; i < this->batch_num; ++i) {
        this->count_grad(i);
    }
    // std::cout << "layer backward finished!\n";
}
void layer::_backward(size_t batch_id) {
    this->count_grad(batch_id);    ////////最后一层，没有邻接矩阵
}
void layer::backward(float* next_grad) {
}
void layer::backward(float* next_grad, size_t batch_id) {
}
void layer::backward_grad() {
}
void layer::backward_grad(size_t batch_id) {
}

/// @brief 打印输入输出
void layer::_print_val(size_t batch_id) {
    std::cout << "Layer input:\n";
    float* input_data = (this->batch_input + batch_id)->data;
    for(int i = 0; i < this->n; ++i) 
        std::cout << input_data[i] << ' ';
    std::cout << "\nLayer output:\n";
    float* output_data = (this->batch_output + batch_id)->data;
    for(int i = 0; i < this->n; ++i) 
        std::cout << output_data[i] << ' ';
    std::cout << std::endl;
}

/// @brief 打印梯度
void layer::_print_grad(size_t batch_id) {
    std::cout << "Layer grad:\n";
    float* grad_data = (this->batch_grad + batch_id)->data;
    for(int i = 0; i < this->n; ++i) 
        std::cout << grad_data[i] << ' ';
    std::cout << std::endl;
}

#ifdef USE_DEBUG

void layer::_forward_D() {
    for(int i = 0; i < this->batch_num; ++i) {
        this->count_output(i);
    }
}
void layer::_forward_D(size_t batch_id) {
    this->count_output(batch_id);
}
void layer::_backward_D() {
    // dynamic_cast<layer*>(this);
    for(int i = 0; i < this->batch_num; ++i) {
        this->count_grad(i);
    }
}
void layer::_backward_D(size_t batch_id) {
    this->count_grad(batch_id);    ////////最后一层，没有邻接矩阵
}

#endif

/// @brief 设置输入
void layer::set_input_value(float* data, size_t batch_id) {
    assert(batch_id < this->batch_num);
    std::cout << "set input value" <<std::endl;
    std::cout << "n is: " << this->n <<std::endl;
    // std::copy(data, data + this->n, (this->batch_input + batch_id)->data);
    for(int i = 0; i < this->n; ++i) {
        (this->batch_input + batch_id)->data[i] = data[i];
    }
}
void layer::set_input_value(std::vector<std::vector<float>>& data) {
    assert(data.size() == this->batch_num && data[0].size() == this->n);   
    for(int batch_id = 0; batch_id < batch_num; ++batch_id)
        for(int i = 0; i < this->n; ++i) {
            (this->batch_input + batch_id)->data[i] = data[batch_id][i];
        }
    return;
}

/// @brief 计算梯度
void layer::count_loss_grad(std::vector<float>& label, loss_type loss_tp, size_t batch_id) {
    assert(this->n == label.size());
    float* label_grad = new float[this->n](); //最后一层的标签损失！！

    // for(int i = 0; i < this->n; ++i) ///////////////
    //     std::cout << "label value is:" << label[i] <<"  ";

    switch (loss_tp)  // 计算输出端grad
    {
    case loss_type::mse: {
        float temp;
        for(int i = 0; i < this->n; ++i) {
            temp = label[i] - *((this->batch_output + batch_id)->data + i);
            *(label_grad + i) = temp * temp; 
        }
        
    }
    case loss_type::cross_entropy: {
        float y, y0;
        for(int i = 0; i < this->n; ++i) {
            y0 = label[i];
            y =  *((this->batch_output + batch_id)->data + i);
            // std::cout << "y0 is: "<< y0<<",  y is: "<<y;
            //std::cout <<(std::min(-y0 * std::log(y) - (1 - y0)*std::log(1-y), (float)100)) <<"     ";
            *(label_grad + i) = std::min(-y0 * std::log(y) - (1 - y0)*std::log(1-y), (float)100); 
            //std::cout << std::endl;
        }
        /////////////////////////////////////////////////////
        for(int i = 0; i < this->n; ++i) {
            *(label_grad + i) = label[i]; 
        }
    }
    default:
        break;
    
    }
    for(int i = 0; i < this->n; ++i) {
        ((batch_grad + batch_id)->data)[i] = label_grad[i]; // 直接把标签损失赋值给最后一层的 grad
    }
    delete[] label_grad;
    //this->count_grad(label_grad, batch_id);  // 此处 next_grad 等价于label_grad
    // （该处实现放到了 count_label_grad 中）
    //std::cout << "      label loss counted!\n";
}

} // namespace dtensor