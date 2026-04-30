#include"dtensor.h"

namespace base {
    size_t get_tensor_size(::dtensor::dtensor_base* t) {
        return t->n;
    }
}
namespace dtensor{
using _size = std::pair<size_t, size_t>;

dtensor_base* dtensor_base::get_next() {
    return this->op_next[0]->output;
}

#ifdef USE_DEBUG
    int layer::layer_index = 0;
    int tensor2D_float::wm_index = 0;
void get_info(std::string input) {
    std::cout << input <<"\n";
    std::string info;
    while(1) {    
        std::cout << "Input info U need\n";
        std::cin >> info;
        if(info == "n") {
            std::cout << "go next...\n";
            return;
        }
        if(info == "ts") {
            std::cout << "Choose tensors \n";
            std::cin >> info;
            if(info == "n") return;
            size_t batch_id;
            bool inc_grad;
            if(info == "w") {
                std::cout << "include grad? \n";
                std::cin >> batch_id >> inc_grad;
            }
            
        }


    }
}
#endif
void dtensor_base::print(bool inc_grad, size_t batch_id, bool rec) {
    if(have_printed && rec) 
        return;
    if(batch_id == batch_num) {
        for(size_t i = 0; i < batch_num; ++i) {
            std::cout << "The " << i <<"-th batch:\n";
            _print_val(i);
            if(inc_grad)
                _print_grad(i);
        }
        return;
    }
    std::cout << "The " << batch_id <<"-th batch:\n";
    _print_val(batch_id);
    if(inc_grad)
        _print_grad(batch_id);

    if(rec) // 需要记录是否打印过
        have_printed = true;
}

void dtensor_base::print_info() {
    std::cout << "Tensor info: \n";
    std::cout << "  is_param: " << this->is_param << "\n";
    std::cout << "  lock_grad: " << this->lock_grad << "\n";
    std::cout << "  tensor_type: ";
    switch(this->tstp) {
        case tensor_type::common:
            std::cout << "common\n";
            break;
        case tensor_type::tensor2D:
            std::cout << "tensor2D\n";
            break;
        case tensor_type::layer:
            std::cout << "layer\n";
            break;
        default:
            std::cout << "unknown\n";
            break;
    }
}
void dtensor_base::reset_count() {
    for(int i = 0; i < batch_num; ++i) {
        temp_n[i] = count_n ;
        have_forwarded[i] = false;
        have_updated = false;
    }

}
void dtensor_base::forward() {
    for(int i = 0; i < batch_num; ++i)
        this->forward(i);
}
void dtensor_base::forward(size_t batch_id) { // 变量tensor前向传播可使的前面op就绪输入计数++
    if(have_forwarded[batch_id])
        return;
    //std::cout << " forward by count n = : " << this->count_n << std::endl;
    this->_forward(batch_id);
    if(!is_param) {//动态tensor
        for(auto &otpt : op_next) {
            (otpt->temp_n)[batch_id]++;
        }
    }
    have_forwarded[batch_id] = true;
}
void dtensor_base::backward() {
    for(int i = 0; i < batch_num; ++i)
        this->backward(i);
}
void dtensor_base::backward(size_t batch_id) {
    // 如果 is_param 如weight metrix， 直接退出
    if(is_param || temp_n[batch_id] != 0) //尚未就绪
        return;
    this->_backward(batch_id);
}

#ifdef USE_DEBUG
void dtensor_base::forward_D() {
    for(int i = 0; i < batch_num; ++i)
        this->forward_D(i);
}
void dtensor_base::forward_D(size_t batch_id) {
    if(have_forwarded[batch_id])
        return;
    this->_forward_D(batch_id);
    if(!is_param) {//动态tensor
        for(auto &otpt : op_next) {
            (otpt->temp_n)[batch_id]++;
        }
    }
    have_forwarded[batch_id] = true;
}
void dtensor_base::backward_D() {
    for(int i = 0; i < batch_num; ++i)
        this->backward_D(i);
}
void dtensor_base::backward_D(size_t batch_id) {
    // 如果 is_param 如weight metrix， 直接退出
    if(is_param || temp_n[batch_id] != 0) //尚未就绪
        return;
    this->_backward_D(batch_id);
}
#endif


// get_shape 
std::vector<size_t> multi_dim_tensor::get_shape() {
    return this->shape;
}
std::vector<size_t> tensor2D_float::get_shape() {
    return {this->shape.first, this->shape.second};
}
std::vector<size_t> layer::get_shape() {
    return this->shape;
}


metrix_float* multi_dim_tensor::_allocdata(size_t bias) {
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

/// @brief 得到内部矩阵引用
/// @return 
metrix_float& multi_dim_tensor::get_input_metrix_ref(size_t batch_id) {
    return this->batch_val[batch_id];
}
metrix_float& multi_dim_tensor::get_grad_metrix_ref(size_t batch_id) {
    return this->batch_grad[batch_id];
}
metrix_float& multi_dim_tensor::get_output_metrix_ref(size_t batch_id) {
    return this->batch_val[batch_id];
}
// tensor2D_float
metrix_float& tensor2D_float::get_input_metrix_ref(size_t batch_id) {
    return this->x->get_batch_input()[batch_id];  /////待修正
}
metrix_float& tensor2D_float::get_grad_metrix_ref(size_t batch_id) {
    return this->batch_grad[batch_id];
}
metrix_float& tensor2D_float::get_output_metrix_ref(size_t batch_id) {
    return this->next->get_batch_output()[batch_id];  /////待修正
}

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
/// @return 

// multi_dim_tensor
metrix_float* multi_dim_tensor::get_input_metrix_ptr() { //一样，输入输出相同
    return this->batch_val;
}
metrix_float* multi_dim_tensor::get_grad_metrix_ptr() {
    return this->batch_grad;
}
metrix_float* multi_dim_tensor::get_output_metrix_ptr() {
    return this->batch_val;
}

// tensor2D_float
metrix_float* tensor2D_float::get_input_metrix_ptr() { //一样，输入输出相同
    return this->weight;
}
metrix_float* tensor2D_float::get_grad_metrix_ptr() {
    return this->batch_grad;
}
metrix_float* tensor2D_float::get_output_metrix_ptr() {
    return this->weight;
}

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
/// @return 
float* multi_dim_tensor::get_input_data_ptr(size_t batch_id) { //一样，输入输出相同
    return this->batch_val[batch_id].data;
}
float* multi_dim_tensor::get_grad_data_ptr(size_t batch_id) {
    return this->batch_grad[batch_id].data;
}
float* multi_dim_tensor::get_output_data_ptr(size_t batch_id) {
    return this->batch_val[batch_id].data;
}

float* tensor2D_float::get_input_data_ptr(size_t batch_id) { //一样，输入输出相同
    return weight->data; // 这个函数在tensor2D_float中没有实际意义，因为它没有输入输出的区分，直接返回nullptr
}
float* tensor2D_float::get_grad_data_ptr(size_t batch_id) {
    return (batch_grad + batch_id)->data; // 这个函数在tensor2D_float中没有实际意义，因为它没有输入输出的区分，直接返回nullptr
}
float* tensor2D_float::get_output_data_ptr(size_t batch_id) {
    return weight->data; // 这个函数在tensor2D_float中没有实际意义，因为它没有输入输出的区分，直接返回nullptr
}

float* layer::get_input_data_ptr(size_t batch_id) { //一样，输入输出相同
    return this->batch_input[batch_id].data;
}
float* layer::get_grad_data_ptr(size_t batch_id) {
    return this->batch_grad[batch_id].data;
}
float* layer::get_output_data_ptr(size_t batch_id) {
    return this->batch_output[batch_id].data;
}

// tensor2D_float 分配和释放内存
metrix_float* tensor2D_float::_allocdata() {

#ifdef USE_DEBUG 
    wm_index++;
    this->index = wm_index;
#endif 

    // 步骤 1: 分配原始内存
    // operator new[] 只分配内存，不调用构造函数
    std::cout << "    1...\n";             /////////////////////////////////////////////////////////////
    *pMemory = operator new[](this->batch_num * sizeof(metrix_float));
    // 步骤 2: 在分配的内存上构造对象
    std::cout << "    2...\n";             /////////////////////////////////////////////////////////////
    metrix_float* myArray = static_cast<metrix_float*>(*pMemory);
    std::cout << "    3...\n";             /////////////////////////////////////////////////////////////
    for (int i = 0; i < this->batch_num; ++i) {
        // placement new: 在指定地址 (myArray + i) 上构造一个 A 对象
        new (myArray + i) metrix_float(this->shape.first, this->shape.second, "simple"); 
    }
    std::cout << "    4...\n";             /////////////////////////////////////////////////////////////
    return myArray;
}

void tensor2D_float::_release_data(metrix_float* myArray) {
    // --- 当不再需要数组时，必须手动销毁 ---
    try{
        // 步骤 A: 手动调用每个对象的析构函数
        for (int i = 0; i < this->batch_num; ++i) {
            myArray[i].~metrix_float();
        }
        // 步骤 B: 释放原始内存
        operator delete[](*this->pMemory);
    } catch (const char* err) { // 捕获字符串类型异常
        std::cout << "exception when releasing weight metrix data: " << err << std::endl;
    } catch (...) { // 兜底捕获其他异常
        std::cout << "Unkown error! " << std::endl;
    }
}

// layer 分配释放内存
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

// clear , update 更新梯度  
void multi_dim_tensor::clear_grad() {
    //
}
void multi_dim_tensor::clear_value() {
    //
}
void tensor2D_float::clear_grad() {
    for(int batch_id = 0; batch_id < this->batch_num; ++batch_id) {
        float* wg = (batch_grad + batch_id)->data;
        int sz = (this->shape).first * (this->shape).second;
        for(int i = 0; i < sz; ++i)
            wg[i] = 0;
    }
}
void tensor2D_float::clear_value() {
    // 无操作
}
void layer::clear_grad() {
    for(int batch_id = 0; batch_id < this->batch_num; ++batch_id) {
        float* g = (this->batch_grad + batch_id)->data;
        for(int i = 0; i < this->n; ++i)
            g[i] = 0;
    }
}
void layer::clear_value() {
    for(int batch_id = 0; batch_id < this->batch_num; ++batch_id) {
        float *g1, *g2;
        g1 = (this->batch_input + batch_id)->data;
        g2 = (this->batch_output + batch_id)->data;     
        for(int i = 0; i < this->n; ++i) {
            g1[i] = 0;g2[i] = 0;
        }
    }
}
float* layer::get_bias_data() {
    return this->b->data;
}
void multi_dim_tensor::update(double lr) {
    // 目前没有参数需要更新，后续如果有了再实现
}
void tensor2D_float::update(double lr) {
    if(this->lock_grad || have_updated) // 不更新梯度或已经更新过了
        return;
    size_t row = this->weight->shape.first, col = this->weight->shape.second;
    float dw; float* w;
    float temp;
    for(int i = 0; i < row * col; ++i) {
        dw = 0; 
        for(int batch_id = 0; batch_id < this->batch_num; ++batch_id){
            w = (this->batch_grad + batch_id)->data;
            temp = w[i];
            if(temp > 1) temp = 1;
            else if(temp < -1) temp = -1;
            dw += temp;
        }

        dw /= ((float)this->batch_num); //梯度取平均值
        this->weight->data[i] -= lr * dw;
    }
    have_updated = true;
    return;
}
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


// _forward() 和 _backward() 的重载版本，分别用于处理整个batch和单个样本的情况
void tensor2D_float::_forward() { //无需操作
}
void tensor2D_float::_forward(size_t batch_id) { //无需操作
}
void tensor2D_float::_backward() { // 无需任何操作，grad即为传入值，在op的backward中已计算
}
void tensor2D_float::_backward(size_t batch_id) { // 无需任何操作，grad即为传入值，在op的backward中已计算
}
void tensor2D_float::backward(float* next_grad) {
}
void tensor2D_float::backward(float* next_grad, size_t batch_id) {
}
void tensor2D_float::backward_grad() {
}
void tensor2D_float::backward_grad(size_t batch_id) {
}
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
void multi_dim_tensor::_forward() {
}
void multi_dim_tensor::_forward(size_t batch_id) {
}
void multi_dim_tensor::_backward() {
}
void multi_dim_tensor::_backward(size_t batch_id) {
}
void multi_dim_tensor::backward(float* next_grad) {
}
void multi_dim_tensor::backward(float* next_grad, size_t batch_id) {
}
void multi_dim_tensor::backward_grad() {
}
void multi_dim_tensor::backward_grad(size_t batch_id) {
}

#ifdef USE_DEBUG
void layer::_forward_D() {
    for(int i = 0; i < this->batch_num; ++i) {
        this->count_output(i);
    }
}
void layer::_forward_D(size_t batch_id) {
    this->count_output(batch_id);
    get_info("after layer forward of batch " + std::to_string(batch_id) + "...\n");
}
void layer::_backward_D() {
    // dynamic_cast<layer*>(this);
    for(int i = 0; i < this->batch_num; ++i) {
        this->count_grad(i);
    }
}
void layer::_backward_D(size_t batch_id) {
    this->count_grad(batch_id);    ////////最后一层，没有邻接矩阵
    get_info("after layer backward of batch " + std::to_string(batch_id) + "...\n");
}

void multi_dim_tensor::_forward_D() {
}
void multi_dim_tensor::_forward_D(size_t batch_id) {
}
void multi_dim_tensor::_backward_D() {
}
void multi_dim_tensor::_backward_D(size_t batch_id) {
}
void tensor2D_float::_forward_D() { //无需操作
}
void tensor2D_float::_forward_D(size_t batch_id) { //无需操作
}
void tensor2D_float::_backward_D() { // 无需任何操作，grad即为传入值，在op的backward中已计算
}
void tensor2D_float::_backward_D(size_t batch_id) { // 无需任何操作，grad即为传入值，在op的backward中已计算
}

#endif



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

// print
void multi_dim_tensor::_print_val(size_t batch_id) {

}
void tensor2D_float::_print_val(size_t batch_id) {
    assert(batch_id <= batch_num);
    std::cout << "tensor2D_float value:\n";
    this->weight->print(); //只有一个weight值
}
void layer::_print_val(size_t batch_id) {
    std::cout << "Layer input:\n";
    this->batch_input[batch_id].print();
    std::cout << "Layer output:\n";
    this->batch_output[batch_id].print();
}
void multi_dim_tensor::_print_grad(size_t batch_id) {

}
void tensor2D_float::_print_grad(size_t batch_id) {
    std::cout << "tensor2D_float grad:\n";
    this->batch_grad[batch_id].print();
}

void layer::_print_grad(size_t batch_id) {
    std::cout << "Layer grad:\n";
    this->batch_grad[batch_id].print();
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

// 用于调试，设置输入值
void multi_dim_tensor::set_input_value(float* data, size_t batch_id) {
    assert(batch_id < this->batch_num);
    std::copy(data, data + this->n, (this->batch_val + batch_id)->data);

}
void multi_dim_tensor::set_input_value(std::vector<std::vector<float>>& data) {
    assert(data.size() == this->batch_num);
    /// ???????
}

void tensor2D_float::set_input_value(float* data, size_t batch_id) {
    assert(batch_id == 0); // tensor2D_float 没有batch的概念，直接断言 batch_id 必须为0
    std::cout << "set input value" <<std::endl;
    std::cout << "n is: " << this->n <<std::endl;
    // std::copy(data, data + this->n, this->weight->data);
    for(int i = 0; i < n; ++i) {
        this->weight->data[i] = data[i];
    }
}
void tensor2D_float::set_input_value(std::vector<std::vector<float>>& data) {
    throw("Could not set input value for Weight Metrix!"); 
}

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

// 反向传播计算损失
void dtensor_base::count_loss_grad(std::vector<std::vector<float>>& label, loss_type loss_tp) {
    for(int i = 0; i < this->batch_num; ++i) {
        this->count_loss_grad(label[i], loss_tp, i);
    }
}
void multi_dim_tensor::count_loss_grad(std::vector<float>& label, loss_type loss_tp, size_t batch_id){

}
void tensor2D_float::count_loss_grad(std::vector<float>& label, loss_type loss_tp, size_t batch_id){

}
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


layer* layer_tool(int n, size_t batch_num, sub_type stp) {
    switch (stp)
    {
    case sub_type::sigmoid:
        return new sigmoid(n, batch_num);
    case sub_type::relu:
        return new relu(n, batch_num);
    case sub_type::softmax:
        return new softmax(n, batch_num);
    case sub_type::origin:
        return new origin(n, batch_num);
    default:
        std::cerr << "Unsupported layer type! \n";
    }
    return nullptr;
}

}// namespace dtensor