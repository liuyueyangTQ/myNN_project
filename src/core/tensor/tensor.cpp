#include <cassert>
#include "tensor.h"
// #include "metrix.h" //调用到了matrix.h中 的 _matmul函数，且"tensor.h"有头文件保护
namespace tensor{
using _size = std::pair<size_t, size_t>;
using namespace base; // 确保 metrix_float， _matmul这些函数能查询到

#ifdef USE_DEBUG
    int layer::layer_index = 0;
    int tensor2D_float::wm_index = 0;
#endif

void tensor_base::print_info() {
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

void common_tensor::add(tensor_base *p) {
    // 可能导致死循环
    // tensor_base* temp = new tensor_base(this->is_param,this->lock_grad);
    // this->next->push_back(temp);
    
    // p->next->push_back(temp); // p为基类指针是否拥有next成员？是否要强转
    

    // temp->last->push_back(this);
    // temp->last->push_back(p);
    // temp->back_ops.push_back(ops::add);
    // temp->back_ops.push_back(ops::add);    
}

void common_tensor::sub(tensor_base *p) {} // 待实现
void common_tensor::matmul(tensor_base *p) {} // 待实现



void tensor2D_float::check() {  // 循环包含，要放到.cpp文件中实现
    assert(this->x && this->next);
    assert(this->x->n == this->weight->shape.second);
    assert(this->next->n == this->weight->shape.first);
}
// 最后一层
void tensor2D_float::count_grad(metrix_float* next_grad, int n) { // 由   z = w * x + b  ， 则  dw = dz * x^T
    metrix_float temp({n, 1}, next_grad);
    _matmul(temp, *(this->x->output), false, true, this->grad->data);
}
// 最后一层 ， 多batch
void tensor2D_float::count_grad(metrix_float* next_grad, size_t batch_id, int n) {
    metrix_float temp({n, 1}, next_grad);
    _matmul(temp, *(this->x->batch_output + batch_id), false, true, (this->batch_grad + batch_id)->data);
}
// 非最后一层
void tensor2D_float::count_grad() { // 由   z = w * x + b  ， 则  dw = dz * x^T
    metrix_float* g = this->next->grad; 
    _matmul(*g, *(this->x->output), false, true, this->grad->data); // dw = dz * x^T
}
// 非最后一层 ， 多batch
void tensor2D_float::count_grad(size_t batch_id) { // 由   z = w * x + b  ， 则  dw = dz * x^T
    metrix_float* g = this->next->batch_grad + batch_id; 
    _matmul_add(*g, *(this->x->batch_output + batch_id), false, true, (this->batch_grad + batch_id)->data); // dw = dz * x^T
}
void tensor2D_float::forward() { // 计算所有样本 的梯度并传递至其之前的层！！！
}
void tensor2D_float::forward(size_t batch_id) { // 计算 第 batch_id 个样本 的梯度并传递至其之前的层！！！
}

void tensor2D_float::backward() { // 全部计算
   // metrix_float* g = this->next->grad; // z = w * x + b
   // _matmul(*g, *(x->output), false, true, this->grad->data);        // dw = dz * x^T
   for(int i = 0; i < this->batch_num; ++i)
        this->count_grad(i);


}
void tensor2D_float::backward(size_t batch_id) { // 只计算一个 sample
   // metrix_float* g = this->next->grad; // z = w * x + b
   // _matmul(*g, *(x->output), false, true, this->grad->data);        // dw = dz * x^T

    this->count_grad(batch_id);
}
//void tensor2D_float::backward_mul() { // 之后在考虑}
void tensor2D_float::backward(float* next_grad) { // 用于尾部层  通过损失函数传递的梯度
}
void tensor2D_float::backward(float* next_grad, size_t batch_id) { // 用于尾部层  通过损失函数传递的 第 batch_id 个样本 的梯度
}
void tensor2D_float::backward_grad() { // 计算所有样本 的梯度并传递至其之前的层！！！
}
void tensor2D_float::backward_grad(size_t batch_id) { // 计算 第 batch_id 个样本 的梯度并传递至其之前的层！！！
}

void tensor2D_float::update() {
    if(this->lock_grad) // 不更新梯度
        return;

    size_t row = this->weight->shape.first, col = this->weight->shape.second;
    float dw; float* w;
    float temp;
    for(int i = 0; i < row * col; ++i) {
        dw = 0; 
        for(int batch_id = 0; batch_id < this->batch_num; ++batch_id){
            w = (this->batch_grad + batch_id)->data;
            temp = *(w + i);
            if(temp > 1) temp = 1;
            else if(temp < -1) temp = -1;
            dw += temp;
        }

        dw /= ((float)this->batch_num); //梯度取平均值
        *(this->weight->data + i) -= this->lr * dw;
    }
    return;
}


void tensor2D_float::update_mul(size_t group_index, size_t group_size) {  //供线程池调用的多线程梯度计算接口 !!!!
    size_t row = this->weight->shape.first, col = this->weight->shape.second;
    float dw; float* w;
    for(int i = group_index * group_size; i < std::min(row * col, (group_index + 1) * group_size); ++i) {
        dw = 0; 
        for(int batch_id = 0; batch_id < this->batch_num; ++batch_id){
            w = (this->batch_grad + batch_id)->data;
        // for(int j = 0; j < this->weight->shape.second; ++j)
        //    *( *(w + i) + j) -= lr * *(*(dw + i) + j); 
            dw += *(w + i);
        }
        dw /= ((float)this->batch_num); //梯度取平均值
        *(this->weight->data + i) -= this->lr * dw;
    }
    return;
}



void tensor2D_float::print(bool inc_grad) {
    std::cout << "Tensor data is: " << std::endl;
    this->weight->print();
    if(inc_grad) {
        std::cout << "Tensor grad is: " << std::endl;
        for(int i = 0; i < this->batch_num; ++i)
            (this->batch_grad + i)->print();            
    }
}
void tensor2D_float::print_param() {
    std::cout << "weight metrix is:\n";
    this->weight->print();
}

void tensor2D_float::print_grad() {
    for(int i = 0 ; i < this->batch_num; ++i){
        auto p = (this->batch_grad + i);
        std::cout << "    The WEIGHT METRIX GRAD of " << i + 1<< " th sample is:\n";
        std::cout << "    ";
        p->print();
        std::cout << std::endl;
    } 
}

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


// layer
void layer::_set_batch() {
    if(batch_num == 1)
        return;
    // 为每个batch 分配内存
    this->batch_grad = _alloc_m_data(0);
    this->batch_input = _alloc_m_data(1);

    this->batch_output = _alloc_m_data(2);
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


        // // 前置检查：避免空指针解引用
        // if (!pMemory) {
        //     std::cerr << "pMemory is null! bias: " << bias << '\n';
        //     return nullptr;
        // }

        // // 步骤1：计算单个metrix_float对象的内存大小（含缓存行填充）
        // // 每个对象对齐到缓存行，避免伪共享
        // const size_t obj_size = sizeof(metrix_float);
        // const size_t aligned_obj_size = ((obj_size + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE) * CACHE_LINE_SIZE;
        // const size_t total_size = batch_num * aligned_obj_size; // 总内存大小

        // // 步骤2：分配对齐的内存（按缓存行对齐，提升缓存命中率）
        // void* raw_mem = memory_utils::aligned_alloc(total_size, CACHE_LINE_SIZE);
        // if (!raw_mem) {
        //     std::cerr << "failed to allocate aligned memory! size: " << total_size 
        //               << ", align: " << CACHE_LINE_SIZE << ", errno: " << errno << '\n';
        //     return nullptr;
        // }

        // // 保存分配的内存地址到pMemory（替代原*(pMemory + bias)）
        // pMemory[bias] = raw_mem; // 简化指针操作：pMemory[bias] 等价于 *(pMemory + bias)

        // // 步骤3：批量构造metrix_float对象（优化构造效率）
        // metrix_float* myArray = static_cast<metrix_float*>(raw_mem);
        // try {
        //     // 循环构造每个对象（按对齐后的地址）
        //     for (size_t i = 0; i < batch_num; ++i) {
        //         // 计算当前对象的地址（对齐后的偏移）
        //         char* obj_addr = static_cast<char*>(raw_mem) + i * aligned_obj_size;
        //         // placement new：构造对象（参数n,1）
        //         new (obj_addr) metrix_float(this->n, 1);

        //         // 可选：编译器优化提示（强制循环展开）
        //         #pragma unroll 4 // 按CPU核心数展开（如4/8）
        //     }

        //     // 替代方案：若metrix_float支持默认构造，可用STL批量构造（效率更高）
        //     // std::uninitialized_default_construct_n(myArray, batch_num);
        //     // 然后批量初始化rows/cols/data（需额外处理）
        // } catch (const std::exception& e) {
        //     std::cerr << "failed to construct metrix_float objects! " << e.what() << '\n';
        //     // 构造失败：回滚已分配的内存（析构已构造的对象 + 释放内存）
        //     for (size_t i = 0; i < batch_num; ++i) {
        //         char* obj_addr = static_cast<char*>(raw_mem) + i * aligned_obj_size;
        //         static_cast<metrix_float*>(static_cast<void*>(obj_addr))->~metrix_float();
        //     }
        //     memory_utils::aligned_free(raw_mem);
        //     pMemory[bias] = nullptr;
        //     return nullptr;
        // }

        // return myArray;
}
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


    // if (!pMemory || !pMemory[bias]) return;

    // void* raw_mem = pMemory[bias];
    // const size_t obj_size = sizeof(metrix_float);
    // const size_t aligned_obj_size = ((obj_size + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE) * CACHE_LINE_SIZE;

    // // 手动析构每个对象
    // for (size_t i = 0; i < batch_num; ++i) {
    //     char* obj_addr = static_cast<char*>(raw_mem) + i * aligned_obj_size;
    //     static_cast<metrix_float*>(static_cast<void*>(obj_addr))->~metrix_float();
    // }

    // // 释放对齐内存
    // memory_utils::aligned_free(raw_mem);
    // pMemory[bias] = nullptr;
}

void layer::update() { // 更新参数 （b）
    float db;
    float* grad_b;
    float temp;
    for(int i = 0; i < this->n; ++i) {
        db = 0;
        for(int batch_id = 0; batch_id < this->batch_num; ++batch_id){
            grad_b = (this->batch_grad + batch_id)->data;
            temp = *(grad_b + i);

            if(temp < -1)temp = -1; //梯度裁剪
            else if(temp > 1)temp = 1;

            db += temp;
        }
        db /= ((float)this->batch_num); //梯度取平均值
        ////////////////////////////////////////////////////
        //std::cout << "    db for b[" << i << "] is: " << db << ' ';
        ////////////////////////////////////////////////////
        *(this->b->data + i) -= this->lr * db;

        // Gradient Clipping
        // if(*(this->b->data + i) < -1) *(this->b->data + i) = -1;
        // else if(*(this->b->data + i) > 1) *(this->b->data + i) = 1;
    }
    ////////////////////////////////////////////////////
    //std::cout << std::endl;
    ////////////////////////////////////////////////////

}
void layer::update_mul(size_t group_index, size_t group_size) { //   更新参数 （b）

    float dw; float* w;
    for(int i = group_index * group_size; i < std::min((size_t)this->n, (group_index + 1) * group_size); ++i) {
        dw = 0; 
        for(int batch_id = 0; batch_id < this->batch_num; ++batch_id){
            w = (this->batch_grad + batch_id)->data;
        // for(int j = 0; j < this->weight->shape.second; ++j)
        //    *( *(w + i) + j)-= lr * *(*(dw + i) + j); 
            dw += *(w + i);
        }
        dw /= ((float)this->batch_num); //梯度取平均值
        *(this->b->data + i) -= this->lr * dw;
    }
    return;
}

void layer::get_input(std::vector<std::vector<float>>& samples) {
#ifdef USE_DEBUG 
    assert(this->n == samples[0].size());
    assert(this->batch_num == samples.size());
#endif   
    float* p;
    for(int batch_id = 0; batch_id < this->batch_num; ++batch_id) { // 逐样本
        p = (this->batch_input + batch_id)->data;
        for(int i = 0; i < this->n; ++i)  //逐维度赋值
            *(p + i) = samples[batch_id][i];
    } 
}
void layer::get_input(std::vector<float>& sample, size_t batch_id) {
#ifdef USE_DEBUG 
    assert(this->n == sample.size());
#endif   
    float* p = (this->batch_input + batch_id)->data;
    for(int i = 0; i < this->n; ++i)  //逐维度赋值
        *(p + i) = sample[i];
}

void layer::forward() { //layer 和可能存在的权重矩阵同时前向传播
    this->count_output(); // 输入已知
    if(this->w_next.size() > 0) {  // 可以没有邻接矩阵，此时 w_next 为空，layer为 batch_norm 或 layer_norm层    
        for(int i = 0; i < this->w_next.size(); ++i) {
            if(this->w_next[i]->is_identity) {
                // 直接传递
                float* p = (this->output)->data;
                float* w = ((this->next)[i]->input)->data;
                for(int j = 0; j < this->n; ++j){
                    *(w + j) += *(p + j); //一定是累加，而不是赋值！！
                }
                continue;
            }
            _matmul(*(w_next[i]->weight), *output, false, false, 
                ((this->next)[i]->input->data)  
            );  // 结合权重矩阵计算下一层的输入 z = w * x + b
        }
    }
    else if(this->next.size() > 0){  // 没有权重矩阵， 直接把这层输出传给下一层（下层存在）
        float* p = (this->next)[0]->input->data;
        float* q = this->output->data;
        for(int i = 0; i < this->n; ++i){
            *(p + i) += *(q + i);
        }
    }
}

void layer::forward(size_t batch_id) {
    this->count_output(batch_id); // 输入已知
    if(this->w_next.size() > 0) {  // 可以没有邻接矩阵，此时 w_next 为空，layer为 batch_norm 或 layer_norm层
#ifdef USE_DEBUG 
      //  std::cout << "counting the matmul at the " << batch_id << "-th batch sample, the " << this->index <<"-th layer...\n";
#endif 
        try
        {   
#ifdef USE_DEBUG
            assert(this->w_next.size() == this->next.size()); //确保一一对应
            for(int i = 0; i < this->w_next.size(); ++i) {
                assert((this->w_next[i]->weight->shape).second == (batch_output + batch_id)->n);
                assert((this->next[i]->batch_input + batch_id)->n == (*(this->w_next[i]->weight)).shape.first);
            }

#endif 
    ////
    //std::cout << "doing forward in batch " << batch_id <<std::endl;
    ////
            assert(this->w_next.size() == this->next.size()); //确保一一对应
            for(int i = 0; i < this->w_next.size(); ++i) {
                if(this->w_next[i]->is_identity) {
                    // 直接传递
                    float* v = (this->batch_output + batch_id)->data;
                    float* w = (this->next[i]->batch_input + batch_id)->data;
                    for(int j = 0; j < this->n; ++j){
                        *(w + j) += *(v + j); //一定是累加，而不是赋值！！
                    }
                    continue;
                }
                _matmul_add(*(this->w_next[i]->weight), *(this->batch_output + batch_id), false, false,   // 计算下一层的input
                    (this->next[i]->batch_input + batch_id)->data // 下一层第 batch_id 个样本
                );  // 下一层 第 batch_id 个样本的输入 z = w * x_i + b
            }


        }
        catch(const std::exception& e)
        {
            std::cerr <<"layer forward matmul failed! " << e.what() << '\n';
        }

    }
    else if(this->next.size() > 0) {  // 没有权重矩阵， 直接把这层输出传给下一层（下层存在）
        float* v = (this->batch_output + batch_id)->data;
        for(int j = 0; j < this->next.size(); ++j){
            float* w = (this->next[j]->batch_input + batch_id)->data;

            for(int i = 0; i < this->n; ++i){
                *(w + i) += *(v + i); //一定是累加，而不是赋值！！
            }
        }
    }

}
// 已实现统一的 backward 方法，故最后包含损失的层不需要和普通层分开实现
// // 非最后一层 （一定有next， 但不一定有 w_next ）  
// void layer::backward() { // 通过下一层的尾部梯度传递损失
//     this->count_grad();
//     this->backward_grad();
// }
// // 非最后一层 batch （一定有next， 但不一定有 w_next ）
// void layer::backward(size_t batch_id){ // 通过下一层的尾部梯度传递损失 

//     this->count_grad(batch_id); // 一定要有batch_id

//     this->backward_grad(batch_id); // 计算 第 batch_id 个样本 的梯度并传递至其之前的层！！！（重要，扩展至resnet的关键！）
//     // 邻接矩阵的反向传播逻辑要放在这里面，和forward保持一致，进而从model_backward里面移除相应的逻辑


// }

void layer::backward() { // 通过下一层的 ##尾部梯度## 传递损失
    // if(this->next) {
    //     metrix_float* ng = this->next->grad;   // z = w * x + b
    //     _matmul(*(this->w_next->weight),*ng, true, false, this->grad->data);        // dx = w^float * dz  是这一层输出的grad
    // }
    this->count_grad();    ////////最后一层，没有邻接矩阵
    // 此时 next_grad 对应最后一层的标签损失

    this->backward_grad();// 计算 第 batch_id 个样本 的梯度并传递至其之前的层！！！（重要，扩展至resnet的关键！）
}

void layer::backward(size_t batch_id) {   // 这个可以弃用！！！
    // if(this->next) {
    //     metrix_float* ng = this->next->batch_grad + batch_id;   
    //     float* tg = (this->batch_grad + batch_id)->data;
    //     if(this->w_next) {// 有邻接矩阵的情况下
    //         _matmul(*(this->w_next->weight),*ng, true, false, tg);   // dx = w^T * dz  计算这一层 第 batch_id 个样本输出对应的grad
    //     } // 没邻接矩阵 （如 layer_norm， 则直接计算grad ）
    //     count_grad((this->next->batch_grad + batch_id)->data, batch_id);//  再计算这一层输入的 grad (需要知道下面一层对应的 grad)       
    // }
    this->count_grad(batch_id);  //先计算这一层的输入对应的梯度 
    // std::cout << "count grad finished for batch " << batch_id << std::endl;
    this->backward_grad(batch_id); 
    // 此时 next_grad 对应最后一层的标签损失！！（在layer::backward_last中）

    // this->update();
}
void layer::backward(float* next_grad) { // 计算所有样本 的梯度并传递至其之前的层！！！
}
void layer::backward(float* next_grad, size_t batch_id) { // 计算所有样本 的梯度并传递至其之前的层！！！
}

void layer::backward_grad() { // 计算所有样本 的梯度并传递至其之前的层！！！
}

void layer::backward_grad(size_t batch_id) { // 计算所有样本 的梯度并传递至其之前的层！！！
    // z = w * x + b  (z是这一层输入，x是上一层输出，w 是上一层的 w_next成员，同时是这一层的w_last成员)
    // dx = w^T * dz 要加到上一层的 输出的 grad 上去
    // std::cout << "  doing backward_grad at the " << batch_id << "-th batch sample...\n";
    // std::cout << "    this layer has " << this->w_last.size() << " w_last...\n";
    // std::cout << "    this layer has " << this->last.size() << " last...\n";
    assert(this->w_last.size() == this->last.size()); // 确保一一对应
    for(int i = 0; i < this->w_last.size(); ++i){
        _matmul_add(*(this->w_last[i]->weight), *(this->batch_grad + batch_id), true, false, (this->last[i]->batch_grad + batch_id)->data);
        //std::cout << "    backward_grad matmul_add finished for the " << i << "-th last layer...\n";
    }

}

void layer::count_loss_grad(std::vector<float>& label, loss_type loss_tp) {
    assert(this->n == label.size());
    float* label_grad = new float[this->n](); //最后一层的标签损失！！
    switch (loss_tp)
    {
    case loss_type::mse:{
        float temp;
        for(int i = 0; i < this->n; ++i) {
            temp = label_grad[i] - *(this->output->data + i);
            *(label_grad + i) = temp * temp; 
        }
    }
    case loss_type::cross_entropy:{
        float y, y0;
        for(int i = 0; i < this->n; ++i) {
            y0 = label_grad[i];
            y =  *(this->output->data + i);

            *(label_grad + i) = -y0 * std::log(y) - (1 - y0)*std::log(1-y); 
        }

        //////////////////////////////////////////
        for(int i = 0; i < this->n; ++i) {


            *(label_grad + i) = label[i]; 
        }
        /////////////////////////////////////////////////////
        // std::cout << "the last layer output GRAD is: \n";
        // for(int i = 0; i < this->n; ++i) {
        //     std::cout << *(label_grad + i) <<' ';
        // }
        // std::cout << std::endl;
        /////////////////////////////////////////////////////
    }

    
    default:
        break;
    }
    for(int i = 0; i < this->n; ++i) {
        this->grad->data[i] = *(label_grad + i); // 直接把标签损失赋值给最后一层的 grad
    }
    delete[] label_grad;
}

void layer::count_loss_grad(std::vector<float>& label, loss_type loss_tp, size_t batch_id) {
    assert(this->n == label.size());
    float* label_grad = new float[this->n](); //最后一层的标签损失！！


    // for(int i = 0; i < this->n; ++i) ///////////////
    //     std::cout << "label value is:" << label[i] <<"  ";

    switch (loss_tp)  // 计算输出端grad
    {

    case loss_type::mse:{
        float temp;
        for(int i = 0; i < this->n; ++i) {
            temp = label[i] - *((this->batch_output + batch_id)->data + i);
            *(label_grad + i) = temp * temp; 
        }
        
    }
    case loss_type::cross_entropy:{
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
        *((this->batch_grad + batch_id)->data + i) = label_grad[i]; // 直接把标签损失赋值给最后一层的 grad
    }
    delete[] label_grad;
    //this->count_grad(label_grad, batch_id);  // 此处 next_grad 等价于label_grad
    // （该处实现放到了 count_label_grad 中）
    //std::cout << "      label loss counted!\n";
}


// void layer::forward(size_t batch_id) { //重复定义了
//     this->count_output(batch_id); // 输入已知
// }

void layer::print(bool inc_grad) {
    this->print_layer(inc_grad);
}

void layer::print_param() {
    std::cout << "layer_param is: ";
    auto p = this->b->data;
    for(int i = 0; i < this->n; ++i)
        std::cout << p[i] << ' ';
    std::cout << std::endl;
}
void layer::print_grad() {

    for(int i = 0 ; i < this->batch_num; ++i){
        float* p = (this->batch_grad + i)->data;
        std::cout << "    The grad of " << i + 1<< " th sample is:\n";
        std::cout << "    ";
        for(int j = 0; j < this->n; ++j)
            std::cout << *(p + j) << ' ';
        std::cout << std::endl;
    } 
}

// sigmoid
void sigmoid::count_output() {
    float* otpt = this->output->data;
    float* ipt = this->input->data;
    float* bpt = this->b->data;
    for(int i = 0; i < this->n; ++i) {
        otpt[i] += 1 / (1 + exp( -ipt[i] - bpt[i]));
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
        otpt[i] += 1 / (1 + exp(- ipt[i] - bpt[i]));
    }
}

void sigmoid::count_grad() { //反向传播必须知道下面一层的梯度
#ifdef USE_DEBUG 
    assert(this->grad);
    assert(this->output);
#endif 
    //  output = 1 / (1 + e^(-x))  ;  input + b = 
            // grad = f(x)(1-f(x))
    float* g = this->grad->data;
    float* otpt = this->output->data;
    for(int i = 0; i < this->n; ++i) {
        g[i] += otpt[i] * (1 - otpt[i]) * g[i];
    }  
}

void sigmoid::count_grad(size_t batch_id) { //反向传播必须知道下面一层的梯度
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
        g[i] += otpt[i] * (1 - otpt[i]) * g[i];
    }  
}

void sigmoid::print_layer(bool inc_grad) {
    std::cout << "Sigmoid layer parameter is: " << std::endl;
    std::cout << "bias: ";
    auto p = this->b->data;
    for(int i = 0; i < this->n; ++i)
        std::cout << *(p + i) << ' ';
    if(inc_grad){ // 多打印一下 grad

    }
    std::cout << std::endl;
}

void sigmoid::print_batches() {
    std::cout << "Sigmoid layer batches is: " << std::endl;
    for(int i = 0 ; i < this->batch_num; ++i){
        float * p = (this->batch_input + i)->data;
        std::cout << "    The input of " << i + 1<< " th sample is:\n";
        std::cout << "    ";
        for(int j = 0; j < this->n; ++j)
            std::cout << *(p + j) << ' ';
        std::cout << std::endl;
    }
    for(int i = 0 ; i < this->batch_num; ++i){
        float * p = (this->batch_output + i)->data;
        std::cout << "    The output of " << i + 1<< " th sample is:\n";
        std::cout << "    ";
        for(int j = 0; j < this->n; ++j)
            std::cout << *(p + j) << ' ';
        std::cout << std::endl;
    }
}

//layer_norm ！！
void layer_norm::count_output() {  // batch = 1的情况 
#ifdef USE_DEBUG 
    assert(this->grad);
    assert(this->output);
#endif   
    float* otpt = this->output->data;
    float* ipt = this->input->data;
    float* bpt = this->b->data;

    *this->min_dim = *ipt + *bpt;  //初始化最小值
    float max_dim = *ipt + *bpt;   //初始化最大值
    for(int i = 0; i < this->n; ++i) { //求最大最小值
        max_dim = (max_dim < (ipt[i] + bpt[i])) ? (ipt[i] + bpt[i]) : (max_dim);
        *this->min_dim = (*this->min_dim <= (ipt[i] + bpt[i])) ? (*this->min_dim) : (ipt[i] + bpt[i]);
    }
    *this->diff = max_dim - *this->min_dim + 0.000001; // 求diff
    for(int i = 0; i < this->n; ++i) {
        otpt[i] += 2 * (ipt[i] + bpt[i] - *this->min_dim) / *this->diff - 1; // 归一化到 [-1, 1] 区间 
    }
}


void layer_norm::count_output(size_t batch_id) { // batch > 1的情况
#ifdef USE_DEBUG 
    assert(this->batch_grad + batch_id);
    assert(this->batch_input + batch_id);
#endif     
    float* otpt = (this->batch_output + batch_id)->data;  // 计算 batch 里面第 batch_id 个样本对应的输出
    float* ipt = (this->batch_input + batch_id)->data;
    float* minpt = this->min_dim + batch_id;  // 第 batch_id 个样本的最小维度
    float* bpt = this->b->data;

    *minpt = *ipt;         //初始化最小值
    float max_dim = *ipt;  //初始化最大值
    for(int i = 0; i < this->n; ++i) { //求最大最小值
        max_dim = (max_dim < (ipt[i] + bpt[i])) ? (ipt[i] + bpt[i]) : (max_dim);
        *minpt = (*minpt <= (ipt[i] + bpt[i])) ? (*minpt) : (ipt[i] + bpt[i]);
    }
    float* diffpt = this->diff + batch_id; // 拿到diff指针
    *diffpt = max_dim - *minpt + 0.000001; // 求diff (第batch_id个)
    for(int i = 0; i < this->n; ++i) {
        otpt[i] += 2 * (ipt[i] - *minpt) / *diffpt - 1; // 归一化到 [-1, 1] 区间 
    }
}

void layer_norm::count_grad() { //反向传播必须知道下面一层的梯度
    //  output = 1 / (1 + e^(-x))  ;  input + b = 
            // grad = f(x)(1-f(x))
    float* g = this->grad->data;
    // float* otpt = this->output->data; // 不需要输出值来计算梯度
    for(int i = 0; i < this->n; ++i) {
        g[i] += g[i] * 2 / *this->diff;
    }  
}

void layer_norm::count_grad(size_t batch_id) { //反向传播必须知道下面一层的梯度
#ifdef USE_DEBUG 
    assert(this->batch_grad + batch_id);
    assert(this->batch_output + batch_id);
    assert(this->batch_input + batch_id);
#endif  
    //  output = 1 / (1 + e^(-x))  ;  input + b = 
            // grad = f(x)(1-f(x))

    float* g = (this->batch_grad + batch_id)->data;
    float* diffpt = this->diff + batch_id; // 拿到diff指针
    for(int i = 0; i < this->n; ++i) {
        g[i] += g[i] * 2 / *diffpt;
    }  
}

void layer_norm::print_layer(bool inc_grad) {
    std::cout << "Layer_norm layer parameter is: " << std::endl;
    std::cout << "diff: ";
    for(int i = 0; i < this->batch_num; ++i)
        std::cout << *(this->diff + i) << ' ';
    std::cout << std::endl;
    std::cout << "min_dim: ";
    for(int i = 0; i < this->batch_num; ++i)
        std::cout << *(this->min_dim + i) << ' ';
    std::cout << std::endl;
    std::cout << "bias: ";
    auto p = this->b->data;
    for(int i = 0; i < this->n; ++i)
        std::cout << *(p + i) << ' ';
    std::cout << std::endl;

    if(inc_grad){ // 多打印一下 grad

    }
}

void layer_norm::print_batches() {
    std::cout << "Layer_norm layer batches is: " << std::endl;
    for(int i = 0 ; i < this->batch_num; ++i){
        float * p = (this->batch_input + i)->data;
        std::cout << "    The input of " << i + 1<< " th sample is:\n";
        std::cout << "    ";
        for(int j = 0; j < this->n; ++j)
            std::cout << *(p + j) << ' ';
        std::cout << std::endl;
    }
    for(int i = 0 ; i < this->batch_num; ++i){
        float * p = (this->batch_output + i)->data;
        std::cout << "    The output of " << i + 1<< " th sample is:\n";
        std::cout << "    ";
        for(int j = 0; j < this->n; ++j)
            std::cout << *(p + j) << ' ';
        std::cout << std::endl;
    }
}




// relu
void relu::count_output() {
    float* otpt = this->output->data;
    float* ipt = this->input->data;
    float* bpt = this->b->data;

    for(int i = 0; i < this->n; ++i) {
        otpt[i] = std::max(ipt[i] + bpt[i], (float)0);
    }
}

void relu::count_output(size_t batch_id) {
    float* otpt = (*(this->batch_output + batch_id)).data;
    float* ipt = (*(this->batch_input + batch_id)).data;
    float* bpt = this->b->data;
    for(int i = 0; i < this->n; ++i) {
        otpt[i] = std::max(ipt[i] + bpt[i], (float)0);
    }
    //////////////////////////////////////////////////////////////////
    /*
    std::cout << "THE RELU LAYER INPUT is:\n";
    for(int i = 0; i < this->n; ++i)
        std::cout << ipt[i] << ' ';
    std::cout << std::endl;
    std::cout << "THE RELU LAYER OUTPUT is:\n";  
    for(int i = 0; i < this->n; ++i)
        std::cout << otpt[i] << ' ';
    std::cout << std::endl;
    */
    //////////////////////////////////////////////////////////////////
}

void relu::count_grad() {
    //  output = 1 / (1 + e^(-x))  ;  input + b = 
            // grad = f(x)(1-f(x))
    float* g = this->grad->data;
    float* otpt = this->output->data;
    for(int i = 0; i < this->n; ++i) {
        g[i] = (otpt[i] > 0) ? g[i] : 0;
    }  
}

void relu::count_grad(size_t batch_id) {
#ifdef USE_DEBUG 
    assert(this->batch_grad + batch_id);
    assert(this->batch_output + batch_id);
    assert(this->batch_input + batch_id);
#endif  
    //  output = 1 / (1 + e^(-x))  ;    dout / dx = e^(-x) / (1 + e^(-x))^2              input + b = 
            // grad = f(x)(1-f(x))
    float* g = (this->batch_grad + batch_id)->data;
    float* otpt = (this->batch_output + batch_id)->data;
    for(int i = 0; i < this->n; ++i) {
        g[i] = (otpt[i] > 0) ? g[i] : 0; // 要乘上 next_grad
    }  
    /////////////////////////////////////////////////////
    // std::cout << "The RELUE Layer INPUT GRAD is: \n";
    // auto q = (this->batch_grad + batch_id)->data;
    // for(int i = 0; i < this->n; ++i) {
    //     std::cout << *(q + i) <<' ';
    // }
    // std::cout << std::endl;
    /////////////////////////////////////////////////////
}

void relu::print_layer(bool inc_grad) {
    std::cout << "Relu layer parameter is: " << std::endl;
    std::cout << "bias: ";
    auto p = this->b->data;
    for(int i = 0; i < this->n; ++i)
        std::cout << *(p + i) << ' ';
    if(inc_grad){ // 多打印一下 grad

    }
    std::cout << std::endl;
}

void relu::print_batches() {
    std::cout << "Relu layer batches is: " << std::endl;
    for(int i = 0 ; i < this->batch_num; ++i){
        float * p = (this->batch_input + i)->data;
        std::cout << "    The input of " << i + 1 << " th sample is:\n";
        std::cout << "    ";
        for(int j = 0; j < this->n; ++j)
            std::cout << *(p + j) << ' ';
        std::cout << std::endl;
    }
    for(int i = 0 ; i < this->batch_num; ++i){
        float * p = (this->batch_output + i)->data;
        std::cout << "    The output of " << i + 1<< " th sample is:\n";
        std::cout << "    ";
        for(int j = 0; j < this->n; ++j)
            std::cout << *(p + j) << ' ';
        std::cout << std::endl;
    }
}

void softmax::count_output() {
    std::vector<float> otpt_exp(this->n, 0);
    auto p = this->output->data;
    for(int i = 0; i < this->n; ++i) {

    }
    //////////////////////////

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
    /////////////////////////////////////////
}
void softmax::count_grad() {

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
        g[i] = otpt[i] - g[i];  //优化过的交叉熵和softmax梯度
        // 是直接赋值而非+=
    }  

    /////////////////////////////////////////////////////
    // std::cout << "The Last Layer INPUT GRAD is: \n";
    // auto q = (this->batch_grad + batch_id)->data;
    // for(int i = 0; i < this->n; ++i) {
    //     std::cout << *(q + i) <<' ';
    // }
    // std::cout << std::endl;
    /////////////////////////////////////////////////////
}
void softmax::print_layer(bool inc_grad) {}
void softmax::print_batches() {
    std::cout << "Softmax layer batches is: " << std::endl;
    for(int i = 0 ; i < this->batch_num; ++i){
        float * p = (this->batch_input + i)->data;
        std::cout << "    The input of " << i + 1<< " th sample is:\n";
        std::cout << "    ";
        for(int j = 0; j < this->n; ++j)
            std::cout << *(p + j) << ' ';
        std::cout << std::endl;
    }
    for(int i = 0 ; i < this->batch_num; ++i){
        float * p = (this->batch_output + i)->data;
        std::cout << "    The output of " << i + 1<< " th sample is:\n";
        std::cout << "    ";
        for(int j = 0; j < this->n; ++j)
            std::cout << *(p + j) << ' ';
        std::cout << std::endl;
    }
}

// origin
void origin::count_output() {  //原始输入层没有bias！！
    float* otpt = this->output->data;
    float* ipt = this->input->data;
    float* bpt = this->b->data;

    for(int i = 0; i < this->n; ++i) {
        otpt[i] = ipt[i];  // + bpt[i];
    }
}

void origin::count_output(size_t batch_id) {
    float* otpt = (*(this->batch_output + batch_id)).data;
    float* ipt = (*(this->batch_input + batch_id)).data;
    float* bpt = this->b->data;
    for(int i = 0; i < this->n; ++i) {
        otpt[i] = ipt[i]; //+ bpt[i];
    }
}

void origin::count_grad() {
    //  output = 1 / (1 + e^(-x))  ;  input + b = 
            // grad = f(x)(1-f(x))
    float* g = this->grad->data;
    float* otpt = this->output->data;
    for(int i = 0; i < this->n; ++i) {
        g[i] = 0;//*(next_grad + i);
    }  
}

void origin::count_grad(size_t batch_id) {
#ifdef USE_DEBUG 
    assert(this->batch_grad + batch_id);
    assert(this->batch_output + batch_id);
    assert(this->batch_input + batch_id);
#endif  
    //  output = 1 / (1 + e^(-x))  ;    dout / dx = e^(-x) / (1 + e^(-x))^2              input + b = 
            // grad = f(x)(1-f(x))
    float* g = (this->batch_grad + batch_id)->data;
    float* otpt = (this->batch_output + batch_id)->data;
    for(int i = 0; i < this->n; ++i) {
        g[i] = 0;// *(next_grad + i); // 要乘上 next_grad
    }  
}

void origin::print_layer(bool inc_grad) {
    std::cout << "Origin layer parameter is: " << std::endl;
    std::cout << "bias: ";
    auto p = this->b->data;
    for(int i = 0; i < this->n; ++i)
        std::cout << *(p + i) << ' ';
    if(inc_grad){ // 多打印一下 grad

    }
    std::cout << std::endl;
}

void origin::print_batches() {
    std::cout << "Origin layer batches is: " << std::endl;
    for(int i = 0 ; i < this->batch_num; ++i){
        float * p = (this->batch_input + i)->data;
        std::cout << "    The input of " << i + 1 << " th sample is:\n";
        std::cout << "    ";
        for(int j = 0; j < this->n; ++j)
            std::cout << *(p + j) << ' ';
        std::cout << std::endl;
    }
    for(int i = 0 ; i < this->batch_num; ++i){
        float * p = (this->batch_output + i)->data;
        std::cout << "    The output of " << i + 1<< " th sample is:\n";
        std::cout << "    ";
        for(int j = 0; j < this->n; ++j)
            std::cout << *(p + j) << ' ';
        std::cout << std::endl;
    }
}


layer* layer_tool(int n, size_t batch_num, layer_type ltp) {
    switch (ltp)
    {
    case layer_type::sigmoid:
        return new sigmoid(n, batch_num);
    case layer_type::relu:
        return new relu(n, batch_num);
    case layer_type::softmax:
        return new softmax(n, batch_num);
    case layer_type::origin:
        return new origin(n, batch_num);
    default:
        break;
    }
    return nullptr;
}

} // namespace tensor