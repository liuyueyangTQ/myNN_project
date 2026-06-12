#include <cstring>
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

void dtensor_base::__set_isparam_lockgrad__(bool is_param, bool lock_grad) {
    this->is_param = is_param;
    this->lock_grad = lock_grad;
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
    if(!is_param) {//动态tensor, 只有不是param 才能进行梯度传播
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

/// @brief 反向传播计算损失
void dtensor_base::count_loss_grad(std::vector<std::vector<float>>& label, loss_type loss_tp) {
    for(int i = 0; i < this->batch_num; ++i) {
        this->count_loss_grad(label[i], loss_tp, i);
    }
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

}// namespace dtensor