#include"ops.h"
namespace dtensor{

using namespace base;

op::op(std::string name, dtensor_base* a, dtensor_base* b) : 
    name(name), batch_num(a->get_batch_num()) , output(nullptr)
{

    assert(this->inputs.size() == 0);

    // 改变输入tensor的count_n计数
    temp_n = new size_t[batch_num]();
    have_backwarded = new bool[batch_num]();
    this->count_n = !(a->is_param) + !(b->is_param); //是动态张量才 + 1 (不是不算)
    if(!(a->is_param)) (a->count_n)++;
    if(!(b->is_param)) (b->count_n)++;

    this->inputs.push_back(a);
    this->inputs.push_back(b);
    // 放入输入指针
    this->metrix_inputs.push_back(a->get_output_metrix_ptr());
    this->metrix_inputs.push_back(b->get_output_metrix_ptr());
    // 放入输入梯度指针
    this->metrix_inputs_grad.push_back(a->get_grad_metrix_ptr());
    this->metrix_inputs_grad.push_back(b->get_grad_metrix_ptr());
    // 张量指向算子
    a->op_next.push_back(this);
    b->op_next.push_back(this);
}

op::op(std::string name, std::vector<dtensor_base*>& inputs) : 
    name(name), inputs(inputs), output(nullptr) ,
//要深拷贝inputs
    batch_num(inputs[0]->get_batch_num()) 
{
    assert(inputs[0]->get_batch_num() == inputs[1]->get_batch_num()); //// 目前先支持两个输入的情况，后续再考虑更多输入的情况
    this->batch_num = inputs[0]->get_batch_num();

    count_n = 0; 
    temp_n = new size_t[this->batch_num]();
    have_backwarded = new bool[batch_num]();

    for(auto& input : inputs) {
        // 放入输入指针, 放入输入梯度指针
        this->metrix_inputs.push_back(input->get_output_metrix_ptr());
        this->metrix_inputs_grad.push_back(input->get_grad_metrix_ptr());
        // 张量指向算子
        input->op_next.push_back(this);
        //是动态张量才 + 1 (不是不算)
        count_n += !(input->is_param);
        // 改变输入tensor的count_n计数
        if(!(input->is_param)) (input->count_n)++;
    }
}

void op::print_info() {
    std::cout << "Operator Name: " << this->name << std::endl;
    std::cout << "Number of Inputs: " << this->inputs.size() << std::endl;
    std::cout << "Batch Number: " << this->batch_num << std::endl;
    std::cout << "Shape output: ("; for(size_t i = 0; i < this->shape_output.size(); ++i) 
        std::cout << this->shape_output[i] << " "; std::cout << ")" << std::endl;
    std::cout << "Number of Input Tensors: " << this->metrix_inputs.size() << std::endl;
}
std::vector<dtensor_base*> op::get_inputs() {
    return this->inputs;
}


add_op::add_op(std::vector<dtensor_base*>& inputs) : op("add", inputs) {
    assert(inputs.size() >= 2);
    assert(this->inputs.size() == 0);
    for(auto &input : inputs) {
        this->inputs.push_back(input);
        this->metrix_inputs.push_back(input->get_output_metrix_ptr());
        this->metrix_inputs_grad.push_back(input->get_grad_metrix_ptr());
    } 
}

std::pair<tensor_type, tensor_type> op::get_type_pair(dtensor_base* a, dtensor_base* b) {
    return std::make_pair(a->tstp, b->tstp);
}
tensor_type op::get_type(std::vector<dtensor_base*>& tensors) {
    return tensors[0]->tstp;
}

void op::do_op(tensor_type p, sub_type q) {
    assert(this->inputs.size() != 0);
    // 实现加法操作的逻辑
    dtensor_base* new_node = nullptr;
    auto shape = this->inputs[0]->get_shape();
    for(size_t i = 0; i < shape.size(); ++i)
        shape_output.push_back(shape[i]);
    switch (p)
    {
    case tensor_type::common:
        new_node = new multi_dim_tensor(inputs[0]->get_shape(), inputs[0]->get_batch_num());
        new_node->set_type(tensor_type::common);
        break;

    case tensor_type::layer:
        new_node = layer_tool(inputs[0]->get_n(), inputs[0]->get_batch_num(), q); // origin 作为加法操作的输出层
        new_node->set_type(tensor_type::layer);
        break;
    
    default:
        break;
    }
    //统一处理
    new_node->set_op_last(this);
    this->output= new_node;
    return;
}
dtensor_base* op::set_output(tensor_type tp, sub_type stp) {  
    this->do_op(tp, stp);
    return this->output;
}
void matmul_op::_check_type(dtensor_base* a, dtensor_base* b) {
    assert((a->tstp == tensor_type::tensor2D || a->tstp == tensor_type::layer) && (b->tstp == tensor_type::tensor2D || b->tstp == tensor_type::layer));
}
void matmul_op::do_op(tensor_type p, sub_type q) {
    assert(this->inputs.size() != 0);
    // 实现矩阵乘法操作的逻辑
    dtensor::dtensor_base* new_node = nullptr;
    
    ///// 打印矩阵维度信息
    std::cout << "matmul input shapes: (" << this->inputs[0]->get_shape()[0] << ", " << this->inputs[0]->get_shape()[1] << ") and (" 
              << this->inputs[1]->get_shape()[0] << ", " << this->inputs[1]->get_shape()[1] << ")" << std::endl;
    
    auto shape_a = this->inputs[0]->get_shape();
    auto shape_b = this->inputs[1]->get_shape();

    std::pair<size_t, size_t> tensor_shape = get_matmul_output_shape({shape_a[0], shape_a[1]}, {shape_b[0], shape_b[1]});
    this->shape_output.push_back(tensor_shape.first);
    this->shape_output.push_back(tensor_shape.second);


    std::cout << "matmul output shape: (" << tensor_shape.first << ", " << tensor_shape.second << ")" << std::endl;
    switch (p)
    {
    case tensor_type::common:
        new_node = new multi_dim_tensor(tensor_shape, this->inputs[0]->get_batch_num());
        new_node->set_type(tensor_type::common);
        break;

    case tensor_type::layer:
        assert(tensor_shape.second == 1); // 目前先支持输出为列向量的情况，后续再考虑更多输出形状的情况
        new_node = layer_tool(tensor_shape.first, inputs[0]->get_batch_num(), q); 
        new_node->set_type(tensor_type::layer);
        break;
    
    default:
        break;
    }
    new_node->set_op_last(this);
    this->output = new_node;
    return;

}

// void dot_op::do_op(tensor_type p) {
//     assert(this->inputs.size() != 0);
//     // 实现点积操作的逻辑
//     dtensor::dtensor_base* new_node = nullptr;
//     switch (p)
//     {
//     case tensor_type::common:
//         new_node = new multi_dim_tensor(this->inputs[0]->get_shape(), this->inputs[0]->get_batch_num());
//         new_node->set_type(tensor_type::common);
//         new_node->set_op_last(this);
//         this->output = new_node;
//         break;

//     case tensor_type::layer:
//         new_node = new origin(this->inputs[0]->get_n(), this->inputs[0]->get_batch_num()); // origin 作为加法操作的输出层
//         new_node->set_type(tensor_type::layer);
//         new_node->set_op_last(this);
//         this->output = new_node;
//         break;
    
//     default:
//         break;
//     }
//     return;
// }

void concat_op::do_op(tensor_type p, sub_type q) {
    assert(this->inputs.size() != 0);
    // 实现拼接操作的逻辑
    if(this->concat_dim == 0) //检测剩余维度匹配
        for(int i = 0; i < this->inputs.size() - 1; ++i)
            assert((this->inputs[i]->get_shape())[1] == (this->inputs[i + 1]->get_shape())[1]);
    else if(this->concat_dim == 1)
        for(int i = 0; i < this->inputs.size(); ++i)
            assert((this->inputs[i]->get_shape())[0] == (this->inputs[i + 1]->get_shape())[0]);

    // 计算每个矩阵维度的拼接索引
    this->shared_dim_size = this->concat_dim == 0 ? this->inputs[0]->get_shape()[1] : this->inputs[0]->get_shape()[0];

    size_t idx = 0;
    for(int i = 0; i < this->inputs.size(); ++i){
        this->concat_dim_indexes.push_back(idx);
        idx += this->inputs[i]->get_shape()[this->concat_dim];
    }
    this->concat_dim_size = idx;
    this->concat_dim_indexes.push_back(idx); // idx 从 0 ~ 总大小，共 n + 1 个索引

    _size concat_shape = this->concat_dim == 0 ? std::make_pair(idx, this->shared_dim_size) : std::make_pair(this->shared_dim_size, idx);
    dtensor_base* new_node = new multi_dim_tensor(concat_shape, this->batch_num);
    new_node->set_type(tensor_type::common);
    new_node->set_op_last(this);
    this->output = new_node;
    return;
}

void op::reset_count() {
    for(int i = 0; i < batch_num; ++i) {
        temp_n[i] = 0;
        have_backwarded[i] = false;
    }

}

/////////////////
//// forward ////
/////////////////
void op::forward(size_t batch_id) {
    if(temp_n[batch_id] != count_n) {      // 指向该 op 的tensor只要有一个未准备就绪，则无法前向传播
        assert(0);
        return;
    }
    // std::cout << " forward by OP \n" ;
    //std::cout << "temp_n is: " << temp_n[batch_id]  <<std::endl;
    this->_forward(batch_id);
}
void op::forward() {
    for(int i = 0; i <batch_num; ++i)
        this->forward(i);
}
void op::backward(size_t batch_id) { // 只要轮到op便能执行backward，因为其只有唯一一个出口
    if(have_backwarded[batch_id])
        return;
    // std::cout << " backward by OP \n" ;
    this->_backward(batch_id);
    for(auto &input_ptr : inputs)  // op的backward导致其动态输入计数值 -1
        if(!(input_ptr->is_param))
            (input_ptr->temp_n)[batch_id]--;
    have_backwarded[batch_id] = true;
}
void op::backward() {
    for(int i = 0; i <batch_num; ++i)
        this->backward(i);
}

// add
void add_op::_forward(size_t batch_id) {
    float* otpt = this->output->get_input_data_ptr(batch_id); // 也可以考虑对输出作复制优化，而非每次都算一遍
    // 不考虑和param相加
    base::_add_tensors(this->metrix_inputs, otpt, batch_id);

}
void add_op::_forward() {
    for(int i = 0; i < this->batch_num; ++i)
        this->_forward(i);
}
// sub
void sub_op::_forward(size_t batch_id) {
#ifdef USE_DEBUG
    assert(this->metrix_inputs.size() == 2);
#endif
    float* otpt = this->output->get_input_data_ptr(batch_id);
    size_t idx1 = this->inputs[0]->is_parameter() ? 0 : batch_id;
    size_t idx2 = this->inputs[1]->is_parameter() ? 0 : batch_id;
    base::_sub_tensors((this->metrix_inputs[0])[idx1], (this->metrix_inputs[1])[idx2], otpt);
}
void sub_op::_forward() {
    for(int i = 0; i < this->batch_num; ++i)
        this->_forward(i);
}
// matmul
void matmul_op::_forward(size_t batch_id) {
#ifdef USE_DEBUG
    assert(this->metrix_inputs.size() == 2);
#endif
    float* otpt = this->output->get_input_data_ptr(batch_id);

    size_t idx1 = this->inputs[0]->is_parameter() ? 0 : batch_id;
    size_t idx2 = this->inputs[1]->is_parameter() ? 0 : batch_id;

    base::_matmul_tensors((this->metrix_inputs[0])[idx1], (this->metrix_inputs[1])[idx2], otpt);
}
void matmul_op::_forward() {
    for(int i = 0; i < this->batch_num; ++i)
        this->_forward(i);
}
// dot
void dot_op::_forward(size_t batch_id) {
#ifdef USE_DEBUG
    assert(this->metrix_inputs.size() == 2);
#endif
    std::cout << "counting output...\n";
    float* otpt = this->output->get_input_data_ptr(batch_id);
    size_t idx1 = this->inputs[0]->is_parameter() ? 0 : batch_id;
    size_t idx2 = this->inputs[1]->is_parameter() ? 0 : batch_id;
    base::_dot_tensors((this->metrix_inputs[0])[idx1], (this->metrix_inputs[1])[idx2], otpt);
}
void dot_op::_forward() {
    for(int i = 0; i < this->batch_num; ++i)
        this->_forward(i);
}
// concat
void concat_op::_forward(size_t batch_id) {
    // 拼接操作的前向计算逻辑
    float* otpt = this->output->get_input_data_ptr(batch_id);
    base::_concat_tensors(this->metrix_inputs, this->concat_dim_indexes, otpt, 
    this->concat_dim, this->concat_dim_size, this->shared_dim_size, batch_id);
}
void concat_op::_forward() {
    for(int i = 0; i < this->batch_num; ++i)
        this->_forward(i);
}

//////////////////
//// backward ////
//////////////////

// add
void add_op::_backward(size_t batch_id) {
    // float* nxt_grad = this->output->get_grad_data_ptr(batch_id); // 也可以考虑对输出作复制优化，而非每次都算一遍
    // for(int i = 0; i < this->inputs.size(); ++i)
    //     base::_add_tensors(this->metrix_inputs_grad[i][batch_id], nxt_grad);
    // 搞错了！！！ 应该是加到输入的grad上面，而不是输入的grad加到输出的grad上面
    metrix_float* next_grad_m = this->output->get_grad_metrix_ptr();
    float* input_m_grad;
    for(int i = 0; i < this->inputs.size(); ++i) {
        input_m_grad = (this->metrix_inputs_grad[i][batch_id]).data;
        base::_add_tensors(next_grad_m[batch_id], input_m_grad);
    }
    return;
}
void add_op::_backward() {
    for(int i = 0; i < this->batch_num; ++i)
        this->_backward(i);
}
// sub
void sub_op::_backward(size_t batch_id) {
    metrix_float* next_grad_m = this->output->get_grad_metrix_ptr(); // 也可以考虑对输出作复制优化，而非每次都算一遍
#ifdef USE_DEBUG
    assert(this->metrix_inputs_grad.size() == 2);
#endif
    float *input_grad1 = (this->metrix_inputs_grad[0][batch_id]).data, 
          *input_grad2 = (this->metrix_inputs_grad[1][batch_id]).data;
    base::_add_tensors(next_grad_m[batch_id], input_grad1);
    base::_sub_tensors(next_grad_m[batch_id], input_grad2);
}
void sub_op::_backward() {
    for(int i = 0; i < this->batch_num; ++i)
        this->_backward(i);
}
// matmul
void matmul_op::_backward(size_t batch_id) {  //还需仔细检查！！
    //std::cout << "count matmul op grad!!...\n";
    metrix_float* nxt_grad = this->output->get_grad_metrix_ptr(); // 也可以考虑对输出作复制优化，而非每次都算一遍
#ifdef USE_DEBUG
    assert(this->metrix_inputs_grad.size() == 2);  
#endif
    // 计算第一个输入张量的梯度     l2 = W * l1  =>  dW = dl2 * l1^T ; dl1 = W^T * dl2

    //std::cout << "LAYER CONTENT is: \n";
    //this->metrix_inputs[1][batch_id].print();
    //std::cout << "WEIGHT METRIX CONTENT is: \n";
    //this->metrix_inputs[0]->print();

    base::_matmul_tensors(nxt_grad[batch_id], this->metrix_inputs[1][batch_id], false, true, this->metrix_inputs_grad[0][batch_id].data);
    base::_matmul_tensors(this->metrix_inputs[0][0], nxt_grad[batch_id], true, false, this->metrix_inputs_grad[1][batch_id].data);
}
void matmul_op::_backward() {
    for(int i = 0; i < this->batch_num; ++i)
        this->_backward(i);
}
// dot
void dot_op::_backward(size_t batch_id) {
    metrix_float* nxt_grad = this->output->get_grad_metrix_ptr() + batch_id; 
#ifdef USE_DEBUG
    assert(this->metrix_inputs_grad.size() == 2);   
#endif
    base::_dot_tensors(this->metrix_inputs[1][batch_id], *nxt_grad, this->metrix_inputs_grad[0][batch_id].data);
    base::_dot_tensors(this->metrix_inputs[0][batch_id], *nxt_grad, this->metrix_inputs_grad[1][batch_id].data);
}
void dot_op::_backward() {
    assert(0);
    for(int i = 0; i < this->batch_num; ++i)
        this->_backward(i);
}
// concat
void concat_op::_backward(size_t batch_id) {
    // 拼接操作的反向计算逻辑
    assert(0);
}

} // namespace dtensor