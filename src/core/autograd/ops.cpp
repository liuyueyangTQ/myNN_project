#include "ops.h"
#include "dtensor_tools.h"

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
    name(name), inputs(inputs), output(nullptr),
//要深拷贝inputs
    batch_num(inputs[0]->get_batch_num()) 
{
    assert(inputs[0]->get_batch_num() == inputs[1]->get_batch_num()); //// 目前先支持两个输入的情况，后续再考虑更多输入的情况
    this->batch_num = inputs[0]->get_batch_num();

    this->count_n = 0; 
    temp_n = new size_t[this->batch_num]();
    have_backwarded = new bool[batch_num]();

    for(auto& input : inputs) {
        // 放入输入指针, 放入输入梯度指针
        this->metrix_inputs.push_back(input->get_output_metrix_ptr());
        this->metrix_inputs_grad.push_back(input->get_grad_metrix_ptr());
        // 张量指向算子
        input->op_next.push_back(this);
        //是动态张量才 + 1 (不是不算)
        this->count_n += !(input->is_param);
        // 改变输入tensor的count_n计数
        if(!(input->is_param)) (input->count_n)++;
    }
}

void op::print_info() {
    std::cout << "Operator Name: " << this->name << std::endl;
    std::cout << "Number of Inputs: " << this->inputs.size() << std::endl;
    std::cout << "Batch Number: " << this->batch_num << std::endl;
    std::cout << "Shape output: ("; for(size_t i = 0; i < this->shape_output.size(); ++i) std::cout << this->shape_output[i] << " "; 
                std::cout << ")" << std::endl;
    std::cout << "Number of Input Tensors: " << this->metrix_inputs.size() << std::endl;
    std::cout << "Input tensor address: "; for(size_t i = 0; i < this->inputs.size(); ++i) std::cout << inputs[i] << ' '; std::cout << std::endl;
}
std::vector<dtensor_base*> op::get_inputs() {
    return this->inputs;
}

// ===== do op ===== // 
// 用于结合两个已有的tensor，通过当前的op（ru add/sub等），生成新的tensor
void op::do_op(tensor_type p, sub_type q) {
    assert(this->inputs.size() != 0);
    // 实现加法操作的逻辑
    dtensor_base* new_node = nullptr;
    std::vector<size_t> new_shape = this->inputs[0]->get_shape();
    shape_output.clear();
    for(size_t i = 0; i < new_shape.size(); ++i)
        shape_output.push_back(new_shape[i]);

    // switch (p)
    // {
    // case tensor_type::common:
    //     new_node = new multi_dim_tensor(inputs[0]->get_shape(), inputs[0]->get_batch_num());
    //     new_node->set_type(tensor_type::common);
    //     break;

    // case tensor_type::tensor2D: { // 此时不用管 sub_type
    //     std::vector<size_t> tensor2D_shape = inputs[0]->get_shape();
    //     new_node = new tensor2D_float({tensor2D_shape[0], tensor2D_shape[1]}, inputs[0]->get_batch_num()); // 二维矩阵张量
    //     new_node->set_type(tensor_type::tensor2D);
    //     break;
    // }

    // case tensor_type::layer:
    //     new_node = layer_tool(inputs[0]->get_n(), inputs[0]->get_batch_num(), q); // origin 作为加法操作的输出层
    //     new_node->set_type(tensor_type::layer);
    //     break;
    
    // default:
    //     break;
    // }
    // //统一处理
    // new_node->set_op_last(this);
    // this->output= new_node;
    if(p == tensor_type::layer) {
        int temp = new_shape[0];
        new_shape.clear();
        new_shape.push_back(temp);
    } 
    dtensor_factory(p, new_shape, q, this->batch_num, this); /// ???? 
    return;
}

// ===== forward ===== //
void op::forward(size_t batch_id) {
    if(temp_n[batch_id] != count_n) {      // 指向该 op 的tensor只要有一个未准备就绪，则无法前向传播
        throw(1);
        return;
    }
    this->_forward(batch_id);
}

void op::forward() {
    for(int i = 0; i < batch_num; ++i)
        this->forward(i);
}

// ===== backward ===== //
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

// ===== tool function ===== //
dtensor_base* op::set_output(tensor_type tp, sub_type stp) {  
    this->do_op(tp, stp);
    return this->output;
}

void op::reset_count() {
    for(int i = 0; i < batch_num; ++i) {
        temp_n[i] = 0;
        have_backwarded[i] = false;
    }
}

std::pair<tensor_type, tensor_type> op::get_type_pair(dtensor_base* a, dtensor_base* b) {
    return std::make_pair(a->tstp, b->tstp);
}

tensor_type op::get_type(std::vector<dtensor_base*>& tensors) {
    return tensors[0]->tstp;
}

} // namespace dtensor