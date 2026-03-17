#pragma once
#include<string>
#include<iostream>
#include<vector>
#include<cmath>
#include<cassert>
#include<map>
#include "enum_type.h"
#include "metrix.h"
#include "dtensor.h"
using _size = std::pair<size_t, size_t>;
namespace base{

class metrix_float;
_size _get_size(metrix_float &m1, metrix_float &m2);
float* _alloc_data(metrix_float &m1, metrix_float &m2);
float* _matmul(metrix_float &m1, metrix_float &m2); // 直接返回两矩阵相乘得到的 矩阵指针 （需要调用allocate data创建数据）
void _matmul(metrix_float &m1, metrix_float &m2,float* data);  // 将得到的两个矩阵相乘结果放到指针data中 （假设都不是转置矩阵）
void _matmul(metrix_float &m1, metrix_float &m2, bool t1, bool t2, float* data); // 将得到的两个矩阵相乘结果放到指针data中 （考虑转置）
void _matmul_add(metrix_float& m1, metrix_float& m2, bool t1, bool t2, float* data);

void _add_tensors(metrix_float &m1, metrix_float &m2, float* data);
void _add_tensors(std::vector<metrix_float*>& ms, float* data, size_t batch_id);
void _sub_tensors(metrix_float &m1, metrix_float &m2, float* data);
void _matmul_tensors(metrix_float& m1, metrix_float& m2, float* data); //不指定转置形式
void _dot_tensors(metrix_float &m1, metrix_float &m2, float* data);
void _concat_tensors(std::vector<metrix_float*>& ms, std::vector<size_t>& concat_dim_indexes, float* data, size_t concat_dim, size_t concat_dim_size, size_t shared_dim_size, size_t batch_id);

_size get_matmul_output_shape(_size shape_a, _size shape_b);
}
namespace dtensor{
using namespace base;
class dtensor_base;
class common_tensor;
class tensor2D_float;
class layer;
class multi_dim_tensor;

}
// std::map<std::pair<tensor_type, tensor_type>, tensor_pair> type_map = {
//     { {tensor_type::layer, tensor_type::layer}, tensor_pair::layer_layer },
//     { {tensor_type::layer, tensor_type::tensor2D}, tensor_pair::layer_tensor2D },
//     { {tensor_type::tensor2D, tensor_type::tensor2D}, tensor_pair::tensor2D_tensor2D },
//     { {tensor_type::common, tensor_type::common}, tensor_pair::common_common }
// };
namespace base {
    size_t get_tensor_size(dtensor::dtensor_base* t);
}

namespace dtensor{
layer* layer_tool(int n, size_t batch_num, layer_type ltp);




class op {
    friend class dtensor_base;
protected:
    std::string name;
    std::pair<tensor_type, tensor_type> get_type_pair(dtensor_base* a, dtensor_base* b);

    dtensor_base* output;
    size_t batch_num;
    std::vector<size_t> shape_output;

    std::vector<dtensor_base*> inputs;
    float* data_output;
    std::vector<metrix_float*> metrix_inputs; // 存储每个输入张量（不是每个batch，而是组成算子的不同张量）的 **首个** 输出metrix指针
    std::vector<metrix_float*> metrix_inputs_grad; // 存储每个输入张量的 **首个** 梯度metrix指针

public:
    op(std::string name, dtensor_base* a, dtensor_base* b) ;
    op(std::string name, std::vector<dtensor_base*>& inputs) ;
    void print_info();
    virtual void forward() = 0;
    virtual void forward(size_t batch_id) = 0;
    virtual void backward() = 0;
    virtual void backward(size_t batch_id) = 0;
    virtual ~op() {}
    tensor_type get_type(std::vector<dtensor::dtensor_base*>& tensors);
    inline dtensor_base* get_output() {
        return this->output;
    }
    virtual void do_op(tensor_type p = tensor_type::common, sub_type q = sub_type::none); //让两个动态张量结合，生成新的张量, 默认情况与输入张量类型一致
    dtensor_base* set_output(tensor_type tp, sub_type stp); // 生成输出张量，并将其类型设置为tp
    virtual void _check_type(dtensor_base* a, dtensor_base* b) {}
    std::vector<dtensor_base*> get_inputs();

};

class add_op : public op {

public:
    add_op(dtensor_base* a, dtensor_base* b) : op("add", a, b) {
        assert(base::get_tensor_size(a) == base::get_tensor_size(b));
    }
    add_op(std::vector<dtensor_base*>& inputs);
    // void do_op(tensor_type p = tensor_type::common) override;
    void forward() override;
    void forward(size_t batch_id) override;
    void backward() override;
    void backward(size_t batch_id) override;
};



class sub_op : public op {
private:

public:
    sub_op(dtensor_base* a, dtensor_base* b) : op("sub", a, b) {
        assert(base::get_tensor_size(a) == base::get_tensor_size(b));
    }
    // void do_op(tensor_type p = tensor_type::common) override;
    void forward() override;
    void forward(size_t batch_id) override;
    void backward() override;
    void backward(size_t batch_id) override;
};

class matmul_op : public op {

public:
    matmul_op(dtensor_base* a, dtensor_base* b) : op("matmul", a, b) {
        this->_check_type(a, b);
    }
    void do_op(tensor_type p = tensor_type::common, sub_type q = sub_type::none) override;
    void forward() override;
    void forward(size_t batch_id) override;
    void backward() override;
    void backward(size_t batch_id) override;
    void _check_type(dtensor_base* a, dtensor_base* b) override;
};

class dot_op : public op {

public:
    dot_op(dtensor_base* a, dtensor_base* b) : op("dot", a, b) {}
    // void do_op(tensor_type p = tensor_type::common) override;
    void forward() override;
    void forward(size_t batch_id) override;
    void backward() override;
    void backward(size_t batch_id) override;
};

class concat_op : public op {
private:
    size_t concat_dim;
    size_t shared_dim_size;
    size_t concat_dim_size;
    std::vector<size_t> concat_dim_indexes;
public:
    concat_op(dtensor_base* a, dtensor_base* b, size_t concat_dim = 0) : op("concat", a, b), concat_dim(concat_dim) {}
    concat_op(std::vector<dtensor_base*>& inputs, size_t concat_dim = 0) : op("concat", inputs), concat_dim(concat_dim) {}
    void do_op(tensor_type p = tensor_type::common, sub_type q = sub_type::none) override;
    void forward() override;
    void forward(size_t batch_id) override;
    void backward() override;
    void backward(size_t batch_id) override;
};
} // namespace dtensor












