#pragma once
#include "enum_types.h"
#include "ops.h"

namespace dtensor{
class dtensor_base;

class add_op : public op {
public:
    add_op(dtensor_base* a, dtensor_base* b) : op("add", a, b) {
        assert(base::get_tensor_size(a) == base::get_tensor_size(b));
    }
    add_op(std::vector<dtensor_base*>& inputs);
    // void do_op(tensor_type p = tensor_type::common) override;
    void _forward() override;
    void _forward(size_t batch_id) override;
    void _backward() override;
    void _backward(size_t batch_id) override;
};

class sub_op : public op {
public:
    sub_op(dtensor_base* a, dtensor_base* b) : op("sub", a, b) {
        assert(base::get_tensor_size(a) == base::get_tensor_size(b));
    }
    // void do_op(tensor_type p = tensor_type::common) override;
    void _forward() override;
    void _forward(size_t batch_id) override;
    void _backward() override;
    void _backward(size_t batch_id) override;
};

class matmul_op : public op {
public:
    matmul_op(dtensor_base* a, dtensor_base* b) : op("matmul", a, b) {
        this->_check_type(a, b);
    }
    void do_op(tensor_type p = tensor_type::common, sub_type q = sub_type::none) override;
    void _forward() override;
    void _forward(size_t batch_id) override;
    void _backward() override;
    void _backward(size_t batch_id) override;
    void _check_type(dtensor_base* a, dtensor_base* b) override;
};

class dot_op : public op {
public:
    dot_op(dtensor_base* a, dtensor_base* b) : op("dot", a, b) {}
    // void do_op(tensor_type p = tensor_type::common) override;
    void _forward() override;
    void _forward(size_t batch_id) override;
    void _backward() override;
    void _backward(size_t batch_id) override;
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
    void _forward() override;
    void _forward(size_t batch_id) override;
    void _backward() override;
    void _backward(size_t batch_id) override;
};

} // namespace dtensor