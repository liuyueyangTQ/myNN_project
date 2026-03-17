#pragma once
#include<iostream>
#include<cassert>
#include<string>
#include<algorithm>
#include<random>
#include<vector>
#include"buffer_util.h"
// #include"tensor.h" // 不能include， 不然编译会报错

namespace tensor{
class tensor_base;   
class common_tensor; 
class tensor2D_float;   
} // namespace tensor

namespace base{
enum class init_type{
simple,
xavier,
he
};
using _size = std::pair<size_t,size_t>;

class __metrix_base{
private:
    void randomUniformInit(int fan_in = 1, int fan_out = 1, init_type init_type = init_type::xavier);
    void initialize(int fan_in = 1, int fan_out = 1, init_type init_type = init_type::xavier);
    void _allocdata(int rol, int col); 
public:
    float* data;
    _size shape;
    size_t n;

public:
    __metrix_base(int row, int col) : // 构造新数据
        shape({row, col}), n(row * col)
    {   
        this->_allocdata(row, col);
    }
    __metrix_base(int row, int col, float* p) :
        shape({row, col}), n(row * col)
    {
        this->_allocdata(row, col);
        std::copy(p, p + n, this->data);
    }
    __metrix_base(_size shape) :
        shape(shape), n(shape.first * shape.second)
    {
        this->_allocdata(shape.first, shape.second);
    }
    __metrix_base(_size shape, float* p) :
        shape(shape),
        n(shape.first * shape.second)
    {
        this->_allocdata(shape.first, shape.second);
        std::copy(p, p + n, this->data);
    }

    __metrix_base(int row, int col, init_type init_type) : // 引入随机初始化
    // 专用于 size(z) * size(x) 的矩阵 ( 输入为 x， 输出为 z ) ( row 对应 fan_out, col对应 fan_in )
        shape({row, col}), n(row * col)
    {
        this->_allocdata(row, col);
        this->initialize(col, row, init_type); // 顺序相反
    }

    __metrix_base(_size shape, init_type init_type) : // 引入随机初始化
    // 专用于 size(z) * size(x) 的矩阵 ( 输入为 x， 输出为 z ) ( row 对应 fan_out, col对应 fan_in )
        shape(shape), n(shape.first * shape.second)
    {
        this->_allocdata(shape.first, shape.second);
        this->initialize(shape.second, shape.first, init_type); // 顺序相反
    }

    ~__metrix_base() {delete[] this->data;}
};


class metrix_float;


_size _get_size(metrix_float &m1, metrix_float &m2);
_size _get_size(metrix_float* m1, metrix_float* m2);
std::vector<size_t> _get_size_mul_dim(metrix_float &m1, metrix_float &m2);
float* _alloc_data(metrix_float &m1, metrix_float &m2);
float* _matmul(metrix_float &m1, metrix_float &m2); // 直接返回两矩阵相乘得到的 矩阵指针 （需要调用allocate data创建数据）

void _matmul(metrix_float &m1, metrix_float &m2,float* data);  // 将得到的两个矩阵相乘结果放到指针data中 （假设都不是转置矩阵）
void _matmul(metrix_float &m1, metrix_float &m2, bool t1, bool t2, float* data); // 将得到的两个矩阵相乘结果放到指针data中 （考虑转置）
void _matmul_add(metrix_float& m1, metrix_float& m2, bool t1, bool t2, float* data); //指定转置形式


void _add_tensors(metrix_float &m1, metrix_float &m2, float* data);
void _add_tensors(std::vector<metrix_float*>& ms, float* data, size_t batch_id);
void _sub_tensors(metrix_float &m1, metrix_float &m2, float* data);
void _matmul_tensors(metrix_float& m1, metrix_float& m2, float* data); //不指定转置形式
void _matmul_tensors(metrix_float& m1, metrix_float& m2, bool t1, bool t2, float* data);  //指定转置形式
void _dot_tensors(metrix_float &m1, metrix_float &m2, float* data);
void _concat_tensors(std::vector<metrix_float*>& ms, std::vector<size_t>& concat_dim_indexes, float* data, size_t concat_dim, size_t concat_dim_size, size_t shared_dim_size, size_t batch_id);

void _equal_tensors(metrix_float &m, float* data);
void _equal_neg_tensors(metrix_float &m, float* data);

_size get_matmul_output_shape(_size shape_a, _size shape_b);

class metrix_float : public __metrix_base{
private:
    bool t;
    // 新增：填充成员，让总大小=缓存行大小（64字节）
    // 计算需填充的字节数：64 - (4+4+8) = 48字节
    char padding[CACHE_LINE_SIZE - (sizeof(bool) + sizeof(__metrix_base))];    
public:
    friend class tensor::tensor_base;
    friend class tensor::common_tensor;
    friend class tensor::tensor2D_float;

    friend float* _alloc_data(metrix_float &m1, metrix_float &m2);
    friend _size _get_size(metrix_float &m1, metrix_float &m2);
    friend _size _get_size(metrix_float* m1, metrix_float* m2);
    friend std::vector<size_t> _get_size_mul_dim(metrix_float &m1, metrix_float &m2);
    friend float* _matmul(metrix_float &m1, metrix_float &m2);
    friend void _matmul(metrix_float &m1, metrix_float &m2, bool t1, bool t2, float* data);
    friend void _matmul(metrix_float &m1, metrix_float &m2,float* data);
    friend void _matmul_add(metrix_float& m1, metrix_float& m2, bool t1, bool t2, float* data);



    friend void _add_tensors(metrix_float &m1, metrix_float &m2, float* data, size_t batch_id);
    friend void _add_tensors(std::vector<metrix_float*>& ms, float* data, size_t batch_id);
    friend void _sub_tensors(metrix_float &m1, metrix_float &m2, float* data);
    friend void _matmul_tensors(metrix_float& m1, metrix_float& m2, float* data); 
    friend void _matmul_tensors(metrix_float& m1, metrix_float& m2, bool t1, bool t2, float* data); 
    friend void _dot_tensors(metrix_float &m1, metrix_float &m2, float* data);
    friend void _concat_tensors(std::vector<metrix_float*>& ms, std::vector<size_t>& concat_dim_indexes, float* data, size_t concat_dim, size_t concat_dim_size, size_t shared_dim_size, size_t batch_id);
    
    friend void _equal_tensors(metrix_float &m, float* data);
    friend void _equal_neg_tensors(metrix_float &m, float* data);
    
    // using __metrix_base::__metrix_base;
    metrix_float(int row, int col, bool t = false) : __metrix_base(row,col), t(t) {}
    metrix_float(int row, int col, float* p, bool t = false) : __metrix_base(row,col,p), t(t) {
        // std::cout<<"dsadasf"<<this->shape.first << this->shape.second<<std::endl;
    }
    metrix_float(_size shape, bool t = false) : __metrix_base(shape), t(t) {}
    metrix_float(_size shape, float* p, bool t = false) : __metrix_base(shape, p), t(t) {}
    metrix_float(_size shape, init_type init_type, bool t = false) : __metrix_base(shape, init_type), t(t) {}
    metrix_float(int row, int col, init_type init_type, bool t = false) : __metrix_base(col, row, init_type), t(t) {}
    metrix_float(metrix_float&& m) noexcept : __metrix_base(m.shape, m.data) { //  ?????
        // this->data = m.data;
        this->t = m.t;
        // this->shape = m.shape;
    }
    metrix_float(const metrix_float &other) : __metrix_base(other.shape), t(other.t) {
         std::copy(other.data, other.data + this->n, this->data); // 深拷贝
    }

    // ~metrix_float() {~_metrix_base();}
public:
    void add(metrix_float &m);
    void sub(metrix_float &m);
    void matmul(metrix_float &m);
    void print();
    std::vector<size_t> get_shape();
};

class __multidimension_metrix_base{

protected:
    size_t dim;
    size_t* shape;
    size_t n;
    float* data;
private:
    void randomUniformInit(int fan_in = 1, int fan_out = 1, init_type init_type = init_type::xavier);
    void initialize(int fan_in = 1, int fan_out = 1, init_type init_type = init_type::xavier);
    void _allocdata(int rol, int col); 
public:
    __multidimension_metrix_base(size_t dim, size_t* shape) : dim(dim), shape(shape) {
        n = 1;
        for(size_t i = 0; i < dim; ++i) {
            n *= shape[i];
        }
        data = new float[n]();
    }
    ~__multidimension_metrix_base() {
        delete[] data;
        delete[] shape;
    }

};

class multidimension_metrix_float;

class multidimension_metrix_float : public __multidimension_metrix_base{
private:
    // 新增：填充成员，让总大小=缓存行大小（64字节）
    // 计算需填充的字节数：64 - (4+8) = 52字节
    char padding[CACHE_LINE_SIZE - (sizeof(__multidimension_metrix_base))];
};

} // namespace base

