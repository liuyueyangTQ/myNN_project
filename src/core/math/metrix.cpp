#include <iostream>
#include <cassert>
#include<random>
#include "metrix.h"
namespace base{
class metrix_float;

// __metrix_base
void __metrix_base::randomUniformInit(int fan_in, int fan_out, init_type init_type) {  // 随机初始化数组（均匀分布）
    // 初始化随机数引擎 
    std::random_device rd;  // 真随机种子（需编译器支持，Windows 可能需用其他种子）
    std::mt19937 gen(rd()); // 梅森旋转算法引擎（高效随机）
    float a, b;
    if (init_type == init_type::xavier) {
        // Xavier 均匀分布：[-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out))]
        float limit = std::sqrt(6.0f / (fan_in + fan_out));
        a = -limit;
        b = limit;
    } else if (init_type == init_type::he) {
        // He 均匀分布：[-sqrt(3/fan_in), sqrt(3/fan_in)]
        float limit = std::sqrt(3.0f / fan_in);
        a = -limit;
        b = limit;
    } else {
        // 简单均匀分布：[-0.5, 0.5]
        a = -0.5f;
        b = 0.5f;
    }
    std::uniform_real_distribution<float> dist(a, b);  // 均匀分布
    for (int i = 0; i < this->n; ++i) 
        *(this->data + i) = dist(gen);  // 生成随机数赋值
}
void __metrix_base::_allocdata(int row, int col) {
    try {
        data = new float[row * col]();
    } catch (const char* err) { // 捕获字符串类型异常
        std::cout << "exception when initializing metrix data!: " << err << std::endl;
    } catch (...) { // 兜底捕获其他异常
        std::cout << "Unkown error when initializing metrix data!! " << std::endl;
    }    
}

void __metrix_base::initialize(int fan_in, int fan_out, init_type init_type){
    this->randomUniformInit(fan_in, fan_out, init_type);
}

// util functions
_size _get_size(metrix_float &m1, metrix_float &m2) {
    if(!m1.t&&!m2.t) {
        assert(m1.shape.second == m2.shape.first);
        return {m1.shape.first,m2.shape.second};
    }
    else if(!m1.t&&m2.t) {
        assert(m1.shape.second == m2.shape.second);
        return {m1.shape.first,m2.shape.first};
    }else if(m1.t&&!m2.t) {
        assert(m1.shape.first == m2.shape.first);
        return {m1.shape.second,m2.shape.second};
    } 
    else {
        assert(m1.shape.first == m2.shape.second);
        return {m1.shape.second,m2.shape.first};
    }
}
_size _get_size(metrix_float* m1, metrix_float* m2) {
    if(!m1->t&&!m2->t) {
        assert(m1->shape.second == m2->shape.first);
        return {m1->shape.first,m2->shape.second};
    }
    else if(!m1->t&&m2->t) {
        assert(m1->shape.second == m2->shape.second);
        return {m1->shape.first,m2->shape.first};
    }else if(m1->t&&!m2->t) {
        assert(m1->shape.first == m2->shape.first);
        return {m1->shape.second,m2->shape.second};
    } 
    else {
        assert(m1->shape.first == m2->shape.second);
        return {m1->shape.second,m2->shape.first};
    }
}

_size get_matmul_output_shape(_size shape_a, _size shape_b) {
    assert(shape_a.second == shape_b.first);
    return {shape_a.first, shape_b.second};
}

std::vector<size_t> _get_size_mul_dim(metrix_float &m1, metrix_float &m2) { //待完善！！！！
    if(!m1.t&&!m2.t) {
        assert(m1.shape.second == m2.shape.first);
        return {m1.shape.first,m2.shape.second};
    }
    else if(!m1.t&&m2.t) {
        assert(m1.shape.second == m2.shape.second);
        return {m1.shape.first,m2.shape.first};
    }else if(m1.t&&!m2.t) {
        assert(m1.shape.first == m2.shape.first);
        return {m1.shape.second,m2.shape.second};
    } 
    else {
        assert(m1.shape.first == m2.shape.second);
        return {m1.shape.second,m2.shape.first};
    }
}


float* _alloc_data(metrix_float& m1, metrix_float& m2) {
    if(!m1.t&&!m2.t) {
 //       std::cout<<"m1.shape.second is "<<m1.shape.second<<"m2.shape.second is "<< m2.shape.second<<std::endl;
        assert(m1.shape.second == m2.shape.first);
        float* data = new float[m1.shape.first * m2.shape.second]();
        return data;
    }
    else if(!m1.t&&m2.t) {
 //       std::cout<<"m1.shape.second is "<<m1.shape.second<<"m2.shape.second is "<< m2.shape.second<<std::endl;
        assert(m1.shape.second == m2.shape.second);
        float* data = new float[m1.shape.first * m2.shape.first]();
        return data;
    }
    else if(m1.t&&!m2.t) {
        assert(m1.shape.first == m2.shape.first);
        float* data = new float[m1.shape.second * m2.shape.second]();
        return data;
    }
    else{
        assert(m1.shape.first == m2.shape.second);
        float* data = new float[m1.shape.second * m2.shape.first]();
        return data;
    }    
}

float* _matmul(metrix_float &m1, metrix_float &m2) {
    float* data = _alloc_data(m1, m2);
    float s;
    size_t row, col, l1, l2;
    l1 = m1.shape.second;
    l2 = m2.shape.second;
    if(!m1.t && !m2.t) {
        row = m1.shape.first;
        col = m2.shape.second;

        for(int i = 0; i < row; ++i)
            for(int j = 0; j < col; ++j) {
                s = 0;
                for(int k = 0;k < m1.shape.second; ++k)
                    s += m1.data[i * l1 + k] * m2.data[k * l2 + j];
                data[i * col + j] = s;
            }

    }
    else if(!m1.t && m2.t) {
        row = m1.shape.first;
        col = m2.shape.first;
        // l1 = m1.shape.second;
        // l2 = m2.shape.fsecond;
        for(int i = 0; i < row; ++i)
            for(int j = 0; j < col; ++j) {
                s = 0;
                for(int k = 0; k < m1.shape.second; ++k)
                    s += m1.data[i * l1 + k] * m2.data[j * l2 + k];
                data[i * col + j] = s;
            }
    }
    else if(m1.t && !m2.t) {
        row = m1.shape.second;
        col = m2.shape.second;
        // l1 = m1.shape.second;

        for(int i = 0; i < row; ++i)
            for(int j = 0; j < col; ++j) {
                s = 0;
                for(int k = 0; k < m1.shape.first; ++k)
                    s += m1.data[k * l1 + i] * m2.data[k * l2 + j];
                data[i * col + j] = s;
            }
    }
    else{
        row = m1.shape.second;
        col = m2.shape.first;
       // l = m1.shape.first;
        for(int i = 0; i < row; ++i)
            for(int j = 0; j < col; ++j) {
                s = 0;
                for(int k = 0; k < m1.shape.first; ++k)
                    s += m1.data[k * l1 + i] * m2.data[j * l2 + k];
                data[i * col + j] = s;
            }
    }
    return data;
}

void _matmul(metrix_float &m1, metrix_float &m2,float* data) {
    //float* data = _alloc_data(m1,m2);
    float s;
    size_t row, col, l1,l2;
    l1 = m1.shape.second;
    l2 = m2.shape.second;
    if(!m1.t && !m2.t) {
        row = m1.shape.first;
        col = m2.shape.second;

        for(int i = 0; i < row; ++i)
            for(int j = 0; j < col; ++j) {
                s = 0;
                for(int k = 0; k < m1.shape.second; ++k)
                    s += m1.data[i * l1 + k] * m2.data[k * l2 + j];
                data[i * col + j] = s;
            }

    }
    else if(!m1.t && m2.t) {
        row = m1.shape.first;
        col = m2.shape.first;
        // l1 = m1.shape.second;
        // l2 = m2.shape.fsecond;
        for(int i = 0; i < row; ++i)
            for(int j = 0; j < col; ++j) {
                s = 0;
                for(int k = 0; k < m1.shape.second; ++k)
                    s += m1.data[i * l1 + k] * m2.data[j * l2 + k];
                data[i * col + j] = s;
            }
    }
    else if(m1.t && !m2.t) {
        row = m1.shape.second;
        col = m2.shape.second;
        // l1 = m1.shape.second;

        for(int i = 0; i < row; ++i)
            for(int j = 0; j < col; ++j) {
                s = 0;
                for(int k = 0; k < m1.shape.first; ++k)
                    s += m1.data[k * l1 + i] * m2.data[k * l2 + j];
                data[i * col + j] = s;
            }
    }
    else{
        row = m1.shape.second;
        col = m2.shape.first;
       // l = m1.shape.first;
        for(int i = 0; i < row; ++i)
            for(int j = 0; j < col; ++j) {
                s = 0;
                for(int k = 0; k < m1.shape.first; ++k)
                    s += m1.data[k * l1 + i] * m2.data[j * l2 + k];
                data[i * col + j] = s;
            }
    }
    //return data;
}

void _matmul(metrix_float& m1, metrix_float& m2, bool t1, bool t2, float* data) {
  //  float* data = _alloc_data(m1,m2);
#ifdef USE_DEBUG        
    assert(&m1);

    assert(!m1.t && !m2.t);
    //std::cout << "m1 shape is: " << '(' << m1.shape.first << ',' << m1.shape.second << ')' << std::endl;
    //std::cout << "m2 shape is: " << '(' << m2.shape.first << ',' << m2.shape.second << ')' << std::endl;
#endif


    float s;
    size_t row, col, l1, l2;
    l1 = m1.shape.second;
    l2 = m2.shape.second;
    if(!t1 && !t2) {

        assert(m1.shape.second == m2.shape.first);
        row = m1.shape.first;
        col = m2.shape.second;
    //    std::cout << "row is: " << row << ' ' << "col is: " << col <<std::endl;
    //    std::cout << "the index is: ";
        
        for(int i = 0; i < row; ++i)
            for(int j = 0; j < col; ++j) {
                s = 0;
                for(int k = 0; k < m1.shape.second; ++k)
                    s += m1.data[i * l1 + k] * m2.data[k * l2+j];
                //std::cout << (i * col + j) << ' ';
                data[i * col + j] = s;
                
            }
#ifdef USE_DEBUG        
        // std::cout << "max index is: " << (col * row - 1) << std::endl;
#endif  
    }
    else if(!t1 && t2) {
        row = m1.shape.first;
        col = m2.shape.first;
        // l1 = m1.shape.second;
        // l2 = m2.shape.fsecond;
        // std::cout << "The Metrix index value is: "; 
        for(int i = 0; i < row; ++i)
            for(int j = 0; j < col; ++j) {
                s = 0;
                for(int k = 0; k < m1.shape.second; ++k)
                    s += m1.data[i * l1 + k] * m2.data[j * l2 + k];
                data[i * col + j] = s;

               // std::cout << (i * col + j) << ' ';
            }
    }
    else if(t1 && !t2) {
        row = m1.shape.second;
        col = m2.shape.second;
        // l1 = m1.shape.second;

        for(int i = 0; i < row; ++i)
            for(int j = 0; j < col; ++j) {
                s = 0;
                for(int k = 0; k < m1.shape.first; ++k)
                    s += m1.data[k * l1 + i] * m2.data[k * l2 + j];
                data[i * col + j] = s;
            }
    }
    else{
        row = m1.shape.second;
        col = m2.shape.first;
       // l = m1.shape.first;
        for(int i = 0; i < row; ++i)
            for(int j = 0; j < col; ++j) {
                s = 0;
                for(int k = 0; k < m1.shape.first; ++k)
                    s += m1.data[k * l1 + i] * m2.data[j * l2 + k];
                data[i * col + j] = s;
            }
    }
  //  return data;
}

void _matmul_add(metrix_float& m1, metrix_float& m2, bool t1, bool t2, float* data) {
  //  float* data = _alloc_data(m1,m2);
#ifdef USE_DEBUG        
    assert(&m1);

    assert(!m1.t && !m2.t);
    //std::cout << "m1 shape is: " << '(' << m1.shape.first << ',' << m1.shape.second << ')' << std::endl;
    //std::cout << "m2 shape is: " << '(' << m2.shape.first << ',' << m2.shape.second << ')' << std::endl;
#endif

    float s;
    size_t row, col, l1, l2;
    l1 = m1.shape.second;
    l2 = m2.shape.second;
    if(!t1 && !t2) {

        assert(m1.shape.second == m2.shape.first);
        row = m1.shape.first;
        col = m2.shape.second;
    //    std::cout << "row is: " << row << ' ' << "col is: " << col <<std::endl;
    //    std::cout << "the index is: ";
        
        for(int i = 0; i < row; ++i)
            for(int j = 0; j < col; ++j) {
                s = 0;
                for(int k = 0; k < m1.shape.second; ++k)
                    s += m1.data[i * l1 + k] * m2.data[k * l2+j];
                //std::cout << (i * col + j) << ' ';
                data[i * col + j] += s;
                
            }
#ifdef USE_DEBUG        
        // std::cout << "max index is: " << (col * row - 1) << std::endl;
#endif  
    }
    else if(!t1 && t2) {
        row = m1.shape.first;
        col = m2.shape.first;
        // l1 = m1.shape.second;
        // l2 = m2.shape.fsecond;
        // std::cout << "The Metrix index value is: "; 
        for(int i = 0; i < row; ++i)
            for(int j = 0; j < col; ++j) {
                s = 0;
                for(int k = 0; k < m1.shape.second; ++k)
                    s += m1.data[i * l1 + k] * m2.data[j * l2 + k];
                data[i * col + j] += s;

               // std::cout << (i * col + j) << ' ';
            }
    }
    else if(t1 && !t2) {
        row = m1.shape.second;
        col = m2.shape.second;
        // l1 = m1.shape.second;

        for(int i = 0; i < row; ++i)
            for(int j = 0; j < col; ++j) {
                s = 0;
                for(int k = 0; k < m1.shape.first; ++k)
                    s += m1.data[k * l1 + i] * m2.data[k * l2 + j];
                data[i * col + j] += s;
            }
    }
    else{
        row = m1.shape.second;
        col = m2.shape.first;
       // l = m1.shape.first;
        for(int i = 0; i < row; ++i)
            for(int j = 0; j < col; ++j) {
                s = 0;
                for(int k = 0; k < m1.shape.first; ++k)
                    s += m1.data[k * l1 + i] * m2.data[j * l2 + k];
                data[i * col + j] += s;
            }
    }
  //  return data;
}



void _matmul_tensors(metrix_float& m1, metrix_float& m2, float* data) { //不指定转置形式
#ifdef USE_DEBUG        
    if(!m1.t&&!m2.t) 
        assert(m1.shape.second == m2.shape.first);
    else if(!m1.t&&m2.t) 
        assert(m1.shape.second == m2.shape.second);
    
    else if(m1.t&&!m2.t) 
        assert(m1.shape.first == m2.shape.first);
    
    else
        assert(m1.shape.first == m2.shape.second);
    
#endif

    float s;
    size_t row, col, l1, l2;
    l1 = m1.shape.second;
    l2 = m2.shape.second;
    if(!m1.t && !m2.t) {
        row = m1.shape.first;
        col = m2.shape.second;
        for(int i = 0; i < row; ++i)
            for(int j = 0; j < col; ++j) {
                s = 0;
                for(int k = 0; k < m1.shape.second; ++k)
                    s += m1.data[i * l1 + k] * m2.data[k * l2+j];
                data[i * col + j] += s;
                
            }
#ifdef USE_DEBUG        
        // std::cout << "max index is: " << (col * row - 1) << std::endl;
#endif  
    }
    else if(!m1.t && m2.t) {
        row = m1.shape.first;
        col = m2.shape.first;
        for(int i = 0; i < row; ++i)
            for(int j = 0; j < col; ++j) {
                s = 0;
                for(int k = 0; k < m1.shape.second; ++k)
                    s += m1.data[i * l1 + k] * m2.data[j * l2 + k];
                data[i * col + j] += s;

            }
    }
    else if(m1.t && !m2.t) {
        row = m1.shape.second;
        col = m2.shape.second;
        // l1 = m1.shape.second;

        for(int i = 0; i < row; ++i)
            for(int j = 0; j < col; ++j) {
                s = 0;
                for(int k = 0; k < m1.shape.first; ++k)
                    s += m1.data[k * l1 + i] * m2.data[k * l2 + j];
                data[i * col + j] += s;
            }
    }
    else {
        row = m1.shape.second;
        col = m2.shape.first;
        for(int i = 0; i < row; ++i)
            for(int j = 0; j < col; ++j) {
                s = 0;
                for(int k = 0; k < m1.shape.first; ++k)
                    s += m1.data[k * l1 + i] * m2.data[j * l2 + k];
                data[i * col + j] += s;
            }
    }
}

void _matmul_tensors(metrix_float& m1, metrix_float& m2, bool t1, bool t2, float* data) {
  //  float* data = _alloc_data(m1,m2);
#ifdef USE_DEBUG        
    assert(&m1);

    assert(!m1.t && !m2.t);
    //std::cout << "m1 shape is: " << '(' << m1.shape.first << ',' << m1.shape.second << ')' << std::endl;
    //std::cout << "m2 shape is: " << '(' << m2.shape.first << ',' << m2.shape.second << ')' << std::endl;
#endif

    float s;
    size_t row, col, l1, l2;
    l1 = m1.shape.second;
    l2 = m2.shape.second;
    if(!t1 && !t2) {

        assert(m1.shape.second == m2.shape.first);
        row = m1.shape.first;
        col = m2.shape.second;
    //    std::cout << "row is: " << row << ' ' << "col is: " << col <<std::endl;
    //    std::cout << "the index is: ";
        
        for(int i = 0; i < row; ++i)
            for(int j = 0; j < col; ++j) {
                s = 0;
                for(int k = 0; k < m1.shape.second; ++k)
                    s += m1.data[i * l1 + k] * m2.data[k * l2+j];
                //std::cout << (i * col + j) << ' ';
                data[i * col + j] += s;
                
            }
#ifdef USE_DEBUG        
        // std::cout << "max index is: " << (col * row - 1) << std::endl;
#endif  
    }
    else if(!t1 && t2) {
        row = m1.shape.first;
        col = m2.shape.first;
        // l1 = m1.shape.second;
        // l2 = m2.shape.fsecond;
        // std::cout << "The Metrix index value is: "; 
        for(int i = 0; i < row; ++i)
            for(int j = 0; j < col; ++j) {
                s = 0;
                for(int k = 0; k < m1.shape.second; ++k)
                    s += m1.data[i * l1 + k] * m2.data[j * l2 + k];
                data[i * col + j] += s;

               // std::cout << (i * col + j) << ' ';
            }
    }
    else if(t1 && !t2) {
        row = m1.shape.second;
        col = m2.shape.second;
        // l1 = m1.shape.second;

        for(int i = 0; i < row; ++i)
            for(int j = 0; j < col; ++j) {
                s = 0;
                for(int k = 0; k < m1.shape.first; ++k)
                    s += m1.data[k * l1 + i] * m2.data[k * l2 + j];
                data[i * col + j] += s;
            }
    }
    else{
        row = m1.shape.second;
        col = m2.shape.first;
       // l = m1.shape.first;
        for(int i = 0; i < row; ++i)
            for(int j = 0; j < col; ++j) {
                s = 0;
                for(int k = 0; k < m1.shape.first; ++k)
                    s += m1.data[k * l1 + i] * m2.data[j * l2 + k];
                data[i * col + j] += s;
            }
    }
  //  return data;
}
void _add_tensors(metrix_float &m, float* data) {
    for(int i = 0; i < m.n; ++i) {
        data[i] += m.data[i];
    }
}
void _add_tensors(metrix_float &m1, metrix_float &m2, float* data) {
    for(int i = 0; i < m1.n; ++i) {
        data[i] += m1.data[i] + m2.data[i];
    }
}
void _add_tensors(std::vector<metrix_float*>& ms, float* data, size_t batch_id) {
    for(auto m : ms) {
        auto p = (m + batch_id)->data;  // std::vector<metrix_float*>& ms并不是batch的形式存储的，而是所有指向该算子的tensor，它们对应的batch需要通过batch_id来索引
        for(int i = 0; i < m->n; ++i) {
            data[i] += p[i];
        }
    }
}
void _sub_tensors(metrix_float &m, float* data) {
    for(int i = 0; i < m.n; ++i) {
        data[i] -= m.data[i];
    }
}
void _sub_tensors(metrix_float &m1, metrix_float &m2, float* data) {
    for(int i = 0; i < m1.n; ++i) {
        data[i] += m1.data[i] - m2.data[i];
    }
}
void _dot_tensors(metrix_float &m1, metrix_float &m2, float* data) {
    for(int i = 0; i < m1.n; ++i) {
        data[i] += m1.data[i] * m2.data[i];
    }
}
void _concat_tensors(std::vector<metrix_float*>& ms, std::vector<size_t>& concat_dim_indexes, float* data, 
    size_t concat_dim, size_t concat_dim_size, size_t shared_dim_size, size_t batch_id) { //要写测试集来验证拼接的正确性！！！！
    for(int m_idx = 0; m_idx < ms.size(); ++m_idx) {
        metrix_float* m = ms[m_idx];
        auto p = (m + batch_id)->data;
        int row = concat_dim == 0 ? m->shape.first : shared_dim_size;
        int col = concat_dim == 0 ? shared_dim_size : m->shape.second;
        int current_dim_size = concat_dim_indexes[m_idx];
        if(concat_dim == 0) {
            // 按行拼接
            for(int i = 0; i < row; ++i) {
                for(int j = 0; j < col; ++j) {
                    data[(i + current_dim_size) * col + j] = p[i * col + j];
                }
            }
        } else {
            // 按列拼接
            for(int i = 0; i < row; ++i) {
                for(int j = 0; j < col; ++j) {
                    data[i * concat_dim_size + j + current_dim_size] = p[i * col + j];
                }
            }
        }
    }

}

void _equal_tensors(metrix_float &m, float* data) {
    for(int i = 0; i < m.n; ++i) {
        data[i] = m.data[i];
    }
}
void _equal_neg_tensors(metrix_float &m, float* data) {
    for(int i = 0; i < m.n; ++i) {
        data[i] = -m.data[i];
    }
}

// metrix_float
void metrix_float::add(metrix_float &m) {
    for(int i = 0; i < this->n; ++i) {
        this->data[i] += m.data[i];
    }
}

void metrix_float::sub(metrix_float &m) {
    for(int i = 0; i < this->n; ++i) {
        this->data[i] -= m.data[i];
    }
}

void metrix_float::matmul(metrix_float &m) {
//    std::cout<<"dsdsda m1.shape.second is "<<this->shape.second<<"m2.shape.second is "<< m.shape.second<<std::endl;
    _size temp = _get_size(*this, m);
    //   std::cout<<"m1.shape.second is "<<this->shape.second<<"m2.shape.second is "<< m.shape.second<<std::endl;
    delete[] this->data;
    this->data = _matmul(*this, m);
    this->t = false;
    this->shape = temp;
    return;
}

void metrix_float::print() {
    if(!this->t)
        for(int i = 0; i < this->shape.first; ++i) {
            for(int j = 0; j < this->shape.second; ++j)
                std::cout<<this->data[i * ((this->shape).second) + j]<<' ';
            std::cout<<std::endl;
        }
    else
        for(int i = 0; i < this->shape.second; ++i) {
            for(int j = 0; j<this->shape.first; ++j)
                std::cout<<this->data[j * ((this->shape).second) + i]<<' ';
            std::cout<<std::endl;
        }

}
std::vector<size_t> metrix_float::get_shape() {
    return {this->shape.first, this->shape.second};
}

// multidimension_metrix_base
void __multidimension_metrix_base::randomUniformInit(int fan_in, int fan_out, init_type init_type) {  // 随机初始化数组（均匀分布）
    // 初始化随机数引擎 
    std::random_device rd;  // 真随机种子（需编译器支持，Windows 可能需用其他种子）
    std::mt19937 gen(rd()); // 梅森旋转算法引擎（高效随机）
    float a, b;
    if (init_type == init_type::xavier) {
        // Xavier 均匀分布：[-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out))]
        float limit = std::sqrt(6.0f / (fan_in + fan_out));
        a = -limit;
        b = limit;
    } else if (init_type == init_type::he) {
        // He 均匀分布：[-sqrt(3/fan_in), sqrt(3/fan_in)]
        float limit = std::sqrt(3.0f / fan_in);
        a = -limit;
        b = limit;
    } else {
        // 简单均匀分布：[-0.5, 0.5]
        a = -0.5f;
        b = 0.5f;
    }
    std::uniform_real_distribution<float> dist(a, b);  // 均匀分布
    for (int i = 0; i < this->n; ++i) 
        *(this->data + i) = dist(gen);  // 生成随机数赋值
}
void __multidimension_metrix_base::_allocdata(int rol, int col) {
    try {
        data = new float[rol * col]();
    } catch (const char* err) { // 捕获字符串类型异常
        std::cout << "exception when initializing metrix data!: " << err << std::endl;
    } catch (...) { // 兜底捕获其他异常
        std::cout << "Unkown error when initializing metrix data!! " << std::endl;
    }    
}

void __multidimension_metrix_base::initialize(int fan_in, int fan_out, init_type init_type){
    this->randomUniformInit(fan_in, fan_out, init_type);
}


} // namespace base