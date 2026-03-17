#include "metrix.h"
#include "dtensor.h"  //要先包含dtensor.h， 因为ops.h 依赖于 dtensor.h 中的定义
#include "ops.h"
#include "nn.h"
#include <iostream>
using namespace std;
using namespace dtensor;
void test_matmul_tensors() {
    float *p1 = new float[12]{1,2,3,4,5,6,7,8,9,10,11,12};
    metrix_float m1(3, 4, p1);
    // 1  2  3  4
    // 5  6  7  8
    // 9 10 11  12
    m1.print();
    metrix_float m2(4, 3, p1);
    // 1   2   3
    // 4   5   6
    // 7   8   9
    //10  11  12
    m2.print();
    float *dt1 = new float[9]();
    _matmul_tensors(m1, m2, false, false, dt1);
    std::cout <<"result is:\n";
    for(int i = 0; i < 3; ++i){
        for(int j = 0; j < 3; ++j)
            cout << dt1[i*3 +j]<< ' ';
        cout<<endl;
    }
    // 70 80 90
    // 158 184 210
    // 246 288 330

    metrix_float m3(3, 4, p1);
    // 1 5 9
    // 2 6 10
    // 3 7 11
    // 4 8 2
    m3.print();
    metrix_float m4(3, 4, p1);
    m4.print();
   // 1  2  3  4
    // 5  6  7  8
    // 9 10 11  12
    float *dt2 = new float[16]();
    _matmul_tensors(m3, m4, true, false, dt2);
    std::cout <<"result is:\n";
    for(int i = 0; i < 4; ++i){
        for(int j = 0; j < 4; ++j)
            cout << dt2[i*4 +j]<< ' ';
        cout<<endl;
    }
    // 107 122 137 152
    // 122 140 158 176
    // 137 158 179 200
    // 152 176 200 224

    metrix_float m5(4, 3, p1);
    // 1 2 3 
    // 4 5 6
    // 7 8 9
    // 10 11 12
    m5.print();
    metrix_float m6(4, 3, p1);
    m6.print();
    // 1 4 7 10
    // 2 5 8 11
    // 3 6 9 12
    float *dt3 = new float[16]();
    _matmul_tensors(m5, m6, false, true, dt3);
    std::cout <<"result is:\n";
    for(int i = 0; i < 4; ++i){
        for(int j = 0; j < 4; ++j)
            cout << dt3[i*4 +j]<< ' ';
        cout<<endl;
    }



    metrix_float m7(4, 3, p1);
    // 1 4 7 10
    // 2 5 8 11
    // 3 6 9 12
    m7.print();
    metrix_float m8(3, 4, p1);
    m8.print();
    // 1 5 9
    // 2 6 10
    // 3 7 11
    // 4 8 12
    float *dt4 = new float[9]();
    _matmul_tensors(m7, m8, true, true, dt4);
    std::cout <<"result is:\n";
    for(int i = 0; i < 3; ++i){
        for(int j = 0; j < 3; ++j)
            cout << dt4[i*3 +j]<< ' ';
        cout<<endl;
    }
    // 70 158 246
    // 80 184 288
    // 90 210 330

}
void test_fun2() {
    std::cout << "start func2...\n";
    int batch_num = 1;
    dtensor_base* l1 = new relu(6, batch_num);  
    dtensor_base* w = new tensor2D_float({5,6}, batch_num);
    // w->add_nopp(l1, *new op("matmul", l1, w)); // 连接操作
    std::cout << "111\n";
    op* my_op = new matmul_op(w, l1); // 创建一个矩阵乘法操作，连接 w 和 l1
    std::cout << "222\n";
    my_op->set_output(tensor_type::layer, sub_type::softmax); // 生成输出张量
    std::cout << "333\n";
    //my_op->get_output()->print(1); // 打印输出张量的梯度值

    float data1[6] = {1, 0.5, 1, -1, 1, 0.5};
    l1->set_input_value(data1, 0); // 设置 l1 的输入值
    float data2[30] = {0, 0.1, 0.5,   -0.5, 0.3, 0.4,  
                        0.1, 0.2, 0.3, 1, -0.5, -2,
                        1, -1, 0.5,    1, -2, 0.5,
                        -0.5, 1, 1,    0.5, 0.4, -0.2,
                        1, -1, 0.4,    -0.3, 0.2, 2}; // 5*6 的权重矩阵，初始化为全零
    w->set_input_value(data2, 0); // 设置 w 的输入值
    std::cout << "444\n";
    my_op->print_info(); // 打印操作信息
    std::cout << "555\n";
    l1->forward(); // 执行 l1 的前向传播，计算输出
    w->forward(); // 执行 w 的前向传播，计算输出
    my_op->forward(); // 执行前向传播
    std::cout << "666\n";
    dtensor_base* otpt = my_op->get_output();
    otpt->forward(); // 执行输出张量的前向传播，计算最终输出
    otpt->print(0); // 打印输出张量的值
    std::cout << "777\n";

    std::vector<std::vector<float>> label = {{0, 1, 0, 0, 0}}; // 假设这是一个 one-hot 标签，表示正确类别是第二个
    otpt->count_loss_grad(label, loss_type::cross_entropy); // 计算损失函数的梯度
    std::cout << "grad count finished\n";
    
    
    std::cout << "starting count grad!!!!!\n";
    // dynamic_cast<layer*>(otpt); // 执行输出张量的反向传播，计算输出的梯度
    otpt->backward(); // 执行操作的反向传播，计算输入张量的梯度
    otpt->print(1); // 打印输出张量的梯度值
    std::cout << "finishing count grad!!!!!\n";
    my_op->backward();//matmul算子执行反向传播

    auto inputs = my_op->get_inputs();
    std::cout << "INPUT SIZE: " << inputs.size() << std::endl;
    for(auto &it : inputs) {
        it->backward();
    }
    for(auto &it : inputs) {
        it->print(1); // 打印输入张量的梯度值
    }  
    l1->backward(); // 执行 l1 的反向传播，计算输入的梯度
    w->backward(); // 执行 w 的反向传播，计算输入的梯度
    l1->print(1); // 打印 l1 的梯度值
    w->print(1); // 打印 w 的梯度值

    otpt->update(0.001);    
    w->update(0.001); // 更新权重参数
    l1->update(0.001); // 更新权重参数
    std::cout << "parameter after updated:...\n";
    otpt->print(1); // 打印输出张量的梯度值
    l1->print(1); // 打印 l1 的梯度值
    w->print(1); // 打印 w 的梯度值
    std::cout <<"finished\n";
    return;
}
int main() { 
  //  test_fun1();
test_matmul_tensors() ;
    test_fun2();
    
}