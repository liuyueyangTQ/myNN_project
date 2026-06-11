#include "common_definitions.h"
#include <vector>
using namespace dtensor;
std::vector<float> nums1 = {0.0, 0.5, 0.5, 0.6, 
                            0.2, 0.3, 0.4, 0.7,
                            1.0, 0.9, 0.0, 0.3};
std::vector<float> nums2 = {0.1, 0.2, 0.3, 0.1, 
                            1.0, 0.9, 0.8, 0.5,
                            0.2, 0.3, 0.6, 0.7};
void test_add_opp() {
    std::cout << " ======== Add Op Test ======== \n";
    int batch_size = 1;  // 只能选 1， 因为 weight metrix 只有一个通道
    // 初始化一个权重矩阵
    dtensor_base* t1 = new tensor2D_float({3,4}, batch_size, nums1);
    t1->__set_isparam_lockgrad__(true, false); // 不向后传播梯度，但允许更新梯度
    std::cout << "Metrix 1 initial val is: \n";
    t1->_print_val(0);
    std::cout << "Metrix 1 initial grad is: \n";
    t1->clear_grad();
    for(int i = 0; i < batch_size; ++i) t1->_print_grad(i);

    // 初始化一个普通tensor
    dtensor_base* t2 = new tensor2D_float({3,4}, batch_size, nums2);
    t2->__set_isparam_lockgrad__(true, false); // 不向后传播梯度，但允许更新梯度
    std::cout << "Metrix 2 initial val is: \n";
    t2->_print_val(0);
    std::cout << "Metrix 2 initial grad is: \n";
    t2->clear_grad();
    for(int i = 0; i < batch_size; ++i) t2->_print_grad(i);

    // 运算结合得到新的tensor
    op* addop = new add_op(t1, t2);
    dtensor_base* t3 = addop->set_output(tensor_type::tensor2D, sub_type::none);
    t3->__set_isparam_lockgrad__(false, false); // 向后传播梯度，且允许更新梯度！！！
    std::cout << "OUT PUT Metrix initial val is: \n";
    t3->_print_val(0);
    std::cout << "OUT PUT Metrix initial grad is: \n";
    t3->clear_grad();
    for(int i = 0; i < batch_size; ++i) t2->_print_grad(i);

    // 前向传播
    std::cout << "DO forward operation...\n";
    addop->forward();
    std::cout << "t3 OUTPUT Metrix val after forward is: \n";
    t3->_print_val(0);

    // add 结果：
    // 0.1 0.7 0.8 0.7
    // 1.2 1.2 1.2 1.2
    // 1.2 1.2 0.6 1.0

    // 设置 t3 的梯度
    std::vector<std::vector<float>> test_grad = {{1,2,3,4,5,6,7,8,9,0,11,0}};
    t3->__set_grads__(test_grad);
    // 反向传播
    std::cout << "DO forward operation...\n";
    addop->backward();
    std::cout << "t1 Grad val after backward is: \n";
    for(int i = 0; i < batch_size; ++i) t1->_print_grad(i);
    std::cout << "t2 Grad val after backward is: \n";
    for(int i = 0; i < batch_size; ++i) t2->_print_grad(i);
}

void test_sub_opp() {
    std::cout << " ======== Sub Op Test ======== \n";
    int batch_size = 1;  // 只能选 1， 因为 weight metrix 只有一个通道
    // 初始化一个权重矩阵
    dtensor_base* t1 = new tensor2D_float({3,4}, batch_size, nums1);
    t1->__set_isparam_lockgrad__(true, false); // 不向后传播梯度，但允许更新梯度
    std::cout << "Metrix 1 initial val is: \n";
    t1->_print_val(0);
    std::cout << "Metrix 1 initial grad is: \n";
    t1->clear_grad();
    for(int i = 0; i < batch_size; ++i) t1->_print_grad(i);

    // 初始化一个普通tensor
    dtensor_base* t2 = new tensor2D_float({3,4}, batch_size, nums2);
    t2->__set_isparam_lockgrad__(true, false); // 不向后传播梯度，但允许更新梯度
    std::cout << "Metrix 2 initial val is: \n";
    t2->_print_val(0);
    std::cout << "Metrix 2 initial grad is: \n";
    t2->clear_grad();
    for(int i = 0; i < batch_size; ++i) t2->_print_grad(i);

    // 运算结合得到新的tensor
    op* addop = new sub_op(t1, t2);
    dtensor_base* t3 = addop->set_output(tensor_type::tensor2D, sub_type::none);
    t3->__set_isparam_lockgrad__(false, false); // 向后传播梯度，且允许更新梯度！！！
    std::cout << "OUT PUT Metrix initial val is: \n";
    t3->_print_val(0);
    std::cout << "OUT PUT Metrix initial grad is: \n";
    t3->clear_grad();
    for(int i = 0; i < batch_size; ++i) t2->_print_grad(i);

    // 前向传播
    std::cout << "DO forward operation...\n";
    addop->forward();
    std::cout << "t3 OUTPUT Metrix val after forward is: \n";
    t3->_print_val(0);

    // sub 结果
    // -0.1 0.3 0.2 0.5
    // -0.8 -0.6 -0.4 0.2
    // 0.8 0.6 -0.6 -0.4

    // 设置 t3 的梯度
    std::vector<std::vector<float>> test_grad = {{1,2,3,4,5,6,7,8,9,0,11,0}};
    t3->__set_grads__(test_grad);
    // 反向传播
    std::cout << "DO forward operation...\n";
    addop->backward();
    std::cout << "t1 Grad val after backward is: \n";
    for(int i = 0; i < batch_size; ++i) t1->_print_grad(i);
    std::cout << "t2 Grad val after backward is: \n";
    for(int i = 0; i < batch_size; ++i) t2->_print_grad(i);
}
int main() {
    test_add_opp(); // 成功！！！
    test_sub_opp(); // 成功！！！
    return 0;
}