#include <vector>
#include "dtensor.h"
#include "dtensors/dtensor_1D.h"
#include "dtensors/dtensor_2D.h"
#include "dtensors/layers/layers.h"
#include "ops.h"
#include "ops/basic_ops.h"
using namespace dtensor;
std::vector<float> nums1 = {0.0, 0.5, 0.5, 0.6, 
                            0.2, 0.3, 0.4, 0.7,
                            1.0, 0.9, 0.0, 0.3};
std::vector<float> nums2 = {0.1, 0.2, 0.3, 0.1, 
                            1.0, 0.9, 0.8, 0.5,
                            0.2, 0.3, 0.6, 0.7};
void test_add_tensor2D() {
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

void test_sub_tensor2D() {
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

void test_add_layer() {
    std::cout << " ======== Add Op Layer Test ======== \n";
    int batch_size = 3;  // 只能选 1， 因为 weight metrix 只有一个通道

    // 初始化 l1
    layer* l1 = new origin(8, batch_size);
    std::vector<std::vector<float>> inputs1 = 
    {
        {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7},
        {0.4, 0.3, 0.2, 0.1, 0.0, 0.7, 0.6, 0.5},
        {0.0, 0.1, 0.2, 0.7, 0.6, 0.5, 0.4, 0.3}
    };
    l1->__set_inputs__(inputs1);

    // 初始化 l2
    layer* l2 = new sigmoid(8, batch_size);
    std::vector<std::vector<float>> inputs2 = 
    {
        {-0.5, 1.0, 0.5, 0.6, 0.4, -0.5, -0.6, -0.7},
        {1.2, 1.3, -0.2, 1.1, 2.0, 2.7, -1.6, 0.5},
        {0.0, -0.1, -1.2, 0.2, 0.3, 0.6, 0.8, -0.3}
    };
    l2->__set_inputs__(inputs2);

    // 初始化 l3
    layer* l3 = new relu(8, batch_size);
    std::vector<std::vector<float>> inputs3 = 
    {
        {-0.5, 1.0, 0.5, 0.6, 0.4, -0.5, -0.6, -0.7},
        {1.2, 1.3, -0.2, 1.1, 2.0, 2.7, -1.6, 0.5},
        {0.0, -0.1, -1.2, 0.2, 0.3, 0.6, 0.8, -0.3}
    };
    l3->__set_inputs__(inputs3);

    // 运算结合得到新的 layer
    std::vector<dtensor_base*> input_tensors = {static_cast<dtensor_base*>(l1), 
                                                static_cast<dtensor_base*>(l2),
                                                static_cast<dtensor_base*>(l3)};

    // 创建算子
    op* addop = new add_op(input_tensors);
    dtensor_base* l4 = addop->set_output(tensor_type::layer, sub_type::origin);
    addop->print_info();

    // 打印各层输出
    std::cout << "\nLayer 1 initial val is: \n\n";
    l1->forward(); // 必须在 addop 创建之后 forward！！！  才能使 forward 运行后 计入 addop 的 temp_n 的值！！
    for(int i = 0; i < batch_size; ++i) l1->_print_val(i);
    std::cout << "\nLayer 2 initial val is: \n\n";
    l2->forward();
    for(int i = 0; i < batch_size; ++i) l2->_print_val(i);
    l3->forward(); 
    std::cout << "\nLayer 3 initial val is: \n\n";
    for(int i = 0; i < batch_size; ++i) l3->_print_val(i);

    // 设置 l4 梯度
    std::vector<std::vector<float>> grads = 
    {
        {-1.0, 1.0, 0.5, -0.5, -1.0, -2.0, 2.0, 1.0},
        {1.0, -1.5, -1.0, 1.0, 2.0, -2.0, -1.0, 0.5},
        {0.0, -1.1, -1.2, 1.2, 0.4, 0.6, 0.8, -0.5}
    };
    l4->__set_grads__(grads);
    std::cout << "\nOUT PUT layer initial val is: \n";
    for(int i = 0; i < batch_size; ++i) l4->_print_val(i);
    std::cout << "\nOUT PUT Metrix initial grad is: \n";
    for(int i = 0; i < batch_size; ++i) l4->_print_grad(i);

    // 前向传播
    std::cout << "\nDO forward operation...\n\n";
    addop->forward();
    std::cout << "l4 OUTPUT Metrix val after forward is: \n";
    for(int i = 0; i < batch_size; ++i) l4->_print_val(i);

    // === output === //
    // origin
    // 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 
    // 0.4 0.3 0.2 0.1 0 0.7 0.6 0.5 
    // 0 0.1 0.2 0.7 0.6 0.5 0.4 0.3 
    // sigmoid
    // 0.377541 0.731059 0.622459 0.645656 0.598688 0.377541 0.354344 0.331812 
    // 0.768525 0.785835 0.450166 0.75026 0.880797 0.937027 0.167982 0.622459 
    // 0.5 0.475021 0.231475 0.549834 0.574443 0.645656 0.689974 0.425557 
    // relu
    // 0 1 0.5 0.6 0.4 0 0 0 
    // 1.2 1.3 0 1.1 2 2.7 0 0.5 
    // 0 0 0 0.2 0.3 0.6 0.8 0 
    // === 输出layer input（等于三者之和）===
    // 0.377541 1.83106 1.32246 1.54566 1.39869 0.877541 0.954344 1.03181 
    // 2.36852 2.38583 0.650166 1.95026 2.8808 4.33703 0.767982 1.62246 
    // 0.5 0.575021 0.431475 1.44983 1.47444 1.74566 1.88997 0.725558 

    // 反向传播
    std::cout << "\nDO backward operation...\n\n";
    addop->backward();
    l1->_backward(); l2->_backward(); l3->_backward();
    std::cout << "l1 Grad val after backward is: \n";
    for(int i = 0; i < batch_size; ++i) l1->_print_grad(i);
    std::cout << "l2 Grad val after backward is: \n";
    for(int i = 0; i < batch_size; ++i) l2->_print_grad(i);
    std::cout << "l3 Grad val after backward is: \n";
    for(int i = 0; i < batch_size; ++i) l3->_print_grad(i);
}

int main() {
    // test_add_tensor2D(); // 成功！！！
    // test_sub_tensor2D(); // 成功！！！
    test_add_layer(); // 成功！！！
    return 0;
}