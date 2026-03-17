#include "metrix.h"
#include "tensor.h"
#include <iostream>
using namespace tensor;
void test_fun1() {
    std::cout << "start func1...\n";
    layer* l1 = new sigmoid(20,5);
    layer* l2 = new relu(20,5);
    layer* l3 = new layer_norm(10,5);

    tensor2D_float* t = new tensor2D_float({8,6}, 5);

    l1->print(false);
    l1->print_batches();
    l2->print_batches();
    l3->print_batches();
    t->print(false);

    delete l1;
    delete l2;
    delete l3;
    delete t;
}
void test_fun2() {
    std::cout << "start func2...\n";
    int batch_num = 5;
    auto l1 = new sigmoid(200, batch_num);  // 数字大了会发生段错误？？
    auto w = new tensor2D_float({150,200}, batch_num);
    l1->add_wnext(w);
    w->set_layer_x(l1);
    auto l2 = new sigmoid(150, batch_num);
    w->set_layer_next(l2);
    l2->add_wlast(w);
    l1->set_layer_next(l2);
    // for(int i = 0 ;i < 5; ++i) {
    //     std::cout << "w weight address is: " << w->get_weight() << std::endl;
    //     std::cout << "l1 batch_out address is: " << (l1->get_batch_output() + i) << std::endl;
    //     std::cout << "l2 batch_input address is: " << (l1->get_next()->get_batch_input() + i) << std::endl;
    // }
    float* temp = new float[150];
    for(int i = 0 ;i < batch_num; ++i) {
        std::cout << " matmul the" << i+1 << "-th batch...\n";
        std::cout << "w weight address is: " << w->get_weight() << std::endl;
        std::cout << "l1 batch_out address is: " << (l1->get_batch_output() + i) << std::endl;
        std::cout << "l2 batch_input address is: " << (l1->get_next()->get_batch_input() + i) << std::endl;
        //auto f = _matmul(*(w->get_weight()), *(l1->get_batch_output() + i));
        _matmul(*(w->get_weight()), *(l1->get_batch_output() + i), false, false,   // 计算下一层的input
                (l1->get_next()->get_batch_input() + i)->data // 下一层第 batch_id 个样本
            ); 
    }
    for(int i = 0 ;i< 150; ++i)std::cout << (*(temp +i)) <<' ';
    // w->print();
    // l1->print_batches();
    // l2->print_batches();
    
}
int main(){
  //  test_fun1();

    test_fun2();
    
}