#include "metrix.h"
#include "dtensor.h"  //先包含dtensor.h，再包含ops.h
#include "ops.h"
#include "nn.h"
#include <iostream>
using namespace std;
using namespace dtensor;
int main() {
    for(int i = 0; i < 10000; ++i) {
        metrix_float m(100, 100);
    }
    cout << "metrix test passed!\n";
    for(int i = 0; i < 10000; ++i) {
        tensor2D_float t({100, 100}, 10);
    }
    cout << "tensor test passed!\n";
    for(int i = 0; i < 1000; ++i) {
        nn::Linear_NN nn(10);
        for(int j = 0; j < 10; ++j)
            nn.add_layer(100, sub_type::relu);
    }
    cout << "nn test passed!\n";
    return 0;
}