#include"metrix.h"
#include<iostream>
using namespace std;
using namespace base;
int main() {
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
    m1.matmul(m2);
    std::cout <<"result is:\n";
    m1.print();
    // 70 80 90
    // 158 184 210
    // 246 288 330

    metrix_float m3(3, 4, p1, true);
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
    m3.matmul(m4);
    std::cout <<"result is:\n";
    m3.print();

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
    metrix_float m6(4, 3, p1, true);
    m6.print();
    // 1 4 7 10
    // 2 5 8 11
    // 3 6 9 12
    m5.matmul(m6);
    std::cout <<"result is:\n";
    m5.print();


    metrix_float m7(4, 3, p1, true);
    // 1 4 7 10
    // 2 5 8 11
    // 3 6 9 12
    m7.print();
    metrix_float m8(3, 4, p1, true);
    m8.print();
    // 1 5 9
    // 2 6 10
    // 3 7 11
    // 4 8 12
    m7.matmul(m8);
    std::cout <<"result is:\n";
    m7.print();

    // 70 158 246
    // 80 184 288
    // 90 210 330


   cout << "_matmul result1 is:\n";
    metrix_float n1(3, 4, p1), n2(3, 4, p1);
    // 1  2  3  4
    // 5  6  7  8
    // 9 10 11  12   
    float* data1 = new float[3 * 3]();
    _matmul(n1,n2,false,true,data1);
    for(int i = 0; i < 3; ++i){
        for(int j = 0; j < 3; ++j)
            cout << data1[i*3 +j]<< ' ';
        cout<<endl;
    }

    cout << "_matmul result2 is:\n";

    auto p2 = new float[12]{2,3,1,6,5,4,7,9,8,12,11,10};
    metrix_float n3(4, 3, p1), n4(4, 3, p2);

    // 1 2 3
    // 4 5 6
    // 7 8 9
    // 10 11 12

    // 2 3 1
    // 6 5 4
    // 7 9 8
    // 12 11 10
    
    // 195 196 173
    // 222 224 196
    // 249 252 219




    float* data2 = new float[3 * 3]();
    _matmul(n3,n4,true,false,data2);
    for(int i = 0 ;i<3; ++i){
        for(int j = 0;j<3; ++j)
            cout << data2[i*3 +j]<< ' ';
        cout<<endl;
    }

    float* data3 = new float[3 * 5]();
    cout << "_matmul result3 is:\n";

    auto p3 = new float[20]{2,3,1,6,5, 4,7,9,8,12, 11,10,20,19,18, 17,16,15,14,13};
    // 2,3,1,6,5,
    // 4,7,9,8,12,
    // 11,10,20,19,18
    // 17,16,15,14,13

    metrix_float n5(4, 3, p1), n6(4, 5, p3);
    _matmul(n5, n6, true, false, data3);
    for(int i = 0 ;i < 3; ++i){
        for(int j = 0; j < 5; ++j)
            cout << data3[i * 5 + j]<< ' ';
        cout<<endl;
    }

    float *P = new float[64];
    for(int i = 0 ; i < 64; ++i) *(P + i) = (i + 1);
    metrix_float N1(16, 4, P), N2(4, 16, P);
    float* DATA = new float[16 * 16]();
    _matmul(N1, N2, false, false, DATA);
    for(int i = 0; i < 16; ++i){
        for(int j = 0; j < 16; ++j)
            cout << DATA[i * 16 + j]<< ' ';
        cout<<endl;
    }    


    return 0;
}