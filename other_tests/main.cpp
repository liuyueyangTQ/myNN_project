#include<iostream>
using namespace std;
class A{
    int a, b;
public:
    A(int x, int y): a(x), b(y) {}
};
int main() {
    A** p = new A*[10];
    return 0;
}