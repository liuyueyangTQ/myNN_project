#include <algorithm>
#include <execution>
#include <cstring>
#include <chrono>
#include <iostream>
// 使用 __restrict 修饰指针，解除别名限制
void _parallel_add(float* __restrict a, float* __restrict b, int n) {
    // 零线程开销，纯单线程硬件寄存器加速
    std::transform(std::execution::unseq, 
                   a, a + n, 
                   b, 
                   a, 
                   [](float di, float si) { return di + si; });
}

int main() {
    int n, times;
    std::cout << "Please input the n and times: \n";
    std::cin >> n >> times;
    float* a = new float[n]();
    float* b = new float[n]();
    for(int i = 0; i < n; ++i) {
        b[i] = 1; a[i] = 0;
    }
        
    std::cout << "Add with normal loop...:\n";
    memset(a, 0, n * sizeof(float));
    auto start_time = std::chrono::high_resolution_clock::now();
    for(int t = 0; t < times; ++t) {
        //memset(a, 0, n * sizeof(float));
        for(int i = 0; i < n; ++i) {
            a[i] += + b[i];
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto res =  std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    std::cout << "Cost time: " << res << "\n";
    std::cout << "res is: "; for(int i = 0; i < 10; ++i) std::cout << a[i] << ' '; std::cout << "...\n";

    std::cout << "Add with SIMD acceleration...:\n";
    memset(a, 0, n * sizeof(float));
    start_time = std::chrono::high_resolution_clock::now();
    for(int t = 0; t < times; ++t) {
        _parallel_add(a, b, n);
    }
    end_time = std::chrono::high_resolution_clock::now();
    res =  std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    std::cout << "Cost time: " << res << "\n";
    std::cout << "res is: "; for(int i = 0; i < 10; ++i) std::cout << a[i] << ' '; std::cout << "...\n";

}