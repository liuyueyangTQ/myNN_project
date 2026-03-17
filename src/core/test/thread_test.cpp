#include<iostream>
#include<chrono>
#include"metrix.h"
#include"tensor.h"
#include"model.h"
#include"threadpool.h"
using namespace tensor;
using namespace base;
void thread_func(loss_type tp, size_t id) {
    std::cout << "The " << id << "-th function is running...\n";
}
int pp[100];
void thread_func1(loss_type tp, size_t id) {
    *(pp+id) = id;
    
}
void time_func() {
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
}
int main (){
ThreadPool pool(8);
for(int j = 0; j < 4;j ++ ){
pool.set_task_nums(16);

for(int i = 0; i < 16; ++i) {
    pool.enqueue(thread_func, loss_type::mse, i + 1);
}
while(!pool.have_finished_works()) ;
std::cout <<std::endl;
std::cout << "Task " << j << " finished!\n";
std::cout <<std::endl;
}
std::cout << "All tasks finished!!!!";

pool.set_task_nums(100);
for(int i = 0; i < 100; ++i) {
    pool.enqueue(thread_func1, loss_type::mse, i);
}
std::cout << "count result is: \n";
for(int i = 0; i < 100; ++i)
std::cout << pp[i]<<' ';

std::cout << "start time tasks...\n";
pool.set_task_nums(8);
for(int i = 0; i < 8; ++i) {
    pool.enqueue(time_func);
}
std::cout << "waiting for time tasks to finish...\n";
pool.wait_for_finish();
std::cout << "time tasks finished!\n";

}