#pragma once
#include <iostream>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <atomic>
#include <stdexcept>
#include <cassert>
#include "enum_types.h"
#include "metrix.h"
#include "dtensor.h"
#include "nn.h"

namespace nn{
    class Linear_Resnet;
    class module_base;
    class Linear_NN;
}
namespace dtensor {
using namespace nn;
class ThreadPool {
public:
    // 构造函数：创建指定数量的工作线程
    explicit ThreadPool(size_t threads = std::thread::hardware_concurrency())
        : stop(false), task_nums(0)
    {
        if (threads == 0) threads = 1; // 至少1个线程
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        // 等待任务或停止信号
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] {
                            return this->stop || !this->tasks.empty();
                            });
                        // 如果收到停止信号且任务队列为空，则退出线程
                        if (this->stop && this->tasks.empty())
                            return;
                        // 获取下一个任务
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    // 执行任务
                    task();
                }
                });
        }
    }

    // 添加任务到线程池
    void enqueue(module_base* obj, void (module_base::*func)(size_t), size_t batch_id, bool sub_count);
    void enqueue(module_base* obj, void (module_base::*func)(std::vector<float>&, size_t, loss_type), std::vector<float>& label, size_t batch_id, loss_type tp, bool sub_count);
    // 析构函数：停止所有线程
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        // 等待所有线程完成
        for (std::thread& worker : workers)
            worker.join();
    }

    // 获取线程池大小
    size_t size();
    void set_task_nums(size_t num);
    void add_task_nums(size_t num);
    bool have_finished_works();
    // 等待所有任务完成（低开销，替代外部while轮询）
    void wait_for_finish();

private:
    void enqueue_impl(std::function<void()> task_core);
    // 工作线程集合
    std::vector<std::thread> workers;
    // 任务队列
    std::queue<std::function<void()>> tasks; //function
    // 同步原语
    std::mutex queue_mutex;
    std::mutex finish_mutex;  // 配合条件变量的互斥锁
    std::condition_variable condition;
    std::condition_variable finish_cv;  // 用于等待所有任务完成
    // 停止标志
    std::atomic<bool> stop;
    std::atomic<size_t> task_nums;
};

class ThreadPoolImp {
public:
    // 构造函数：创建指定数量的工作线程
    explicit ThreadPoolImp(size_t threads = std::thread::hardware_concurrency())
        :  task_nums(0)
    {

        if (threads == 0) threads = 1; // 至少1个线程
        tasks = std::vector<std::function<void()>>(threads);
        for (size_t i = 0; i < threads; ++i) 
            locks[i].clear();
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this, i] {
                while (true) {
                    std::function<void()> task;
                    while(locks[i].test_and_set(std::memory_order_acquire)); // 等待获取锁, 只能由外部清零
                    task = this->tasks[i];
                    // 执行任务2
                    task();
                    this->task_nums.fetch_sub(1, std::memory_order_release);
                }
            }
            );
        }
    }

    // 添加任务到线程池
    void enqueue(module_base* obj, void (module_base::*func)(size_t), size_t batch_id, bool sub_count);
    void enqueue(module_base* obj, void (module_base::*func)(std::vector<float>&, size_t, loss_type), std::vector<float>& label, size_t batch_id, loss_type tp, bool sub_count);
    // 析构函数：停止所有线程/
    ~ThreadPoolImp() {
        for (std::thread& worker : workers)
            worker.join();
    }

    void set_task_nums(size_t num);
    bool have_finished_works();

private:
    // 工作线程集合
    std::vector<std::thread> workers;
    // 任务槽
    std::vector<std::function<void()>> tasks; //function
    // 任务槽开关
    std::atomic_flag locks[24];
    // 任务数
    std::atomic<size_t> task_nums;
};

} // namespace dtensor