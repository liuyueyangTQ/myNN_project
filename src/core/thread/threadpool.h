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
#include "metrix.h"
#include "tensor.h"
#include "model.h"

namespace tensor{

using func = void (*)(tensor::loss_type, size_t);
using func2 = void (*)(size_t);
using func3 = void (*)();
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
                    
                    int current_num = task_nums--;  // 原子递减，获取递减前的值
                    // 当任务计数变为0时，唤醒等待线程
                    if (current_num == 1) {  // 递减前是1，递减后为0
                        std::lock_guard<std::mutex> lock(finish_mutex);
                        finish_cv.notify_one();  // 唤醒一个等待线程
                    }
                }
                });
        }
    }


    // 添加任务到线程池
    void enqueue(func f, tensor::loss_type loss_tp, size_t batch_id);//Linear_NN::forward(std::vector<float>& sample, size_t batch_id)
    void enqueue(layer* obj, void (layer::*mem_func)(size_t), size_t batch_id); 
    void enqueue(layer* obj, void (layer::*mem_func)(std::vector<float>&, tensor::loss_type, size_t), std::vector<float>& label, tensor::loss_type loss_tp, size_t batch_id);
    void enqueue(tensor2D_float* obj, void (tensor2D_float::*mem_func)(size_t), size_t batch_id); 
    void enqueue(Linear_NN* obj, void (Linear_NN::*mem_func)(std::vector<float>&, size_t), std::vector<float>& sample, size_t batch_id);
    void enqueue(func3 f);
    // 析构函数：停止所有线程
    ~ThreadPool() {
        {
            // 设置停止标志
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }

        // 通知所有线程
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

} // namespace tensor