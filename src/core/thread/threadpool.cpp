#include"threadpool.h"
namespace tensor{
size_t ThreadPool::size() {
    // 加锁访问tasks.size()，确保线程安全
    std::unique_lock<std::mutex> lock(queue_mutex);
    return tasks.size();
}
void ThreadPool::set_task_nums(size_t num) {
    assert(this->task_nums.load() == 0);
    this->task_nums.store(num);
}
void ThreadPool::add_task_nums(size_t num) {
    this->task_nums.fetch_add(num);
}
void ThreadPool::enqueue(func f, tensor::loss_type loss_tp, size_t batch_id) 
{   
    if (f == nullptr) {
        throw std::invalid_argument("func cannot be nullptr");
    }

    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        // 线程池已停止 → 拒绝提交
        if (stop) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }
        // 构造任务并加入队列
        auto wrapped_task = //std::bind(f, loss_tp, batch_id);
            [this, f, loss_tp, batch_id]() {
                try {
                    // 第一步：执行目标函数f(loss_tp, batch_id)
                    f(loss_tp, batch_id);
                } catch (const std::exception& e) {
                    // 捕获f的异常，避免任务崩溃导致计数不递减
                    std::cerr << "Task " << batch_id <<" failed: " << e.what() << std::endl;
                } catch (...) {
                    std::cerr << "Task " << batch_id <<" failed with unknown error" << std::endl;
                }

                // 第二步：无论f是否抛出异常，都执行原子计数--
                //this->task_nums.fetch_sub(1); //这一段放进了构造函数里

                // // （可选）如果计数为0，通知外部所有任务完成
                // if (this->task_nums.load() == 0) {
                //     this->cv_idle.notify_all();
                // }
            };
        tasks.emplace(std::move(wrapped_task));
     
    }

    // 通知一个等待的线程
    condition.notify_one();
}

void ThreadPool::enqueue(layer* obj, void (layer::*mem_func)(size_t), size_t batch_id) {   
    if (mem_func == nullptr) {
        throw std::invalid_argument("func cannot be nullptr");
    }

    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        // 线程池已停止 → 拒绝提交
        if (stop) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }
        // 构造任务并加入队列
        auto wrapped_task = //std::bind(f, loss_tp, batch_id);
            [this, obj, mem_func, batch_id]() {
                try {
                    // 第一步：执行目标函数f(loss_tp, batch_id)
                    (obj->*mem_func)(batch_id);
                } catch (const std::exception& e) {
                    // 捕获f的异常，避免任务崩溃导致计数不递减
                    std::cerr << "Task " << batch_id <<" failed: " << e.what() << std::endl;
                } catch (...) {
                    std::cerr << "Task " << batch_id <<" failed with unknown error" << std::endl;
                }

                // 第二步：无论f是否抛出异常，都执行原子计数--
                //this->task_nums.fetch_sub(1);

                // // （可选）如果计数为0，通知外部所有任务完成
                // if (this->task_nums.load() == 0) {
                //     this->cv_idle.notify_all();
                // }
            };
        tasks.emplace(std::move(wrapped_task));
     
    }

    // 通知一个等待的线程
    condition.notify_one();
}

void ThreadPool::enqueue(layer* obj, void (layer::*mem_func)(std::vector<float>&, tensor::loss_type, size_t), std::vector<float>& label, tensor::loss_type loss_tp, size_t batch_id) {
    if (mem_func == nullptr) {
        throw std::invalid_argument("func cannot be nullptr");
    }

    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        // 线程池已停止 → 拒绝提交
        if (stop) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }
        // 构造任务并加入队列
        auto wrapped_task = //std::bind(f, loss_tp, batch_id);
            [this, obj, mem_func, &label, loss_tp, batch_id]() {
                try {
                    // 第一步：执行目标函数f(loss_tp, batch_id)
                    (obj->*mem_func)(label, loss_tp, batch_id);
                } catch (const std::exception& e) {
                    // 捕获f的异常，避免任务崩溃导致计数不递减
                    std::cerr << "Task " << batch_id <<" failed: " << e.what() << std::endl;
                } catch (...) {
                    std::cerr << "Task " << batch_id <<" failed with unknown error" << std::endl;
                }

                // 第二步：无论f是否抛出异常，都执行原子计数--
                //this->task_nums.fetch_sub(1);

                // // （可选）如果计数为0，通知外部所有任务完成
                // if (this->task_nums.load() == 0) {
                //     this->cv_idle.notify_all();
                // }
            };
        tasks.emplace(std::move(wrapped_task));
     
    }

    // 通知一个等待的线程
    condition.notify_one();
}

void ThreadPool::enqueue(tensor2D_float* obj, void (tensor2D_float::*mem_func)(size_t), size_t batch_id) {   
    if (mem_func == nullptr) {
        throw std::invalid_argument("func cannot be nullptr");
    }

    {
        
        // 线程池已停止 → 拒绝提交
        if (stop) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }
        // 构造任务并加入队列
        auto wrapped_task = //std::bind(f, loss_tp, batch_id);
            [this, obj, mem_func, batch_id]() {
                try {
                    // 第一步：执行目标函数f(loss_tp, batch_id)
                    (obj->*mem_func)(batch_id);
                } catch (const std::exception& e) {
                    // 捕获f的异常，避免任务崩溃导致计数不递减
                    std::cerr << "Task " << batch_id <<" failed: " << e.what() << std::endl;
                } catch (...) {
                    std::cerr << "Task " << batch_id <<" failed with unknown error" << std::endl;
                }

                // 第二步：无论f是否抛出异常，都执行原子计数--
                //this->task_nums.fetch_sub(1);

                // // （可选）如果计数为0，通知外部所有任务完成
                // if (this->task_nums.load() == 0) {
                //     this->cv_idle.notify_all();
                // }
            };
        std::unique_lock<std::mutex> lock(queue_mutex);
        tasks.emplace(std::move(wrapped_task));
     
    }

    // 通知一个等待的线程
    condition.notify_one();
}

void ThreadPool::enqueue(Linear_NN* obj, void (Linear_NN::*mem_func)(std::vector<float>&, size_t), std::vector<float>& sample, size_t batch_id) {
    {

        // 线程池已停止 → 拒绝提交
        if (stop) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }



        auto wrapped_task = // 构造任务并加入队列
                
            [this, obj, mem_func, &sample, batch_id]() {
                try {
                    // 第一步：执行目标函数f(loss_tp, batch_id)
                    (obj->*mem_func)(sample, batch_id);
                } catch (const std::exception& e) {
                    // 捕获f的异常，避免任务崩溃导致计数不递减
                    std::cerr << "Task " << batch_id <<" failed: " << e.what() << std::endl;
                } catch (...) {
                    std::cerr << "Task " << batch_id <<" failed with unknown error" << std::endl;
                }

                // 第二步：无论f是否抛出异常，都执行原子计数--
                //this->task_nums.fetch_sub(1);
                
            };
        std::unique_lock<std::mutex> lock(queue_mutex);
        tasks.emplace(std::move(wrapped_task));
     
    }
    // 通知一个等待的线程
    condition.notify_one();
}

void ThreadPool::enqueue(func3 f) {
    {
        
        // 线程池已停止 → 拒绝提交
        if (stop) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }



        auto wrapped_task = // 构造任务并加入队列
                
            [this, f]() {
                try {
                    // 第一步：执行目标函数f(loss_tp, batch_id)
                    f();
                } catch (const std::exception& e) {
                    // 捕获f的异常，避免任务崩溃导致计数不递减
                    std::cerr << "Task failed: " << e.what() << std::endl;
                } catch (...) {
                    std::cerr << "Task failed with unknown error" << std::endl;
                }

                // 第二步：无论f是否抛出异常，都执行原子计数--
                //this->task_nums.fetch_sub(1);
                
            };
        std::unique_lock<std::mutex> lock(queue_mutex);  //减小粒度！！！！！
        tasks.emplace(std::move(wrapped_task));
     
    }
    // 通知一个等待的线程
    condition.notify_one();
}


void ThreadPool::wait_for_finish() {
    std::unique_lock<std::mutex> lock(finish_mutex);
    // while循环避免虚假唤醒：即使被唤醒，也要再次检查num是否为0
    finish_cv.wait(lock, [this]() { return task_nums == 0; });
    return;
}
void ThreadPool::enqueue_impl(std::function<void()> task_core) {

}
bool ThreadPool::have_finished_works() {
    return this->task_nums.load() == 0;
}
}