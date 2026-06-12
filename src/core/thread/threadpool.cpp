#include"threadpool.h"

namespace dtensor {
void ThreadPool::enqueue(module_base* obj, void (module_base::*func)(size_t), size_t batch_id, bool sub_count) {
    // 线程池已停止 → 拒绝提交
    if (stop) {
        throw std::runtime_error("enqueue on stopped ThreadPool");
    }
    auto wrapped_task = // 构造任务并加入队列    
        [this, obj, func, batch_id, sub_count]() {
                (obj->*func)(batch_id);      
                if(sub_count) 
                    this->task_nums--; // 原子递减，获取递减前的值
        };
    std::unique_lock<std::mutex> lock(queue_mutex);
    tasks.emplace(std::move(wrapped_task));
    // 通知一个等待的线程
    condition.notify_one();
}
void ThreadPool::enqueue(module_base* obj, void (module_base::*func)(std::vector<float>&, size_t, loss_type), std::vector<float>& label, size_t batch_id, loss_type tp, bool sub_count) {
    // 线程池已停止 → 拒绝提交
    if (stop) {
        throw std::runtime_error("enqueue on stopped ThreadPool");
    }
    auto wrapped_task = // 构造任务并加入队列    
        [this, obj, func, &label, batch_id, tp, sub_count]() {
                (obj->*func)(label, batch_id, tp); 
                if(sub_count) 
                    this->task_nums--; // 原子递减，获取递减前的值           
        };
    std::unique_lock<std::mutex> lock(queue_mutex);
    tasks.emplace(std::move(wrapped_task));
    // 通知一个等待的线程
    condition.notify_one();
}

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


void ThreadPoolImp::enqueue(module_base* obj, void (module_base::*func)(size_t), size_t batch_id, bool sub_count) {
    auto wrapped_task = // 构造任务并加入任务槽  
        [this, obj, func, batch_id]() {
                (obj->*func)(batch_id);      
        };
    tasks[batch_id] = wrapped_task;
    locks[batch_id].clear(std::memory_order_release); // 将锁置 0， 此时内部worker能看到，立即执行任务
}
void ThreadPoolImp::enqueue(module_base* obj, void (module_base::*func)(std::vector<float>&, size_t, loss_type), std::vector<float>& label, size_t batch_id, loss_type tp, bool sub_count) {
    auto wrapped_task = // 构造任务并加入队列    
        [this, obj, func, &label, batch_id, tp, sub_count]() {
                (obj->*func)(label, batch_id, tp);           
        };
    tasks[batch_id] = wrapped_task;
    locks[batch_id].clear(std::memory_order_release);
}

void ThreadPoolImp::set_task_nums(size_t num) {
    task_nums.store(num, std::memory_order_seq_cst);
}

bool ThreadPoolImp::have_finished_works() {
    return task_nums.load(std::memory_order_acquire) == 0;
}

} // namespace dtensor