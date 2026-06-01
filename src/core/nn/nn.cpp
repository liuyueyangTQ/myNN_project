#include"metrix.h"
#include"dtensor.h"
#include"ops.h"
#include"threadpool.h"
#include"nn.h"
namespace nn{
using namespace dtensor;
void NNParams::check() {
    std::cout << "====== Checking NN Parameters: ======\n";
    std::cout << "Model type is: " << (model_type == nn_type::Linear_NN ? "LinearNN" : "Linear_ResNet") << "\n"; 
    assert(batch_size == thread_num);
    assert(layer_sizes.size() == layer_num);
    assert(layer_types.size() == layer_num);
    assert(input_output_dim.first == layer_sizes[0]);
    assert(input_output_dim.second == layer_sizes.back());
    assert(samples % batch_size == 0);
    std::cout << "layer_num is: " << layer_num << "\n";
    std::cout << "Batch Size is: " << batch_size << "\n";
    std::cout << "Thread Num is: " << thread_num << "\n"; 
    std::cout << "Using Multithread Training: " << (use_multithread ? "[Yes]" : "[No]") << "\n";
    std::cout << "Input Dimension: " << input_output_dim.first << "\n";
    std::cout << "Output Dimension: " << input_output_dim.second << "\n";
    std::cout << "Sample Number: " << samples << "\n";
    std::cout << "Layer Sizes: "; for(int i = 0; i < layer_num; ++i) std::cout << layer_sizes[i] << ' '; std::cout << "\n";
    std::cout << "Layer Types: "; for(int i = 0; i < layer_num; ++i) std::cout << (static_cast<int>(layer_types[i])) << ' '; std::cout << "\n";
    std::cout << "====== Finished checking! ======\n";
}
module_base* model_data::get_model() {
    return this->model_ptr;
}
void model_data::set_model(module_base* nn) {
    this->model_ptr = nn;
}
void model_data::check_model() {
    assert(this->model_ptr != nullptr);
    assert(this->layer_sizes.size() == this->layer_types.size());
    assert(this->layer_sizes.size() == this->outputs[0].size() && this->layer_sizes.size() == this->param_b.size() && this->layer_sizes.size() == this->param_w.size() + 1);
    for(int i = 0; i < this->layer_sizes.size(); ++i) {
        assert(this->param_w[i].size() == this->layer_sizes[i + 1] && this->param_w[i][0].size() == this->layer_sizes[i]);
    }
    return;
}

void module_base::set_learning_rate(double lr) {
    this->lr = lr;
}
void module_base::set_thread_pool(ThreadPool* pool) {
    this->thread_pool = pool;
}
void module_base::set_thread_pool(ThreadPoolImp* pool) {
    this->thread_pool_imp = pool;
}
void module_base::reshuffle_data() {
    assert(samples > 0 && groups > 0);
    assert(samples == (this->train_data).size() * this->batch_num);
    std::vector<int> indices(samples);
    for(int i = 0; i < samples; ++i)
        indices[i] = i;
    // 打乱索引
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    // 根据打乱的索引重新排列数据和标签
    for(int i = 0; i < groups; ++i) {
        for(int j = 0; j < this->batch_num; ++j) {
            int idx = i * this->batch_num + j;
            train_data[i][j] = this->data[indices[idx]];
            this->train_labels[i][j] = this->labels[indices[idx]];
        }
    }
    return;
}
void module_base::train_one_epoch(double lr) {
    this->reshuffle_data();
    //std::cout << " train one epoch...\n";
    for(int i = 0; i < groups; ++i) {
        this->clear_grad();
        this->clear_samples(); //非常重要！！！因为 _add_tensors 等实现逻辑是累加式的！！
        this->reset_count();   // 将每个tensor和op的temp_n重置！非常重要！
        //std::cout << " grad cleared...\n";
        this->set_input_value(this->train_data[i]);
        //std::cout << " input set...\n";
        this->forward();
        //std::cout << " forward finished!!!!!!!!!!!!...\n";        
        this->backward(this->train_labels[i], loss_type::cross_entropy);
        //std::cout << " backward finished...\n";
        this->update_parameters(lr);
        //std::cout << " parameter updated...\n";
    }
}
void module_base::train_model(int epochs, double lr) {
    for(int i = 0; i < epochs; ++i)  {
        train_one_epoch(lr);
    }
}
void module_base::train_one_epoch_mul(double lr, size_t pool_type) {
    this->reshuffle_data();
    for(int i = 0; i < groups; ++i) {
        this->clear_grad();
        this->clear_samples(); //非常重要！！！因为 _add_tensors 等实现逻辑是累加式的！！
        this->reset_count();   // 将每个tensor和op的temp_n重置！非常重要！
        this->set_input_value(this->train_data[i]);
        if(pool_type == 1) {
            // 设置同步任务数，等待前向传播全部完成后才能继续
            thread_pool->set_task_nums(this->batch_num); //设置任务计数用于同步
            for(size_t batch_id = 0; batch_id < this->batch_num; ++batch_id) // 并行计算这层每个样本的前向传播
                this->thread_pool->enqueue(this, &module_base::forward, batch_id, true);
            while(!this->thread_pool->have_finished_works()); // 等待所有任务完成 
            thread_pool->set_task_nums(this->batch_num); //设置任务计数用于同步
            for(size_t batch_id = 0; batch_id < this->batch_num; ++batch_id) // 并行计算这层每个样本的前向传播
                this->thread_pool->enqueue(this, &module_base::backward, this->train_labels[i][batch_id], batch_id, loss_type::cross_entropy, true);
            while(!this->thread_pool->have_finished_works());
        } else if(pool_type == 2) {
            // 设置同步任务数，等待前向传播全部完成后才能继续
            thread_pool_imp->set_task_nums(this->batch_num); //设置任务计数用于同步
            for(size_t batch_id = 0; batch_id < this->batch_num; ++batch_id) // 并行计算这层每个样本的前向传播
                this->thread_pool_imp->enqueue(this, &module_base::forward, batch_id, true);
            while(!this->thread_pool_imp->have_finished_works()); // 等待所有任务完成 
            thread_pool_imp->set_task_nums(this->batch_num); //设置任务计数用于同步
            for(size_t batch_id = 0; batch_id < this->batch_num; ++batch_id) // 并行计算这层每个样本的前向传播
                this->thread_pool_imp->enqueue(this, &module_base::backward, this->train_labels[i][batch_id], batch_id, loss_type::cross_entropy, true);
            while(!this->thread_pool_imp->have_finished_works());
        } else throw(1);
 
        this->update_parameters(lr);
    }
}
void module_base::train_model_mul_with_pool(int epochs, double lr, int thread_num, size_t pool_type) {
    assert(this->thread_pool != nullptr);
    for(int i = 0; i < epochs; ++i)  {
        train_one_epoch_mul(lr, pool_type);
    }
}
void module_base::train_model_multi_thread(int epochs, double lr, int thread_num) { 
    std::vector<std::thread> threads(thread_num);
    for(int i = 0; i < epochs; ++i)  {
        this->reshuffle_data();
        for(int grp = 0; grp < groups; ++grp) {
            this->clear_grad();
            this->clear_samples(); //非常重要！！！因为 _add_tensors 等实现逻辑是累加式的！！
            this->reset_count();   // 将每个tensor和op的temp_n重置！非常重要！
            this->set_input_value(this->train_data[grp]);
            for(int batch_id = 0; batch_id < thread_num; ++batch_id) {
                threads[batch_id] = std::thread([this, batch_id, grp]() {
                    this->forward(batch_id);
                    this->backward(this->train_labels[grp][batch_id], batch_id, loss_type::cross_entropy);
                });
            }
            // std::thread(static_cast<void (Linear_Resnet::*)(size_t)>(&Linear_Resnet::forward), this, (size_t)batch_id);
            // for(int batch_id = 0; batch_id < thread_num; ++batch_id) {
            //     threads[batch_id] = std::thread(static_cast<void (Linear_Resnet::*)(std::vector<float>& , size_t , loss_type )>(&Linear_Resnet::backward), this, std::ref(this->train_labels[grp][batch_id]), batch_id, loss_type::cross_entropy);
            // }
            for(int t = 0; t < thread_num; ++t) {
                threads[t].join();
            } // 省去中间一步join可使速度从12.5s提升到7.5s左右
            this->update_parameters(lr);
        }   
    }
}
void module_base::get_train_data(std::vector<std::vector<float>>& data, std::vector<std::vector<float>>& labels) {
    assert(data.size() > 0 && data.size() == labels.size() && data.size() % this->batch_num == 0);
    this->samples = data.size();
    this->groups = samples / batch_num;
    (this->data).clear();
    (this->labels).clear();
    assert((this->data).size() == 0 && (this->labels).size() == 0);
    this->data = data;
    this->labels = labels;
    assert((this->data).size() > 0 && (this->labels).size() > 0);
    (this->train_data).clear();
    (this->train_labels).clear();
    for(int i = 0; i < groups; ++i) {
        std::vector<std::vector<float>> temp_data, temp_label;
        for(int batch_id = 0; batch_id < batch_num; ++batch_id) {
            temp_data.push_back(this->data[i * batch_num + batch_id]);
            temp_label.push_back(this->labels[i * batch_num + batch_id]);
        }
        this->train_data.push_back(temp_data);
        this->train_labels.push_back(temp_label);

    }
    // std::cout << "trandata size: " << train_data.size() << ' ' << train_data[0].size() << ' ' << train_data[0][0].size() <<std::endl;
    // std::cout << "tranlabel size: " << train_labels.size() << ' ' << train_labels[0].size() << ' ' << train_labels[0][0].size() <<std::endl;
    // std::cout << "Finished clearing!\n";
    return;
}
void module_base::validate() {
    assert(this->train_data.size() && this->train_labels.size());
    double precision_rate = 0.0;
    for(int i = 0; i < groups; ++i) {
        this->clear_samples();
        this->reset_count(); // 非常重要！！！
        this->set_input_value(train_data[i]);
        this->forward();
        for(int batch_id = 0; batch_id < this->batch_num; ++batch_id) {
            auto output_logits = this->last_layer->get_output_metrix_ptr() + batch_id;
            auto label = (this->train_labels)[i][batch_id]; // 标准类别
            int pred_label = 0; // 预测标签所在类别的索引
            float max_logit = output_logits->data[0];
            for(int j = 1; j < this->last_layer->get_n(); ++j) {
                if(output_logits->data[j] > max_logit) {
                    max_logit = output_logits->data[j];
                    pred_label = j;
                }
            }
            if(label[pred_label] == 1.0)
                precision_rate += 1.0;
        }
    }
    precision_rate /= samples;
    std::cout << "Validation precision: " << precision_rate * 100.0 << "%\n";
    return;
}
std::vector<std::vector<float>> module_base::get_param_b() {
    auto p = this->first_layer;
    std::vector<std::vector<float>> res;
    while(p) {
        int n = p->get_n();
        std::vector<float> temp(n);
        auto q = dynamic_cast<layer*>(p);
        if (!q) {
            std::cout << "Error: Tensor is not of type 'layer'. Cannot retrieve bias data.\n";
            return res; // 或者抛出异常，根据你的错误处理策略
        }
        float* data = q->get_bias_data();
        for(int i = 0; i < n; ++i) {
            temp[i] = data[i];
        }
        res.push_back(temp);
        p = p->get_op_next() ? p->get_op_next()->get_output() : nullptr;
    }
    return res;
}
std::vector<std::vector<std::vector<float>>> module_base::get_outputs() {
    std::vector<std::vector<std::vector<float>>> res;
    for(int batch_id = 0; batch_id < this->batch_num; ++batch_id) {
        dtensor_base* p = this->first_layer;
        std::vector<std::vector<float>> batch_outputs;
        while(p) {
            int n = p->get_n();
            std::vector<float> temp(n);
            auto q = dynamic_cast<layer*>(p);
            if (!q) {
                std::cout << "Error: Tensor is not of type 'layer'. Cannot retrieve bias data.\n";
                return res; // 或者抛出异常，根据你的错误处理策略
            }
            float* data = p->get_output_data_ptr(batch_id); // 获取当前层第batch_id个batch的输出数据指针
            for(int i = 0; i < n; ++i) {
                temp[i] = data[i];
            }
            batch_outputs.push_back(temp);
            p = p->get_op_next() ? p->get_op_next()->get_output() : nullptr;
        }
        res.push_back(batch_outputs);
    }
    return res;
}
std::vector<std::vector<std::vector<float>>> module_base::get_param_w() {
    dtensor_base* p = this->first_layer;
    std::vector<std::vector<std::vector<float>>> res;
    while(p) {
        op* op_next = p->get_op_next();
        if(!op_next)
            break;
        dtensor_base* wm = op_next->get_inputs()[0]; // 当前层的权重矩阵
        float* data = wm->get_input_metrix_ptr()->data;
        std::vector<size_t> shape = wm->get_shape();
        assert(shape.size() == 2);
        int h = shape[0], w = shape[1];
        std::vector<std::vector<float>> temp(h, std::vector<float>(w));
        for(int i = 0; i < h; ++i) {
            for(int j = 0; j < w; ++j) {
                temp[i][j] = data[i * w + j];
            }
        }
        res.push_back(temp);
        p = op_next->get_output();
    }
    return res;
}
Linear_NN::~Linear_NN() {
    dtensor_base* cur_layer = first_layer;
    op *cur_op, *temp_op;
    while(cur_layer) {
        cur_op = cur_layer->get_op_next();
        temp_op = cur_op;
        if(!cur_op) {// 最后一层无next_op
            delete cur_layer;
            break;
        }
        for(auto &input : cur_op->get_inputs()) // 先让计算图输入tensor前向传播
            delete input;
        cur_layer = cur_op->get_output();
        delete temp_op;
    } 
}
void Linear_NN::add_layer(int num, sub_type tp) {
    if(!this->first_layer) {
        first_layer = layer_tool(num, batch_num, sub_type::origin);
        last_layer = first_layer;
        return;
    }
    int l = num, w = last_layer->get_n();
    dtensor_base* weight_metrix = new tensor2D_float({l, w} , batch_num, init_type::simple);
    op* mat_op = new matmul_op(weight_metrix, this->last_layer);    
    dtensor_base* new_layer = mat_op->set_output(tensor_type::layer, tp); 
    last_layer = new_layer;
    return;
}
void Linear_NN::forward(size_t batch_id) {
    dtensor_base* cur_layer = first_layer;
    op* cur_op;
    while(cur_layer) {
        cur_op = cur_layer->get_op_next();
        if(!cur_op) // 最后一层无next_op
            break;
        for(auto &input : cur_op->get_inputs()) // 先让计算图输入tensor前向传播
            input->forward(batch_id);
        cur_op->forward(batch_id);  // 再让算子计算对应输出(重要)
        cur_layer = cur_op->get_output();
    } 
    cur_layer->forward(batch_id);
    return;
}
void Linear_NN::forward() {
#ifdef USE_DEBUG 

#endif
    // 实现前向传播逻辑
    for(int batch_id = 0; batch_id < batch_num; ++batch_id) {
        forward(batch_id);
    }
}
void Linear_NN::backward(std::vector<float>& label, size_t batch_id, loss_type tp) {
    dtensor_base* cur_layer = last_layer;
    op* cur_op = cur_layer->get_op_last();
    cur_layer->count_loss_grad(label, tp, batch_id);   
    cur_layer->backward(batch_id);
    do {
        cur_op = cur_layer->get_op_last();
        if(cur_op) {
            // 最后一层无next_op
            cur_op->backward(batch_id);
        } else break;
        auto last_tensors = cur_op->get_inputs();
        if(!last_tensors.size()) 
            break;
        for(auto &ts : last_tensors)
            ts->backward(batch_id);
        cur_layer = last_tensors[1]; // 得到op inputs的第 2 个元素
    } while(cur_layer);
    return;
}
void Linear_NN::backward(std::vector<std::vector<float>>& labels, loss_type tp) {
#ifdef USE_DEBUG 

#endif
    // 实现前向传播逻辑
    for(int batch_id = 0; batch_id < batch_num; ++batch_id) {
        backward(labels[batch_id], batch_id, tp);
    }  
    return;
}
void Linear_NN::update_parameters(double lr) {
    dtensor_base* cur_layer = first_layer;
    op* temp_op;
    while(cur_layer) {
        temp_op = cur_layer->get_op_next();
        if(!temp_op)
            break;
        auto tensor_vec = temp_op->get_inputs();
        for(auto &ts : tensor_vec) 
            ts->update(lr);
        cur_layer = temp_op->get_output();
    }
    cur_layer->update(lr); // 最后一层输出层更新
    return;
}
void Linear_NN::set_input_value(std::vector<std::vector<float>>& data) {
    this->first_layer->set_input_value(data);
}
void Linear_NN::clear_grad() {
    dtensor_base* cur_layer = first_layer;
    op* temp_op;
    while(cur_layer) {
        temp_op = cur_layer->get_op_next();
        if(!temp_op)
            break;
        auto tensor_vec = temp_op->get_inputs();
        for(auto &ts : tensor_vec) 
            ts->clear_grad();
        cur_layer = temp_op->get_output();
    }
    cur_layer->clear_grad();
    return;
}
void Linear_NN::clear_samples() {
    dtensor_base* cur_layer = first_layer;
    op* temp_op;
    while(cur_layer) {
        temp_op = cur_layer->get_op_next();
        if(!temp_op)
            break;
        auto tensor_vec = temp_op->get_inputs();
        for(auto &ts : tensor_vec) 
            ts->clear_value();
        cur_layer = temp_op->get_output();
    }
    cur_layer->clear_value();
    return;
}
void Linear_NN::check_net() { // 待完善！！！
    std::cout << "start checking the net size...\n";
    auto p = this->first_layer;
    op* temp_op;
    int ct = 0;
    while(p->get_op_next()) {
        std::cout << "checking the " << ct + 1 << "-th layer size...\n";
        ct++;
        temp_op = p->get_op_next();
        auto vec = temp_op->get_inputs();
        for(auto &input : vec)
            input->print(1);
        p = temp_op->get_output();
    }
    p->print(1);

    ///还要检查所有 w 的 x 和 next 指针 不为 nullptr
    // 以及所有 layer 的 next 和 last 不为空
    std::cout << "check finished!\n";
}
void Linear_NN::print_count_n() {
    auto p = this->first_layer;
    op* temp_op;
    int ct = 0;
    while(p->get_op_next()) {
        std::cout << "Printing the " << ct + 1 << "-th layer count num...\n";
        ct++; 
        temp_op = p->get_op_next();
        auto vec = temp_op->get_inputs();
        std::cout << "OP address is: " << temp_op << std::endl;
        for(auto &input : vec) {
            std::cout << "count n is: " << input->get_count_n() << std::endl;
            std::cout << "input address is: " << input << std::endl;
        }
        p = temp_op->get_output();
    }
    std::cout << "LAST LAYER count n is: " << p->get_count_n() <<std::endl;
    std::cout << "LAST LAYER address n is: " << p <<std::endl;
}
void Linear_NN::reset_count() {
    auto p = this->first_layer;
    op* temp_op;
    while(p->get_op_next()) {
        temp_op = p->get_op_next();
        temp_op->reset_count();
        auto vec = temp_op->get_inputs();
        for(auto &input : vec)
            input->reset_count();
        p = temp_op->get_output();
    }
    p->reset_count();
}
void Linear_NN::print_all_layers(size_t batch_id, bool inc_grad) {

}
Linear_Resnet::~Linear_Resnet() {
    dtensor_base* cur_layer = first_layer;
    op *cur_op, *temp_op;
    while(cur_layer) {
        cur_op = cur_layer->get_op_next();
        temp_op = cur_op;
        if(!cur_op) {// 最后一层无next_op
            delete cur_layer;
            break;
        }
        delete cur_op->get_inputs()[0];// 待完善！！！
        cur_layer = cur_op->get_output();
        delete temp_op;
    } 
}
void Linear_Resnet::add_layer(int num, sub_type tp) {
    if(!this->first_layer) {
        first_layer = layer_tool(num, batch_num, sub_type::origin);
        last_layer = first_layer;
        return;
    }
    int l = num, w = last_layer->get_n();
    dtensor_base* weight_metrix = new tensor2D_float({l, w} , batch_num, init_type::xavier);
    op* mat_op = new matmul_op(weight_metrix, this->last_layer);    
    dtensor_base* new_layer = mat_op->set_output(tensor_type::layer, tp); 
    last_layer = new_layer;
    return;
}
void Linear_Resnet::add_res_layer(int num, sub_type tp) {
    assert(this->last_layer);
    assert(this->last_layer->get_n() == num);
    int l = num, w = last_layer->get_n();
    dtensor_base* weight_metrix = new tensor2D_float({l, w}, batch_num, init_type::xavier);
    op* mat_op = new matmul_op(weight_metrix, this->last_layer);  
    dtensor_base* new_layer = mat_op->set_output(tensor_type::layer, tp);
    op* ad_op = new add_op(last_layer, new_layer);
    new_layer = ad_op->set_output(tensor_type::layer, sub_type::origin);  // 仅仅是把二者相加，并不需要用 relu！！！
    last_layer = new_layer;
    return;
}
void Linear_Resnet::forward(size_t batch_id) {
    dtensor_base* cur_layer = first_layer;
    op* cur_op;
    //std::cout << "first_layer address is: " << cur_layer << std::endl;
    while(cur_layer) {
        cur_op = cur_layer->get_op_next();
        if(!cur_op) // 最后一层无next_op
            break;
        for(auto &input : cur_op->get_inputs()) // 先让计算图输入tensor前向传播
            input->forward(batch_id);
        cur_op->forward(batch_id);  // 再让算子计算对应输出(重要)
        //std::cout << "current op address is: " << cur_op << std::endl;
        cur_layer = cur_op->get_output();
    } 
    cur_layer->forward(batch_id);
    // std::cout << "forward finished!!!\n\n";

    return;
}

void Linear_Resnet::forward() {
    // 实现前向传播逻辑
    for(int batch_id = 0; batch_id < batch_num; ++batch_id) {
        forward(batch_id);
    }
}
void Linear_Resnet::backward(std::vector<float>& label, size_t batch_id, loss_type tp) {
    dtensor_base* cur_layer = last_layer;
    op* cur_op = cur_layer->get_op_last();
    cur_layer->count_loss_grad(label, tp, batch_id);   
    cur_layer->backward(batch_id);
    // std::cout << "last_layer address is: " << last_layer << std::endl;
    do {
        cur_op = cur_layer->get_op_last();
        //std::cout << "1111\n";
        if(cur_op) {
            // 最后一层无next_op
            //std::cout << "op address is: " << cur_op << std::endl;
            cur_op->backward(batch_id);
        } else break;
        //std::cout << "2222\n";
        auto last_tensors = cur_op->get_inputs();
        if(!last_tensors.size()) 
            break;
        for(auto &ts : last_tensors)
            ts->backward(batch_id);
        //std::cout << "3333\n";
        cur_layer = last_tensors[1]; // 得到op inputs的第 2 个元素, 即为残差的output
        //std::cout << "layer address is: " << cur_layer << std::endl;
    } while(cur_layer);
    // std::cout << "backward finished!!!\n\n";

    return;
}
void Linear_Resnet::backward(std::vector<std::vector<float>>& labels, loss_type tp) {
#ifdef USE_DEBUG
    assert(labels.size() == batch_num);
#endif
    // 实现前向传播逻辑
    for(int batch_id = 0; batch_id < batch_num; ++batch_id) {
        backward(labels[batch_id], batch_id, tp);
    }  
    return;
}
void Linear_Resnet::print_all_layers(size_t batch_id, bool inc_grad) {
    dtensor_base* cur_layer = first_layer;
    op* cur_op;
    //std::cout << "first_layer address is: " << cur_layer << std::endl;
    while(cur_layer) {
        cur_op = cur_layer->get_op_next();
        if(!cur_op) // 最后一层无next_op
            break;
        for(auto &input : cur_op->get_inputs()) {// 先让计算图输入tensor前向传播
            input->forward(batch_id);
            input->print(true, batch_id, 1);
        }
        cur_op->forward(batch_id);  // 再让算子计算对应输出(重要)
        //std::cout << "current op address is: " << cur_op << std::endl;
        cur_layer = cur_op->get_output();
    } 
    cur_layer->forward(batch_id);
    return;
}

void Linear_Resnet::clear_grad() {
    dtensor_base* cur_layer = first_layer;
    op* temp_op;
    while(cur_layer) {
        temp_op = cur_layer->get_op_next();
        if(!temp_op)
            break;
        auto tensor_vec = temp_op->get_inputs();
        for(auto &ts : tensor_vec) 
            ts->clear_grad();
        cur_layer = temp_op->get_output();
    }
    cur_layer->clear_grad();
    return;
}
void Linear_Resnet::clear_samples() {
    dtensor_base* cur_layer = first_layer;
    op* temp_op;
    while(cur_layer) {
        temp_op = cur_layer->get_op_next();
        if(!temp_op)
            break;
        auto tensor_vec = temp_op->get_inputs();
        for(auto &ts : tensor_vec) 
            ts->clear_value();
        cur_layer = temp_op->get_output();
    }
    cur_layer->clear_value();
    return;
}
void Linear_Resnet::reset_count() {
    auto p = this->first_layer;
    op* temp_op;
    while(p->get_op_next()) {
        temp_op = p->get_op_next();
        temp_op->reset_count();
        auto vec = temp_op->get_inputs();
        for(auto &input : vec)
            input->reset_count();
        p = temp_op->get_output();
    }
    p->reset_count();
}
void Linear_Resnet::set_input_value(std::vector<std::vector<float>>& data) {
    this->first_layer->set_input_value(data);
}
void Linear_Resnet::update_parameters(double lr) {
    dtensor_base* cur_layer = first_layer;
    op* temp_op;
    while(cur_layer) {
        temp_op = cur_layer->get_op_next();
        if(!temp_op)
            break;
        auto tensor_vec = temp_op->get_inputs();
        for(auto &ts : tensor_vec) 
            ts->update(lr);
        cur_layer = temp_op->get_output();
    }
    cur_layer->update(lr); // 最后一层输出层更新
    return;
}
void Linear_Resnet::check_net() { // 待完善！！！
    std::cout << "start checking the net size...\n";
    auto p = this->first_layer;
    op* temp_op;
    int ct = 0;
    while(p->get_op_next()) {
        std::cout << "checking the " << ct + 1 << "-th layer size...\n";
        ct++;
        temp_op = p->get_op_next();
        auto vec = temp_op->get_inputs();
        for(auto &input : vec)
            input->print(1);
        p = temp_op->get_output();
    }
    p->print(1);

    ///还要检查所有 w 的 x 和 next 指针 不为 nullptr
    // 以及所有 layer 的 next 和 last 不为空
    std::cout << "check finished!\n";
}
void Linear_Resnet::print_count_n() {
    auto p = this->first_layer;
    op* temp_op;
    int ct = 0;
    while(p->get_op_next()) {
        std::cout << "Printing the " << ct + 1 << "-th layer count num...\n";
        ct++; 
        temp_op = p->get_op_next();
        auto vec = temp_op->get_inputs();
        std::cout << "OP address is: " << temp_op << std::endl;
        for(auto &input : vec) {
            std::cout << "count n is: " << input->get_count_n() << std::endl;
            std::cout << "input address is: " << input << std::endl;
        }
        p = temp_op->get_output();
    }
    std::cout << "LAST LAYER count n is: " << p->get_count_n() <<std::endl;
    std::cout << "LAST LAYER address n is: " << p <<std::endl;
}
model_data run_model(const NNParams& params) {
    model_data res;

    auto data_gen = make_classification(params.samples, params.input_output_dim.first, params.input_output_dim.second,
                            params.input_output_dim.first - 2, 2);
    auto& data = data_gen.first;
    auto& labels = data_gen.second;
    std::cout << "Sample num is: " << data.size() << std::endl;
    std::cout <<"data size is: " << data[0].size() <<std::endl;
    std::cout <<"label size is: " << labels[0].size() <<std::endl;
    auto model_type = params.model_type;  
    auto layer_sizes = params.layer_sizes;
    int layer_num = params.layer_num;
    auto layer_types = params.layer_types; 
    int batch_size = params.batch_size;
    int epochs = params.epochs;
    double lr = params.lr;
    loss_type lstp = params.lstp;
    std::cout << "information: \n";
    std::cout << "model_type:" << (model_type == nn_type::Linear_NN ? "Linear NN" : "Linear Resnet") << std::endl;
    std::cout << "layers: "; for(int i = 0 ; i < layer_sizes.size(); ++i) std::cout << layer_sizes[i] << 
        (layer_types[i] == sub_type::relu ? "relu":(layer_types[i] == sub_type::origin ? "origin" : "softmax")) << "   ";
    std::cout << std::endl;
    std::cout << "epochs: " << epochs << std::endl;
    std::cout << "batch_size: " << batch_size << std::endl;
    std::cout << "learing rate: " << lr << std::endl;

    nn::Linear_NN* nn = new nn::Linear_NN(batch_size);
    assert(layer_sizes[0] == data[0].size() && layer_sizes[layer_num - 1] == labels[0].size());
    for(int i = 0; i < layer_num; ++i) {
        nn->add_layer(layer_sizes[i], layer_types[i]);
    }

    nn->get_train_data(data, labels);
    nn->train_model(epochs, lr);
    nn->validate();
    auto nn_ptr =  static_cast<nn::module_base*>(nn);
    res.layer_sizes = layer_sizes;
    res.layer_types = layer_types;
    res.set_model(nn_ptr);
    res.param_b = nn->get_param_b();
    res.param_w = nn->get_param_w();
    res.outputs = nn->get_outputs();
    return res;
}

} // namespace nn

namespace dtensor {
// Tools
sub_type strToSubType(const std::string& st) {
    if (st == "relu")       return sub_type::relu;
    if (st == "sigmoid")    return sub_type::sigmoid;
    if (st == "softmax")    return sub_type::softmax;
    if (st == "layer_norm") return sub_type::layer_norm;
    if (st == "none")       return sub_type::none;
    return sub_type::origin;
}

std::string subTypeToStr(sub_type t) {
    switch (t) {
        case sub_type::origin:     return "origin";
        case sub_type::relu:       return "relu";
        case sub_type::sigmoid:    return "sigmoid";
        case sub_type::softmax:    return "softmax";
        case sub_type::layer_norm: return "layer_norm";
        case sub_type::none:       return "none";
        default: return "origin";
    }
}

}