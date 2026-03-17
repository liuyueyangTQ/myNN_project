#include<iostream>
#include<string>
#include<vector>
#include<map>
#include"metrix.h"
#include"tensor.h"
#include"model.h"

namespace tensor{
layer* Linear_NN::layer_tool(int n, layer_type tp) {    //考虑 resnet ？
    switch (tp)
    {
    case layer_type::sigmoid:
        /* code */
        return new sigmoid(n, this->batch_num);
    case layer_type::relu:
        return new relu(n, this->batch_num);
    case layer_type::layer_norm:
        return new layer_norm(n, this->batch_num);
    case layer_type::softmax:
        return new softmax(n, this->batch_num);
    case layer_type::origin:
        return new origin(n, this->batch_num);
    default:
        return nullptr;
    }
}
void Linear_NN::_add_layer(int n, layer_type tp) {
    if(!this->first_layer) {
        first_layer = this->layer_tool(n, tp);
        layers.push_back(first_layer);
        assert(first_layer);  // 不能为空
        cur_layer = first_layer;
        last_layer = first_layer;
    } else {
        last_layer = this->layer_tool(n, tp); //新的层

        cur_layer->next.push_back(last_layer);  // 互相设置前面和后面的指针
        last_layer->last.push_back(cur_layer);
        layers.push_back(last_layer);
        assert(first_layer);  // 不能为空
        cur_layer = last_layer;
    }
}

void Linear_NN::_add_linear(int n, layer_type tp) { // n 是添加层的神经元数， 故矩阵形式为 n * m
    int m = cur_layer->n;
    tensor2D_float *w;
    try
    {
        std::cout << "allocating tensor2D_float...\n";
        w = new tensor2D_float({n, m}, this->batch_num, init_type::xavier);
    }
    catch(const std::exception& e)
    {
        std::cerr <<"tensor2D_float can't be allocated! " << e.what() << '\n';
    }
    

    assert((w->shape).second == this->cur_layer->n);

    layer* new_layer; 
    try
    {
        new_layer = this->layer_tool(n, tp);
    }
    catch(const std::exception& e)
    {
        std::cerr <<"new_layer can't be allocated! " << e.what() << '\n';
    }
    
    w->x = this->cur_layer;  // 为 当前层&权重矩阵 互相添加 对方的指针
    this->cur_layer->w_next.push_back(w);
    new_layer->w_last.push_back(w);   

    this->cur_layer->next.push_back(new_layer); //为新建层，当前层，wm设置前后层对应的指向
    new_layer->last.push_back(this->cur_layer); // new_layer 指向当前层

    layers.push_back(new_layer);

    w->next = new_layer;     
       

    this->cur_layer = new_layer;  //将当前层指针移动到新建层
    this->last_layer = new_layer; //将神经网络最后一层指针移动到新建层
}

void Linear_NN::add_layer(int n, layer_type tp, bool req_wm) {
    assert(!(!this->first_layer && req_wm));    //要验证 刚开始加第一层和邻接矩阵 不能同时存在
    layer_shape.push_back(n);
    if(!this->first_layer || !req_wm) { // 刚开始加第一层 或 不需要邻接矩阵， 则仅添加层
#ifdef USE_DEBUG        
        std::cout << "added single layer!\n";
#endif
        _add_layer(n, tp);
    }
    else{
#ifdef USE_DEBUG        
        std::cout << "added layer & weight metrix!\n";
#endif
        _add_linear(n, tp);
    }
#ifdef USE_DEBUG
    this->layer_num++;
    std::cout << "Successfully added the " << layer_num << "-th layer" << std::endl;
#endif

}

void Linear_NN::add_block(int layer1, int layer2) {  // 第 layer1 层的 输出 到 layer2 层的 输入 添加线性矩阵
    auto w = new tensor2D_float({layers[layer2]->n, layers[layer1]->n}, this->batch_num, init_type::xavier);
    layers[layer1]->w_next.push_back(w);
    layers[layer2]->w_last.push_back(w);
    layers[layer1]->next.push_back(layers[layer2]);
    layers[layer2]->last.push_back(layers[layer1]);
    w->x = layers[layer1];
    w->next = layers[layer2];
    return;
}
void Linear_NN::add_residual_block(int layer1, int layer2) { // 第 layer1 层的 输出 到 layer2 层的 输出 添加线性矩阵
    assert(layers[layer1]->n == layers[layer2]->n); // 残差连接要求神经元数相同
    auto w = layers[layer2]->w_next[0]; // 残差连接使用已有的权重矩阵
    layers[layer1]->w_next.push_back(w);
    layers[layer2 + 1]->w_last.push_back(w);
    layers[layer1]->next.push_back(layers[layer2 + 1]);
    layers[layer2 + 1]->last.push_back(layers[layer1]);
    return;
}

void Linear_NN::forward(std::vector<std::vector<float>>& samples) { //计算一组样本前向梯度（单线程）
    assert(samples.size() == this->batch_num);
    
    this->first_layer->get_input(samples);  
   // std::cout << "   got the input\n";
    auto p = this->first_layer;


    while(p) {    
#ifdef USE_DEBUG
    //std::cout << "Forward propagation at the "<< p->get_layer_index() << "-th layer" << std::endl;
#endif
        for(int batch_id = 0; batch_id < this->batch_num; ++batch_id) { //逐个样本前向传播
            p->forward(batch_id);

        }   
        if(p->next.size() > 0)
            p = p->next[0];
        else break;
    }
}

void Linear_NN::forward(std::vector<float>& sample, size_t batch_id) { //计算一个样本前向梯度（可扩展至多线程）
    assert(sample.size() == this->first_layer->n);

    this->first_layer->get_input(sample, batch_id);  
    auto p = this->first_layer; //重置指针指向
    while(p) {

        p->forward(batch_id);
        if(p->next.size() > 0)
            p = p->next[0];
        else break;
    }
}
void Linear_NN::clear_grad() {
    auto p = this->first_layer;
    while(p) {
        // 清零 layer 的梯度
        for(int batch_id = 0; batch_id < this->batch_num; ++batch_id) {
            float* g = (p->batch_grad + batch_id)->data;
            for(int i = 0; i < p->n; ++i)
                g[i] = 0;
        }
        // 清零邻接矩阵的梯度
        for(auto& w : p->w_next)
            for(int batch_id = 0; batch_id < this->batch_num; ++batch_id) {
                float* wg = (w->batch_grad + batch_id)->data;
                int sz = (w->shape).first * (w->shape).second;
                for(int i = 0; i < sz; ++i)
                    wg[i] = 0;
            }
        
        if(p->next.size() > 0)
            p = p->next[0];
        else break;
    }
}
void Linear_NN::clear_samples() {
    auto p = this->first_layer;
    while(p) {
        // 清零 layer 的样本输入输出
        for(int batch_id = 0; batch_id < this->batch_num; ++batch_id) {
            float* in = (p->batch_input + batch_id)->data;
            float* out = (p->batch_output + batch_id)->data;
            for(int i = 0; i < p->n; ++i) {
                *(in + i) = 0; 
                *(out + i) = 0;
            }
        }
        if(p->next.size() > 0)
            p = p->next[0];
        else break;
    }
}

#ifdef USE_THREAD
void Linear_NN::forward(std::vector<std::vector<float>>& samples, ThreadPool& pool) { //计算一个样本前向梯度（可扩展至多线程）
#ifdef USE_DEBUG  
    assert(samples[0].size() == this->first_layer->n);
#endif
    this->first_layer->get_input(samples);  
    auto p = this->first_layer; //重置指针指向
    while(p) {
        pool.set_task_nums(this->batch_num); //设置任务计数用于同步
        for(size_t batch_id = 0; batch_id < this->batch_num; ++batch_id) // 并行计算这层每个样本的前向传播
            pool.enqueue(p, &layer::forward, batch_id);

        //while(!pool.have_finished_works()); // 等待所有任务完成 (高开销)

        pool.wait_for_finish(); // 等待所有任务完成（低开销版）
        if(p->next.size() > 0)
            p = p->next[0];
        else break;
    }
}
void Linear_NN::backward(std::vector<std::vector<float>>& labels, ThreadPool& pool) { // 一组label
#ifdef USE_DEBUG    
    assert(labels[0].size() == this->last_layer->n);
#endif
    auto p = this->last_layer;
    pool.set_task_nums(this->batch_num);    
    for(size_t i = 0 ; i < this->batch_num; ++i)  //最后一层反向传播
         pool.enqueue(p, &layer::count_loss_grad, labels[i], this->loss_tp, i);  // 每个label对应 batch 的一个样本
    
    // while(!pool.have_finished_works()); // 同步 (高开销)

    pool.wait_for_finish(); // 等待所有任务完成（低开销版）
    
    // p = p->last[0]; // 最后一层传递完之后调用上一层 layer
    while (p) {
        pool.set_task_nums(this->batch_num); 
        for(size_t batch_id = 0; batch_id < this->batch_num; ++batch_id) // 每个sample都反向传播一次
            pool.enqueue(p, &layer::backward, batch_id);
        pool.wait_for_finish(); // 等待所有任务完成（低开销版）

        if(p->w_next.size() > 0) { // 还要考虑邻接权重矩阵的反向传播
            for(auto& w : p->w_next){
                pool.set_task_nums(this->batch_num);
                for(size_t batch_id = 0; batch_id < this->batch_num; ++batch_id)
                    pool.enqueue(w, &tensor2D_float::backward, batch_id); // 不包含update
                pool.wait_for_finish(); // 等待所有任务完成（低开销版）
            }
        }
        //while(!pool.have_finished_works()); // 同步 ???? 为什么多线程反而没有单线程快？？？？


        if(p->last.size() > 0)
            p = p->last[0];
        else break;
    }
    
}

#endif


void Linear_NN::backward(std::vector<std::vector<float>>& label) { // 一组label
#ifdef USE_DEBUG
    assert(label.size() == this->batch_num);
#endif
    auto p = this->last_layer;
    for(int batch_id = 0 ; batch_id < this->batch_num; ++batch_id)  //最后一层先计算标签损失对应的梯度
        p->count_loss_grad(label[batch_id], this->loss_tp, batch_id);  // 每个label对应 batch 的一个样本
    //this->cur_layer->update(); //不在这里面更新

    // std::cout << "    last layer finished!\n";

    // p = p->last[0]; // 传递完之后调用上一层 layer
    while(p){  // 从最后一层开始逐层向前传递梯度
#ifdef USE_DEBUG
    //std::cout << "Backward propagation at the "<< p->get_layer_index() << "-th layer" << std::endl;
#endif
        for(int batch_id = 0; batch_id < this->batch_num; ++batch_id) // 每个sample都反向传播一次
            p->backward(batch_id);  

        if(p->w_last.size() > 0) { //同时更新权重矩阵梯度
            for(int batch_id = 0; batch_id < this->batch_num; ++batch_id)
                for(auto& w : p->w_last)
                    w->backward(batch_id); // 不包含update
        }
        if(p->last.size() > 0)
            p = p->last[0];
        else break;
    }
}        

// 用于多线程 （待修正！！！）
void Linear_NN::backward(std::vector<float>& label, size_t batch_id) {  // 一个 sample 单独反向传播 
    // 如果包含 batch_norm层的话， 则不能单个样本并行计算 ！
#ifdef USE_DEBUG
    //std::cout << "Backward propagation at the "<< last_layer->get_layer_index() << "-th layer" << std::endl;
#endif
    auto p = this->last_layer;
    p->count_loss_grad(label, this->loss_tp, batch_id);
    // p = p->last[0]; // 传递完之后调用上一层 layer
    while(p){  // 从最后一层开始逐层向前传递梯度
#ifdef USE_DEBUG
    //std::cout << "Backward propagation at the "<< cur_layer->get_layer_index() << "-th layer" << std::endl;
#endif
        // p->backward((p->batch_grad + batch_id)->data, batch_id); //这个地方一开始搞错了！！
        // 按照原来的逻辑应该：  p->backward((p->next->batch_grad + batch_id)->data, batch_id);
        // 现在改成：
        p->backward(batch_id);
        if(p->w_last.size() > 0) { //同时更新权重矩阵梯度
            for(auto& w : p->w_last)
                 w->backward(batch_id);
        }
        if(p->last.size() > 0)
            p = p->last[0];
        else break;
    }
}

void Linear_NN::update() {
    auto p = this->last_layer;
    int ct = (this->layer_shape).size() - 1;
    while(p) {
        assert(layer_shape[ct] == p->n);
        ct--;
        p->update();  //层更新参数b
        for(auto& w : p->w_next)  //邻接矩阵更新参数
            w->update(); //邻接矩阵更新参数
        if(p->last.size() > 0)
            p = p->last[0];
        else break;
    }
    return;
}

void Linear_NN::set_train_data(std::vector<std::vector<float>>& data, std::vector<std::vector<float>>& labels, int epochs, loss_type loss_tp, int batches, double lr) {
#ifdef USE_DEBUG
    assert(this->cur_layer && this->first_layer && this->last_layer);
#endif
    this->loss_tp = loss_tp;
    this->batch_num = batches;
    this->lr = lr;
    this->data = &data;
    this->labels = &labels;
#ifdef USE_DEBUG
    std::cout << "spliting the data...\n";
#endif

    this->split_data();
    this->split_labels();
#ifdef USE_DEBUG
    std::cout << "spliting done!\n";
#endif
    return;
}

void Linear_NN::validate() {
    assert(this->train_data && this->train_labels);
    double precision_rate = 0.0;
    for(int i = 0; i < this->train_data->size(); ++i) {
        this->clear_samples();
        this->forward((*this->train_data)[i]);
        for(int batch_id = 0; batch_id < this->batch_num; ++batch_id) {
            auto output_logits = this->last_layer->batch_output + batch_id;
            auto label = (*this->train_labels)[i][batch_id]; // 标准类别

            int pred_label = 0; // 预测标签所在类别的索引
            float max_logit = output_logits->data[0];
            for(int j = 1; j < this->last_layer->n; ++j) {
                if(output_logits->data[j] > max_logit) {
                    max_logit = output_logits->data[j];
                    pred_label = j;
                }
            }

            if(label[pred_label] == 1.0)
                precision_rate += 1.0;
        }
    }
    precision_rate /= (this->train_data->size() * this->batch_num);
    std::cout << "the validate precision rate is: " << precision_rate << std::endl;
    return;
}

void Linear_NN::train(int epochs, loss_type loss_tp, int batches, double lr) {
    //this->initialize();
    std::vector<float> losses;
    float loss;
    for(int epoch = 0; epoch < epochs; ++epoch) { 
        this->reshuffle_data(); // 每个epoch 重新打乱数据顺序
         
        if(epoch % 20 == 0)
        std::cout << "the " << epoch + 1 << "-th epoch training...\n";
        // this->print_parameters(); // 打印模型参数
        loss = 0; 
        for(int i = 0; i < this->train_data->size(); ++i) {
            loss += count_batch_loss(i);
            if(epoch % 20 == 0);
            this->clear_grad(); // 每个epoch 清零梯度
            this->clear_samples(); // 清零样本输入输出缓存

            //std::cout << "    batch " << (i + 1) << " training...\n";

            // 计算时间统计
            auto start_time = std::chrono::high_resolution_clock::now();

            this->forward((*this->train_data)[i]);
            if(epoch % 20 == 0);
            //std::cout << "    forward finished!\n";
            this->backward((*this->train_labels)[i]);
            if(epoch % 20 == 0);
            //std::cout << "    backward finished!\n";

            auto end_time = std::chrono::high_resolution_clock::now();
            this->cost_ms += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

            this->update();
            if(epoch % 20 == 0);
            //std::cout << "    batch " << i + 1 << " training finished\n";
            if(epoch == epochs - 1) {
                std::cout << "The " << i + 1 <<"-th batch predict result is:\n";
                for(int k = 0; k < ((*this->train_data)[0]).size(); ++k) {
                    auto q = (this->last_layer->batch_output + k)->data;
                    for(int j = 0; j < this->last_layer->n; ++j)
                        std::cout << q[j] << ' ';
                    std::cout << std::endl;
                }
            }
            // if(epoch == epochs - 1) {
            //     std::cout << "The " << i+1 <<"-th batch grad is:\n";
            //     this->print_grad();
            // }
        }


        loss /= this->train_data->size();
        losses.push_back(loss);

    }
    std::cout << "the train loss under totally " << epochs << " epoches is:\n";

    for(int i = 0; i < epochs; ++i){   
        if(i % 10 == 0 || i <= 20)
            std::cout << losses[i] << ' ';
    }
    std::cout << std::endl;
    this->print_parameters(); // 打印模型参数
    this->last_layer->print_batches();

    return;
}

#ifdef USE_THREAD
    void Linear_NN::multithread_train(int epochs, loss_type loss_tp, int batches, double lr) {
    #ifdef USE_DEBUG
        assert(this->cur_layer && this->first_layer && this->last_layer);
    #endif
        //this->initialize();
        std::vector<float> losses;
        float loss;
        ThreadPool pool(this->batch_num * 2 > std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : this->batch_num); //线程池大小取决于硬件支持的最大线程数和batch大小的最小值
        for(int epoch = 0; epoch < epochs; ++epoch) {
            this->reshuffle_data(); // 每个epoch 重新打乱数据顺序
            this->clear_grad(); // 每个epoch 清零梯度
            if(epoch % 20 == 0)
            std::cout << "the " << epoch + 1 << "-th epoch training...\n";
            // this->print_parameters(); // 打印模型参数
            loss = 0; 
            for(int i = 0; i < this->train_data->size(); ++i) {
                loss += count_batch_loss(i);
                this->clear_grad(); // 每个epoch 清零梯度
                this->clear_samples(); // 清零样本输入输出缓存
                
                // 计算时间统计
                auto start_time = std::chrono::high_resolution_clock::now();

                // 多线程计算forward
                this->forward((*this->train_data)[i], pool);
                // 多线程计算backward
                this->backward((*this->train_labels)[i], pool);

                auto end_time = std::chrono::high_resolution_clock::now();
                this->cost_ms += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

                // 更新参数
                this->update();

                if(epoch == epochs - 1) {
                    std::cout << "The " << i + 1 <<"-th batch predict result is:\n";
                    for(int k = 0; k < ((*this->train_data)[0]).size(); ++k) {
                        auto q = (this->last_layer->batch_output + k)->data;
                        for(int j = 0; j < this->last_layer->n; ++j)
                            std::cout << ( *(q + j)) << ' ';
                        std::cout << std::endl;
                    }
                }
                // if(epoch == epochs - 1) { // 不需要打印梯度了
                //     std::cout << "The " << i+1 <<"-th batch grad is:\n";
                //     this->print_grad();
                // }
            }


            loss /= this->train_data->size();
            losses.push_back(loss);

        }
        std::cout << "the train loss under totally " << epochs << " epoches is:\n";

        for(int i = 0; i < epochs; ++i){   
            if(i % 10 == 0 || i <= 20)
                std::cout << losses[i] << ' ';
        }
        std::cout << std::endl;
        this->print_parameters(); // 打印模型参数
        this->last_layer->print_batches();

        return;
        }
#endif

void Linear_NN::split_data() {
    assert(this->data->size() % this->batch_num == 0);
    for(int i = 0; i < this->data->size() / this->batch_num; ++i) {
        std::vector<std::vector<float>> temp;
        for(int j = i * this->batch_num; j < (i + 1) * this->batch_num; ++j)
            temp.push_back((*this->data)[j]);
        this->train_data->push_back(temp);
    }
#ifdef USE_DEBUG
    std::cout << "The data shape is: " 
              << (*train_data).size() << ',' 
              << (*train_data)[0].size() << ',' 
              << (*train_data)[0][0].size() << std::endl;
#endif
}


void Linear_NN::split_labels() {
    assert(this->labels->size() % this->batch_num == 0);
    for(int i = 0; i < this->labels->size() / this->batch_num; ++i) {
        std::vector<std::vector<float>> temp;
        for(int j = i * this->batch_num; j < (i + 1) * this->batch_num; ++j)
            temp.push_back((*this->labels)[j]);
        this->train_labels->push_back(temp);
    }
#ifdef USE_DEBUG
    std::cout << "The label shape is: " 
              << (*train_labels).size() << ',' 
              << (*train_labels)[0].size() << ',' 
              << (*train_labels)[0][0].size() << std::endl;
#endif
}

void Linear_NN::reshuffle_data() {
    assert(this->data->size() == this->labels->size());
    std::vector<int> indices(this->data->size());
    for(int i = 0; i < indices.size(); ++i)
        indices[i] = i;
    // 打乱索引
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    // 根据打乱的索引重新排列数据和标签
    assert(indices.size() == this->train_data->size() * this->batch_num);
    for(int i = 0; i < this->train_data->size(); ++i) {
        for(int j = 0; j < this->batch_num; ++j) {
            int idx = i * this->batch_num + j;
            (*this->train_data)[i][j] = (*this->data)[indices[idx]];
            (*this->train_labels)[i][j] = (*this->labels)[indices[idx]];
        }
    }
    return;
}

void Linear_NN::check_net() { // 待完善！！！
    std::cout << "start checking the net size...\n";
    auto p = this->first_layer;
    int ct = 0;
    while(p) {
        std::cout << "checking the " << ct + 1 << "-th layer size...\n";
        assert(p->n == this->layer_shape[ct]);
        assert(layers[ct] == p);
        ct++;
        p->print_batches();
        if(p->w_next.size() > 0)
            for(auto& w : p->w_next)
                w->print_param();

        std::cout << "The parameter of layer " << ct << " is:\n";
        std::cout << "w_next size: " << p->w_next.size() << std::endl;
        std::cout << "next size: " << p->next.size() << std::endl;
        std::cout << "last size: " << p->last.size() << std::endl;
        std::cout << "w_last size: " << p->w_last.size() << std::endl;


        if(p->next.size() > 0)
            p = p->next[0];
        else break;
    }


    ///还要检查所有 w 的 x 和 next 指针 不为 nullptr
    // 以及所有 layer 的 next 和 last 不为空
    std::cout << "check finished!\n";
}
void Linear_NN::print_parameters() {
    assert(this->first_layer->n == this->layer_shape.front());
    assert(this->last_layer->n == this->layer_shape.back());
    auto p = this->first_layer;
    int ct = 1;
    std::cout << "model parameter is: \n";
    while(p) {
        assert(p->n == this->layer_shape[ct-1]);
        std::cout << "The " << ct << "-th layer parameter is: ";
        p->print_param();
        if(p->w_next.size() > 0)
            for(auto& w : p->w_next)
                w->print_param();
        ct++;
        if(p->next.size() > 0)
            p = p->next[0];
        else break;
    }

}
void Linear_NN::print_grad() {
    auto p = this->last_layer;
    while(p) {
        p->print_grad();
        if(p->w_next.size() > 0)
            for(auto& w : p->w_next)
                w->print_grad();
        if(p->last.size() > 0)
            p = p->last[0];
        else break;
    }
}
float Linear_NN::count_batch_loss(size_t batch_index) {
    auto p = this->last_layer;
    float loss = 0;
    auto batch_label = (*this->train_labels)[batch_index];
    for(int batch_id = 0; batch_id < this->batch_num; ++batch_id){
        auto otpt = (p->batch_output + batch_id)->data;
        loss += count_loss(otpt, batch_label[batch_id]);
    }
    loss /= this->batch_num;
    return loss;
}
float Linear_NN::count_loss(float* otpt, std::vector<float>& label) {
    float loss = 0;
    switch (this->loss_tp)
    {
    case loss_type::mse:{
        
        for(int i = 0; i < label.size(); ++i)
            loss += std::pow(label[i] - otpt[i], 2);
        break;
    }
    case loss_type::cross_entropy:{

  
        float y, y0;
        for(int i = 0; i < label.size(); ++i) {
            y0 = label[i];
            y =  otpt[i];
            //std::cout << "y0 is: "<< y0<<",  y is: "<<y;
            //std::cout <<(std::min(-y0 * std::log(y) - (1 - y0)*std::log(1-y), (float)100)) <<"     ";
            loss += std::min(- y0 * std::log(y) - (1 - y0)*std::log(1 - y), (float)100); 
        }
        break;
      }
    default:
        break;
    }
    return loss;
}

std::vector<std::vector<float>> Linear_NN::get_layer_output() {
    std::vector<std::vector<float>> outputs;
    auto p = this->first_layer;
    while(p) {
        std::vector<float> layer_output(p->n);
        for(int i = 0; i < p->n; ++i) {
            float val = (p->batch_output->data)[i]; // 取第一个batch的输出
            layer_output[i] = val;
        }
        outputs.push_back(layer_output);
        if(p->next.size() > 0)
            p = p->next[0];
        else break;
    }
    return outputs; 
}
std::vector<std::vector<std::vector<float>>> Linear_NN::get_weight_metrix() {
    std::vector<std::vector<std::vector<float>>> metrixs;
    auto p = this->first_layer;
    while(p) {
        if(p->w_next.size() > 0) {
            std::vector<std::vector<float>> metrix;
            for(int i = 0; i < p->w_next[0]->shape.first; ++i) {
                std::vector<float> row(p->w_next[0]->shape.second);
                auto q = p->w_next[0]->weight->data;
                for(int j = 0; j < p->w_next[0]->shape.second; ++j) {
                    float val = *(q + i * p->w_next[0]->shape.second + j);
                    row[j] = val;
                }
                metrix.push_back(row);
            }
            metrixs.push_back(metrix);
        }
        if(p->next.size() > 0)
            p = p->next[0];
        else break;
    }
    return metrixs;
}
long long Linear_NN::get_model_cost_time() {
    return this->cost_ms;
}

} // namespace tensor