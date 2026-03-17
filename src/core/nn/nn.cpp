#include"nn.h"

namespace nn{
using namespace dtensor;

void module_base::set_learning_rate(double lr) {
    this->lr = lr;
}
void Linear_NN::get_train_data(std::vector<std::vector<float>>& data, std::vector<std::vector<float>>& labels) {
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
    std::cout << "trandata size: " << train_data.size() << ' ' << train_data[0].size() << ' ' << train_data[0][0].size() <<std::endl;
    std::cout << "tranlabel size: " << train_labels.size() << ' ' << train_labels[0].size() << ' ' << train_labels[0][0].size() <<std::endl;
    std::cout << "Finished clearing!\n";
    return;
}
void Linear_NN::reshuffle_data() {
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
        cur_layer->forward(batch_id);
        cur_op = cur_layer->get_op_next();
        if(!cur_op) // 最后一层无next_op
            break;
        cur_op->forward(batch_id);
        cur_layer = cur_op->get_output();
    } 
    cur_layer->forward(batch_id);
    //std::cout << "last layer is:!!!!\n";
    //last_layer->print(1);
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
        //std::cout << "1111\n";
        if(cur_op) {
            // 最后一层无next_op
            cur_op->backward(batch_id);
        } else break;
        //std::cout << "2222\n";
        auto last_tensors = cur_op->get_inputs();
        if(!last_tensors.size()) 
            break;
        for(auto &ts : last_tensors)
            ts->backward(batch_id);
        //std::cout << "3333\n";
        cur_layer = last_tensors[1]; // 得到op inputs的第 2 个元素
    } while(cur_layer);
    //std::cout << "batch_id: "<< batch_id << " FINISHED BACKWARD!!\n";
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
void Linear_NN::train_model(int epochs, double lr) {
    for(int i = 0; i < epochs; ++i)  {
        train_one_epoch(lr);
    }
}

void Linear_NN::train_one_epoch(double lr) {
    this->reshuffle_data();
    //std::cout << " train one epoch...\n";
    for(int i = 0; i < groups; ++i) {
        this->clear_grad();
        this->clear_samples();
        //std::cout << " grad cleared...\n";
        this->first_layer->set_input_value(this->train_data[i]);
        //std::cout << " input set...\n";
        this->forward();
        //std::cout << " forward finished...\n";
        this->backward(this->train_labels[i], loss_type::cross_entropy);
        //std::cout << " backward finished...\n";
        this->update_parameters(lr);
        //std::cout << " parameter updated...\n";
    }
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

void Linear_NN::validate() {
    assert(this->train_data.size() && this->train_labels.size());
    double precision_rate = 0.0;
    for(int i = 0; i < groups; ++i) {
        this->clear_samples();
        //std::cout << "finished clearing\n";
        this->first_layer->set_input_value(train_data[i]);
        //std::cout << "succesfully set input\n";
        this->forward();
        //std::cout << "finished forward\n";
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
    std::cout << "the validate precision rate is: " << precision_rate << std::endl;
    return;
}



}