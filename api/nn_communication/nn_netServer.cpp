#include <iostream>
#include <string>
#include "nn_netServer.h"
using namespace dtensor;
namespace netService {
void nn_netServer::handle_request(int &idx) {
    temp_fd = fds[idx].fd;
    int status;
    if(blocking) {
        status = _read_msg_blk(idx);
    } else {
        status = _read_msg_nonblk(idx);
    }
    if (status == -1) {
        close_conn(idx);
        return;
    }
    handle_msg(idx, status);
}
void nn_netServer::handle_msg(int& idx, int len) {
    printf("received message from fd [%d], len: %d\n", temp_fd, len);
    // 根据协议解析消息, 处理函数调用等
    model_agent.get_binary(data_buf, len); // 获取二进制数据
    // 定义训练参数
    nn::NNParams params;
    params.model_type = nn::nn_type::Linear_NN;
    params.layer_num = 4;
    params.layer_sizes = {10, 20, 20, 5};
    params.layer_types = {sub_type::origin, sub_type::relu, sub_type::relu, sub_type::softmax};
    params.batch_size = 4;
    params.epochs = 1000;
    params.lr = 0.001;

    model_agent.train_model(params); // 训练模型
    std::cout << "Start constructing the proto  data...\n";
    model_agent.network2proto(); // 构建网络结构的 proto 对象, 并生成二进制流
    std::string response = model_agent.get_binary(); // 获得二进制流
    send_data(response, temp_fd); // 发送响应
    std::cout << "Successfully handled the request of the " << idx << "-th connection, the " << temp_fd << "-th fd!\n"; 
}

void nn_netClient::connect(const std::string& server_ip) {
    _connect_to_server(server_ip);
}
void nn_netClient::handle_request() {
    std::cout << "handling request from fd: " << fd << std::endl;
    int status;
    status = _read_msg_blk();
    if (status == -1) {
        closesocket(fd);
        std::cout << "Connection closed by server." << std::endl;
        return;
    }
    std::cout << "recv len is: " << status << std::endl;
    handle_msg(status);
}
void nn_netClient::receiving_loop() {
    while (true) {
        handle_request();
    }
}
void nn_netClient::handle_msg(int len) {
    printf("Client received message from server, len: %d\n", len);
    // 根据协议解析消息, 处理函数调用等
    model_agent.get_binary(data_buf, len); // 获取二进制数据
    model_agent.proto2network(); // 还原网络结构
    std::cout << "Print the received network structure and parameters: \n";
    model_agent.print_network();
}



} // namespace netService