#pragma once
#include "proto_utils.h"
#include "nn.h"
#include "enum_type.h"
#include "server.h"
using namespace proto;
class proto::proto_utils;
namespace netService {
class nn_netServer : public __netServer_base {
private:
    proto_utils model_agent; // 用于解析 protobuf 二进制数据
protected:
    void handle_msg(int& idx, int len) override;
public:
    nn_netServer(bool blocking = false, int port = PORT, int conn_num = 10) : __netServer_base(blocking, port, conn_num) {}
    void handle_request(int &idx) override;
};

class nn_netClient : public __netClient_base {
private:
    proto_utils model_agent; // 用于解析 protobuf 二进制数据
protected:
    void handle_msg(int len) override;
public:
    nn_netClient(int port = PORT) : __netClient_base(port) {}
    void connect(const std::string& server_ip);
    void handle_request() override;
    void receiving_loop(); // 用于持续接收服务器消息的循环
};
} // namespace netService