#include "server.h"
namespace netService {
void __netServer_base::start_service() {
    if(blocking)
        _start_service_blk();
    else    
        _start_service_nonblk();
}
void __netServer_base::_start_service_blk() {
    std::cout << "Starting service with blocking mode...\n";
#ifdef _WIN32
    // 1. 初始化 Winsock
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        printf("Failed to initialize Winsock\n");
        throw std::runtime_error("Failed to initialize Winsock");
        return;
    }
    // 2. 创建监听 Socket
    SOCKET listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd == INVALID_SOCKET) {
        printf("Failed to create socket\n");
        WSACleanup();
        throw std::runtime_error("Failed to create socket");
        return;
    }
    // 3. 绑定地址 + 监听
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(this->port);
    if (bind(listen_fd, (struct sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        printf("Failed to bind socket\n");
        closesocket(listen_fd);
        WSACleanup();
        throw std::runtime_error("Failed to bind socket");
        return;
    }
    listen(listen_fd, 5);
    printf("started on port %d\n", this->port);
    // 4. WSAPoll 事件数组
    // 添加监听 socket，关注可读事件（有新连接）
    fds[0].fd = listen_fd;
    fds[0].events = POLLIN; //监听可读事件
    nfds = 1;
    while (1) {
        // 5. 等待事件（-1=无限等待）
        int ret = WSAPoll(fds, nfds, -1);
        if (ret <= 0) break;
        // 遍历所有活跃 fd
        int i;
        for (i = 0; i < nfds; i++) {
            std::cout << "checking the " << i << "-th fd: " << fds[i].fd << std::endl;
            // 只处理真正有事件的 fd
            if (!(fds[i].revents & (POLLIN | POLLHUP | POLLERR)))
                continue;
            // 先判断错误/挂断事件
            if (fds[i].revents & (POLLHUP | POLLERR))
            {
                printf("net has closed connection fd = %d\n", (int)fds[i].fd);
                CLOSESOCKET(fds[i].fd);
                fds[i] = fds[--nfds];
                std::cout << "The rest connection num is: " << nfds - 1 << "\n";
                i--; // 避免跳过
                continue;
            }
            // 新客户端连接
            if (fds[i].fd == listen_fd) {
                SOCKET client_fd = accept(listen_fd, NULL, NULL);
                if (client_fd != INVALID_SOCKET && nfds < conn_num) {
                    printf("New connection: %d\n", client_fd);
                    fds[nfds].fd = client_fd;
                    fds[nfds].events = POLLIN;
                    fds[nfds].revents = 0; //要初始化
                    nfds++;
                }
                continue;
            }
            // 处理客户端消息
            handle_request(i);
        }
    }
    // 清理
    closesocket(listen_fd);
    WSACleanup();
#else
    // Linux 平台
    // Implementation for Linux
#endif
    return;
}

void __netServer_base::_start_service_nonblk() {
    // // 只处理真正有事件的 fd
    // if (!(fds[i].revents & (POLLIN | POLLOUT | POLLHUP | POLLERR)))
    //     continue;
    // // 先判断错误/挂断事件
    // if (fds[i].revents & POLLERR)
    // {
    //     printf("net has closed connection fd = %d\n", (int)fds[i].fd);
    //     CLOSESOCKET(fds[i].fd);
    //     fds[i] = fds[--nfds];
    //     std::cout << "The rest connection num is: " << nfds - 1 << "\n";
    //     i--; // 避免跳过
    //     continue;
    // }
    // std::cout << "Starting service with non-blocking mode...\n";
    // bool real_close = false;
    // if (fds[i].revents & POLLHUP)
    // {
    //     // 尝试读 0 字节，判断是否真断开
    //     char tmp[2];
    //     int ret = recv((SOCKET)fds[i].fd, tmp, 0, 0);

    //     // Windows 规则：
    //     // recv 返回 0 → 真关闭
    //     // recv 不返回 0 → 只是非阻塞连接成功，不是关闭
    //     if (ret == 0)
    //     {
    //         real_close = true;
    //     }
    // }

    // if (real_close)
    // {
    //     printf("close fd = %d\n", (int)fds[i].fd);
    //     CLOSESOCKET(fds[i].fd);
    //     fds[i] = fds[--nfds];
    //     i--;
    // }
    // if(!blocking) { // 非阻塞模式
    //     u_long mode = 1; // 1=非阻塞 0=阻塞
    //     if (ioctlsocket(client_fd, FIONBIO, &mode) != 0) {
    //         printf("Failed to set non-blocking mode for client socket\n");
    //     }
    // }
}
void __netServer_base::send_data(std::string& data, int fd) {
    send(fd, data.c_str(), data.size(), 0);
}
void __netServer_base::send_data(char* data, int len, int fd) {
    send(fd, data, len, 0);
}
void __netServer_base::close_conn(int& idx) {
    SOCKET fd = fds[idx].fd;
    closesocket(fd);
    printf("fd [%d] has received message: %s\n", fd, buf);
    // 从数组删除该 fd（用最后一个覆盖当前 idx）
    fds[idx] = fds[--nfds];
    idx--; // 避免跳过
    return;
}
int __netServer_base::_read_msg_blk(int& idx) {
    int len = recv(temp_fd, buf, sizeof(buf) - 1, 0);
    if (len <= 0) {
        return -1;
    }
    memcpy(data_buf, buf, len);
    return len;
}
int __netServer_base::_read_msg_nonblk(int& idx) {
    int total = 0;
    int nRet;
    while (true) {
        // 非阻塞接收
        std::cout << "recving data nonblocking..\n";
        nRet = recv(temp_fd, buf, sizeof(buf) - 1, 0);
        if (nRet > 0) { // 读到数据，拼接
            memcpy(data_buf + total, buf, nRet);
            total += nRet;
        }
        else if (nRet == 0) {
            // 关闭连接
            return -1;
        }
        else {
            // nRet == SOCKET_ERROR
            int err = WSAGetLastError();
            if(err == WSAEWOULDBLOCK) {
                // 无更多数据，内核缓冲区已读完，正常退出
                break;
            }
            else { // 真正网络错误
                printf("recv error:%d\n", err);
                return -1;
            }
        }
    }
    return total;
}

void demoServer::handle_request(int &idx) {
    temp_fd = fds[idx].fd;
    std::cout << "Handling request on fd [" <<  temp_fd << "]...\n";
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
    std::cout << "recv len is: " << status << std::endl;
    handle_msg(idx, status);
}
void demoServer::handle_msg(int& idx, int len) {
    temp_fd = fds[idx].fd;
    printf("fd [%d] has received message: %.*s\n", temp_fd, len, data_buf);
    send_data(data_buf, len, temp_fd); // 回显
}

void __netClient_base::send_data(std::string&& data) {
    send(fd, data.c_str(), data.size(), 0);
}
void __netClient_base::send_data(char* data, int len) {
    send(fd, data, len, 0);
}
void __netClient_base::_connect_to_server(const std::string& server_ip) {
    SOCKET sockfd = INVALID_SOCKET;
    struct sockaddr_in server_addr;
    // 1. 初始化 Windows Socket
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        printf("WSAStartup failed! error code: %d\n", WSAGetLastError());
        this->fd = INVALID_SOCKET;
        return;
    }
    // 2. 创建客户端 socket
    sockfd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sockfd == INVALID_SOCKET) {
        printf("socket failed! error code: %d\n", WSAGetLastError());
        WSACleanup();
        this->fd = INVALID_SOCKET;
        return;
    }
    // 3. 设置服务端地址信息
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(this->port);  // 端口（类成员）
    // IP 字符串转网络格式
    if (inet_pton(AF_INET, server_ip.c_str(), &server_addr.sin_addr) <= 0) {
        printf("IP address format error!\n");
        closesocket(sockfd);
        WSACleanup();
        this->fd = INVALID_SOCKET;
        return;
    }
    // 4. 连接服务器
    if (connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == SOCKET_ERROR) {
        printf("connect failed! error code: %d\n", WSAGetLastError());
        closesocket(sockfd);
        WSACleanup();
        this->fd = INVALID_SOCKET;
        return;
    }
    this->fd = (int)sockfd;  // Windows SOCKET 是指针类型，强转int保存
    printf("Client successfully connected to server: %s:%d\n", server_ip.c_str(), this->port);
}
int __netClient_base::_read_msg_blk() {
    int len = recv(fd, data_buf, DATA_BUF_SIZE, 0);
    if (len <= 0) {
        return -1;
    }
    data_buf[len] = 0;
    return len;
}
void demoClient::connect(const std::string& server_ip) {
    _connect_to_server(server_ip);
}
void demoClient::handle_request() {
    // 客户端发送消息的处理逻辑
    // 这里可以实现客户端特定的消息处理，例如解析服务器响应等
    printf("Client received message: %s\n", buf);
}
void demoClient::handle_msg(int len) {
    printf("Client received message from fd [%d], len: %d\n", fd, len);
    // 根据协议解析消息, 处理函数调用等
    // 这里可以添加对服务器响应的处理逻辑
}
} // namespace netService