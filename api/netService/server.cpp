#include "server.h"
// const int myServer::max_fds = MAX_FDS;
void myServer::start_service() {
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
            // 客户端消息
            handle_request(i);
            std::cout << "ok" << std::endl;
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

void myServer::handle_request(int &idx) {
    SOCKET fd = fds[idx].fd;
    char buf[1024];
    std::cout << "handling request from fd: " << fd << std::endl;
    int len = recv(fd, buf, sizeof(buf), 0);
    std::cout << "recv len is: " << len << std::endl;
    std::cout << "buf len is: " << sizeof(buf) << std::endl;
    if (len <= 0) {
        closesocket(fd);
        // 从数组删除该 fd（用最后一个覆盖当前 idx）
        fds[idx] = fds[--nfds];
        idx--; // 避免跳过
        return;
    }
    buf[len] = 0;
    printf("fd [%d] has received message: %s\n", fd, buf);
    send(fd, buf, len, 0); // 回显
    
}