#ifdef _WIN32
    // Windows 平台
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #include <ws2tcpip.h>
    // 链接 Winsock 库
    #pragma comment(lib, "ws2_32.lib")
    typedef SOCKET socket_t;
    #define CLOSESOCKET closesocket
    #define POLLWS    WSAPOLLFD
    #define POLL_FUNC WSAPoll
    #define socklen_t int
#else
    // Linux 平台
    #include <unistd.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <poll.h>
    typedef int socket_t;
    #define CLOSESOCKET close
    #define POLLWS    struct pollfd
    #define POLL_FUNC poll
#endif
#include <stdexcept>
#include <iostream>
#include <cassert>
#include <string>
#define PORT 8888
#define MAX_FDS 1024
#define BUF_SIZE 1024 * 8
#define DATA_BUF_SIZE 100000
// 包含tensor库
//#include "nn.h"

namespace netService {
class __netServer_base {
protected:
#ifdef _WIN32
    WSADATA wsaData;
    WSAPOLLFD* fds;
#endif
    int port;
    int conn_num;
    int nfds;
    bool blocking;
    SOCKET temp_fd;
    char buf[BUF_SIZE];
    char* data_buf;
    void _start_service_blk();
    void _start_service_nonblk();
    virtual void handle_msg(int& idx, int len) = 0;
    void close_conn(int& idx);
    int _read_msg_blk(int& idx);
    int _read_msg_nonblk(int& idx);
public:
    __netServer_base(bool blocking = false, int port = PORT, int conn_num = 1) : port(port), conn_num(conn_num), blocking(blocking) {
        assert(conn_num <= MAX_FDS);
        fds = new WSAPOLLFD[conn_num];
        data_buf = new char[DATA_BUF_SIZE];
        nfds = 0; 
    }
    ~__netServer_base() {
        delete[] fds;
        delete[] data_buf;
    }
    void start_service();
    virtual void handle_request(int &idx) = 0;
    void send_data(std::string& data, int fd);
    void send_data(char* data, int len, int fd);
};
class demoServer : public __netServer_base {
protected:
    void handle_msg(int& idx, int len) override;
public:
    demoServer(int blocking = false, int port = PORT, int conn_num = 1) : __netServer_base(blocking, port, conn_num) {}
    void handle_request(int &idx) override;
};

class __netClient_base {
protected:
#ifdef _WIN32
    WSADATA wsaData;
#endif
    int port;
    char buf[BUF_SIZE];
    char* data_buf;
    int fd;
    virtual void handle_msg(int len) = 0;
    void _connect_to_server(const std::string& server_ip);
    int _read_msg_blk();
public:
    __netClient_base(int port = PORT) : port(port) {
        data_buf = new char[DATA_BUF_SIZE];
    }
    ~__netClient_base() {
        delete[] data_buf;
    }
    void send_data(std::string&& data);
    void send_data(char* data, int len);
    virtual void handle_request() = 0;
};
class demoClient : public __netClient_base {
protected:
    void handle_msg(int len) override;
public:
    demoClient(int port = PORT) : __netClient_base(port) {}
    void connect(const std::string& server_ip);
    void handle_request() override;
};
} // namespace netService