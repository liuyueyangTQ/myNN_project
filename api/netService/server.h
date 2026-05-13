#ifdef _WIN32
    // Windows 平台
    #include <winsock2.h>
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
#define PORT 8888
#define MAX_FDS 1024
#define BUF_SIZE 1024
// 包含tensor库
//#include "nn.h"


class myServer {
private:
#ifdef _WIN32
    WSADATA wsaData;

#endif
    int port;
    int conn_num;
    int nfds;
    WSAPOLLFD* fds;
    //char buf[1024];
    //static const int max_fds;
    void start_service();
public:
    myServer(int port = PORT, int conn_num = 1) : port(port), conn_num(conn_num) {
        fds = new WSAPOLLFD[conn_num];
        nfds = 0; 
        start_service();
    }
    ~myServer() {
        delete[] fds;
    }
    void handle_request(int &idx);
};
