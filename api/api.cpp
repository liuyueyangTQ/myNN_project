#include <QApplication>
#include "QT_nn_interface.h"
int main(int argc, char *argv[]) {
    // Qt 应用程序入口
    QApplication a(argc, argv);
    
    // 创建主窗口并显示
    api::MainWindow w;
    w.show();

    // 运行事件循环
    return a.exec();
}