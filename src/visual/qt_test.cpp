#include <QApplication>
#include "main_window.h"
#include "qt_helper.h"
int main(int argc, char *argv[]) {
    // Qt 应用程序入口
    QApplication a(argc, argv);
    
    // 创建主窗口并显示
    MainWindow w;
    w.show();

    // 运行事件循环
    return a.exec();
}