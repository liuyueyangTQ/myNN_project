// mainwindow.cpp
#include "main_window.h"
#include <QMessageBox>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    // 设置主窗口
    setWindowTitle("神经网络模型选择");
    setFixedSize(300, 150);

    // 中心窗口（Qt 主窗口必须设置 centralWidget）
    QWidget* centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    // 1. 创建按钮
    QPushButton* linearNNBtn = new QPushButton("Linear NN", this);
    QPushButton* linearResnetBtn = new QPushButton("Linear Resnet", this);

    // 2. 布局（垂直布局）
    QVBoxLayout* mainLayout = new QVBoxLayout(centralWidget);
    mainLayout->addWidget(linearNNBtn);
    mainLayout->addWidget(linearResnetBtn);
    mainLayout->setSpacing(20);
    mainLayout->setContentsMargins(50, 30, 50, 30);

    // 3. 绑定信号槽
    connect(linearNNBtn, &QPushButton::clicked, this, &MainWindow::onLinearNNClicked);
    connect(linearResnetBtn, &QPushButton::clicked, this, &MainWindow::onLinearResnetClicked);
}

// 点击 Linear NN 按钮：弹出参数窗口
void MainWindow::onLinearNNClicked() {
    ParamWindow* paramWin = new ParamWindow("Linear NN", this);
    // 绑定参数确认信号
    connect(paramWin, &ParamWindow::paramsConfirmed, this, &MainWindow::onNNParamsReceived);
    paramWin->exec(); // 以模态方式显示窗口（阻塞，直到关闭）
}

// 点击 Linear Resnet 按钮：弹出参数窗口
void MainWindow::onLinearResnetClicked() {
    ParamWindow* paramWin = new ParamWindow("Linear Resnet", this);
    connect(paramWin, &ParamWindow::paramsConfirmed, this, &MainWindow::onNNParamsReceived);
    paramWin->exec();
}

// 接收参数：调用后台模型
void MainWindow::onNNParamsReceived(const NNParams& params) {
    // 调用后台逻辑
    runNNModel(params);
}

// 模拟后台：打印参数并执行模型（替换为你的实际逻辑）
void MainWindow::runNNModel(const NNParams& params) {
    // 1. 打印参数（调试用）
    qDebug() << "===== 神经网络参数 =====";
    qDebug() << "模型类型：" << QString::fromStdString(params.model_type);
    qDebug() << "各层神经元数：";
    for (int size : params.layer_sizes) {
        qDebug() << "  - " << size;
    }
    qDebug() << "激活函数：" << QString::fromStdString(params.activation);
    qDebug() << "========================";

    // 2. 提示用户模型开始执行
    QMessageBox::information(this, "执行成功", 
        QString("已启动 %1 模型训练！\n"
                "层配置：%2\n"
                "激活函数：%3")
        .arg(QString::fromStdString(params.model_type))
        .arg(QString::fromStdString(
            [&]() {
                std::string s;
                for (size_t i=0; i<params.layer_sizes.size(); ++i) {
                    s += std::to_string(params.layer_sizes[i]);
                    if (i != params.layer_sizes.size()-1) s += ",";
                }
                return s;
            }()
        ))
        .arg(QString::fromStdString(params.activation))
    );

    // 3. 替换为你的实际后台逻辑：
    // 例如：调用你的 ResNet/LinearNN 训练函数
    // MyNNLibrary::train(params.model_type, params.layer_sizes, params.activation);
}