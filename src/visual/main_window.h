#pragma once
#include <QMainWindow>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>
#include "qt_helper.h"

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override = default;

private slots:
    // 点击 Linear NN 按钮
    void onLinearNNClicked();
    // 点击 Linear Resnet 按钮
    void onLinearResnetClicked();
    // 接收参数窗口传递的参数，调用后台逻辑
    void onNNParamsReceived(const NNParams& params);

private:
    // 模拟后台：执行神经网络模型
    void runNNModel(const NNParams& params);
};
