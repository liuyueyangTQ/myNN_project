#pragma once
#include <QLabel>
#include <QDialog>
#include <QLineEdit>
#include <QVBoxLayout>
#include <QPushButton>
#include <QMessageBox>
#include <vector>
#include <sstream>

// 神经网络参数结构体（前后台共用）
struct NNParams {
    std::string model_type;       // 模型类型："LinearNN" / "LinearResnet"
    std::vector<int> layer_sizes; // 各层神经元数（如 [784, 256, 128, 10]）
    std::string activation;       // 激活函数："ReLU" / "Sigmoid" / "Tanh"
    bool is_resnet;
};

// 参数输入窗口类
class ParamWindow : public QDialog {
    Q_OBJECT
public:
    // 构造函数：接收模型类型（LinearNN/LinearResnet）
    explicit ParamWindow(const QString& modelType, QWidget *parent = nullptr);
    ~ParamWindow() override = default;

signals:
    // 确认参数后触发的信号：传递封装好的 NNParams
    void paramsConfirmed(const NNParams& params);

private slots:
    // 确认按钮点击事件
    void onConfirmClicked();
    // 取消按钮点击事件
    void onCancelClicked();

private:
    // 控件定义
    QLineEdit* m_layerEdit;    // 输入各层神经元数（逗号分隔）
    QLineEdit* m_actEdit;      // 输入激活函数类型
    QString m_modelType;       // 保存模型类型
    bool is_resnet;
    // 解析输入的层大小（字符串转vector<int>）
    std::vector<int> parseLayerSizes(const QString& text);
    bool parseLayerTypes(const QString& text, std::vector<int>& sizes, bool is_resnet);
};
