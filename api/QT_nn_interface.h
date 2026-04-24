#pragma once
#include <QMainWindow>
#include <QWidget>
#include <QLabel>
#include <QDialog>
#include <QLineEdit>
#include <QVBoxLayout>
#include <QPushButton>
#include <QMessageBox>
#include <QCheckBox>  

#include <QPainter>
#include <QPen>
#include <QBrush>
#include <QColor>
#include <QFont>
#include <vector>
#include <cctype> 
#include <sstream>
#include "enum_type.h"
#include "nn.h"
namespace nn{
class module_base;
class Linear_NN;
class Linear_Resnet;
struct NNParams;
class model_data;
}
namespace api{
using namespace dtensor;
using namespace std;
using namespace nn;
std::string get_model_type(nn_type model_type);
sub_type string_to_type(std::string s);
std::string type_to_string(sub_type ltp);
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
    void set_style();
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
    QCheckBox* m_threadCheck;   // 多线程勾选框
    QLineEdit* m_threadEdit;     // 线程数输入框
    QLabel* m_threadLabel;       // 线程数文字

    NNParams params;
    // 解析输入的层大小（字符串转vector<int>）
    bool parseLayerSizes(const QString& text);
    bool parseLayerTypes(const QString& text);
    void set_style();
};


class NNVisualWidget : public QWidget
{
    Q_OBJECT
public:
    explicit NNVisualWidget(QWidget *parent = nullptr);

    // 设置网络结构 [784,256,128,10]
    void setLayerSizes(const vector<int>& sizes);

    // 设置激活值（0~1），必须和 layer size 对应
    void setActivations(const vector<vector<float>>& activations);

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    vector<int> m_layerSizes;
    vector<vector<float>> m_activations; // 0~1

    const int NODE_RADIUS = 14;
    const int MAX_DRAW_NODES = 20;
};


} // namespace api
