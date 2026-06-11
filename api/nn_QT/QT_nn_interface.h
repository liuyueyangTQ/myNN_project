#pragma once
#include <QMainWindow>
#include <QWidget>
#include <QLabel>
#include <QDialog>
#include <QLineEdit>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QMessageBox>
#include <QCheckBox>
#include <QSlider>
#include <QSpinBox>
#include <QApplication>
#include <QComboBox>

#include <QPainter>
#include <QPen>
#include <QBrush>
#include <QColor>
#include <QFont>
#include <vector>
#include <cctype> 
#include <sstream>
#include "enum_types.h"
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
signals:
    // 确认参数后触发的信号：传递封装好的 NNParams
    void TrainingFinished();
private slots:
    // 点击 Fully Connected 按钮
    void onFullyConnectedClicked();
    // 点击 Convolutional 按钮
    void onConvolutionalClicked();
    // 点击 Custom Module 按钮
    void onCustomModuleClicked();

private:
    QPushButton* createOptionCard(const QString& icon, const QString& title, // 三张选项卡片
                              const QString& desc, bool available);
    void set_style();
};

class PlaceholderWindow : public QDialog {
    Q_OBJECT
public:
    explicit PlaceholderWindow(const QString& title, const QString& message,
                               QWidget *parent = nullptr);
};

// 参数输入窗口类
class ParamWindow : public QDialog {
    Q_OBJECT
public:
    // 构造函数：接收模型类型（LinearNN/LinearResnet）
    explicit ParamWindow(QWidget *parent = nullptr);
    ~ParamWindow() override = default;

signals:
    // 确认参数后触发的信号：传递封装好的 NNParams
    void paramsConfirmed(const NNParams& params);

private slots:
    // 确认按钮点击事件
    void onConfirmClicked();
    // 取消按钮点击事件
    void onCancelClicked();
    // 可视化按钮点击事件
    void onVisualizeClicked();

private:
    QComboBox* m_modelCombo = nullptr;
    // layer count slider
    QSlider* m_layerCountSlider = nullptr;
    QLabel* m_layerCountLabel = nullptr;
    // dynamic layer rows container
    QWidget* m_layersContainer = nullptr;
    QVBoxLayout* m_layersLayout = nullptr;
    // 控件定义
    QLineEdit* m_layerEdit = nullptr;    // 输入各层神经元数（逗号分隔）, \@ 暂时不用
    QLineEdit* m_actEdit = nullptr;      // 输入激活函数类型 ， \@暂时不用
    nn_type m_modelType;       // 保存模型类型
    QCheckBox* m_threadCheck = nullptr;   // 多线程勾选框
    QLineEdit* m_threadEdit = nullptr;     // 线程数输入框
    QLabel* m_threadLabel = nullptr;       // 线程数文字

    QLabel* m_statusLabel = nullptr; // 训练状态标签
    QPushButton* m_vizButton = nullptr; // 可视化按钮

    vector<int> m_layerSizes;
    vector<vector<float>> m_resultActs;
    bool m_trainingDone = false;

    NNParams params;
    // 解析输入的层大小（字符串转vector<int>）
    bool parseLayerSizes(const QString& text);
    bool parseLayerTypes(const QString& text);
    void set_style();
    void rebuildLayerRows(int count);
    // 模拟后台：执行神经网络模型
    model_data runNNModel(const NNParams& params);
    // 训练完毕的信号处理槽函数
    void TrainingStateFinished();
    void TrainingStateStarted();
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
    friend class ParamWindow; // 允许 ParamWindow 访问私有成员
    vector<int> m_layerSizes;
    vector<vector<float>> m_activations; // 0~1
    const int NODE_RADIUS = 14;
    const int MAX_DRAW_NODES = 20;
};


} // namespace api
