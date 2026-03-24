#include "QT_nn_interface.h"
namespace api {
using namespace dtensor;
using namespace nn;
std::string get_model_type(nn_type model_type) {
    switch (model_type)
    {
    case nn_type::Linear_Resnet:
        return "Linear Resnet";
    case nn_type::Linear_NN:
        return "Linear NN";
    default:
        return "Invalid Model!";
    }
}
MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    // 设置主窗口
    setWindowTitle("神经网络训练器");
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
    qDebug() << "模型类型：" << get_model_type(params.model_type);
    qDebug() << "各层神经元数(激活函数): ";
    for (int i = 0; i < params.layer_sizes.size(); ++i) {
        qDebug() << "  - " << params.layer_sizes[i] << " - " << type_to_string(params.layer_types[i]);
    }
    // 执行相应的 ResNet/LinearNN 训练函数
    nn::run_model(params);
}
// 构造函数：初始化界面
ParamWindow::ParamWindow(const QString& modelType, QWidget *parent)
    : QDialog(parent), m_modelType(modelType) {
    if(modelType == "Linear NN") {
        params.model_type = nn_type::Linear_NN;
    } else if(modelType == "Linear Resnet") {
        params.model_type = nn_type::Linear_Resnet;
    }
    // 设置窗口标题
    setWindowTitle(QString("配置 %1 参数").arg(modelType));
    setFixedSize(400, 200); // 固定窗口大小
    // 1. 创建控件
    QLabel* layerLabel = new QLabel("Layer sizes (comma separated, e.g. 784,256,10):");
    m_layerEdit = new QLineEdit(this);
    m_layerEdit->setPlaceholderText("Example: 784,256,128,10");

    QLabel* actLabel = new QLabel("Activation function (ReLU/Sigmoid/Tanh):");
    m_actEdit = new QLineEdit(this);
    m_actEdit->setPlaceholderText("Example: ReLU");

    QPushButton* confirmBtn = new QPushButton("OK", this);
    QPushButton* cancelBtn = new QPushButton("Cancel", this);

    // 2. 布局管理（垂直布局）
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    mainLayout->addWidget(layerLabel);
    mainLayout->addWidget(m_layerEdit);
    mainLayout->addWidget(actLabel);
    mainLayout->addWidget(m_actEdit);
    
    // 按钮水平布局
    QHBoxLayout* btnLayout = new QHBoxLayout();
    btnLayout->addWidget(confirmBtn);
    btnLayout->addWidget(cancelBtn);
    mainLayout->addLayout(btnLayout);

    setLayout(mainLayout);

    // 3. 绑定信号槽
    connect(confirmBtn, &QPushButton::clicked, this, &ParamWindow::onConfirmClicked);
    connect(cancelBtn, &QPushButton::clicked, this, &ParamWindow::onCancelClicked);
}

// 解析层大小："784,256,10" → [784,256,10]
bool ParamWindow::parseLayerSizes(const QString& text) {
    std::vector<int> layer_size;
    std::string str = text.toStdString();
    std::stringstream ss(str);
    std::string token;

    while (std::getline(ss, token, ',')) {
        try {
            int num = std::stoi(token);
            if (num <= 0) { // 校验神经元数为正
                QMessageBox::warning(this, "invalid input", "number of neurons must be positive integers");
                return false;
            }
            layer_size.push_back(num);
        } catch (...) { // 非数字输入
            QMessageBox::warning(this, "invalid input", "please enter valid number!");
            return false;
        }
    }
    if (layer_size.size() < 2) { // 只有一层
        QMessageBox::warning(this, "invalid input", "please enter at least two layers!!");
    }
    // 正确后赋值
    this->params.layer_sizes = std::move(layer_size);
    return true;
}
sub_type string_to_type(std::string str) {
    for (size_t i = 0; i < str.size(); ++i) {
        // 强转 unsigned char 避免负数（如扩展 ASCII 字符）
        str[i] = static_cast<char>(toupper(static_cast<unsigned char>(str[i])));
    }
    if(str == "ORIGIN")
        return sub_type::origin;
    if(str == "RELU")
        return sub_type::relu;
    if(str == "SOFTMAX")
        return sub_type::softmax;
    if(str == "SIGMOID")
        return sub_type::sigmoid;
    return sub_type::none;
}
std::string type_to_string(sub_type ltp) {
    switch (ltp)
    {
    case sub_type::origin:
        return "ORIGIN";
    case sub_type::relu:
        return "RELU";
    case sub_type::softmax:
        return "SOFTMAX";
    case sub_type::sigmoid:
        return "SIGMOID";
    default:
        return "NONE!";
    }
}
bool ParamWindow::parseLayerTypes(const QString& text) {
    std::string str = text.toStdString();
    std::stringstream ss(str);
    std::string token;
    // assert(this->params.layer_sizes.size() > 0);
    if(this->params.layer_sizes.size() == 0) {
        QMessageBox::warning(this, "Sequence fault!","Layer numbers should be mentioned first!!");
        return false;
    }
    int len = this->params.layer_sizes.size();
    if(params.model_type == nn_type::Linear_Resnet) {
        if(len > 2) {
            int size_mid = params.layer_sizes[1];
            for(int i = 1; i < len - 1; ++i) {
                if(params.layer_sizes[i] != size_mid) {
                    QMessageBox::warning(this, "invalid number!","Resnet Layers must have equal neurons!!");
                    return false;
                }
            }
        } 
    }
    std::vector<sub_type> temp;
    while (std::getline(ss, token, ',')) {
        temp.push_back(string_to_type(token));
    }
    if(len != temp.size()) { // 校验神经元数和type数相同
        QMessageBox::warning(this, "wrong layers!","layer numbers and types should be equal!");
        return false;
    }
    if (temp[0] != sub_type::origin) { 
        QMessageBox::warning(this, "invalid input!", "the first layer must be origin type!");
        return false;
    }
    if (temp[len - 1] != sub_type::softmax) { // 校验神经元数为正
        QMessageBox::warning(this, "invalid layer type!","The last layer must be Softmax type!");
        return false;
    }
    // 全部符合后，再赋值
    params.layer_types = std::move(temp);
    return true;
}

// 确认按钮：校验并发送参数
void ParamWindow::onConfirmClicked() {
    // 1. 获取输入
    QString layerText = m_layerEdit->text().trimmed();
    QString actText = m_actEdit->text().trimmed().toUpper(); // 统一转大写
    // 2. 校验输入
    if (layerText.isEmpty() || actText.isEmpty()) {
        QMessageBox::warning(this, "input empty", "please enter the parameters!");
        return;
    }
    // 3. 解析层大小
    bool is_valid1 = parseLayerSizes(layerText);
    if (!is_valid1) {
        QMessageBox::warning(this, "wrong neuron num!", "Please check the layer numbers!");
        return;
    }
    // 4. 校验激活函数
    bool is_valid2 = parseLayerTypes(actText);
    if(!is_valid2) 
        return;

    // 5. 封装参数（已经在校验函数中实现了）

    // 6. 发送信号（传递参数）+ 关闭窗口
    emit paramsConfirmed(this->params);
    this->accept(); // 关闭对话框并返回 Accepted
}

// 取消按钮：关闭窗口
void ParamWindow::onCancelClicked() {
    this->reject(); // 关闭对话框并返回 Rejected
}

} // namespace api