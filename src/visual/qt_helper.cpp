#include "qt_helper.h"
// 构造函数：初始化界面
ParamWindow::ParamWindow(const QString& modelType, QWidget *parent)
    : QDialog(parent), m_modelType(modelType) {
    if(modelType == "Linear NN") {
        is_resnet = false;
    } else if(modelType == "Linear Resnet") {
        is_resnet = true;
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
std::vector<int> ParamWindow::parseLayerSizes(const QString& text) {
    std::vector<int> sizes;
    std::string str = text.toStdString();
    std::stringstream ss(str);
    std::string token;

    while (std::getline(ss, token, ',')) {
        try {
            int size = std::stoi(token);
            if (size <= 0) { // 校验神经元数为正
                QMessageBox::warning(this, "invalid input", "number of neurons must be positive integers");
                return {};
            }
            sizes.push_back(size);
        } catch (...) { // 非数字输入
            QMessageBox::warning(this, "invalid input", "please enter valid number!");
            return {};
        }
    }

    if (sizes.empty()) { // 空输入
        QMessageBox::warning(this, "invalid input", "please enter at least one layer");
    }
    return std::move(sizes);
}

bool ParamWindow::parseLayerTypes(const QString& text, std::vector<int>& sizes, bool is_resnet) {
    std::string str = text.toStdString();
    std::stringstream ss(str);
    std::string token;
    std::vector<std::string> layer_types;
    int len = sizes.size();
    if(is_resnet) {
        if(len > 2) {
            int size_mid = sizes[1];
            for(int i = 1; i < len - 1; ++i) {
                if(sizes[i] != size_mid) {
                    QMessageBox::warning(this, "invalid number!","Resnet Layers must have equal neurons!!");
                    return false;
                }
            }
        } 
    }
    while (std::getline(ss, token, ',')) {
        layer_types.push_back(token);
    }

    if(len != layer_types.size()) { // 校验神经元数和type数相同
        QMessageBox::warning(this, "wrong layers!","layer numbers and types should be equal!");
        return false;
    }
    if (layer_types[0] != "ORIGIN") { 
        QMessageBox::warning(this, "invalid input!", "the first layer must be origin type!");
        return false;
    }

    if (layer_types[len-1] != "SOFTMAX") { // 校验神经元数为正
        QMessageBox::warning(this, "invalid layer type!","The last layer must be Softmax type!");
        return false;
    }
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
    std::vector<int> layerSizes = parseLayerSizes(layerText);
    if (layerSizes.size() <=1) {
        QMessageBox::warning(this, "wrong neuron num!", "number of neuron layers must more than 2!");
        return;
    }


    // 4. 校验激活函数

    bool is_valid = parseLayerTypes(actText, layerSizes, this->is_resnet );
    if(!is_valid) return;

    // 5. 封装参数
    NNParams params;
    params.model_type = m_modelType.toStdString();
    params.layer_sizes = layerSizes;
    params.activation = actText.toStdString();
    params.is_resnet = this->is_resnet;

    // 6. 发送信号（传递参数）+ 关闭窗口
    emit paramsConfirmed(params);
    this->accept(); // 关闭对话框并返回 Accepted
}

// 取消按钮：关闭窗口
void ParamWindow::onCancelClicked() {
    this->reject(); // 关闭对话框并返回 Rejected
}