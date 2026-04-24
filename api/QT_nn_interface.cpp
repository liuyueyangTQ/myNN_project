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

    // 设置样式
    this->set_style();

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
    nn::model_data res = nn::run_model(params);
    res.check_model(); // 检查模型输出的正确性
    // 创建绘图窗口
    NNVisualWidget* viz = new NNVisualWidget();
    // 设置网络结构
    vector<int> layers = params.layer_sizes;
    viz->setLayerSizes(layers);
    // 设置激活值 0~1
    vector<vector<float>> acts = res.outputs[0]; // 最后一层输出作为激活值示例
    // {
    //     {0.0, 0.2, 0.4, 0.6, 0.8, 1.0},   // 输入层
    //     {0.1, 0.3, 0.5, 0.7, 0.9},        // 隐藏层1
    //     {0.2, 0.4, 0.6, 0.8},             // 隐藏层2
    //     {0.5, 0.5, 0.5, 0.5, 0.5}         // 输出层
    // };
    viz->setActivations(acts);
    viz->show();

}
void MainWindow::set_style() {
    // ========== 窗口美化 ==========
    this->setStyleSheet(R"(
        QMainWindow {
            background-color: #f0f2f5;
            font-family: Microsoft YaHei;
        }
        QPushButton {
            background-color: #409eff;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 10px 20px;
            font-size: 16px;
        }
        QPushButton:hover {
            background-color: #66b1ff;
        }
        QPushButton:pressed {
            background-color: #337ecc;
        }
    )");
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
    // setFixedSize(500,250); // 固定窗口大小
    // 固定宽度，高度自动变化
    setMinimumWidth(500);
    setMaximumWidth(500);
    // 1. 创建控件
    QLabel* layerLabel = new QLabel("Layer sizes (comma separated, e.g. 784,256,10):");
    m_layerEdit = new QLineEdit(this);
    m_layerEdit->setPlaceholderText("Example: 784,256,128,10");

    QLabel* actLabel = new QLabel("Activation function (ReLU/Sigmoid/Tanh):");
    m_actEdit = new QLineEdit(this);
    m_actEdit->setPlaceholderText("Example: ReLU");

    // ====================== 新增：多线程选项 ======================
    m_threadCheck = new QCheckBox("Enable multi-threading", this);
    m_threadLabel = new QLabel("Number of threads:", this);
    m_threadEdit = new QLineEdit(this);
    m_threadEdit->setPlaceholderText("Example: 4");
    m_threadEdit->setVisible(false);
    m_threadLabel->setVisible(false);

    // 勾选框控制输入框显示/隐藏
    connect(m_threadCheck, &QCheckBox::toggled, [=](bool checked) {
        m_threadLabel->setVisible(checked);
        m_threadEdit->setVisible(checked);
        // 让窗口自动适应内容大小！
        this->adjustSize();
    });

    // 确认 / 取消 按钮
    QPushButton* confirmBtn = new QPushButton("OK", this);
    QPushButton* cancelBtn = new QPushButton("Cancel", this);

    // 2. 布局管理（垂直布局）
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    mainLayout->addWidget(layerLabel);
    mainLayout->addWidget(m_layerEdit);
    mainLayout->addWidget(actLabel);
    mainLayout->addWidget(m_actEdit);
    // 添加多线程控件
    mainLayout->addWidget(m_threadCheck);
    mainLayout->addWidget(m_threadLabel);
    mainLayout->addWidget(m_threadEdit);    

    // 按钮水平布局
    QHBoxLayout* btnLayout = new QHBoxLayout();
    btnLayout->addWidget(confirmBtn);
    btnLayout->addWidget(cancelBtn);
    mainLayout->addLayout(btnLayout);

    setLayout(mainLayout);

    this->set_style(); // 设置样式
    // 给取消按钮设置对象名，方便单独美化
    cancelBtn->setObjectName("CancelBtn");
    // ========== 优化布局间距（更美观、不拥挤） ==========
    mainLayout->setSpacing(12);
    mainLayout->setContentsMargins(24, 20, 24, 20);
    btnLayout->setSpacing(12);
    btnLayout->addStretch(); // 按钮靠右对齐

    // 3. 绑定信号槽
    connect(confirmBtn, &QPushButton::clicked, this, &ParamWindow::onConfirmClicked);
    connect(cancelBtn, &QPushButton::clicked, this, &ParamWindow::onCancelClicked);
}
void ParamWindow::set_style() {
    // ========== 窗口美化 ==========
    this->setStyleSheet(R"(
        QDialog {
            background-color: #f8f9fa;
            font-family: Microsoft YaHei;
            font-size: 14px;
        }
        QLabel {
            color: #2c3e50;
            font-size: 14px;
            font-weight: 500;
        }
        QLineEdit {
            border: 1px solid #dcdfe6;
            border-radius: 6px;
            padding: 6px 10px;    
            background-color: white;
            font-size: 14px;
            min-height: 20px;
        }
        QLineEdit:focus {
            border: 1px solid #409eff;
            outline: none;
        }
        QPushButton {
            background-color: #409eff;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 10px 24px;  /* 变大内边距，最关键！*/
            font-size: 14px;
            min-width: 100px;  /* 强制最小宽度，不会挤 */
            min-height: 28px;     /* 强制高度 */
        }
        QPushButton:hover {
            background-color: #66b1ff;
        }
        QPushButton:pressed {
            background-color: #337ecc;
        }
        QPushButton#CancelBtn {
            background-color: #909399;
        }
        QPushButton#CancelBtn:hover {
            background-color: #a6a9ad;
        }
    )");
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
    // 5. 解析多线程选项
    bool useThread = m_threadCheck->isChecked();
    this->params.use_multithread = useThread;
    int threadNum = m_threadEdit->text().toInt();
    this->params.thread_num = threadNum > 0 ? threadNum : 4; // 默认线程数为4

    // 6. 封装参数（已经在校验函数中实现了）

    // 7. 发送信号（传递参数）+ 关闭窗口
    emit paramsConfirmed(this->params);
    this->accept(); // 关闭对话框并返回 Accepted
}

// 取消按钮：关闭窗口
void ParamWindow::onCancelClicked() {
    this->reject(); // 关闭对话框并返回 Rejected
}


NNVisualWidget::NNVisualWidget(QWidget *parent) : QWidget(parent)
{
    setStyleSheet("background-color:white;");
    setMinimumSize(900, 700);
}

void NNVisualWidget::setLayerSizes(const vector<int>& sizes)
{
    m_layerSizes = sizes;
    update();
}

void NNVisualWidget::setActivations(const vector<vector<float>>& activations)
{
    m_activations = activations;
    update();
}

void NNVisualWidget::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event);
    if (m_layerSizes.empty()) return;

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    QFont textFont = painter.font();
    textFont.setPointSize(12);
    painter.setFont(textFont);

    int w = width();
    int h = height();
    int layers = m_layerSizes.size();
    int xStep = w / (layers + 1);

    // 画每层神经元 + 连线
    for (int L = 0; L < layers; ++L) {
        int x = (L + 1) * xStep;
        int total = m_layerSizes[L];
        int draw = min(total, MAX_DRAW_NODES);
        int yStep = h / (draw + 1);
        // 画权重连线
        if (L == layers - 1) continue;
        int x2 = (L + 2) * xStep;
        int n2 = min(m_layerSizes[L+1], MAX_DRAW_NODES);
        int y2Step = h / (n2 + 1);

        painter.setPen(QPen(QColor(220,220,220), 1));
        for (int i = 0; i < draw; ++i) {
            int y1 = (i + 1) * yStep;
            for (int j = 0; j < n2; ++j) {
                int y2 = (j + 1) * y2Step;
                painter.drawLine(x, y1, x2, y2);
            }
        }
    }
    // ========== 第二步：再画所有神经元圆圈（置顶盖住线条） ==========
    int lastLayerIdx = layers - 1;
    int outX = (lastLayerIdx + 1) * xStep;
    int arrowLen = xStep / 2;   // 箭头长度
    int textOffsetY = -8;      // 文字往上偏移一点
    for (int L = 0; L < layers; ++L)
    {
        int x = (L + 1) * xStep;
        int total = m_layerSizes[L];
        int draw = min(total, MAX_DRAW_NODES);
        int yStep = h / (draw + 1);

        for (int i = 0; i < draw; ++i)
        {
            int y = (i + 1) * yStep;

            // 0~1：浅灰 → 深蓝
            double val = 0.5;
            if (L < m_activations.size() && i < m_activations[L].size())
                val = m_activations[L][i];
            val = qBound(0.0, val, 1.0);

            int r = 200 - val * 160;
            int g = 220 - val * 180;
            int b = 240;
            QColor fillColor(r, g, b);

            painter.setPen(QPen(Qt::black, 1.5));
            painter.setBrush(fillColor);
            painter.drawEllipse(x - NODE_RADIUS, y - NODE_RADIUS,
                                NODE_RADIUS * 2, NODE_RADIUS * 2);
            // ========== 第三步：仅输出层 画箭头 + 标数值(保留两位小数) ==========
            if (L == lastLayerIdx)
            {
                // 画横向箭头主线
                painter.setPen(QPen(Qt::darkGray, 2));
                painter.drawLine(x + NODE_RADIUS* 1.5, y, x + NODE_RADIUS + arrowLen, y);

                // 画箭头三角
                int arrW = 6;
                int arrH = 4;
                QPoint p1(x + NODE_RADIUS + arrowLen, y);
                QPoint p2(x + NODE_RADIUS + arrowLen - arrW, y - arrH);
                QPoint p3(x + NODE_RADIUS + arrowLen - arrW, y + arrH);
                painter.setBrush(Qt::darkGray);
                painter.drawPolygon(QPolygon({p1,p2,p3}));

                // 标注数值 保留两位小数
                QString numText = QString::asprintf("%.3f", val);
                painter.setPen(QPen(Qt::black,1));
                painter.drawText(x + NODE_RADIUS + 12, y + textOffsetY, numText);
            }
        }
    }
}

} // namespace api