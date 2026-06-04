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
    setFixedSize(640, 520);

    // 中心窗口（Qt 主窗口必须设置 centralWidget）
    QWidget* centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    // 1. 创建按钮
    QVBoxLayout* mainLayout = new QVBoxLayout(centralWidget);
    mainLayout->setSpacing(0);
    mainLayout->setContentsMargins(60, 30, 60, 40);

    QLabel* titleLabel = new QLabel("欢迎使用神经网络集成训练工具");
    titleLabel->setAlignment(Qt::AlignCenter);
    titleLabel->setStyleSheet(
        "font-size: 26px; font-weight: bold; color: #1a3a5c;"
        "padding-bottom: 6px;"
    );
    QLabel* subtitleLabel = new QLabel("模型训练选项");
    subtitleLabel->setAlignment(Qt::AlignCenter);
    subtitleLabel->setStyleSheet(
        "font-size: 15px; color: #7f8c9b;"
        "padding-bottom: 24px;"
    );
    // 2. 布局（垂直布局）
    mainLayout->addStretch(); // 弹簧，把下面内容推到中间
    mainLayout->addWidget(titleLabel);
    mainLayout->addWidget(subtitleLabel);


    QPushButton* card1 = createOptionCard("FC", "全连接层",
        "包含 Linear NN 与 Linear Resnet 两种架构，\n"
        "支持自定义层数、神经元数与激活函数", true);
    QPushButton* card2 = createOptionCard("CNN", "卷积神经网络",
        "CNN 模型训练模块（功能开发中）", false);
    QPushButton* card3 = createOptionCard("CM", "手动搭建",
        "可视化拖拽搭建网络结构，\n"
        "后台自动解析并训练模型（即将推出）", false);

    mainLayout->addWidget(card1);
    mainLayout->addSpacing(12);
    mainLayout->addWidget(card2);
    mainLayout->addSpacing(12);
    mainLayout->addWidget(card3);
    
    // 3. 绑定信号槽
    connect(card1, &QPushButton::clicked, this, &MainWindow::onFullyConnectedClicked);
    connect(card2, &QPushButton::clicked, this, &MainWindow::onConvolutionalClicked);
    connect(card3, &QPushButton::clicked, this, &MainWindow::onCustomModuleClicked);

    mainLayout->addStretch(); // 占位，推送内容到顶部

    QLabel* footerLabel = new QLabel("v1.0  |  QT Neural Network Suite");
    footerLabel->setAlignment(Qt::AlignCenter);
    footerLabel->setStyleSheet("font-size: 11px; color: #b0b8c1; padding-top: 12px;");
    mainLayout->addWidget(footerLabel);

    // 设置样式
    this->set_style();
}


// 点击 全连接 按钮：弹出参数窗口
void MainWindow::onFullyConnectedClicked() {
    ParamWindow* paramWin = new ParamWindow(this);
    paramWin->setWindowModality(Qt::WindowModal);  // 阻止回点 MainWindow，但不阻止其他顶层窗口
    paramWin->setAttribute(Qt::WA_DeleteOnClose);  // 关闭时自动销毁，不堆垃圾
    paramWin->show();                               // 非模态，不跑独立事件循环
    // paramWin->exec()（模态，锁死一切）
}

void MainWindow::onConvolutionalClicked() {
    PlaceholderWindow* pw = new PlaceholderWindow(
        "卷积神经网络", "CNN 开发中，正在开发中，敬请期待！", this);
    pw->exec();
}

void MainWindow::onCustomModuleClicked() {
    PlaceholderWindow* pw = new PlaceholderWindow(
        "手动搭建", "可视化拖拥擭建网络结构功能即将推出，敬请期待！", this);
    pw->exec();
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

QPushButton* MainWindow::createOptionCard(const QString& icon, const QString& title,
                                       const QString& desc, bool available) {
    QPushButton* card = new QPushButton();
    card->setFlat(true);
    card->setCursor(available ? Qt::PointingHandCursor : Qt::ArrowCursor);
    card->setEnabled(available);
    card->setMinimumHeight(80);

    QHBoxLayout* row = new QHBoxLayout(card);
    row->setSpacing(16);
    row->setContentsMargins(18, 14, 18, 14);

    QLabel* iconLabel = new QLabel(icon);
    iconLabel->setFixedSize(52, 52);
    iconLabel->setAlignment(Qt::AlignCenter);
    iconLabel->setStyleSheet(QString(
        "background-color: %1; border-radius: 26px;"
        "font-size: 18px; font-weight: bold; color: white;"
    ).arg(available ? "#409eff" : "#c0c4cc"));

    QVBoxLayout* textCol = new QVBoxLayout();
    textCol->setSpacing(4);
    QLabel* titleLbl = new QLabel(title);
    titleLbl->setStyleSheet(QString(
        "font-size: 16px; font-weight: bold; color: %1; background: transparent;"
    ).arg(available ? "#2c3e50" : "#a0a8b4"));
    QLabel* descLbl = new QLabel(desc);
    descLbl->setWordWrap(true);
    descLbl->setStyleSheet(QString(
        "font-size: 12px; color: %1; background: transparent;"
    ).arg(available ? "#6b7b8d" : "#b0b8c4"));
    textCol->addWidget(titleLbl);
    textCol->addWidget(descLbl);

    row->addWidget(iconLabel);
    row->addLayout(textCol, 1);

    QLabel* badgeLabel = new QLabel(available ? "›" : "敬请期待");
    badgeLabel->setStyleSheet(QString(
        "font-size: %1; color: %2; font-weight: bold; background: transparent;"
    ).arg(available ? "28px" : "11px",
          available ? "#409eff" : "#c0c4cc"));
    badgeLabel->setAlignment(Qt::AlignCenter);
    badgeLabel->setFixedWidth(available ? 20 : 56);
    row->addWidget(badgeLabel);

    card->setStyleSheet(QString(
        "QPushButton {"
        "  background-color: %1;"
        "  border: 1px solid %2;"
        "  border-radius: 10px;"
        "  text-align: left;"
        "}"
        "QPushButton:hover {"
        "  background-color: %3;"
        "  border-color: #409eff;"
        "}"
    ).arg(available ? "#ffffff" : "#f5f6f8",
          available ? "#e0e4ea" : "#e8eaef",
          available ? "#f0f5ff" : "#f5f6f8"));

    return card;
}

PlaceholderWindow::PlaceholderWindow(const QString& title, const QString& message,
                                     QWidget *parent)
    : QDialog(parent) {
    setWindowTitle(title);
    setFixedSize(420, 220);
    setStyleSheet(
        "QDialog { background-color: #f8f9fa; font-family: Microsoft YaHei; }"
    );

    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->setAlignment(Qt::AlignCenter);
    layout->setSpacing(20);

    QLabel* iconLabel = new QLabel("⌛");
    iconLabel->setAlignment(Qt::AlignCenter);
    iconLabel->setStyleSheet("font-size: 48px; color: #c0c4cc;");

    QLabel* msgLabel = new QLabel(message);
    msgLabel->setAlignment(Qt::AlignCenter);
    msgLabel->setWordWrap(true);
    msgLabel->setStyleSheet("font-size: 16px; color: #5a6a7a;");

    QPushButton* backBtn = new QPushButton("返回");
    backBtn->setStyleSheet(
        "QPushButton { background-color: #409eff; color: white; border: none;"
        "border-radius: 6px; padding: 8px 32px; font-size: 14px; }"
        "QPushButton:hover { background-color: #66b1ff; }"
    );
    connect(backBtn, &QPushButton::clicked, this, &QDialog::accept);

    layout->addStretch();
    layout->addWidget(iconLabel);
    layout->addWidget(msgLabel);
    layout->addWidget(backBtn, 0, Qt::AlignCenter);
    layout->addStretch();
}

// 构造函数：初始化界面
ParamWindow::ParamWindow(QWidget *parent)
    : QDialog(parent) {

    setWindowTitle("配置全连接神经网络参数");
    setMinimumWidth(540);
    setMaximumWidth(540);

    QLabel* modelLabel = new QLabel("模型架构:");
    m_modelCombo = new QComboBox();
    m_modelCombo->addItem("Linear NN");
    m_modelCombo->addItem("Linear Resnet");
    connect(m_modelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), [this](int idx) {
        switch(idx) {
            case 0:
                params.model_type = nn_type::Linear_NN;
                break;
            case 1:
                params.model_type = nn_type::Linear_Resnet;
                break;
        }
    });

    QHBoxLayout* modelRow = new QHBoxLayout();
    modelRow->addWidget(modelLabel);
    modelRow->addWidget(m_modelCombo, 1);
    modelRow->addStretch();

    // ========== layer count slider ==========
    QLabel* layerCountTitle = new QLabel("网络层数 (3-8):");
    m_layerCountLabel = new QLabel("4", this);
    m_layerCountLabel->setAlignment(Qt::AlignCenter);
    m_layerCountLabel->setStyleSheet("font-size: 18px; font-weight: bold; color: #409eff;");

    m_layerCountSlider = new QSlider(Qt::Horizontal, this);
    m_layerCountSlider->setRange(3, 8);
    m_layerCountSlider->setValue(4);
    m_layerCountSlider->setTickPosition(QSlider::TicksBelow);
    m_layerCountSlider->setTickInterval(1);

    m_layersContainer = new QWidget(this);
    m_layersLayout = new QVBoxLayout(m_layersContainer);
    m_layersLayout->setSpacing(8);
    m_layersLayout->setContentsMargins(0, 0, 0, 0);

    connect(m_layerCountSlider, &QSlider::valueChanged, this, [this](int val) {
        m_layerCountLabel->setText(QString::number(val));
        rebuildLayerRows(val);
        this->adjustSize();
    });

    QLabel* layersGroupLabel = new QLabel("各层配置:");
    layersGroupLabel->setStyleSheet("font-weight: bold; font-size: 15px; color: #2c3e50; margin-top: 6px;");

    // ====================== 新增：多线程选项 ======================
    m_threadCheck = new QCheckBox("Enable multi-threading", this);
    m_threadLabel = new QLabel("Number of threads:", this);
    m_threadEdit = new QLineEdit(this);
    m_threadEdit->setPlaceholderText("Example: 4");
    m_threadEdit->setVisible(false);
    m_threadLabel->setVisible(false);
    // 勾选框控制输入框显示/隐藏
    connect(m_threadCheck, &QCheckBox::toggled, [this](bool checked) {
        m_threadLabel->setVisible(checked);
        m_threadEdit->setVisible(checked);
        // 让窗口自动适应内容大小！
        this->adjustSize();
    });

    // 确认 / 取消 按钮
    QPushButton* confirmBtn = new QPushButton("开始训练", this);
    QPushButton* cancelBtn = new QPushButton("返回", this);

    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    mainLayout->addLayout(modelRow);
    mainLayout->addSpacing(8);
    mainLayout->addWidget(layerCountTitle);
    QHBoxLayout* sliderRow = new QHBoxLayout();
    sliderRow->addWidget(m_layerCountSlider, 1);
    sliderRow->addWidget(m_layerCountLabel);
    mainLayout->addLayout(sliderRow);
    mainLayout->addWidget(layersGroupLabel);
    mainLayout->addWidget(m_layersContainer);
    mainLayout->addSpacing(8);
    mainLayout->addWidget(m_threadCheck);
    mainLayout->addWidget(m_threadLabel);
    mainLayout->addWidget(m_threadEdit);

    m_statusLabel = new QLabel();
    m_statusLabel->setAlignment(Qt::AlignCenter);
    m_statusLabel->setStyleSheet(
        "font-size: 16px; font-weight: bold; color: #67c23a;"
        "padding: 10px;"
    );
    m_statusLabel->setVisible(false); // (训练完成状态)初始隐藏，训练完成后显示

    mainLayout->addWidget(m_statusLabel);
    mainLayout->addWidget(m_vizButton, 0, Qt::AlignCenter);
    m_vizButton = new QPushButton("可视化"); 
    m_vizButton->setStyleSheet(
        "QPushButton { background-color: #67c23a; color: white; border: none;"
        "border-radius: 6px; padding: 8px 16px; font-size: 14px; }"
        "QPushButton:hover { background-color: #85ce61; }"
    );
    m_vizButton->setVisible(false);
    connect(m_vizButton, &QPushButton::clicked, this, &ParamWindow::onVisualizeClicked);

    QHBoxLayout* btnLayout = new QHBoxLayout();
    btnLayout->addStretch();

    btnLayout->addWidget(m_vizButton); // 可视化按钮位置和 confirmBtn cancelBtn 齐平
    btnLayout->addWidget(confirmBtn);
    btnLayout->addWidget(cancelBtn);
    mainLayout->addLayout(btnLayout);

    setLayout(mainLayout);
    this->set_style();
    cancelBtn->setObjectName("CancelBtn");

    mainLayout->setSpacing(10);
    mainLayout->setContentsMargins(24, 20, 24, 20);
    btnLayout->setSpacing(12);

    connect(confirmBtn, &QPushButton::clicked, this, &ParamWindow::onConfirmClicked);
    connect(cancelBtn, &QPushButton::clicked, this, &ParamWindow::onCancelClicked);

    rebuildLayerRows(4);
}

// 模拟后台：打印参数并执行模型（替换为你的实际逻辑）
model_data ParamWindow::runNNModel(const NNParams& params) {
    // 1. 打印参数（调试用）
    qDebug() << "===== 神经网络参数 =====";
    qDebug() << "模型类型：" << get_model_type(params.model_type);
    qDebug() << "各层神经元数(激活函数): ";
    for (int i = 0; i < params.layer_sizes.size(); ++i) {
        qDebug() << "  - " << params.layer_sizes[i] << " - " << type_to_string(params.layer_types[i]);
    }
    // 执行相应的 ResNet/LinearNN 训练函数
    model_data res = nn::run_model(params);
    return res;
}

// 接收参数：调用后台模型
void ParamWindow::TrainingStateFinished() {
    m_statusLabel->setText("训练完成！");
    m_statusLabel->setVisible(true); // 训练完成后显示
    m_vizButton->setVisible(true); // 显示可视化按钮
    m_trainingDone = true;
    this->adjustSize(); // 调整窗口大小以适应新的状态（显示状态标签和可视化按钮）
}

void ParamWindow::TrainingStateStarted() {
    m_statusLabel->setText("训练中...");
    m_statusLabel->setVisible(true);
    m_vizButton->setVisible(false); // 训练开始后隐藏可视化按钮，直到训练完成才显示
    m_trainingDone = false;
    this->adjustSize();
}

// 可视化展示
void ParamWindow::onVisualizeClicked() {
    if (!m_trainingDone) {
        std::cerr << "Error: Training not completed yet!" << std::endl;
        return;
    }
    NNVisualWidget* viz = new NNVisualWidget();
    viz->setLayerSizes(m_layerSizes);
    viz->setActivations(m_resultActs);
    viz->show();
    viz->raise();             // 提到最上层
    viz->activateWindow();    // 抢键盘焦点
    // this->accept();        // ParamWindow 保持打开
}

void ParamWindow::rebuildLayerRows(int count) {
    // delete all old child widgets first
    QList<QWidget*> kids = m_layersContainer->findChildren<QWidget*>(QString(), Qt::FindDirectChildrenOnly);
    for (QWidget* w : kids) { delete w; }
    // clear remaining sub-layout items
    while (m_layersLayout->count() > 0) {
        delete m_layersLayout->takeAt(0);
    }
    // build new rows
    for (int i = 0; i < count; ++i) {
        QHBoxLayout* row = new QHBoxLayout();
        row->setSpacing(8);
        QLabel* label = new QLabel(QString("Layer %1:").arg(i + 1));
        label->setFixedWidth(60);
        label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        QSpinBox* neuronBox = new QSpinBox();
        neuronBox->setRange(1, 10000);
        neuronBox->setValue((i == 0 || i == count - 1) ? 10 : 128);
        neuronBox->setSuffix(" neurons");
        QComboBox* typeBox = new QComboBox();
        if (i == 0) { typeBox->addItem("ORIGIN"); typeBox->setEnabled(false); }
        else if (i == count - 1) { typeBox->addItem("SOFTMAX"); typeBox->setEnabled(false); }
        else { typeBox->addItems({"RELU", "SIGMOID"}); }
        row->addWidget(label);
        row->addWidget(neuronBox, 1);
        row->addWidget(typeBox, 1);
        m_layersLayout->addLayout(row);
    }
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
    QList<QSpinBox*> neuronBoxes = m_layersContainer->findChildren<QSpinBox*>();
    QList<QComboBox*> typeBoxes = m_layersContainer->findChildren<QComboBox*>();
    if (neuronBoxes.isEmpty() || typeBoxes.isEmpty()) {
        QMessageBox::warning(this, "input empty", "please configure the layers!");
        return;
    }
    int layerCount = neuronBoxes.size();
    std::vector<int> layer_sizes;
    std::vector<sub_type> layer_types;
    for (int i = 0; i < layerCount; ++i) {
        int neurons = neuronBoxes[i]->value();
        if (neurons <= 0) {
            QMessageBox::warning(this, "invalid input", "number of neurons must be positive integers");
            return;
        }
        layer_sizes.push_back(neurons);
        QString typeStr = typeBoxes[i]->currentText().toUpper();
        layer_types.push_back(string_to_type(typeStr.toStdString()));
    }
    if (layer_types.front() != sub_type::origin) {
        QMessageBox::warning(this, "invalid input!", "the first layer must be ORIGIN type!");
        return;
    }
    if (layer_types.back() != sub_type::softmax) {
        QMessageBox::warning(this, "invalid layer type!", "The last layer must be SOFTMAX type!");
        return;
    }
    if (params.model_type == nn_type::Linear_Resnet && layerCount > 2) {
        int midNeurons = layer_sizes[1];
        for (int i = 1; i < layerCount - 1; ++i) {
            if (layer_sizes[i] != midNeurons) {
                QMessageBox::warning(this, "invalid number!", "Resnet Layers must have equal neurons!!");
                return;
            }
        }
    }
    this->params.layer_sizes = std::move(layer_sizes);
    this->params.layer_types = std::move(layer_types);
    this->params.layer_num = this->params.layer_sizes.size();
    this->params.input_output_dim = {this->params.layer_sizes.front(), this->params.layer_sizes.back()};
    bool useThread = m_threadCheck->isChecked();
    this->params.use_multithread = useThread;
    int threadNum = m_threadEdit->text().toInt();
    this->params.thread_num = threadNum > 0 ? threadNum : 4; // 默认线程数为4

    // 6. 校验参数（已经在校验函数中实现了）
    this->params.check();

    this->TrainingStateStarted(); // 训练开始状态更新

    QApplication::processEvents(); // 强制刷新UI，显示"训练中..."
    // processEvents() 强制 Qt 立刻处理积压的绘制事件，把"训练中..."渲染到屏幕上，然后再进入阻塞的训练函数。

    model_data res = this->runNNModel(this->params); // 调用模型训练（阻塞界面，实际应用中应放在子线程）
    m_layerSizes = res.layer_sizes;
    m_resultActs = res.outputs[0]; // 第一个样本的激活值
    this->TrainingStateFinished(); // 训练完成状态更新
}

// 取消按钮：关闭窗口
void ParamWindow::onCancelClicked() {
    this->reject(); // 关闭对话框并返回 Rejected
}

NNVisualWidget::NNVisualWidget(QWidget *parent) : QWidget(parent)
{
    setWindowFlags(Qt::Window);
    setWindowTitle("神经网络可视化");
    setAttribute(Qt::WA_DeleteOnClose); // 关闭窗口自动析构
    setStyleSheet("background-color:white;");
    resize(900, 700);
    setMinimumSize(600, 400);
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

// 暂时弃用
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
    this->params.layer_num = this->params.layer_sizes.size();
    this->params.input_output_dim = {this->params.layer_sizes.front(), this->params.layer_sizes.back()};
    return true;
}

} // namespace api