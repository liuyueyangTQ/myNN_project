#include <iostream>
#include <cmath>
#include <vector>
#include <memory>
#include "buttons.h"

// --- Button 的静态成员定义 ---
constexpr const char* Button::fragmentShaderSource;
constexpr const char* Button::vertexShaderSource;

int Button::numButtons = 0;
int CircleButton::numCircleButtons = 0;
int RectButton::numRectButtons = 0;
int TriangleButton::numTriangleButtons = 0;
std::vector<Button*> MyWindow::buttons;

// --- MyWindow 的静态成员定义 ---
GLFWwindow* MyWindow::window = nullptr; // 初始化为 nullptr
std::pair<int,int> MyWindow::size = std::pair<int,int>(0, 0); // 初始化一个默认尺寸



// CircleButton::CircleButton(MyWindow& win, std::pair<double,double> pos,double radius,std::vector<float>& color, bool detect = true);

void MyWindow::staticMouseCallback(GLFWwindow* window, int button, int action, int mods) {
    // // 从窗口用户数据中取出类实例指针（关键：获取 this 指针）
    // CircleButton* instance = static_cast<CircleButton*>(glfwGetWindowUserPointer(window));
    // if (instance) {
    //     // 转调非静态成员函数（传递参数，实现逻辑）
    //     instance->onMouseClick(button, action, mods);
    // }
    MyWindow::onMouseClick(MyWindow::window, button, action, mods);
}
void MyWindow::initializeGLFW() {
    // 初始化 GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return;
    }
    // 配置窗口和 OpenGL 版本（3.3 核心模式）
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    return;
}

void MyWindow::initializeGLEW() {
    // 初始化 GLEW（必须在设置当前上下文后）
    glewExperimental = GL_TRUE; // 启用实验性扩展
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return;
    }
    return;
}
void MyWindow::onMouseClick(GLFWwindow* window, int button, int action, int mods) {  // static function
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        // 转换窗口坐标到 OpenGL 标准化设备坐标（NDC）
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        float ndcX = (xpos / width) * 2.0f - 1.0f;  // x: [-1, 1]
        float ndcY = 1.0f - (ypos / height) * 2.0f; // y: [-1, 1]（翻转 y 轴）
        std::cout << "Mouse Click absolute pos: (" << xpos << ", " << ypos << ")" << std::endl;
        // 打印 NDC 坐标
        std::cout << "Mouse Click at NDC: (" << ndcX << ", " << ndcY << ")" << std::endl;
        for(int i = 0; i < MyWindow::buttons.size(); ++i) { // 检测点击事件
            MyWindow::buttons[i]->handleClick({xpos, ypos});
        }
    }

}

/*
void MyWindow::mouse_button_callback(GLFWwindow* window, int button, int action, int mods) { // 不要这个了
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        double xpos, ypos;
        glfwGetCursorPos(MyWindow::get_window(), &xpos, &ypos);
        std::cout << "left click on window position: (" << xpos << ", " << ypos << ")" << std::endl;

        float ndcX, ndcY;
        // MyWindow::windowToNDC(xpos, ypos, ndcX, ndcY); 不要这个了
        std::cout << "transfer to NDC position: (" << ndcX << ", " << ndcY << ")" << std::endl;

        // // 判断是否在圆形内 (圆心在 NDC (0,0))
        // float distance = sqrtf(ndcX * ndcX + ndcY * ndcY);
        // if (distance <= circleRadius)
        // {
        //     std::cout << "===== click inside the circle =====" << std::endl;
        // }
        // else
        // {
        //     std::cout << "click outside the circle" << std::endl;
        // }
    }
}*/
GLFWwindow* MyWindow::initializeWindow(std::pair<int,int> size) {
    MyWindow::initializeGLFW();
    // 创建窗口
    GLFWwindow* window = glfwCreateWindow(size.first, size.second, "OpenGL Window", NULL, NULL);
    if (window == NULL) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return window;
    }
    glfwMakeContextCurrent(window);
    MyWindow::initializeGLEW(); //必须在创建窗口并设置上下文后调用 GLEW 初始化
    // 设置视口（窗口大小对应 OpenGL 坐标）
    glViewport(0, 0, size.first, size.second);
    return window;
}
void MyWindow::initShaders() {
    

    // for(int i = 0; i < MyWindow::buttons.size(); ++i) {
    //     MyWindow::buttons[i]->show();  //包含初始化共享的着色器程序，创建VAO/VBO, 创建顶点数据
    // }   //这个放到下面的draw函数里面了

    // --- 注册回调函数 ---  // 鼠标点击检测
    glfwSetMouseButtonCallback(MyWindow::window, &MyWindow::onMouseClick);
    glfwSetFramebufferSizeCallback(MyWindow::window, [](GLFWwindow* window, int width, int height) {
        glViewport(0, 0, width, height);
    });

    unsigned int shaderProgram;   // 共享的着色器程序
    //  渲染循环 （统一显示）
    while (!glfwWindowShouldClose(MyWindow::window)) {
        // 清空屏幕
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // 使用共享着色器
        glUseProgram(shaderProgram);
        for(int i = 0; i < MyWindow::buttons.size(); ++i)
            MyWindow::buttons[i]->draw(shaderProgram);  // 绘制各个按钮
            // // 绘制圆形1（红色）：通过uniform变量传递颜色
            // glUniform4f(glGetUniformLocation(shaderProgram, "circleColor"), 1.0f, 0.0f, 0.0f, 1.0f);
            // glBindVertexArray(circle1.VAO);
            // glDrawArrays(GL_TRIANGLE_FAN, 0, numSegments + 2);  // +2：圆心+闭合顶点

            // // 绘制圆形2（蓝色）：传递不同颜色
            // glUniform4f(glGetUniformLocation(shaderProgram, "circleColor"), 0.0f, 0.0f, 1.0f, 1.0f);
            // glBindVertexArray(circle2.VAO);
            // glDrawArrays(GL_TRIANGLE_FAN, 0, numSegments + 2);

        // 交换缓冲区和处理事件
        glfwSwapBuffers(MyWindow::window);
        glfwPollEvents();
    }

    // 清理资源（两个圆形的VAO/VBO + 着色器）
    for(int i = 0; i < MyWindow::buttons.size(); ++i)
        MyWindow::buttons[i]->deleteResources();

    glfwTerminate();

}
void MyWindow::createButtonVAO() {
    // for(int i = 0; i < MyWindow::buttons.size(); ++i) {
    //     // std::vector<float> vertices;
    //     // generateCircleVertices(circle.centerX, circle.centerY, circle.radius, vertices);
    //     MyWindow::buttons[i]->show();
    // // 创建VAO和VBO
    // glGenVertexArrays(1, &circle.VAO);
    // glGenBuffers(1, &circle.VBO);

    // glBindVertexArray(circle.VAO);
    // glBindBuffer(GL_ARRAY_BUFFER, circle.VBO);
    // glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    // // 配置顶点属性（x, y, z）
    // glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    // glEnableVertexAttribArray(0);

    // // 解绑
    // glBindBuffer(GL_ARRAY_BUFFER, 0);
    // glBindVertexArray(0);
    //}
}

void Button::mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    
    // // 只处理“按下”事件（忽略“释放”事件，如需检测释放可删除此判断）
    // if (action == GLFW_PRESS) {
    //     // 获取鼠标点击时的窗口坐标（原点在窗口左上角，x向右递增，y向下递增）
    //     double xpos, ypos;
    //     glfwGetCursorPos(this->window, &xpos, &ypos);
    //     // 打印点击信息（区分左键、右键、中键）
    //     if (button == GLFW_MOUSE_BUTTON_LEFT) {
    //         std::cout << "left click! position: (" << xpos << ", " << ypos << ")" << std::endl;
    //     } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
    //         std::cout << "right click! position: (" << xpos << ", " << ypos << ")" << std::endl;
    //     } else if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
    //         std::cout << "mid click! position: (" << xpos << ", " << ypos << ")" << std::endl;
    //     }
    // }
    // MyWindow::onMouseClick(window, button, action, mods);
    // glfwSetMouseButtonCallback(window, &MyWindow::onMouseClick);
            /* @brief 这个放到按钮里去检测更好 */


    // for(int i = 0; i < buttons.size(); ++i) { // 检测点击事件
    //     buttons[i]->handleClick({xpos, ypos});
    // }
}   /// MyWindow class
void Button::deleteResources() {
    glDeleteVertexArrays(1, &this->VAO);
    glDeleteBuffers(1, &this->VBO);
}
void Button::initShaders(unsigned int &shaderProgram) {
    
    // 编译顶点着色器
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &Button::vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    // 编译片段着色器
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &Button::fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    // 链接着色器程序
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // 获取 uniform 变量 "circleColor" 的位置 //////////////////////////////////////
    this->colorLoc = glGetUniformLocation(shaderProgram, "myButtonColor");
    if (this->colorLoc == -1) {
        std::cerr << "cannot find uniform variable myButtonColor" << std::endl;
    }

    // 删除中间着色器（已链接到程序，无需保留）
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

}
void Button::createVAOVBO() {
    // 创建VAO和VBO
    glGenVertexArrays(1, &(this->VAO));   //这两个是地址传递
    glGenBuffers(1, &(this->VBO)); 
    glBindVertexArray(this->VAO);  //这两个是值传递
    glBindBuffer(GL_ARRAY_BUFFER, this->VBO); 
}
void Button::activateTouch() {
    this->activate_touch = true;
    return;
}
void Button::disactivateTouch() {
    this->activate_touch = false;
}


/// CircleButton class
void CircleButton::windowToNDC(double xpos, double ypos, float& ndcX, float& ndcY) {
    int width, height;
    glfwGetWindowSize(this->get_window(), &width, &height);

    // 1. 将坐标原点从左上角移动到左下角
    ypos = height - ypos;

    // 2. 归一化到 [0.0, 1.0] 范围
    float xNorm = (float)xpos / (float)width;
    float yNorm = (float)ypos / (float)height;

    // 3. 转换到 [-1.0, 1.0] 范围
    ndcX = (xNorm * 2.0f) - 1.0f;
    ndcY = (yNorm * 2.0f) - 1.0f;
}    
void CircleButton::mouse_button_callback(GLFWwindow* window, int button, int action, int mods) { //弃用
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
    {
        double xpos, ypos;
        glfwGetCursorPos(this->get_window(), &xpos, &ypos);
        std::cout << "left click on window position: (" << xpos << ", " << ypos << ")" << std::endl;

        float ndcX, ndcY;
        this->windowToNDC(xpos, ypos, ndcX, ndcY);
        std::cout << "transfer to NDC position: (" << ndcX << ", " << ndcY << ")" << std::endl;

        // 判断是否在圆形内 (圆心在 NDC (0,0))
        float distance = sqrtf(ndcX * ndcX + ndcY * ndcY);
        if (distance <= this->NDCradius)  //这个要改！！！！
        {
            std::cout << "===== click inside the circle =====" << std::endl;
        }
        else
        {
            std::cout << "click outside the circle" << std::endl;
        }
    }
}
void CircleButton::generateVertices() {
    // 圆心
    float ct_x = NDCcenter.first; // NDCcenter.first * hw_rate;  不需要乘 hw_rate, 因为NDCcenter.first 本来就已经是 NDC 坐标了
    float ct_y = NDCcenter.second;
    circleVertices.push_back(ct_x);
    circleVertices.push_back(ct_y);
    circleVertices.push_back(0.0f);

    // 圆周上的点
    for (int i = 0; i <= CircleButton::numSegments; ++i) {
        float theta = 2.0f * (float)M_PI * (float)i / (float)CircleButton::numSegments;
        float x = NDCcenter.first  + (NDCradius * cosf(theta)) * hw_rate; //要乘以一个横向的缩放系数
        float y = NDCcenter.second + NDCradius * sinf(theta);

        circleVertices.push_back(x);
        circleVertices.push_back(y);
        circleVertices.push_back(0.0f);
    }
}
void CircleButton::handleClick(std::pair<double,double> pos) {
    if(!this->activate_touch)
        return;
    int x = pos.first, y = pos.second;
    std::cout << "CircleButton handleClick at position: (" << x << ", " << y << ")" << std::endl;
    // glfwGetWindowSize(this->get_window(), &width, &height);

    // 计算点击点到圆心的距离  ????????????????

    // long long distance = std::sqrt(std::pow(this->center.second - this->NDCcenter.first, 2) + std::pow(this->center.second - this->NDCcenter.second, 2));

    // 判断是否点击在圆形内

    if ( std::pow(x - center.first, 2) + std::pow(y - center.second, 2) <= std::pow(radius, 2) ) {
        std::cout << "click "<< this->name <<" button!"<< std::endl;
    } else {
        // std::cout << "click " << std::endl;
    }
}

void CircleButton::draw(unsigned int &shaderProgram) {

    this->initShaders(shaderProgram);  // 初始化着色器程序
    this->createVAOVBO();     // 创建VAO和VBO

    this->generateVertices();  // 先生成圆形的各各顶点

    glBufferData(GL_ARRAY_BUFFER, this->circleVertices.size() * sizeof(float), this->circleVertices.data(), GL_STATIC_DRAW);
    

    // 配置顶点属性（x, y, z）
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);    
    // 解绑
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);    
    // 绘制圆形
    glUseProgram(shaderProgram);

    // 传递颜色变量给 uniform（四个分量对应 r, g, b, a）   ////////////////////////
    glUniform4f(this->colorLoc, this->r, this->g, this->b, this->a);
    glBindVertexArray(this->VAO);


    glDrawArrays(GL_TRIANGLE_FAN, 0, CircleButton::numSegments + 2); // 使用 TRIANGLE_FAN 绘制
}
std::pair<double,double> CircleButton::get_center() {
    return this->center;
}

///  TriangleButton class
void TriangleButton::handleClick(std::pair<double,double> pos) {
    
}

void TriangleButton::draw(unsigned int &shaderProgram) {
    this->initShaders(shaderProgram);  // 初始化着色器程序
    this->createVAOVBO();     // 创建VAO和VBO
    this->generateVertices();  // 先生成三角形的各各顶点

    glBufferData(GL_ARRAY_BUFFER, this->triangleVertices.size() * sizeof(float), this->triangleVertices.data(), GL_STATIC_DRAW);

    // 配置顶点属性（x, y, z）
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);    
    
    // 解绑
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);    
    // 绘制三角形
    glUseProgram(shaderProgram);

    // 传递颜色变量给 uniform（四个分量对应 r, g, b, a）   ////////////////////////
    glUniform4f(this->colorLoc, this->r, this->g, this->b, this->a);
    glBindVertexArray(this->VAO);

    glDrawArrays(GL_TRIANGLE_FAN, 0, 3); //起始索引 0，共 3 个顶点
}
void TriangleButton::generateVertices() {

}
std::pair<double,double> TriangleButton::get_center() {
    std::cout << "triangle button does not have standarized center!"<< std::endl;
    return {0,0};
}


/// RectButton class
void RectButton::draw(unsigned int &shaderProgram) {
    this->initShaders(shaderProgram);  // 初始化着色器程序
    this->createVAOVBO();     // 创建VAO和VBO
    this->generateVertices();  // 先生成平行四边形的各各顶点

    glBufferData(GL_ARRAY_BUFFER, this->rectVertices.size() * sizeof(float), this->rectVertices.data(), GL_STATIC_DRAW);

    // 配置顶点属性（x, y, z）
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);    
    
    // 解绑
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);    
    // 绘制平行四边形
    glUseProgram(shaderProgram);

    // 传递颜色变量给 uniform（四个分量对应 r, g, b, a）   ////////////////////////
    glUniform4f(this->colorLoc, this->r, this->g, this->b, this->a);
    glBindVertexArray(this->VAO);

    glDrawArrays(GL_TRIANGLE_FAN, 0, 6); //起始索引 0，共 6 个顶点
    // glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0); //要用 EBO 才行
}
void RectButton::handleClick(std::pair<double,double> pos) {
    if(!this->activate_touch)
        return;
}
void RectButton::generateVertices() {

}
std::pair<double,double> RectButton::get_center() {
    return this->center;
}

void my_circle_button_test() {
    MyWindow myWin({800,600});
    auto window = MyWindow::get_window();
    std::vector<float> colorOfButton = {};  // 不能取名为color, 会和函数名冲突

    std::cout << "---- first circle button ----" << std::endl;
    CircleButton circleBtn(myWin,{600, 200}, 100.0, colorOfButton);
    // auto window = initializeWindow();
        // --- 3. 生成圆形顶点数据 ---
    // circleBtn.generateVertices();
    // circleBtn.show();
    std::cout << "---- second circle button ----" << std::endl;
    CircleButton circleBtn1(myWin, {300,200}, 150.0, colorOfButton);

    std::cout << "---- first triangle button ----" << std::endl;
    TriangleButton triangleBtn1(myWin, {600,400,700,500,500,500}, colorOfButton);

    std::cout << "---- first rectangle button ----"<< std::endl;
    RectButton rectBtn(myWin, colorOfButton, {100.0,500.0,300.0,500.0,350.0,550.0,150.0,550.0} );

    myWin.initShaders();
}


void draw_neural_network(std::vector<int>& layers) {

    // myWin.initShaders(); //显示窗口内部的神经网络图形
}



void nn_plot::initialize() {
    std::vector<int> layer_index(this->n, 0);  // [0,3,7,10]
    // 计算layer_index
    for(int i = 1; i < this->n; ++i) {
        layer_index[i] = layer_index[i-1] + layers[i-1];
    }
    
    // 计算每层神经元的最大数量（用于垂直居中）
    int maxNeurons = 0;
    int neuronNum = 0;
    for(int i = 0; i < layers.size(); ++i) {
        neuronNum += layers[i];
        if(layers[i] > maxNeurons)
            maxNeurons = layers[i];
    }
    std::vector<float> colorOfButton = {};  // 不能取名为color, 会和函数名冲突
    int width, height;
    glfwGetWindowSize(this->window, &width, &height);

    double layerSpacing = (double)width / (double)(layers.size() + 1); // 水平间距
    double neuronSpacing = (double)height / (double)(maxNeurons + 1);   // 垂直间距
    // std::vector<std::unique_ptr<CircleButton> > neuronBtnsVec;  // 动态数组存储神经元按钮

    for(int i = 0; i < layers.size(); ++i) {
        double xPos = layerSpacing * (i + 1); // 当前层的x位置
        int numNeurons = layers[i];
        CircleButton* tp;
        for(int j = 0; j  < numNeurons; ++j) {
            double yPos = neuronSpacing * (j + 1 + (maxNeurons - numNeurons) / 2.0f); // 当前神经元的y位置，垂直居中
            // neuronBtnsVec.push_back(std::make_unique<CircleButton>(myWin, {xPos, yPos}, (double)20.0, colorOfButton));
            this->neurons.emplace_back(tp = new CircleButton(*(this->mywin), {xPos, yPos}, (double)20.0, colorOfButton));
            tp->disactivateTouch();
            // new CircleButton(myWin, {xPos, yPos}, (double)20.0, colorOfButton);

        }
    }
    // 计算各个连接线对应的矩形 四个顶点的位置， 并生成对应矩形
    double width_2 = this->edge_width / 2;
    double costheta, sintheta;
    for(int i = 0; i < this->n-1; ++i) {
        for(int j = 0; j < layers[i]; ++j)
        for(int k = 0; k < layers[i+1]; ++k) {
            int index1 = layer_index[i] + j;
            int index2 = layer_index[i+1] + k; 
            auto circle1 = neurons[index1];
            auto circle2 = neurons[index2];
            std::pair<double,double> pos1 = circle1->get_center();
            std::pair<double,double> pos2 = circle2->get_center();
            auto dis = sqrtf(std::pow(pos1.first - pos2.first, 2) + std::pow(pos1.second - pos2.second, 2));
            auto dis1 =  sqrtf(std::pow(circle1->get_radius(), 2) - std::pow(width_2, 2));
            auto dis2 =  sqrtf(std::pow(circle2->get_radius(), 2) - std::pow(width_2, 2));
            double dy = pos2.second - pos1.second;
            double dx = pos2.first - pos1.first;
            double theta = atan2(dy, dx);
            costheta = cosf(theta); sintheta = sinf(theta);
            std::pair<double,double> pos2v = {pos2.first - dis2 * costheta, pos2.second - dis2 * sintheta};
            std::pair<double,double> pos1v = {pos1.first + dis1 * costheta, pos1.second + dis1 * sintheta};
            // 对应矩形的四个顶点
            std::vector<double> rectPos = {pos2v.first - width_2 * sintheta, pos2v.second + width_2 * costheta,
                                    pos2v.first + width_2 * sintheta, pos2v.second - width_2 * costheta, 
                                    pos1v.first + width_2 * sintheta, pos1v.second - width_2 * costheta, 
                                    pos1v.first - width_2 * sintheta, pos1v.second + width_2 * costheta};
            edge_pos.emplace_back(rectPos);
            edges.emplace_back(new RectButton(*(this->mywin),{0.0f, 1.0f, 0.0f, 1.0f}, edge_pos.back())); // 绿色矩形边
        }
    }
    return;
}

void nn_plot::initialize(std::vector<std::vector<float>>& layer_values, std::vector<std::vector<std::vector<float>>>& metrix_values) {
    std::vector<int> layer_index(this->n, 0);  // [0,3,7,10]
    // 计算layer_index
    for(int i = 1; i < this->n; ++i) {
        layer_index[i] = layer_index[i-1] + layers[i-1]; // 记录每一层第一个元素的索引
    }
    
    // 计算每层神经元的最大数量（用于垂直居中）
    int maxNeurons = 0;
    int neuronNum = 0;
    for(int i = 0; i < layers.size(); ++i) {
        neuronNum += layers[i];
        if(layers[i] > maxNeurons)
            maxNeurons = layers[i];
    }
    std::vector<float> colorOfButton = {0.2f, 0.5f, 0.7f, 1.0f};  // 不能取名为color, 会和函数名冲突
    int width, height;
    glfwGetWindowSize(this->window, &width, &height);

    double layerSpacing = (double)width / (double)(layers.size() + 1); // 水平间距
    double neuronSpacing = (double)height / (double)(maxNeurons + 1);   // 垂直间距
    // std::vector<std::unique_ptr<CircleButton> > neuronBtnsVec;  // 动态数组存储神经元按钮
    // 
    for(int i = 0; i < layers.size(); ++i) {
        double xPos = layerSpacing * (i + 1); // 当前层的x位置
        int numNeurons = layers[i];
        CircleButton* tp;
        for(int j = 0; j  < numNeurons; ++j) {
            double yPos = neuronSpacing * ( j + 1 + (maxNeurons - numNeurons) / 2.0f); // 当前神经元的y位置，垂直居中
            // neuronBtnsVec.push_back(std::make_unique<CircleButton>(myWin, {xPos, yPos}, (double)20.0, colorOfButton));
            // 读取输出的颜色值
            colorOfButton[3] = ( ((layer_values[i][j] > 1) ? 1 : layer_values[i][j]) + 0.5 ) / 1.5;
            this->neurons.emplace_back(tp = new CircleButton(*(this->mywin), {xPos, yPos}, (double)20.0, colorOfButton));
            tp->disactivateTouch();
            // new CircleButton(myWin, {xPos, yPos}, (double)20.0, colorOfButton);

        }
    }
    // 计算各个连接线对应的矩形 四个顶点的位置， 并生成对应矩形
    double width_2; // =  this->edge_width / 2;
    double costheta, sintheta;
    for(int i = 0; i < this->n-1; ++i) {
        for(int j = 0; j < layers[i]; ++j)
        for(int k = 0; k < layers[i+1]; ++k) {
            int index1 = layer_index[i] + j;
            int index2 = layer_index[i + 1] + k; 
            // 线的宽度
            width_2 =  (metrix_values[i][k][j] + 1 + 0.5) / 1.5 * this->edge_width / 2; // 矩阵大小为  layers[i+1] * layers[i]
            auto circle1 = neurons[index1];
            auto circle2 = neurons[index2];
            std::pair<double,double> pos1 = circle1->get_center();
            std::pair<double,double> pos2 = circle2->get_center();
            auto dis = sqrtf(std::pow(pos1.first - pos2.first, 2) + std::pow(pos1.second - pos2.second, 2));
            auto dis1 =  sqrtf(std::pow(circle1->get_radius(), 2) - std::pow(width_2, 2));
            auto dis2 =  sqrtf(std::pow(circle2->get_radius(), 2) - std::pow(width_2, 2));
            double dy = pos2.second - pos1.second;
            double dx = pos2.first - pos1.first;
            double theta = atan2(dy, dx);
            costheta = cosf(theta); sintheta = sinf(theta);
            std::pair<double,double> pos2v = {pos2.first - dis2 * costheta, pos2.second - dis2 * sintheta};
            std::pair<double,double> pos1v = {pos1.first + dis1 * costheta, pos1.second + dis1 * sintheta};
            // 对应矩形的四个顶点
            std::vector<double> rectPos = {pos2v.first - width_2 * sintheta, pos2v.second + width_2 * costheta,
                                    pos2v.first + width_2 * sintheta, pos2v.second - width_2 * costheta, 
                                    pos1v.first + width_2 * sintheta, pos1v.second - width_2 * costheta, 
                                    pos1v.first - width_2 * sintheta, pos1v.second + width_2 * costheta};
            edge_pos.emplace_back(rectPos);
            edges.emplace_back(new RectButton(*(this->mywin),{0.0f, 1.0f, 0.0f, 1.0f}, edge_pos.back())); // 绿色矩形边
        }
    }
    return;
}
void my_nn_draw_test() {
    /*
    int n,t;
    std::cout<< "enter n:"<<std::endl;
    std::cin>>n;
    std::cout<< "enter the neuron nums of each layer:\n";
    std::vector<int> layers; // 示例：3层神经网络，分别有3、5、2个神经元
    for(int i = 0; i < n; ++i) {
        std::cin >> t;
        layers.push_back(t);
    }*/
    std::vector<int> layers = {3, 4, 4, 3};
    // draw_neural_network(layers);
    MyWindow myWin({800,600});
    std::vector<float> colorOfButton = {};  // 不能取名为color, 会和函数名冲突
    nn_plot mynn(myWin, layers);
    mynn.initialize();
    myWin.initShaders();
}