#pragma once
#include <GL/glew.h>   // 先include glew 因为其中重定义了glad当中的内容，统一按照glew的要求
// #include <glad/glad.h> 不用glad，因为glew已经包含了
// 定义GLFW_INCLUDE_NONE，阻止GLFW包含<GL/gl.h>
#define GLFW_INCLUDE_NONE
// 再包含GLFW头文件
#include <GLFW/glfw3.h>
#include <exception>
#include <cassert>
#include <string>
#include <vector>
#include <iostream>
#include <cmath>
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void mouse_button_callback_circle(GLFWwindow* window, int button, int action, int mods);
// void windowToNDC(GLFWwindow* window, double xpos, double ypos, float& ndcX, float& ndcY);
// void initializeGLFW();
// void initializeGLEW(); 
// GLFWwindow* initializeWindow();

void my_circle_button_test();
void my_button_test();
void my_nn_draw_test();
void draw_neural_network(std::vector<int>& layers);
class Button;
class nn_plot;
class MyWindow{
private:
    friend class Button;
    static GLFWwindow* window;
    static std::pair<int,int> size;
    static std::vector<Button*> buttons;
    inline void addButton(Button* btn) {
        buttons.push_back(btn);
    }
    
public: 
    // 静态成员函数：GLFW 的直接回调（无 this 指针，参数匹配 GLFW 要求）
    static void staticMouseCallback(GLFWwindow* window, int button, int action, int mods);
    static GLFWwindow* initializeWindow(std::pair<int,int> size);
    static void initializeGLFW();
    static void initializeGLEW();
    static inline GLFWwindow* get_window() { return window; }
    static void initShaders();
    static void createButtonVAO();
    // static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);

    MyWindow(std::pair<double,double> size) {
        MyWindow::size.first = size.first;
        MyWindow::size.second = size.second;
        window = initializeWindow(size);
    }
    static void onMouseClick(GLFWwindow* window, int button, int action, int mods);
};

class Button{
public:
    friend class MyWindow;
    friend void draw_neural_network(std::vector<int>& layers);
    friend class nn_plot;
    Button(MyWindow& win, std::string name, std::vector<float> color, bool detect = true) : window(win.get_window()), name(name),mywin(&win) {   
        win.addButton(this);     // 直接为窗口注册按钮
        numButtons++;
        if(color.size() == 4) {
            r = color[0];
            g = color[1];
            b = color[2];
            a = color[3];
        }
        else{
            r = 0.2f; g = 0.5f; b = 0.7f; a = 1.0f;
            std::cout<<"blue button"<<std::endl;
        }
        if(detect) {
            // // 将类实例指针绑定到窗口用户数据（关键步骤）
            // glfwSetWindowUserPointer(win, this);
            // // 注册静态回调函数到 GLFW
            // glfwSetMouseButtonCallback(win, MyWindow::staticMouseCallback);
        }

    }
    // virtual void show(unsigned int &shaderProgram) = 0; //不需要了，功能包含在draw里面
    
protected:
    static int numButtons; 
    inline GLFWwindow* get_window() { return window; }
    MyWindow* mywin;
    std::string name;
    GLFWwindow* window;
    GLuint VAO,VBO;
    bool activate_touch = true;
    float r = 0.2f, g = 0.5f, b = 0.7f, a = 1.0f;// 颜色值
    // unsigned int VAO, VBO;
    int colorLoc;// 片段着色器中颜色变量的位置
    void deleteResources();
    void disactivateTouch();
    void activateTouch();
    static constexpr const char* fragmentShaderSource =   // 可变颜色
        "#version 330 core\n"
        "out vec4 FragColor;\n"
        "uniform vec4 myButtonColor; // 外部传入的颜色变量（r, g, b, a）\n"
        "void main()\n"
        "{\n"
        "   FragColor = myButtonColor; // 使用 uniform 变量作为颜色\n"
        "}\0";\
// --- 着色器代码 ---
    static constexpr const char* vertexShaderSource = 
        "#version 330 core\n"
        "layout (location = 0) in vec3 aPos;\n"
        "void main()\n"
        "{\n"
        "   gl_Position = vec4(aPos, 1.0);\n"
        "}\0";    
    void initShaders(unsigned int &shaderProgram);
    void createVAOVBO();
private:
    virtual void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    virtual void handleClick(std::pair<double,double> pos) = 0; 
    virtual std::pair<double,double> get_center() = 0;
    virtual void draw(unsigned int &shaderProgram) = 0;
    

    // virtual void initShaders() = 0; 可以用show代替
    virtual void generateVertices() = 0;
};

class CircleButton : public Button {
    friend class nn_plot;
public:
    CircleButton(MyWindow& win, std::pair<double,double> pos,double radius,std::vector<float> color, bool detect = true)
        : Button(win,(std::string)"Circle", color, detect), center({pos.first,pos.second}), radius(radius) {
            // 计算 NDC 坐标
            int width, height;
            glfwGetWindowSize(this->get_window(), &width, &height);

            CircleButton::numCircleButtons++;

            NDCcenter.first = ( pos.first -  width / 2 ) / width  * 2.0f;
            NDCcenter.second =  (height - pos.second - height/2) / height *2.f;  // 1.0f - (pos.second / height) * 2.0f;

            std::cout << "Window size: (" << width << ", " << height << ")" << std::endl;
            std::cout << "NDC center: (" << NDCcenter.first << ", " << NDCcenter.second << ")" << std::endl;// (600,200) 对应 (0.5,0.33)

            hw_rate = (double)height / (double)width;
            // std::cout << "height/width rate: " << hw_rate << std::endl;
            NDCradius = radius / height * 2.0f; // 根据高度计算 NDC 半径
        }

    CircleButton(const CircleButton& other) : Button(*(other.mywin), (std::string)"Circle", {other.r,other.g,other.b,other.a}) {
            
            // 计算 NDC 坐标
            int width, height;
            glfwGetWindowSize(this->get_window(), &width, &height);

            CircleButton::numCircleButtons++;

            NDCcenter.first = other.NDCcenter.first;
            NDCcenter.second =  other.NDCcenter.second;

            std::cout << "Window size: (" << width << ", " << height << ")" << std::endl;
            std::cout << "NDC center: (" << NDCcenter.first << ", " << NDCcenter.second << ")" << std::endl;// (600,200) 对应 (0.5,0.33)

            hw_rate = (double)height / (double)width;
            // std::cout << "height/width rate: " << hw_rate << std::endl;
            NDCradius = other.NDCradius;
    }
        // --- 圆形参数 ---

    static constexpr const int numSegments = 100;          // 用于逼近圆形的三角形数量
    std::vector<float> circleVertices;    // 存储圆形顶点数据的向量
    inline double get_radius() {return this->radius;} //返回圆形的半径
private:
    static int numCircleButtons;    // 记录圆形按钮的数量
    std::pair<double,double> center;
    double radius;
    std::pair<double,double> NDCcenter;
    double NDCradius; // 圆形半径 (NDC 坐标)
    double hw_rate;
    friend void my_circle_button_test();
    void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) override;
    // 将窗口坐标 (x, y) 转换为标准化设备坐标 (NDC)
    void windowToNDC(double xpos, double ypos, float& ndcX, float& ndcY);
    // 鼠标点击回调
    void generateVertices() override;
    void handleClick(std::pair<double,double> pos) override;
    void draw(unsigned int &shaderProgram) override;
    std::pair<double,double> get_center() override;

};
class TriangleButton : public Button {
    friend class nn_plot;
public:
    TriangleButton(MyWindow& win, std::vector<double> pos, std::vector<float> color, bool detect = true) : Button(win,(std::string)"Triangle", color, detect) {
        assert(pos.size() ==6);
        TriangleButton::numTriangleButtons++;

        // 计算 NDC 坐标
        int width, height;
        glfwGetWindowSize(this->get_window(), &width, &height);
        for(int i = 0; i < 3; ++i) {
            triangleVertices.emplace_back(( pos[i*2] -  width / 2 ) / width  * 2.0f);
            triangleVertices.emplace_back((height - pos[i*2+1] - height/2) / height *2.f);  // 1.0f - (pos.second / height) * 2.0f;
            triangleVertices.emplace_back(0.0f);
            // hw_rate = (double) height / (double)width;
        }
    }
private:
    static int numTriangleButtons;    // 记录圆形按钮的数量
    void handleClick(std::pair<double,double> pos) override;
    void generateVertices() override;
    void draw(unsigned int &shaderProgram) override;
    std::vector<float> triangleVertices;  // 存储三角形顶点数据的向量
    std::pair<double,double> get_center() override;
};

class RectButton : public Button { //平行四边形
    friend class nn_plot;
public:
    RectButton(MyWindow& win,std::vector<float> color, std::vector<double> pos, bool detect = true) : Button(win,(std::string)"Rectangle", color, detect) {
        assert(pos.size() ==8);
        RectButton::numRectButtons++;

        // 计算 NDC 坐标
        int width, height;
        glfwGetWindowSize(this->get_window(), &width, &height);
        center.first = (pos[0]+pos[4])/2.0 *1.0f;
        center.second = (pos[1]+pos[5])/2.0*1.0f;
        NDCcenter = {( center.first -  width / 2 ) / width  * 2.0f,
                     (height - center.second - height/2) / height *2.f}; 

        // rectVertices.push_back(NDCcenter.first);
        // rectVertices.push_back(NDCcenter.second);
        // rectVertices.push_back(0.0f);
        size_t indices[6] = {0,1,2,0,2,3};
        for(int i = 0; i < 6; ++i) {
            rectVertices.emplace_back(( pos[indices[i]*2] -  width / 2 ) / width  * 2.0f);
            rectVertices.emplace_back((height - pos[indices[i]*2+1] - height/2) / height *2.f);  // 1.0f - (pos.second / height) * 2.0f;
            rectVertices.emplace_back(0.0f);
            // hw_rate = (double) height / (double)width;
        }
        // 计算长和宽（shape）

    }


private:
    std::vector<float> rectVertices;    // 存储矩形顶点数据的向量
    
    std::pair<double, double> shape;
    std::pair<double, double> center;
    std::pair<double, double> NDCcenter;
    // double hw_rate;

    static int numRectButtons;    // 记录矩形按钮的数量
    void handleClick(std::pair<double, double> pos) override;
    void draw(unsigned int &shaderProgram) override;
    void generateVertices() override;
    std::pair<double, double> get_center() override;
};

class nn_plot{
private:
    int n;
    GLFWwindow* window;
    std::vector<int> layers;   // [3,4,4,3]
    std::vector<CircleButton*> neurons;
    std::vector<RectButton*> edges;         
    std::vector<std::vector<double>> edge_pos;
    MyWindow* mywin;
    double edge_width = 3; 
    void initialize(std::vector<std::vector<float>>& layer_values, std::vector<std::vector<std::vector<float>>>& metrix_values);
public:
    void initialize();
    nn_plot(MyWindow& win, std::vector<int> layers) : layers(layers), n(layers.size()), window(MyWindow::get_window()),mywin(&win) {
    }

    nn_plot(MyWindow& win, std::vector<std::vector<float>>& layer_values, std::vector<std::vector<std::vector<float>>>& metrix_values) 
        :  n(layer_values.size()), window(MyWindow::get_window()), mywin(&win) {
            for(int i = 0; i < layer_values.size(); ++i) {
                this->layers.push_back(layer_values[i].size());
                //this->n += layer_values[i].size();
            }

            
            this->initialize(layer_values, metrix_values);
        } 
    
};