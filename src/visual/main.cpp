// 第一步：包含glad.h（必须放在最前面，glad需要先定义OpenGL函数）
#include <GL/glew.h>   // 先include glew 因为其中重定义了glad当中的内容，统一按照glew的要求
// #include <glad/glad.h> 不用glad，因为glew已经包含了
// 第二步：定义GLFW_INCLUDE_NONE，阻止GLFW包含<GL/gl.h>
#define GLFW_INCLUDE_NONE
// 再包含GLFW头文件
#include <GLFW/glfw3.h>
#include"buttons.h"
#include <iostream>
int main() {
    std::cout << "my openGL test 1" << std::endl;
    /*
    // 初始化 GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // 创建一个窗口和 OpenGL 上下文
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "GLFW and GLEW Test", NULL, NULL);
    if (window == NULL) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // 初始化 GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    // 设置窗口关闭回调
    glfwSetWindowCloseCallback(window, [](GLFWwindow* window) {
        glfwDestroyWindow(window);
        glfwTerminate();
        exit(0);
        });

    // 设置窗口大小改变回调
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* window, int width, int height) {
        glViewport(0, 0, width, height);
        });

    // 循环直到用户关闭窗口
    while (!glfwWindowShouldClose(window)) {
        // 渲染逻辑
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // 交换缓冲区和轮询事件
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // 退出清理
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
    */
    initializeGLEW();
    initializeGLFW();
    // 创建窗口
    GLFWwindow* window = glfwCreateWindow(800, 600, "OpenGL 绘制三角形", NULL, NULL);
    if (window == NULL) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);
    // 设置视口（窗口大小对应 OpenGL 坐标）
    glViewport(0, 0, 800, 600);

    // --------------------------
    // 1. 编译并链接着色器程序
    // --------------------------
    // 顶点着色器
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    // 片段着色器
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    // 链接成着色器程序
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    // // 检查顶点着色器编译错误
    // int success;
    // char infoLog[512];
    // glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    // if (!success) {
    //     glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
    //     std::cerr << "顶点着色器编译失败:\n" << infoLog << std::endl;
    // }

    // // 片段着色器
    // unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    // glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    // glCompileShader(fragmentShader);
    // // 检查片段着色器编译错误
    // glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    // if (!success) {
    //     glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
    //     std::cerr << "片段着色器编译失败:\n" << infoLog << std::endl;
    // }

    // // 链接成着色器程序
    // unsigned int shaderProgram = glCreateProgram();
    // glAttachShader(shaderProgram, vertexShader);
    // glAttachShader(shaderProgram, fragmentShader);
    // glLinkProgram(shaderProgram);
    // // 检查链接错误
    // glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    // if (!success) {
    //     glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
    //     std::cerr << "着色器程序链接失败:\n" << infoLog << std::endl;
    // }
    // // 链接后可删除着色器（已被程序包含）
    // glDeleteShader(vertexShader);
    // glDeleteShader(fragmentShader);

    // --------------------------
    // 2. 定义顶点数据（位置 + 颜色）
    // --------------------------
    float vertices[] = {
        // 位置              // 颜色
         0.5f, -0.5f, 0.0f,  1.0f, 0.0f, 0.0f,  // 右下：红色
        -0.5f, -0.5f, 0.0f,  0.0f, 1.0f, 0.0f,  // 左下：绿色
         0.0f,  0.5f, 0.0f,  0.0f, 0.0f, 1.0f   // 顶部：蓝色
    };

    // --------------------------
    // 3. 创建 VAO（顶点数组对象）和 VBO（顶点缓冲对象）
    // --------------------------
    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    // 绑定 VAO（核心模式必须绑定 VAO 才能绘制）
    glBindVertexArray(VAO);

    // 绑定 VBO 并传入顶点数据
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // 配置顶点属性（位置：location=0）
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0); // 启用位置属性

    // 配置顶点属性（颜色：location=1）
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1); // 启用颜色属性

    // 解绑缓冲（可选，避免后续误操作）
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

        // --------------------------
    // 关键：注册鼠标按钮回调函数
    // 当鼠标按钮被点击时，GLFW 会自动调用 mouse_button_callback
    // --------------------------
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    // 窗口关闭回调
    glfwSetWindowCloseCallback(window, [](GLFWwindow* window) {
        glfwDestroyWindow(window);
        glfwTerminate();
        exit(0);
    });

    // --------------------------
    // 窗口回调函数
    // --------------------------
    // 窗口关闭回调
    glfwSetWindowCloseCallback(window, [](GLFWwindow* window) {
        glfwDestroyWindow(window);
        glfwTerminate();
        exit(0);
    });

    // 窗口大小改变回调（调整视口）
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* window, int width, int height) {
        glViewport(0, 0, width, height);
    });

    // --------------------------
    // 渲染循环（持续绘制）
    // --------------------------
    while (!glfwWindowShouldClose(window)) {
        // 1. 清空屏幕（黑色背景）
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f); // 深灰色背景
        glClear(GL_COLOR_BUFFER_BIT);

        // 2. 绘制三角形
        glUseProgram(shaderProgram); // 激活着色器程序
        glBindVertexArray(VAO);      // 绑定 VAO（包含顶点数据配置）
        glDrawArrays(GL_TRIANGLES, 0, 3); // 绘制三角形（3个顶点）

        // 3. 交换缓冲区 + 处理事件
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // 清理资源
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}