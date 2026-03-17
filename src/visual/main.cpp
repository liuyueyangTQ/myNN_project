// #include <glad/glad.h>  // 必须在GLFW之前包含（加载OpenGL函数）
// #include <GLFW/glfw3.h>

// #include <iostream>


// int main() {
//     // -------------------------- 初始化GLFW --------------------------
//     glfwInit();
//     // 设置OpenGL版本（3.3核心模式）
//     glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
//     glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
//     glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 核心模式（现代OpenGL）
// #ifdef __APPLE__
//     glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);  // macOS特殊设置
// #endif

//     // -------------------------- 创建窗口 --------------------------
//     GLFWwindow* window = glfwCreateWindow(800, 600, "GLFW + OpenGL 示例", NULL, NULL);
//     if (window == NULL) {
//         std::cerr << "Failed to create GLFW window" << std::endl;
//         glfwTerminate();
//         return -1;
//     }
//     glfwMakeContextCurrent(window);  // 将窗口的OpenGL上下文设为当前线程的主上下文
//     glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);  // 注册窗口大小变化回调

//     // -------------------------- 初始化GLAD（加载OpenGL函数） --------------------------
//     if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
//         std::cerr << "Failed to initialize GLAD" << std::endl;
//         return -1;
//     }

//     // -------------------------- 定义三角形的顶点数据（位置+颜色） --------------------------
//     float vertices[] = {
//         // 位置              // 颜色
//          0.5f, -0.5f, 0.0f,  1.0f, 0.0f, 0.0f,  // 右下
//         -0.5f, -0.5f, 0.0f,  0.0f, 1.0f, 0.0f,  // 左下
//          0.0f,  0.5f, 0.0f,  0.0f, 0.0f, 1.0f   // 顶部
//     };

//     // -------------------------- 创建顶点缓冲对象（VBO）和顶点数组对象（VAO） --------------------------
//     unsigned int VBO, VAO;
//     glGenVertexArrays(1, &VAO);
//     glGenBuffers(1, &VBO);

//     // 绑定VAO（后续操作会记录到VAO）
//     glBindVertexArray(VAO);

//     // 绑定VBO并复制顶点数据到GPU
//     glBindBuffer(GL_ARRAY_BUFFER, VBO);
//     glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

//     // 配置顶点属性（位置：3个float，步长6个float，偏移0）
//     glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
//     glEnableVertexAttribArray(0);  // 启用位置属性（索引0）

//     // 配置顶点属性（颜色：3个float，步长6个float，偏移3个float）
//     glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
//     glEnableVertexAttribArray(1);  // 启用颜色属性（索引1）

//     // 解绑VBO和VAO（可选，避免后续误操作）
//     glBindBuffer(GL_ARRAY_BUFFER, 0);
//     glBindVertexArray(0);

//     // -------------------------- 定义着色器代码（顶点着色器+片段着色器） --------------------------
//     const char* vertexShaderSource = "#version 330 core\n"
//         "layout (location = 0) in vec3 aPos;\n"  // 位置属性（对应VAO索引0）
//         "layout (location = 1) in vec3 aColor;\n" // 颜色属性（对应VAO索引1）
//         "out vec3 ourColor;\n"  // 输出颜色到片段着色器
//         "void main()\n"
//         "{\n"
//         "   gl_Position = vec4(aPos, 1.0);\n"  // 顶点位置（齐次坐标）
//         "   ourColor = aColor;\n"  // 传递颜色
//         "}\0";

//     const char* fragmentShaderSource = "#version 330 core\n"
//         "in vec3 ourColor;\n"  // 从顶点着色器接收颜色
//         "out vec4 FragColor;\n"  // 输出最终像素颜色
//         "void main()\n"
//         "{\n"
//         "   FragColor = vec4(ourColor, 1.0);\n"  // 颜色+透明度
//         "}\0";

//     // 创建着色器程序
//     Shader ourShader(vertexShaderSource, fragmentShaderSource);
//     if (ourShader.ID == 0) {
//         return -1;  // 着色器创建失败
//     }

//     // -------------------------- 渲染循环 --------------------------
//     while (!glfwWindowShouldClose(window)) {
//         // 处理输入
//         processInput(window);

//         // 渲染指令
//         glClearColor(0.2f, 0.3f, 0.3f, 1.0f);  // 清空屏幕的颜色（深灰）
//         glClear(GL_COLOR_BUFFER_BIT);  // 清空颜色缓冲

//         // 使用着色器程序
//         ourShader.use();

//         // 绑定VAO并绘制三角形（3个顶点）
//         glBindVertexArray(VAO);
//         glDrawArrays(GL_TRIANGLES, 0, 3);

//         // 交换缓冲并轮询事件
//         glfwSwapBuffers(window);  // 交换前后缓冲（避免闪烁）
//         glfwPollEvents();  // 处理鼠标/键盘事件
//     }

//     // -------------------------- 清理资源 --------------------------
//     glDeleteVertexArrays(1, &VAO);
//     glDeleteBuffers(1, &VBO);
//     glfwTerminate();  // 终止GLFW
//     return 0;
// }

// 正确的包含顺序：先定义宏，再包含GLFW，最后包含GLAD
// #define GLFW_INCLUDE_NONE  // 关键：阻止GLFW包含gl.h

// #include <GLFW/glfw3.h>
// #include <iostream>
// #include <cmath>
// #include "glad/glad.h"

// // 窗口尺寸
// const unsigned int SCR_WIDTH = 800;
// const unsigned int SCR_HEIGHT = 600;

// // 顶点着色器
// const char *vertexShaderSource = "#version 330 core\n"
//     "layout (location = 0) in vec3 aPos;\n"
//     "void main()\n"
//     "{\n"
//     "   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
//     "}\0";

// // 片段着色器
// const char *fragmentShaderSource = "#version 330 core\n"
//     "out vec4 FragColor;\n"
//     "uniform float time;\n"
//     "void main()\n"
//     "{\n"
//     "   FragColor = vec4(sin(time) * 0.5 + 0.5, cos(time * 0.7) * 0.5 + 0.5, sin(time * 1.3) * 0.5 + 0.5, 1.0);\n"
//     "}\0";

// // 处理窗口大小变化
// void framebuffer_size_callback(GLFWwindow* window, int width, int height)
// {
//     glViewport(0, 0, width, height);
// }

// // 处理输入
// void processInput(GLFWwindow *window)
// {
//     if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
//         glfwSetWindowShouldClose(window, true);
// }

// // 编译着色器
// unsigned int compileShader(unsigned int type, const char* source)
// {
//     unsigned int shader = glCreateShader(type);
//     glShaderSource(shader, 1, &source, NULL);
//     glCompileShader(shader);
    
//     // 检查编译错误
//     int success;
//     char infoLog[512];
//     glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
//     if(!success)
//     {
//         glGetShaderInfoLog(shader, 512, NULL, infoLog);
//         std::cout << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
//     }
//     return shader;
// }

// // 创建着色器程序
// unsigned int createShaderProgram()
// {
//     unsigned int vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
//     unsigned int fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    
//     unsigned int shaderProgram = glCreateProgram();
//     glAttachShader(shaderProgram, vertexShader);
//     glAttachShader(shaderProgram, fragmentShader);
//     glLinkProgram(shaderProgram);
    
//     // 检查链接错误
//     int success;
//     char infoLog[512];
//     glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
//     if(!success) {
//         glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
//         std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
//     }
    
//     glDeleteShader(vertexShader);
//     glDeleteShader(fragmentShader);
    
//     return shaderProgram;
// }

// int main()
// {

//         // 新增：启动日志（写入文件，确认 main 是否执行）
//     FILE* log_file = fopen("launch_log.txt", "w");
//     if (log_file) {
//         fprintf(log_file, "main function started!\n");
//         fclose(log_file);
//     }

//     std::cout << "program start..." << std::endl;
//     // 初始化GLFW
//     if (!glfwInit())
//     {
//         std::cout << "Failed to initialize GLFW" << std::endl;
//         return -1;
//     }
    
//     // 配置GLFW
//     glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
//     glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
//     glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
//     glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // 新增这行
// // 补充说明
// // •	GLFW_OPENGL_FORWARD_COMPAT：告诉 GLFW 创建 “向前兼容” 的 OpenGL 上下文，不支持已废弃的旧功能（核心模式的要求）；
// // •	这行配置在 macOS 上是强制的，Windows 上虽非官方强制，但多数显卡驱动（如 NVIDIA、AMD）需要它才能正常启用核心模式渲染。

//     // 创建窗口
//     GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "OpenGL Test - Rotating Triangle", NULL, NULL);
//     if (window == NULL)
//     {
//         std::cout << "Failed to create GLFW window" << std::endl;
//         glfwTerminate();
//         return -1;
//     }
//     glfwMakeContextCurrent(window);
//     glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    
//     // 初始化GLAD
//     if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
//     {
//         std::cout << "Failed to initialize GLAD" << std::endl;
//         return -1;
//     }
//     std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl; // 新增：验证版本
//     // 创建着色器程序
//     unsigned int shaderProgram = createShaderProgram();
    
//     // 设置顶点数据
//     float vertices[] = {
//         // 位置
//         -0.5f, -0.5f, 0.0f,  // 左下
//          0.5f, -0.5f, 0.0f,  // 右下
//          0.0f,  0.5f, 0.0f   // 顶部
//     };
    
//     unsigned int VBO, VAO;
//     glGenVertexArrays(1, &VAO);
//     glGenBuffers(1, &VBO);
    
//     glBindVertexArray(VAO);
//     glBindBuffer(GL_ARRAY_BUFFER, VBO);
//     glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
//     glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
//     glEnableVertexAttribArray(0);
    
//     glBindBuffer(GL_ARRAY_BUFFER, 0);
//     glBindVertexArray(0);
    
//     std::cout << "OpenGL Test Started Successfully!" << std::endl;
//     std::cout << "Press ESC to exit" << std::endl;
    
//     // 渲染循环
//     while (!glfwWindowShouldClose(window))
//     {
//         // 输入处理
//         processInput(window);
        
//         // 渲染
//         glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
//         glClear(GL_COLOR_BUFFER_BIT);
        
//         // 使用着色器程序
//         glUseProgram(shaderProgram);
        
//         // 设置时间uniform
//         float timeValue = glfwGetTime();
//         int timeLocation = glGetUniformLocation(shaderProgram, "time");
//         glUniform1f(timeLocation, timeValue);
        
//         // 绘制三角形
//         glBindVertexArray(VAO);
//         glDrawArrays(GL_TRIANGLES, 0, 3);
        
//         // 交换缓冲区和轮询事件
//         glfwSwapBuffers(window);
//         glfwPollEvents();
//     }
    
//     // 清理资源
//     glDeleteVertexArrays(1, &VAO);
//     glDeleteBuffers(1, &VBO);
//     glDeleteProgram(shaderProgram);
    
//     glfwTerminate();
//     return 0;
// }

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