#ifndef SHADER_H
#define SHADER_H

#include <glad/glad.h>  // 需先包含glad（OpenGL函数加载器）
#include <string>

// 着色器程序类：编译、链接顶点/片段着色器
class Shader {
public:
    unsigned int ID;  // 着色器程序ID

    // 构造函数：从字符串加载并编译着色器
    Shader(const char* vertexSource, const char* fragmentSource);
    // 使用/激活着色器程序
    void use();
    // 设置 uniforms（着色器中的全局变量）
    void setBool(const std::string& name, bool value) const;
    void setInt(const std::string& name, int value) const;
    void setFloat(const std::string& name, float value) const;
};

#endif