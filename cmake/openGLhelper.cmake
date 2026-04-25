# cmake/modules/OpenGLHelper.cmake

# 封装OpenGL+GLFW+GLAD的构建逻辑，参数：TARGET_NAME（目标名，如可执行文件）
function(add_opengl_support TARGET_NAME)
    message(STATUS "当前目标名：${TARGET_NAME}")
    # 对于MinGW，需要设置正确的编译器标志
    if(MINGW)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
    endif()

    set(GLFW_INSTALL_DIR "C:/Tools/glfw-3.4")
    list(APPEND CMAKE_PREFIX_PATH "${GLFW_INSTALL_DIR}")
    # 查找必要的包
    find_package(glfw3 3.4 REQUIRED)

    # 打印查找结果
    message(STATUS "=== OpenGL 库信息 ===")
    message(STATUS "是否找到 OpenGL: ${OPENGL_FOUND}")
    message(STATUS "头文件目录: ${OPENGL_INCLUDE_DIR}")
    message(STATUS "所有链接库: ${OPENGL_LIBRARIES}")
    message(STATUS "核心库路径: ${OPENGL_gl_LIBRARY}")
    message(STATUS "GLU 库路径: ${OPENGL_glu_LIBRARY}")
    # 设置GLFW路径（根据你的实际安装路径修改）
    # set(GLFW_ROOT "C:/Tools/glfw-3.4" CACHE PATH "C:/Tools/glfw-3.4/build_glfw_install")

    get_filename_component(PARENT_SOURCE_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)
    get_filename_component(PARENT_SOURCE_DIR ${PARENT_SOURCE_DIR} DIRECTORY)


    target_include_directories(${TARGET_NAME} PRIVATE "C:/Tools/glfw-3.4/include")
    target_include_directories(${TARGET_NAME} PRIVATE "C:/Tools/glew-2.2.0/include")
    target_include_directories(${TARGET_NAME} PRIVATE ${PARENT_SOURCE_DIR}/inc)
    target_include_directories(${TARGET_NAME} PRIVATE
        ${GLFW_INSTALL_DIR}/src/visual
        ${GLFW_INSTALL_DIR}/test
        ${GLFW_INSTALL_DIR}/inc  
        ${PARENT_SOURCE_DIR}/inc   # glad所在目录  
    )

    # 3. 编译器差异化处理 (关键点)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # 如果是 Clang，强制它遵循 MinGW 行为 (解决 -l 找不到库的问题)
        add_compile_options("--target=x86_64-w64-windows-gnu")
        add_link_options("--target=x86_64-w64-windows-gnu")
        
        # Clang 链接时通常不需要手动指定 .dll 名字，而是寻找 .lib 或 .dll.a 导入库
        set(GLEW_LIB_NAME "C:/Tools/glew-2.2.0/lib/Release/x64/glew32.lib")
        set(GLFW_LIB_NAME "C:/Tools/glfw-3.4/lib/glfw3dll.lib")
    else()
        # 如果是 GCC (MinGW)
        set(GLEW_LIB_NAME "glew32.dll")
        set(GLFW_LIB_NAME "glfw3.dll")
    endif()

    target_link_directories(${TARGET_NAME} PRIVATE "C:/Tools/glew-2.2.0/bin/Release/x64")  # 先指定链接的动态库路径 ！！！！
    target_link_directories(${TARGET_NAME} PRIVATE "C:/Tools/glfw-3.4/bin")

    # target_link_libraries(${TARGET_NAME} PRIVATE glew32.dll) # 再指定链接的动态库名字 ！！！！ 必须包含

    # target_link_libraries(${TARGET_NAME} PRIVATE glfw3.dll)



    target_link_libraries(${TARGET_NAME} PRIVATE
        glfw # 链接 GLFW 库本身（提供窗口创建、事件处理等核心功能）。 包含一下（不包含问题也不大）
        opengl32 # Windows 系统 OpenGL 库, 若不包含会缺少必要的系统库链接   !!!!!!!!  
        ${GLEW_LIB_NAME}
        ${GLFW_LIB_NAME}    
        ws2_32                # Windows 网络库（GLFW 依赖）   这些不需要！！
        mswsock               # Windows Winsock 扩展库（GLFW 依赖）
    )


    # 4.2 添加构建后事件：复制 DLL 到 exe 所在目录（CMAKE_CURRENT_BINARY_DIR 即 build 目录）
    add_custom_command(
        TARGET ${TARGET_NAME}  # 关联到目标可执行文件
        POST_BUILD           # 在目标构建完成后执行
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "C:/Tools/glew-2.2.0/bin/Release/x64/glew32.dll"  # 源文件：glew32.dll
            "${CMAKE_CURRENT_BINARY_DIR}/"  # 目标目录：exe 所在目录
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "C:/Tools/glfw-3.4/bin/glfw3.dll"  # 源文件：glfw3.dll
            "${CMAKE_CURRENT_BINARY_DIR}/"  # 目标目录：exe 所在目录
        COMMENT "copying libglew32.dll to executable fold"  # 构建时显示的提示信息
    )
endfunction()
# cmake -G "MinGW Makefiles" -DCMAKE_C_COMPILER="C:/Program Files/LLVM/bin/clang.exe" -DCMAKE_CXX_COMPILER="C:/Program Files/LLVM/bin/clang++.exe" ..