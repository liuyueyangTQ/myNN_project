function(add_QT_support TARGET_NAME)
    # 解析参数：支持可选的自定义宏和包含目录
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEFINES INCLUDES MAIN_DIR LINKS)
    cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    # 设置QT工具源路径
    set(QT_SRC_DIR "C:/Qt/6.10.1")
    # 添加 Qt6 安装路径
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU") 
        # 适配 GCC (MinGW)
        message(STATUS "Detected GCC: Using standard mingw_64 path")
        set(CMAKE_PREFIX_PATH "C:/Qt/6.10.1/mingw_64")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # 适配 Clang (包括 Clang-cl 或 llvm-mingw)
        message(STATUS "Detected Clang: Using llvm-mingw_64 path")
        set(CMAKE_PREFIX_PATH "C:/Qt/6.10.1/llvm-mingw_64")
    else()
        message(WARNING "Unknown compiler: ${CMAKE_CXX_COMPILER_ID}. Defaulting to mingw_64")
        set(CMAKE_PREFIX_PATH "C:/Qt/6.10.1/mingw_64")
    endif()
    # 寻找 QT 依赖
    find_package(Qt6 REQUIRED COMPONENTS Core Widgets)
    message(STATUS "Current Linker Flags: ${CMAKE_EXE_LINKER_FLAGS}")

    # 链接 Qt 库
    target_link_libraries(${TARGET_NAME} PRIVATE Qt6::Core Qt6::Widgets) 
    # 定义目标文件的输出路径
    set_target_properties(${TARGET_NAME} PROPERTIES
        # 可执行文件输出到目录
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_custom_command(
        TARGET ${TARGET_NAME}  # 关联到目标可执行文件
        POST_BUILD           # 在目标构建完成后执行
        COMMAND ${CMAKE_COMMAND} -E copy_if_different #复制 QT 动态库文件到当前可执行文件夹下
            "${QT_SRC_DIR}/mingw_64/bin/Qt6Core.dll"  
            "${CMAKE_CURRENT_BINARY_DIR}/"  
        COMMAND ${CMAKE_COMMAND} -E copy_if_different 
            "${QT_SRC_DIR}/mingw_64/bin/Qt6Gui.dll"  
            "${CMAKE_CURRENT_BINARY_DIR}/"  
        COMMAND ${CMAKE_COMMAND} -E copy_if_different 
            "${QT_SRC_DIR}/mingw_64/bin/Qt6Widgets.dll"  
            "${CMAKE_CURRENT_BINARY_DIR}/"  
        COMMAND ${CMAKE_COMMAND} -E make_directory  # 在build文件夹下创建目录 platform用于链接QT
            "${CMAKE_CURRENT_BINARY_DIR}/platforms"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${QT_SRC_DIR}/mingw_64/plugins/platforms/qwindows.dll"  
            "${CMAKE_CURRENT_BINARY_DIR}/platforms/"  # 目标目录：exe 所在目录
        COMMENT "copying QT DLLto executable fold..."  # 构建时显示的提示信息
    )
endfunction()