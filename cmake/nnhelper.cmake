# 优化版 AddNNExecutable.cmake
function(add_nn_executable NAME)
    # 解析参数：支持可选的自定义宏和包含目录
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEFINES INCLUDES MAIN_DIR)
    cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    # 创建可执行文件
    add_executable(${NAME} ${ARGS_SRCS})
    # 基础包含目录
    set(base_includes
        ${ARGS_MAIN_DIR}/math
        ${ARGS_MAIN_DIR}/tensor
        ${ARGS_MAIN_DIR}/nn
        ${ARGS_MAIN_DIR}/thread
        ${ARGS_MAIN_DIR}/utils 
    )
    # 合并基础+自定义包含目录
    target_include_directories(${NAME} PRIVATE
        ${base_includes}
        ${ARGS_INCLUDES}
    )
    # 合并基础+自定义编译宏
    target_compile_definitions(${NAME} PRIVATE
        ${ARGS_DEFINES}
    )
    # 链接文件
    target_link_libraries(${NAME} PRIVATE
        ${ARGS_LINKS}
    )
endfunction()