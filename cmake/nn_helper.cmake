# 优化版 AddNNSupport.cmake
function(add_nn_support TARGET_NAME)
    # 解析参数：支持可选的自定义宏和包含目录
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEFINES INCLUDES MAIN_DIR LINKS)
    cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    # 基础包含目录
    set(base_includes
        ${ARGS_MAIN_DIR}/common
        ${ARGS_MAIN_DIR}/math
        ${ARGS_MAIN_DIR}/tensor
        ${ARGS_MAIN_DIR}/nn
        ${ARGS_MAIN_DIR}/thread
        ${ARGS_MAIN_DIR}/utils 
    )
    # 合并基础+自定义包含目录
    target_include_directories(${TARGET_NAME} PRIVATE
        ${base_includes}
        ${ARGS_INCLUDES}
    )
    # 合并基础+自定义编译宏
    target_compile_definitions(${TARGET_NAME} PRIVATE
        ${ARGS_DEFINES}
    )
    # 链接文件
    target_link_libraries(${TARGET_NAME} PRIVATE
        ${ARGS_LINKS}
    )
endfunction()