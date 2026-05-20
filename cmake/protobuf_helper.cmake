function(add_protobuf_support TARGET_NAME)
    message(STATUS "当前目标名：${TARGET_NAME}")
    # 解析参数：支持可选的自定义宏和包含目录
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEFINES INCLUDES MAIN_DIR LINKS)
    cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    # 查找 Protobuf
    # 1. 定义你的两个安装路径（根据你的实际路径调整）
    set(ABSL_DIR "C:/Tools/abseil-cpp-master/absl_install")
    set(PROTOBUF_DIR "C:/Tools/protobuf-main/protobuf_install")

    # 2. 告诉 CMake 去哪里找这些库的配置文件
    list(APPEND CMAKE_PREFIX_PATH "${ABSL_DIR}" "${PROTOBUF_DIR}")

    # 3. 显式查找所有需要的包（核心：加上 utf8_range）
    find_package(absl CONFIG REQUIRED)
    message(STATUS "✅ Abseil 配置成功 for target: ${TARGET_NAME}")
    find_package(utf8_range CONFIG REQUIRED) # Protobuf 4.x 强依赖的 utf8 验证库, 在protobuf_install文件夹下
    message(STATUS "✅ utf8_range 配置成功 for target: ${TARGET_NAME}")
    find_package(protobuf CONFIG REQUIRED)
    message(STATUS "✅ Protobuf 配置成功 for target: ${TARGET_NAME}")

    # 链接到 CMake 目标（Target），这会自动引入所有底层依赖（包括 utf8_range 和 absl 的基础库）
    set(Absl_INCLUDE_DIRS "C:/Tools/abseil-cpp-master/absl_install/include")

    get_target_property(target_type ${TARGET_NAME} TYPE) # 获取目标类型（可执行文件、静态库等）
    message(STATUS "目标类型: ${target_type}")
    # 2. 根据类型判断
    if(target_type STREQUAL "EXECUTABLE")
        # 包含头文件路径
        target_include_directories(${TARGET_NAME} PRIVATE 
            ${PROTOBUF_INCLUDE_DIRS}
            ${Absl_INCLUDE_DIRS}
            ${ARGS_INCLUDES}
            ${TOOLS_DIR}/protobuf
        )
    else()
        # 对于库目标，通常不直接包含 Protobuf 头文件路径，而是通过链接来传递依赖关系
        message(STATUS "库目标 ${TARGET_NAME} 将通过链接传递 Protobuf 依赖")
    endif()

    # 链接到可执行文件
    target_link_libraries(${TARGET_NAME} PRIVATE 
        # 1. 业务需要的核心上层目标
        protobuf::libprotobuf
        # 2. 显式补充 Protobuf 3.x/4.x/5.x 强依赖的 utf8 验证库（如果有的话）
        utf8_range::utf8_range
        utf8_range::utf8_validity
        # 3. 链接 protobuf 依赖的 Abseil 组件
        absl::cord
        absl::cord_internal
        absl::cordz_info
        absl::cordz_handle
        absl::log_internal_message
        absl::log_internal_check_op
        absl::log_internal_nullguard
        absl::log_internal_proto
        absl::synchronization
        absl::status
        absl::hash
        absl::base
        # 4. 业务需要的其他库（如之前的 math_libs）
        ${ARGS_LINKS}
    )
    # 合并基础+自定义编译宏
    target_compile_definitions(${TARGET_NAME} PRIVATE
        ${ARGS_DEFINES}
    )
endfunction()

function(gen_protobuf_lib TARGET_NAME)
    
endfunction(gen_protobuf_lib)

