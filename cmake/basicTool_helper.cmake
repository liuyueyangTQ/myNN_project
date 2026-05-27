function(add_json_support TARGET_NAME)
    # 解析参数：支持可选的自定义宏和包含目录
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEFINES INCLUDES MAIN_DIR LINKS FILE_DIR)
    cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    # FILE_DIR 为必选参数
    get_target_property(target_type ${TARGET_NAME} TYPE) # 获取目标类型（可执行文件、静态库等）
    if(NOT target_type)
        message(FATAL_ERROR "❌ 错误: 目标 ${TARGET_NAME} 不存在")
    endif()
    # 2. 如果是可执行文件，必须定义 FILE_DIR
    if(target_type STREQUAL "EXECUTABLE" AND NOT ARGS_FILE_DIR)
        message(FATAL_ERROR 
            "❌ 错误: 调用 add_json_support 时未提供必选参数 FILE_DIR ！（应传入字符串类型）\n"
        )
    endif()
    # 配置 jsonCpp 路径
    set(jsonCpp_INSTALL_DIR "C:/Tools/jsoncpp-1.9.6/jsoncpp_install")
    set(JsonCpp_INCLUDE_DIRS "${jsonCpp_INSTALL_DIR}/include")
    set(JsonCpp_LIBRARIES_DIR "${jsonCpp_INSTALL_DIR}/lib")
    list(APPEND CMAKE_PREFIX_PATH "${jsonCpp_INSTALL_DIR}")
    find_package(jsoncpp CONFIG REQUIRED)
    message(STATUS "✅ jsoncpp 配置成功 ")
    message(STATUS "json头文件路径为: ${JsonCpp_INCLUDE_DIRS}")
    # 2. 现代 CMake 检查 Target 是否存在的标准写法
    if(TARGET jsoncpp_lib)
        message(STATUS "成功找到现代 Target: jsoncpp_lib")
    endif()

    # 包含头文件路径
    target_include_directories(${TARGET_NAME} PRIVATE 
        ${JsonCpp_INCLUDE_DIRS}
        ${ARGS_INCLUDES}
        ${TOOLS_DIR}/json
    )
    target_link_libraries(${TARGET_NAME} PRIVATE
        # jsoncpp_lib
        ${JsonCpp_LIBRARIES_DIR}/libjsoncpp_static.a
        ${ARGS_LINKS}
    )
    target_compile_definitions(${TARGET_NAME} PRIVATE 
        HTML_FILE_DIR="${ARGS_FILE_DIR}"
    )
endfunction()

function(add_web_support TARGET_NAME)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEFINES INCLUDES MAIN_DIR LINKS)
    cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    target_link_libraries(${TARGET_NAME} PRIVATE
        ${ARGS_LINKS}
        ws2_32 # 依赖 ws2_32
    )
endfunction()