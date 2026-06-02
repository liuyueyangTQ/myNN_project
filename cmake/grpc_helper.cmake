function(add_grpc_support TARGET)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEFINES INCLUDES MAIN_DIR LINKS PROTO_DIR)
    cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(PROTOBUF_ROOT  "C:/Tools/protobuf-main/protobuf_install")
    set(GRPC_ROOT      "C:/Tools/grpc-1.81.0/grpc_install")
    set(ABSL_ROOT      "C:/Tools/abseil-cpp-master/absl_install")
    set(RE2_ROOT       "C:/Tools/re2/re2_install")
    set(CARES_ROOT     "C:/Tools/c-ares/c-ares_install")
    set(OPENSSL_ROOT   "C:/Tools/openssl-openssl-4.0.0/openssl_install")
    set(ZLIB_ROOT      "C:/Tools/zlib-1.3.1/zlib_install")

    set(PROTOC       "${PROTOBUF_ROOT}/bin/protoc.exe")
    set(GRPC_PLUGIN  "${GRPC_ROOT}/bin/grpc_cpp_plugin.exe")
    if(NOT ARGS_PROTO_DIR)
        message(FATAL_ERROR 
            "❌ 错误: 调用 add_grpc_support 时未提供必选参数 PROTO_DIR ！（应传入字符串类型）\n"
        )
    endif()
    set(COMMON_INCLUDE
        ${CMAKE_CURRENT_BINARY_DIR}
        ${PROTOBUF_ROOT}/include
        ${GRPC_ROOT}/include
        ${ABSL_ROOT}/include
        ${RE2_ROOT}/include
        ${CARES_ROOT}/include
        ${OPENSSL_ROOT}/include
        ${ZLIB_ROOT}/include
    )

    file(GLOB ABSL_LIBS "${ABSL_ROOT}/lib/libabsl_*.a")

    target_include_directories(${TARGET} PRIVATE 
        ${COMMON_INCLUDE}
        ${ARGS_INCLUDES}
        ${ARGS_PROTO_DIR}
    )
    target_link_directories(${TARGET} PRIVATE
        ${PROTOBUF_ROOT}/lib
        ${GRPC_ROOT}/lib
        ${ABSL_ROOT}/lib
        ${RE2_ROOT}/lib
        ${CARES_ROOT}/lib
        ${OPENSSL_ROOT}/lib64
        ${ZLIB_ROOT}/lib
    )
    # abseil 内部大量交叉引用，必须用 --whole-archive 强制全量链接
    target_link_libraries(${TARGET} PRIVATE
        -Wl,--whole-archive
        ${ABSL_LIBS}
        -Wl,--no-whole-archive
        # gRPC 主库 + 内部依赖
        grpc++
        grpc
        gpr
        address_sorting   # gRPC DNS 地址排序
        upb               # gRPC 内部 proto 运行时
        # protobuf
        protobuf
        utf8_range        # protobuf UTF-8 校验
        utf8_validity     # 另一个可能的命名
        # 其他传递依赖
        re2
        cares
        ssl
        crypto
        z
        # Windows 系统 API（MinGW 需要显式指定）
        ws2_32
        bcrypt
        crypt32
        iphlpapi
        dbghelp
    )
endfunction()


# 用于自动生成 protobuf 和 gRPC 文件的函数
# add_custom_command(
#     OUTPUT ${PROTO_SRC} ${PROTO_HDR} ${GRPC_SRC} ${GRPC_HDR}
#     COMMAND protobuf::protoc
#         --proto_path=${PROTO_DIR}
#         --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
#         ${PROTO_DIR}/${PROTO_FILE}
#     COMMAND protobuf::protoc
#         --proto_path=${PROTO_DIR}
#         --grpc_out=${CMAKE_CURRENT_BINARY_DIR}
#         --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
#         ${PROTO_DIR}/${PROTO_FILE}
#     DEPENDS ${PROTO_DIR}/${PROTO_FILE}
#     COMMENT "Generating gRPC/protobuf files from ${PROTO_FILE}"
# )