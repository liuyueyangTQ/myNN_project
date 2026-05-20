function(add_netService_support TARGET_NAME)
    message(STATUS "当前目标名：${TARGET_NAME}")
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEFINES INCLUDES MAIN_DIR LINKS)
    cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    target_link_libraries(${TARGET_NAME} PRIVATE
        ${ARGS_LINKS}
    )
    if(WIN32)
        message(STATUS "为 Windows 平台设置网络服务支持")
        target_include_directories(${TARGET_NAME} PRIVATE
            ${CMAKE_SOURCE_DIR}/src/tools/netService
        )
        target_link_libraries(${TARGET_NAME} PRIVATE
            ${ARGS_LINKS}
            ws2_32 # 必须放在最后，越底层的库越靠后，ARGS_LINKS 中的库可能依赖 ws2_32
        )
    else()
    endif()
    target_include_directories(${TARGET_NAME} PRIVATE
        ${ARGS_MAIN_DIR}
        ${ARGS_INCLUDES}
    )
    target_compile_definitions(${TARGET_NAME} PRIVATE
        ${ARGS_DEFINES}
    )
endfunction()