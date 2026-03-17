#pragma once
#include <cstddef>
#include <cassert>
#include <stdexcept>
#include <iostream>
#include <cerrno>
#include <algorithm>
#include <memory>
#include <new>
#include <thread>

// 适配不同平台的缓存行大小（C++17推荐用标准常量）
#if __cplusplus >= 201703L
#include <new>
constexpr size_t CACHE_LINE_SIZE = std::hardware_destructive_interference_size;
#else
// 通用缓存行大小（绝大多数CPU为64字节）
constexpr size_t CACHE_LINE_SIZE = 64;
#endif

// 跨平台对齐内存分配/释放（封装成工具函数）
namespace memory_utils {
    // 分配对齐的内存（size：总字节数，align：对齐大小，必须是2的幂）
    void* aligned_alloc(size_t size, size_t align);

    // 释放对齐的内存
    void aligned_free(void* ptr);       
}