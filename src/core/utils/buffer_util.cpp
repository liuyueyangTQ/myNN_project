#include "buffer_util.h"


namespace memory_utils {
void* aligned_alloc(size_t size, size_t align) {
    if (size == 0) return nullptr;
    if ((align & (align - 1)) != 0) { // 检查是否为2的幂
        errno = EINVAL;
        return nullptr;
    }

#if defined(_WIN32) || defined(_WIN64)
    // Windows平台：_aligned_malloc（对齐大小最小8字节）
    return _aligned_malloc(size, std::max(align, (size_t)8UL));
#else
    // Linux/macOS：posix_memalign（对齐大小最小sizeof(void*)）
    void* ptr = nullptr;
    int ret = posix_memalign(&ptr, std::max(align, sizeof(void*)), size);
    return ret == 0 ? ptr : nullptr;
#endif
}

void aligned_free(void* ptr) {
    if (!ptr) return;
#if defined(_WIN32) || defined(_WIN64)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}
} // namespace memory_utils