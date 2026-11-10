#pragma once
#include <ATen/Parallel.h>
#include <chrono>

namespace fr_parallel {
inline double wall_time() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
}
