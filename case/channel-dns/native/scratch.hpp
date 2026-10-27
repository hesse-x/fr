#pragma once
#include <vector>

namespace fr_scratch {
inline std::vector<double>& buffer(int size,int slot) {
    // Slots keep convection and LDG storage independent during a fused call.
    static thread_local std::vector<double> buffers[2];
    auto& value=buffers[slot];
    if(static_cast<int>(value.size())<size) value.resize(size);
    return value;
}
}
