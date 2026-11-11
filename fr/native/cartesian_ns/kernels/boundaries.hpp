#pragma once
// Boundary laws are separate from group traversal and FR/LDG contractions.
#include "avx2.hpp"

namespace fr_ns_boundary {
inline simd::Vec reflected_momentum(simd::Vec value, int component, int axis) {
    if(component==axis+1)
        return simd::Vec(_mm256_xor_pd(value.value,_mm256_set1_pd(-0.)));
    return value;
}
inline simd::Vec primitive_value(int component, double temperature) {
    return component==3 ? temperature:0.;
}
inline simd::Vec stationary_wall_energy(simd::Vec value, double scale,
    double gl, double gr, simd::Vec left, simd::Vec right, bool lower, bool upper) {
    if(lower) value=value-(scale*gl)*left;
    if(upper) value=value-(scale*gr)*right;
    return value;
}
}
