// Four FP64 lanes. The compiler may contract multiply/add into FMA.
#pragma once
#include <immintrin.h>

namespace simd {
struct Vec {
    __m256d value;
    Vec() = default;
    Vec(double x): value(_mm256_set1_pd(x)) {}
    Vec(__m256d x): value(x) {}
};
inline Vec operator+(Vec a,Vec b) { return _mm256_add_pd(a.value,b.value); }
inline Vec operator-(Vec a,Vec b) { return _mm256_sub_pd(a.value,b.value); }
inline Vec operator*(Vec a,Vec b) { return _mm256_mul_pd(a.value,b.value); }
inline Vec operator/(Vec a,Vec b) { return _mm256_div_pd(a.value,b.value); }
inline Vec load(const double* p) { return _mm256_loadu_pd(p); }
inline void store(double* p,Vec x) { _mm256_storeu_pd(p,x.value); }
inline Vec sqrt(Vec x) { return _mm256_sqrt_pd(x.value); }
inline Vec abs(Vec x) { return _mm256_andnot_pd(_mm256_set1_pd(-0.),x.value); }
inline Vec maximum(Vec a,Vec b) {
    // Match std::max's first-operand NaN and equal-value behavior.
    return _mm256_blendv_pd(a.value,b.value,_mm256_cmp_pd(a.value,b.value,_CMP_LT_OQ));
}
inline void transpose(Vec (&a)[4]) {
    __m256d lo0=_mm256_unpacklo_pd(a[0].value,a[1].value);
    __m256d hi0=_mm256_unpackhi_pd(a[0].value,a[1].value);
    __m256d lo1=_mm256_unpacklo_pd(a[2].value,a[3].value);
    __m256d hi1=_mm256_unpackhi_pd(a[2].value,a[3].value);
    a[0]=_mm256_permute2f128_pd(lo0,lo1,0x20);
    a[1]=_mm256_permute2f128_pd(hi0,hi1,0x20);
    a[2]=_mm256_permute2f128_pd(lo0,lo1,0x31);
    a[3]=_mm256_permute2f128_pd(hi0,hi1,0x31);
}
// Four face points at once, each vector holding one position along the line.
// z lines use a register transpose, avoiding AVX2 gathers/scalar scatters.
inline void load_lines(const double* p,int axis,int f,Vec (&a)[4]) {
    if(axis==0) for(int j=0;j<4;++j) a[j]=load(p+j*16+f);
    else if(axis==1) for(int j=0;j<4;++j) a[j]=load(p+(f/4)*16+j*4);
    else {
        for(int j=0;j<4;++j) a[j]=load(p+(f+j)*4);
        transpose(a);
    }
}
inline void store_lines(double* p,int axis,int f,Vec (&a)[4]) {
    if(axis==0) for(int j=0;j<4;++j) store(p+j*16+f,a[j]);
    else if(axis==1) for(int j=0;j<4;++j) store(p+(f/4)*16+j*4,a[j]);
    else {
        transpose(a);
        for(int j=0;j<4;++j) store(p+(f+j)*4,a[j]);
    }
}
inline Vec trace(const double* p,int axis,int f,const double* weights) {
    Vec a[4];
    load_lines(p,axis,f,a);
    Vec result=a[0]*weights[0];
    for(int j=1;j<4;++j) result=result+a[j]*weights[j];
    return result;
}
inline void copy64(double* dst,const double* src) {
    for(int n=0;n<64;n+=4) store(dst+n,load(src+n));
}
}
