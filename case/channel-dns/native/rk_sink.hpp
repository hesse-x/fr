#pragma once
#include "avx2.hpp"
#include <cstdint>

namespace fr_output {
template<class T> struct Lane;
template<> struct Lane<double> {
    static double load(const double* p) { return *p; }
    static void store(double* p,double value) { *p=value; }
    static void stream(double* p,double value) { *p=value; }
};
template<> struct Lane<simd::Vec> {
    static simd::Vec load(const double* p) { return simd::load(p); }
    static void store(double* p,simd::Vec value) { simd::store(p,value); }
    static void stream(double* p,simd::Vec value) { _mm256_stream_pd(p,value.value); }
};

struct Raw {
    template<int C,class T>
    inline void put(double* output,int64_t count,int64_t at,int local,T derivative) const {
        Lane<T>::store(output+C*count+at,derivative);
    }
};

struct PartialRK {
    const double* original;
    const double* packed;
    double* pending;
    int stride;
    double coefficient;
    int combined;

    template<int C,class T>
    inline void put(double* output,int64_t count,int64_t at,int local,T derivative) const {
        if constexpr(C==1 || C==4) {
            // flux_force reads component 1: keep momentum at offset count.
            Lane<T>::store(pending+(C==1 ? count:0)+at,derivative);
        } else {
            T value=Lane<T>::load(packed+C*stride+local);
            if(combined) value=Lane<T>::load(original+C*count+at)*(2./3.)+value*(1./3.);
            Lane<T>::stream(output+C*count+at,value+coefficient*derivative);
        }
    }
};
}
