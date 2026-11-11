#pragma once
#include "avx2.hpp"

namespace fr_sgs {
enum class Model { None, Vreman, Smagorinsky };
struct Config {
    double coefficient=0.;
    double turbulent_prandtl=.9;
    double filter_scale=1.;
};
#if FR_NS_VREMAN
inline constexpr Model compiled_model=Model::Vreman;
#elif FR_NS_SMAGORINSKY
inline constexpr Model compiled_model=Model::Smagorinsky;
#else
inline constexpr Model compiled_model=Model::None;
#endif
inline constexpr int extra_fields=compiled_model==Model::None ? 0:1;
// Model selection is specialized once per cached library, never per solution point.
// Coefficients are runtime plan data; changing them does not trigger compilation.
template<Model model>
inline simd::Vec viscosity(const double* gradient,int gs,int at,double delta,
                           simd::Vec rho,const Config& config) {
    if constexpr(model==Model::None) return simd::Vec(0.);
    else {
        using simd::Vec;
        Vec a[3][3],norm2=0.;
        for(int c=0;c<3;++c) for(int d=0;d<3;++d) {
            a[c][d]=simd::load(gradient+(d*4+c)*gs+at);
            norm2=norm2+a[c][d]*a[c][d];
        }
        if constexpr(model==Model::Smagorinsky) {
            Vec trace=(a[0][0]+a[1][1]+a[2][2])/3.;
            Vec magnitude2=0.;
            for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
                Vec strain=(a[i][j]+a[j][i])*.5;
                if(i==j) strain=strain-trace;
                magnitude2=magnitude2+2.*strain*strain;
            }
            return rho*(config.coefficient*config.coefficient*delta*delta)*simd::sqrt(magnitude2);
        }
        Vec b=0.;
        for(int i=0;i<3;++i) for(int j=i+1;j<3;++j) for(int d=0;d<3;++d) {
            int r=(d+1)%3,q=(d+2)%3;
            Vec cross=a[i][r]*a[j][q]-a[i][q]*a[j][r];
            b=b+cross*cross;
        }
        return rho*(config.coefficient*delta*delta)*simd::sqrt(b/simd::maximum(norm2,Vec(1e-300)));
    }
}
}
