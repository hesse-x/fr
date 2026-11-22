#pragma once
#include <limits>
#include "cartesian_ns/kernels/avx2.hpp"
#include "cartesian_ns/kernels/sgs.hpp"
namespace mapped_simd {
using simd::Vec;
inline double maximum(Vec v){double x[4];simd::store(x,v);return std::max(std::max(x[0],x[1]),std::max(x[2],x[3]));}
template<bool Scan,bool Rotating=false,bool GivenNorm=false>
inline void transport(void** data,int64_t e,int n,const double* gradient,Vec rho,Vec invJ,
                      double gamma,double mu,double kappa,bool owned,
                      Vec& effective_mu,Vec& effective_kappa,double& max_vis,Vec supplied_norm2=0.){
    effective_mu=mu;effective_kappa=kappa;
    if constexpr(fr_sgs::compiled_model!=fr_sgs::Model::None){
        const auto config=*(const fr_sgs::Config*)data[20];
        const auto delta=(const double*)data[21];
        Vec a[3][3];for(int c=0;c<3;++c)for(int x=0;x<3;++x)a[c][x]=simd::load(gradient+(c*3+x)*64+n);
        if constexpr(Rotating){const double* o=(const double*)data[22];
            a[0][1]=a[0][1]-o[2];a[0][2]=a[0][2]+o[1];
            a[1][0]=a[1][0]+o[2];a[1][2]=a[1][2]-o[0];
            a[2][0]=a[2][0]-o[1];a[2][1]=a[2][1]+o[0];}
        Vec turbulent=fr_sgs::tensor_viscosity<fr_sgs::compiled_model>(a,delta[e],rho,config);
        effective_mu=mu+turbulent;
        effective_kappa=kappa+turbulent*(gamma/((gamma-1)*config.turbulent_prandtl));
        if constexpr(Scan){if(owned){
            Vec norm2=0.;
            if constexpr(GivenNorm)norm2=supplied_norm2;
            else {const auto C=(const double*)data[1];
                for(int axis=0;axis<3;++axis)for(int x=0;x<3;++x){Vec h=simd::load(C+(e*9+axis*3+x)*64+n)*invJ;norm2=norm2+h*h;}}
            Vec rate=.25*norm2*simd::maximum(4./3.*effective_mu,(gamma-1)*effective_kappa)/rho;
            max_vis=std::max(max_vis,maximum(rate));
        }}
    }
}
inline bool finite(Vec v){return _mm256_movemask_pd(_mm256_cmp_pd(simd::abs(v).value,_mm256_set1_pd(std::numeric_limits<double>::max()),_CMP_LE_OQ))==15;}
inline bool positive(Vec v){return _mm256_movemask_pd(_mm256_cmp_pd(v.value,_mm256_setzero_pd(),_CMP_GT_OQ))==15;}
template<bool SoA> inline Vec state_load(const double* field,int64_t count,int64_t at,int c){
    if constexpr(SoA)return simd::load(field+c*count+at);
    return Vec(_mm256_setr_pd(field[at*5+c],field[(at+1)*5+c],field[(at+2)*5+c],field[(at+3)*5+c]));
}
template<bool SoA> inline void state_store(double* field,int64_t count,int64_t at,int c,Vec v){
    if constexpr(SoA)simd::store(field+c*count+at,v);
    else {double x[4];simd::store(x,v);for(int i=0;i<4;++i)field[(at+i)*5+c]=x[i];}
}
inline Vec neighbor_trace(const double* field,int components,const int64_t* neighbor,int q,int c){
    int64_t first=neighbor[q],last=neighbor[q+3],face=first/16;
    const double* base=field+face*components*16+c*16;
    if(last-first==3)return simd::load(base+first%16);
    if(last-first==-3)return Vec(_mm256_permute4x64_pd(simd::load(base+last%16).value,0x1b));
    return Vec(_mm256_i64gather_pd(base,_mm256_setr_epi64x(first%16,neighbor[q+1]%16,neighbor[q+2]%16,last%16),8));
}
inline void primitive_unchecked(Vec (&u)[5],double gamma,Vec (&w)[4],Vec& pressure,Vec& sound){
    const Vec inverse_density=1./u[0];
    for(int x=0;x<3;++x)w[x]=u[x+1]*inverse_density;
    pressure=(gamma-1)*(u[4]-.5*u[0]*((w[0]*w[0]+w[1]*w[1])+w[2]*w[2]));
    w[3]=pressure*inverse_density;sound=simd::sqrt(gamma*w[3]);
}
inline bool primitive_vectors(Vec (&u)[5],double gamma,Vec (&w)[4],Vec& pressure,Vec& sound){
    bool good=true;for(int c=0;c<5;++c)good=finite(u[c])&&good;
    good=positive(u[0])&&good;
    const Vec inverse_density=1./u[0];
    for(int x=0;x<3;++x)w[x]=u[x+1]*inverse_density;
    pressure=(gamma-1)*(u[4]-.5*u[0]*((w[0]*w[0]+w[1]*w[1])+w[2]*w[2]));
    good=positive(pressure)&&good;
    w[3]=pressure*inverse_density;sound=simd::sqrt(gamma*w[3]);return good;
}

}
