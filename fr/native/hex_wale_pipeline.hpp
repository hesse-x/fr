#pragma once
#include "wale_invariants.hpp"
// Mapped-cell WALE scheduling across independent vectors; preserve each lane's
// arithmetic expression, including normal division and nested square roots.
namespace mapped_simd {
using fr_sgs::wale_invariants;

template<int Batch,bool SoA,bool Rotating>
inline void wale_pipeline(void** data,int64_t e,int64_t count,const double* input,const double* gradient,double* output){
    const auto config=*(const fr_sgs::Config*)data[20];const auto delta=(const double*)data[21];
    for(int base=0;base<64;base+=Batch*4){
        Vec s2[Batch],d2[Batch],rootS[Batch],rootD[Batch],fourthD[Batch];
        for(int b=0;b<Batch;++b){
            const int n=base+b*4;Vec a[3][3];
            for(int c=0;c<3;++c)for(int x=0;x<3;++x)a[c][x]=simd::load(gradient+(c*3+x)*64+n);
            if constexpr(Rotating){const double* o=(const double*)data[22];
                a[0][1]=a[0][1]-o[2];a[0][2]=a[0][2]+o[1];
                a[1][0]=a[1][0]+o[2];a[1][2]=a[1][2]-o[0];
                a[2][0]=a[2][0]-o[1];a[2][1]=a[2][1]+o[0];}
            wale_invariants(a,s2[b],d2[b]);
        }
        for(int b=0;b<Batch;++b)rootS[b]=simd::sqrt(s2[b]);
        for(int b=0;b<Batch;++b)rootD[b]=simd::sqrt(d2[b]);
        for(int b=0;b<Batch;++b)fourthD[b]=simd::sqrt(rootD[b]);
        for(int b=0;b<Batch;++b){
            Vec denominator=s2[b]*s2[b]*rootS[b]+d2[b]*fourthD[b];
            Vec rho=state_load<SoA>(input,count,e*64+base+b*4,0);
            Vec turbulent=rho*(config.coefficient*config.coefficient*delta[e]*delta[e])*d2[b]*rootD[b]
                /simd::maximum(denominator,Vec(1e-300));
            simd::store(output+base+b*4,turbulent);
        }
    }
}

template<bool Scan,bool GivenNorm=false>
inline void transport_pipeline(void** data,int64_t e,int n,const double* turbulent,Vec rho,Vec invJ,
                      double gamma,double mu,double kappa,bool owned,
                      Vec& effective_mu,Vec& effective_kappa,double& max_vis,Vec supplied_norm2=0.){
    const auto config=*(const fr_sgs::Config*)data[20];Vec viscosity=simd::load(turbulent+n);
    effective_mu=mu+viscosity;
    effective_kappa=kappa+viscosity*(gamma/((gamma-1)*config.turbulent_prandtl));
    if constexpr(Scan){if(owned){
        Vec norm2=0.;
        if constexpr(GivenNorm)norm2=supplied_norm2;
        else{const auto C=(const double*)data[1];
            for(int axis=0;axis<3;++axis)for(int x=0;x<3;++x){Vec h=simd::load(C+(e*9+axis*3+x)*64+n)*invJ;norm2=norm2+h*h;}}
        Vec rate=.25*norm2*simd::maximum(4./3.*effective_mu,(gamma-1)*effective_kappa)/rho;
        max_vis=std::max(max_vis,maximum(rate));
    }}
}
}
