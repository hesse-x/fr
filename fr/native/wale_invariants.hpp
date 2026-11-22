#pragma once
#include "cartesian_ns/kernels/avx2.hpp"
namespace fr_sgs {
using simd::Vec;
inline void wale_invariants(const Vec a[3][3],Vec& s2,Vec& d2){
    Vec diagonal[3];
    for(int i=0;i<3;++i){diagonal[i]=0.;
        for(int k=0;k<3;++k)diagonal[i]=diagonal[i]+a[i][k]*a[k][i];}
    Vec trace=(diagonal[0]+diagonal[1]+diagonal[2])/3.;
    s2=0.;
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){
        Vec s=.5*(a[i][j]+a[j][i]);s2=s2+s*s;}
    auto symmetric_square=[&](int i,int j){Vec forward=0.,reverse=0.;
        for(int k=0;k<3;++k){
            forward=forward+a[i][k]*a[k][j];
            reverse=reverse+a[j][k]*a[k][i];}
        return .5*(forward+reverse);
    };
    Vec sd=.5*(diagonal[0]+diagonal[0])-trace;d2=0.;d2=d2+sd*sd;
    Vec sd01=symmetric_square(0,1);d2=d2+sd01*sd01;
    Vec sd02=symmetric_square(0,2);d2=d2+sd02*sd02;
    d2=d2+sd01*sd01;
    sd=.5*(diagonal[1]+diagonal[1])-trace;d2=d2+sd*sd;
    Vec sd12=symmetric_square(1,2);d2=d2+sd12*sd12;
    d2=d2+sd02*sd02;d2=d2+sd12*sd12;
    sd=.5*(diagonal[2]+diagonal[2])-trace;d2=d2+sd*sd;
}

}
