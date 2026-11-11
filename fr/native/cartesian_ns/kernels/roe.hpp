#pragma once
#include "avx2.hpp"

namespace fr_roe {
using simd::Vec;
inline Vec acoustic_absolute(Vec wave,Vec sound) {
    Vec value=simd::abs(wave),width=.05*sound;
    Vec fixed=.5*(value*value/width+width);
    return Vec(_mm256_blendv_pd(value.value,fixed.value,
        _mm256_cmp_pd(value.value,width.value,_CMP_LT_OQ)));
}
// Conservative Roe |A| jump in four face lanes; acoustic-only entropy fix.
template<int Axis,int Points=16>
inline void dissipation(const double* left,const double* right,int f,double gamma,Vec (&out)[5]) {
    constexpr int F=Points;
    Vec rl=simd::load(left+f),rr=simd::load(right+f);
    Vec sl=simd::sqrt(rl),sr=simd::sqrt(rr),sum=sl+sr;
    Vec ul[3],ur[3],u[3],dm[3],q2=0.,ql2=0.,qr2=0.;
    for(int c=0;c<3;++c) {
        Vec ml=simd::load(left+(c+1)*F+f),mr=simd::load(right+(c+1)*F+f);
        ul[c]=ml/rl; ur[c]=mr/rr;
        u[c]=(sl*ul[c]+sr*ur[c])/sum;
        dm[c]=mr-ml;
        q2=q2+u[c]*u[c]; ql2=ql2+ul[c]*ul[c]; qr2=qr2+ur[c]*ur[c];
    }
    Vec el=simd::load(left+4*F+f),er=simd::load(right+4*F+f);
    Vec pl=(gamma-1)*(el-.5*rl*ql2),pr=(gamma-1)*(er-.5*rr*qr2);
    Vec h=(sl*((el+pl)/rl)+sr*((er+pr)/rr))/sum;
    Vec c2=(gamma-1)*(h-.5*q2),sound=simd::sqrt(c2),normal=u[Axis];
    Vec dr=rr-rl,de=er-el,udm=0.;
    for(int c=0;c<3;++c) udm=udm+u[c]*dm[c];
    Vec dp=(gamma-1)*(de-udm+.5*q2*dr),dn=dm[Axis]-normal*dr;
    Vec am=acoustic_absolute(normal-sound,sound)*(dp-sound*dn)/(2*c2);
    Vec ap=acoustic_absolute(normal+sound,sound)*(dp+sound*dn)/(2*c2);
    Vec ac=simd::abs(normal)*(dr-dp/c2);
    out[0]=am+ac+ap;
    Vec shear_work=0.;
    for(int c=0;c<3;++c) {
        Vec shear=dm[c]-u[c]*dr;
        if(c==Axis) shear=shear-dn;
        out[c+1]=out[0]*u[c]+simd::abs(normal)*shear;
        if(c==Axis) out[c+1]=out[c+1]+sound*(ap-am);
        shear_work=shear_work+u[c]*shear;
    }
    out[4]=am*(h-sound*normal)+ac*.5*q2+ap*(h+sound*normal)+simd::abs(normal)*shear_work;
}
}
