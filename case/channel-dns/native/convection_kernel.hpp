#pragma once
// p=3 CPU FR convection. FMA contraction enabled; no fast-math.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <omp.h>
#include "avx2.hpp"
using simd::Vec;

namespace fr_convection {
constexpr int Q=4, N=64, F=16, C=5;
inline int scratch_capacity(int max_cells,int max_faces) {
    return 19*max_cells*N+10*F+5*max_faces*F;
}
template<int A>
void numerical(const double* minus, const double* plus, double* common,
               const double* offset, double gamma) {
    for(int f=0;f<F;f+=4) {
        Vec rm=simd::load(minus+f),rp=simd::load(plus+f);
        Vec um=simd::load(minus+F+f)/rm,vm=simd::load(minus+2*F+f)/rm,wm=simd::load(minus+3*F+f)/rm;
        Vec up=simd::load(plus+F+f)/rp,vp=simd::load(plus+2*F+f)/rp,wp=simd::load(plus+3*F+f)/rp;
        Vec pm=(gamma-1)*(simd::load(minus+4*F+f)-.5*rm*((um*um+vm*vm)+wm*wm));
        Vec pp=(gamma-1)*(simd::load(plus+4*F+f)-.5*rp*((up*up+vp*vp)+wp*wp));
        Vec am=A==0?um:A==1?vm:wm,ap=A==0?up:A==1?vp:wp;
        Vec sm=simd::abs(am)+simd::sqrt(gamma*pm/rm);
        Vec sp=simd::abs(ap)+simd::sqrt(gamma*pp/rp);
        Vec speed=.5*simd::maximum(sm,sp);
        for(int c=0;c<C;++c) {
            Vec minus_c=simd::load(minus+c*F+f),plus_c=simd::load(plus+c*F+f);
            Vec fm=minus_c*am,fp=plus_c*ap;
            if(c==A+1) { fm=fm+pm; fp=fp+pp; }
            if(c==4) { fm=fm+pm*am; fp=fp+pp*ap; }
            simd::store(common+c*F+f,(.5*(fm+fp)-speed*(plus_c-minus_c))-offset[c]);
        }
    }
}
inline void solve_face(int axis,const double* minus,const double* plus,double* out,
                       const double* reference,double gamma) {
    if(axis==0) numerical<0>(minus,plus,out,reference,gamma);
    else if(axis==1) numerical<1>(minus,plus,out,reference+5,gamma);
    else numerical<2>(minus,plus,out,reference+10,gamma);
}
template<int A>
void volume(const double* packed, const double* primitive, double* flux,
            int count, const double* offset) {
    const int stride=count*N;
    const double* vel=primitive+A*stride;
    const double* pressure=primitive+3*stride;
    for(int c=0;c<C;++c) {
        for(int n=0;n<stride;n+=4) {
            Vec velocity=simd::load(vel+n),p=simd::load(pressure+n);
            Vec value=simd::load(packed+c*stride+n)*velocity;
            if(c==A+1) value=value+p;
            if(c==4) value=value+p*velocity;
            simd::store(flux+c*stride+n,value-offset[c]);
        }
    }
}
struct Group { const double* result; const double* primitive; const double* packed; int count; };
inline Group group(const double* input,int64_t elements,int g,
    const int64_t* group_offsets,const int64_t* cells,const int64_t* face_offsets,
    const int64_t* faces,const int64_t* cell_faces,const double* widths,
    const double* L,const double* R,const double* matrix,const double* gl,const double* gr,
    const double* reference,double gamma,double* scratch,double* timing) {
    double pack_total=0,face_total=0,volume_total=0;
            const int64_t start=group_offsets[g], end=group_offsets[g+1];
            const int count=end-start, stride=count*N;
            const int64_t first_face=face_offsets[g];
            const int nf=face_offsets[g+1]-first_face;
            double* packed=scratch;
            double* primitive=packed+5*stride;
            double* flux=primitive+4*stride;
            double* result=flux+5*stride;
            double* face_states=result+5*stride;
            double* common=face_states+10*F;
            double t=timing ? omp_get_wtime():0;
            for(int c=0;c<C;++c) for(int e=0;e<count;++e) {
                const int64_t old=cells[start+e];
                simd::copy64(packed+c*stride+e*N,input+c*elements*N+old*N);
            }
            // Explicitly pack both sides of every unique group interface.
            // A local endpoint reads packed data; an external endpoint reads
            // only its required four-point face interpolation from global U.
            for(int f=0;f<nf;++f) {
                const int64_t* face=faces+(first_face+f)*6;
                const int axis=face[0], wall=face[5];
                double* face_state=face_states;
                for(int side=0;side<2;++side) {
                    if((wall<0 && side==0)||(wall>0 && side==1)) continue;
                    const int64_t old=face[1+side], local=face[3+side];
                    for(int c=0;c<C;++c) {
                        const double* u=local>=0 ? packed+c*stride+local*N : input+c*elements*N+old*N;
                        const double* weights=side==0 ? R:L;
                        for(int n=0;n<F;n+=4)
                            simd::store(face_state+(side*5+c)*F+n,simd::trace(u,axis,n,weights));
                    }
                }
                if(wall) {
                    const int ghost=wall<0 ? 0:1, interior=1-ghost;
                    for(int c=0;c<C;++c) for(int n=0;n<F;n+=4) {
                        Vec value=simd::load(face_state+(interior*5+c)*F+n);
                        if(c==axis+1) value=Vec(_mm256_xor_pd(value.value,_mm256_set1_pd(-0.)));
                        simd::store(face_state+(ghost*5+c)*F+n,value);
                    }
                }
                solve_face(axis,face_state,face_state+5*F,common+f*5*F,reference,gamma);
            }
            if(timing) { pack_total+=omp_get_wtime()-t; t=omp_get_wtime(); }
            if(timing) { face_total+=omp_get_wtime()-t; t=omp_get_wtime(); }
            for(int n=0;n<stride;n+=4) {
                Vec rho=simd::load(packed+n);
                Vec u=simd::load(packed+stride+n)/rho,v=simd::load(packed+2*stride+n)/rho,w=simd::load(packed+3*stride+n)/rho;
                simd::store(primitive+n,u);
                simd::store(primitive+stride+n,v);
                simd::store(primitive+2*stride+n,w);
                simd::store(primitive+3*stride+n,(gamma-1)*(simd::load(packed+4*stride+n)-.5*rho*((u*u+v*v)+w*w)));
            }
            for(int axis=0;axis<3;++axis) {
                if(axis==0) volume<0>(packed,primitive,flux,count,reference);
                else if(axis==1) volume<1>(packed,primitive,flux,count,reference+5);
                else volume<2>(packed,primitive,flux,count,reference+10);
                for(int c=0;c<C;++c) for(int e=0;e<count;++e) {
                    const int64_t left=cell_faces[(start+e)*6+axis*2];
                    const int64_t right=cell_faces[(start+e)*6+axis*2+1];
                    const double* fc=flux+c*stride+e*N;
                    const double* cl=common+left*5*F+c*F;
                    const double* cr=common+right*5*F+c*F;
                    double scale=2/widths[cells[start+e]*3+axis];
                    for(int f=0;f<F;f+=4) {
                        Vec lines[4],previous[4],updated[4];
                        simd::load_lines(fc,axis,f,lines);
                        double* dst=result+c*stride+e*N;
                        if(axis==0) for(auto& value:previous) value=0.;
                        else
                        simd::load_lines(dst,axis,f,previous);
                        Vec common_l=simd::load(cl+f),common_r=simd::load(cr+f);
                        for(int i=0;i<Q;++i) {
                            Vec value=0.;
                            for(int j=0;j<Q;++j) value=value+lines[j]*matrix[i*Q+j];
                            value=value+gl[i]*common_l;
                            value=value+gr[i]*common_r;
                            updated[i]=previous[i]-scale*value;
                        }
                        simd::store_lines(dst,axis,f,updated);
                    }
                }
            }
            if(timing) { volume_total+=omp_get_wtime()-t; t=omp_get_wtime(); }
    if(timing) { timing[0]+=pack_total; timing[1]+=face_total; timing[2]+=volume_total; }
    return {result,primitive,packed,count};
}
inline void scatter(const double* result,double* output,int64_t elements,int g,
                    const int64_t* offsets,const int64_t* cells) {
    int64_t start=offsets[g],end=offsets[g+1];
    int count=end-start,stride=count*N;
    for(int c=0;c<C;++c) for(int e=0;e<count;++e) {
        simd::copy64(output+c*elements*N+cells[start+e]*N,result+c*stride+e*N);
    }
}
}
