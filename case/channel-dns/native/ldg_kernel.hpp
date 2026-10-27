#pragma once
// Experimental p=3 grouped LDG, with unchanged primal/dual traces and walls.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <omp.h>
#include "rk_sink.hpp"
#include "avx2.hpp"
using simd::Vec;

namespace fr_ldg {
constexpr int Q=4, N=64, F=16;
template<bool Local,class Output=fr_output::Raw>
inline void group(const double* input,const double* convection,double* output,int64_t elements,int g,
    const int64_t* offsets,const int64_t* primitive_ids,const int64_t* active_meta,const int64_t* owned_meta,
    const double* widths,const double* L,const double* R,const double* primal,const double* dual,
    const double* gl,const double* gr,double gamma,double mu,double kappa,double wall_temperature,
    double forcing,double* scratch,double* timing,Output sink={}) {
    double pack_total=0,gradient_total=0,flux_total=0,divergence_total=0,store_total=0;
            int64_t os=offsets[g*3],as=offsets[g*3+1],bs=offsets[g*3+2];
            int no=offsets[(g+1)*3]-os,na=offsets[(g+1)*3+1]-as,nb=offsets[(g+1)*3+2]-bs;
            int ps=nb*N,gs=na*N,vs=no*N;
            double* primitive=scratch;
            double* gradient=primitive+4*ps;
            double* flux=gradient+12*gs;
            double* viscous=flux+4*gs;
            double* wall_work=viscous+4*vs;
            // Owned cells need all directions. A halo cell only needs the
            // flux directions consumed by owned cells' dual traces.
            static thread_local std::vector<unsigned char> needed;
            needed.assign(na,0);
            for(int e=0;e<no;++e) {
                needed[e]=7;
                for(int axis=0;axis<3;++axis) {
                    int neighbor=owned_meta[(os+e)*3+axis];
                    if(neighbor>=0) needed[neighbor]|=1<<axis;
                }
            }
            double t=timing ? omp_get_wtime():0;
            for(int e=0;e<nb;++e) {
                int64_t old=primitive_ids[bs+e];
                for(int n=0;n<N;n+=4) {
                    int64_t at=old*N+n;
                    Vec rho=simd::load(input+at);
                    Vec u=simd::load(input+elements*N+at)/rho;
                    Vec v=simd::load(input+2*elements*N+at)/rho;
                    Vec w=simd::load(input+3*elements*N+at)/rho;
                    Vec pressure=(gamma-1)*(simd::load(input+4*elements*N+at)-.5*rho*((u*u+v*v)+w*w));
                    simd::store(primitive+e*N+n,u);
                    simd::store(primitive+ps+e*N+n,v);
                    simd::store(primitive+2*ps+e*N+n,w);
                    simd::store(primitive+3*ps+e*N+n,pressure/rho);
                }
            }
            if(timing) { pack_total+=omp_get_wtime()-t; t=omp_get_wtime(); }
            for(int axis=0;axis<3;++axis) for(int e=0;e<na;++e) {
                const int64_t* meta=active_meta+(as+e)*5;
                int64_t old=primitive_ids[bs+e];
                double scale=2/widths[old*3+axis];
                for(int c=0;c<4;++c) {
                    // Diagonal velocity derivatives enter every stress trace;
                    // off-diagonals enter row/column directions, dT only its row.
                    int mask=c==3 ? (1<<axis) : (axis==c ? 7 : (1<<axis)|(1<<c));
                    if(!(needed[e]&mask)) continue;
                    const double* field=primitive+c*ps+e*N;
                    for(int f=0;f<F;f+=4) {
                        Vec lines[4],updated[4];
                        simd::load_lines(field,axis,f,lines);
                        Vec wall=c==3 ? wall_temperature:0.;
                        Vec common=(axis==1 && meta[3]) ? wall:
                            simd::trace(primitive+c*ps+meta[axis]*N,axis,f,R);
                        Vec delta=0.;
                        if(axis==1 && meta[4]) delta=wall-simd::trace(field,axis,f,R);
                        for(int i=0;i<Q;++i) {
                            Vec value=0.;
                            for(int j=0;j<Q;++j) value=value+lines[j]*primal[i*Q+j];
                            value=scale*(value+gl[i]*common);
                            if(axis==1 && meta[4]) value=value+(scale*gr[i])*delta;
                            updated[i]=value;
                        }
                        simd::store_lines(gradient+(axis*4+c)*gs+e*N,axis,f,updated);
                    }
                }
            }
            if(timing) { gradient_total+=omp_get_wtime()-t; t=omp_get_wtime(); }
            for(int axis=0;axis<3;++axis) {
                // Halo gradients are recomputed privately, so no global
                // gradient/stress arrays or cross-worker synchronization.
                for(int e=0;e<na;++e) {
                    if(!(needed[e]&(1<<axis))) continue;
                    for(int n=0;n<N;n+=4) {
                        int at=e*N+n;
                        Vec tr=(simd::load(gradient+at)+simd::load(gradient+5*gs+at))+simd::load(gradient+10*gs+at);
                        Vec stress[3];
                        for(int c=0;c<3;++c) {
                            Vec correction=c==axis ? (2./3.)*tr:Vec(0.);
                            stress[c]=mu*((simd::load(gradient+(axis*4+c)*gs+at)+
                                           simd::load(gradient+(c*4+axis)*gs+at))-correction);
                            simd::store(flux+c*gs+at,stress[c]);
                        }
                        Vec work=((Vec(0.)+simd::load(primitive+e*N+n)*stress[0])+
                                  simd::load(primitive+ps+e*N+n)*stress[1])+
                                  simd::load(primitive+2*ps+e*N+n)*stress[2];
                        simd::store(flux+3*gs+at,work+kappa*simd::load(gradient+(axis*4+3)*gs+at));
                        if(axis==1 && e<no) simd::store(wall_work+e*N+n,work);
                    }
                }
                if(timing) { flux_total+=omp_get_wtime()-t; t=omp_get_wtime(); }
                for(int e=0;e<no;++e) {
                    const int64_t* meta=owned_meta+(os+e)*3;
                    const int64_t* boundary=active_meta+(as+e)*5;
                    double scale=2/widths[primitive_ids[bs+e]*3+axis];
                    for(int c=0;c<4;++c) {
                        const double* field=flux+c*gs+e*N;
                        bool upper=axis==1 && boundary[4];
                        const double* neighbor=upper ? field:flux+c*gs+meta[axis]*N;
                        for(int f=0;f<F;f+=4) {
                            Vec lines[4],previous[4],updated[4];
                            simd::load_lines(field,axis,f,lines);
                            double* dst=viscous+c*vs+e*N;
                            if(axis==0) for(auto& value:previous) value=0.;
                            else
                            simd::load_lines(dst,axis,f,previous);
                            Vec common=simd::trace(neighbor,axis,f,upper ? R:L);
                            for(int i=0;i<Q;++i) {
                                Vec value=0.;
                                for(int j=0;j<Q;++j) value=value+lines[j]*dual[i*Q+j];
                                value=scale*(value+gr[i]*common);
                                updated[i]=previous[i]+value;
                            }
                            simd::store_lines(dst,axis,f,updated);
                        }
                    }
                }
                if(timing) { divergence_total+=omp_get_wtime()-t; t=omp_get_wtime(); }
            }
            for(int e=0;e<no;++e) {
                const int64_t* boundary=active_meta+(as+e)*5;
                if(boundary[3] || boundary[4]) {
                    double scale=2/widths[primitive_ids[bs+e]*3+1];
                    const double* work=wall_work+e*N;
                    for(int f=0;f<F;f+=4) {
                        Vec left=simd::trace(work,1,f,L),right=simd::trace(work,1,f,R);
                        Vec values[4];
                        double* dst=viscous+3*vs+e*N;
                        simd::load_lines(dst,1,f,values);
                        for(int i=0;i<Q;++i) {
                            if(boundary[3]) values[i]=values[i]-(scale*gl[i])*left;
                            if(boundary[4]) values[i]=values[i]-(scale*gr[i])*right;
                        }
                        simd::store_lines(dst,1,f,values);
                    }
                }
                int64_t old=primitive_ids[bs+e];
                const double* conv_cell=convection+(Local ? e*N:old*N);
                int64_t conv_stride=Local ? vs:elements*N;
                for(int n=0;n<N;n+=4) {
                    int64_t at=old*N+n;
                    sink.template put<0>(output,elements*N,at,e*N+n,simd::load(conv_cell+n));
                    sink.template put<1>(output,elements*N,at,e*N+n,(simd::load(conv_cell+conv_stride+n)+simd::load(viscous+e*N+n))+forcing);
                    sink.template put<2>(output,elements*N,at,e*N+n,simd::load(conv_cell+2*conv_stride+n)+simd::load(viscous+vs+e*N+n));
                    sink.template put<3>(output,elements*N,at,e*N+n,simd::load(conv_cell+3*conv_stride+n)+simd::load(viscous+2*vs+e*N+n));
                    sink.template put<4>(output,elements*N,at,e*N+n,(simd::load(conv_cell+4*conv_stride+n)+simd::load(viscous+3*vs+e*N+n))+
                                         forcing*simd::load(primitive+e*N+n));
                }
            }
            if(timing) store_total+=omp_get_wtime()-t;
    if(timing) { timing[0]+=pack_total; timing[1]+=gradient_total; timing[2]+=flux_total;
                 timing[3]+=divergence_total; timing[4]+=store_total; }
}
}
