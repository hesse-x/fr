#pragma once
#include "sgs.hpp"
// Central lifted gradient and flux, with primitive velocity/temperature penalty.
// Uses two-sided private halos; no full-grid gradient or residual intermediates.
#include "ldg_kernel.hpp"

namespace fr_br1 {
using fr_ldg::Q; using fr_ldg::N; using fr_ldg::F;
template<bool Local,class Output=fr_output::Raw>
inline void group(const double* input,const double* convection,double* output,int64_t elements,int group_id,
    const int64_t* offsets,const int64_t* primitive_ids,const int64_t* active_meta,const int64_t* owned_meta,
    const double* widths,const double* L,const double* R,const double* primal,const double* dual,
    const double* gl,const double* gr,double gamma,double mu,double kappa,double wall_temperature,
    double forcing,double* scratch,double* timing,Output sink={},double* diffusion_ratio=nullptr,fr_sgs::Config sgs={}) {
    const int g=group_id;
    int64_t os=offsets[g*3],as=offsets[g*3+1],bs=offsets[g*3+2];
    int no=offsets[(g+1)*3]-os,na=offsets[(g+1)*3+1]-as,nb=offsets[(g+1)*3+2]-bs;
    int ps=nb*N,gs=na*N,vs=no*N;
    double* primitive=scratch;
    double* gradient=primitive+4*ps;
    double* flux=gradient+12*gs;
    double* viscous=flux+4*gs;
    double* wall_work=viscous+4*vs;
    double t=timing ? fr_parallel::wall_time():0;
    auto tick=[&](int index) {
        if(timing) { double now=fr_parallel::wall_time(); timing[index]+=now-t; t=now; }
    };
    for(int e=0;e<nb;++e) {
        int64_t old=primitive_ids[bs+e];
        for(int n=0;n<N;n+=4) {
            int64_t at=old*N+n;
            Vec rho=simd::load(input+at);
            Vec u=simd::load(input+elements*N+at)/rho;
            Vec v=simd::load(input+2*elements*N+at)/rho;
            Vec w=simd::load(input+3*elements*N+at)/rho;
            Vec pressure=(gamma-1)*(simd::load(input+4*elements*N+at)-.5*rho*((u*u+v*v)+w*w));
            simd::store(primitive+e*N+n,u); simd::store(primitive+ps+e*N+n,v);
            simd::store(primitive+2*ps+e*N+n,w); simd::store(primitive+3*ps+e*N+n,pressure/rho);
        }
    }
    tick(0);
    auto gradient_sweep=[&](auto wall_tag) {
        for(int axis=0;axis<3;++axis) for(int e=0;e<na;++e) {
            const int64_t* meta=active_meta+(as+e)*8;
            bool lower=axis==1 && meta[6],upper=axis==1 && meta[7];
#if FR_NS_SPLIT_BOUNDARY
            if(bool(lower || upper)!=decltype(wall_tag)::value) continue;
#endif
            double scale=2/widths[primitive_ids[bs+e]*3+axis];
            for(int c=0;c<4;++c) {
                const double* field=primitive+c*ps+e*N;
                for(int f=0;f<F;f+=4) {
                    Vec lines[4],updated[4]; simd::load_lines(field,axis,f,lines);
                    Vec left=simd::trace(field,axis,f,L),right=simd::trace(field,axis,f,R);
                    Vec wall=fr_ns_boundary::primitive_value(c,wall_temperature);
                    Vec common_l=lower ? wall:.5*(left+simd::trace(primitive+c*ps+meta[axis]*N,axis,f,R));
                    Vec common_r=upper ? wall:.5*(right+simd::trace(primitive+c*ps+meta[axis+3]*N,axis,f,L));
                    for(int i=0;i<Q;++i) {
                        Vec value=0.;
                        for(int j=0;j<Q;++j) value=value+lines[j]*(primal[i*Q+j]+gl[i]*L[j]);
                        updated[i]=scale*(value+gl[i]*(common_l-left)+gr[i]*(common_r-right));
                    }
                    simd::store_lines(gradient+(axis*4+c)*gs+e*N,axis,f,updated);
                }
            }
        }
    };
    gradient_sweep(std::false_type{});
#if FR_NS_SPLIT_BOUNDARY
    gradient_sweep(std::true_type{});
#endif
    tick(1);
    for(int axis=0;axis<3;++axis) {
        for(int e=0;e<na;++e) for(int n=0;n<N;n+=4) {
            int at=e*N+n;
            Vec tr=(simd::load(gradient+at)+simd::load(gradient+5*gs+at))+simd::load(gradient+10*gs+at);
            Vec stress[3];
            for(int c=0;c<3;++c) {
                Vec correction=c==axis ? (2./3.)*tr:Vec(0.);
                stress[c]=mu*(simd::load(gradient+(axis*4+c)*gs+at)+simd::load(gradient+(c*4+axis)*gs+at)-correction);
                simd::store(flux+c*gs+at,stress[c]);
            }
            Vec work=((Vec(0.)+simd::load(primitive+e*N+n)*stress[0])+simd::load(primitive+ps+e*N+n)*stress[1])+
                       simd::load(primitive+2*ps+e*N+n)*stress[2];
            simd::store(flux+3*gs+at,work+kappa*simd::load(gradient+(axis*4+3)*gs+at));
            if(axis==1 && e<no) simd::store(wall_work+e*N+n,work);
        }
        tick(2);
        auto divergence_sweep=[&](auto wall_tag) {
            for(int e=0;e<no;++e) {
                const int64_t* meta=owned_meta+(os+e)*6;
                const int64_t* flags=active_meta+(as+e)*8;
                bool lower=axis==1 && flags[6],upper=axis==1 && flags[7];
#if FR_NS_SPLIT_BOUNDARY
                if(bool(lower || upper)!=decltype(wall_tag)::value) continue;
#endif
                double h=widths[primitive_ids[bs+e]*3+axis],scale=2/h;
                double hl=lower ? h:std::min(h,widths[primitive_ids[bs+meta[axis]]*3+axis]);
                double hr=upper ? h:std::min(h,widths[primitive_ids[bs+meta[axis+3]]*3+axis]);
                for(int c=0;c<4;++c) {
                    const double* field=flux+c*gs+e*N;
                    const double* state=primitive+c*ps+e*N;
                    double coefficient=(c==3 ? kappa:mu)*FR_NS_PENALTY*Q*Q;
                    for(int f=0;f<F;f+=4) {
                        Vec lines[4],previous[4],updated[4]; simd::load_lines(field,axis,f,lines);
                        Vec fl=simd::trace(field,axis,f,L),fr=simd::trace(field,axis,f,R);
                        Vec wl=simd::trace(state,axis,f,L),wr=simd::trace(state,axis,f,R);
                        Vec wall=fr_ns_boundary::primitive_value(c,wall_temperature);
                        Vec cl=lower ? fl+(coefficient/hl)*(wl-wall):
                            .5*(fl+simd::trace(flux+c*gs+meta[axis]*N,axis,f,R))+
                            (coefficient/hl)*(wl-simd::trace(primitive+c*ps+meta[axis]*N,axis,f,R));
                        Vec cr=upper ? fr+(coefficient/hr)*(wall-wr):
                            .5*(fr+simd::trace(flux+c*gs+meta[axis+3]*N,axis,f,L))+
                            (coefficient/hr)*(simd::trace(primitive+c*ps+meta[axis+3]*N,axis,f,L)-wr);
                        double* dst=viscous+c*vs+e*N;
                        if(axis==0) for(auto& value:previous) value=0.;
                        else simd::load_lines(dst,axis,f,previous);
                        for(int i=0;i<Q;++i) {
                            Vec value=0.;
                            for(int j=0;j<Q;++j) value=value+lines[j]*(primal[i*Q+j]+gl[i]*L[j]);
                            updated[i]=previous[i]+scale*(value+gl[i]*(cl-fl)+gr[i]*(cr-fr));
                        }
                        simd::store_lines(dst,axis,f,updated);
                    }
                }
            }
        };
        divergence_sweep(std::false_type{});
#if FR_NS_SPLIT_BOUNDARY
        divergence_sweep(std::true_type{});
#endif
        tick(3);
    }
    for(int e=0;e<no;++e) {
        const int64_t* flags=active_meta+(as+e)*8;
        if(flags[6] || flags[7]) {
            double scale=2/widths[primitive_ids[bs+e]*3+1];
            for(int f=0;f<F;f+=4) {
                Vec values[4]; double* dst=viscous+3*vs+e*N;
                simd::load_lines(dst,1,f,values);
                Vec left=simd::trace(wall_work+e*N,1,f,L),right=simd::trace(wall_work+e*N,1,f,R);
                for(int i=0;i<Q;++i) values[i]=fr_ns_boundary::stationary_wall_energy(values[i],scale,
                    gl[i],gr[i],left,right,flags[6],flags[7]);
                simd::store_lines(dst,1,f,values);
            }
        }
    }
    for(int e=0;e<no;++e) {
        int64_t old=primitive_ids[bs+e];
        const double* conv=convection+(Local ? e*N:old*N);
        int64_t stride=Local ? vs:elements*N;
        for(int n=0;n<N;n+=4) {
            int64_t at=old*N+n;
            sink.template put<0>(output,elements*N,at,e*N+n,simd::load(conv+n));
            sink.template put<1>(output,elements*N,at,e*N+n,simd::load(conv+stride+n)+simd::load(viscous+e*N+n)+forcing);
            sink.template put<2>(output,elements*N,at,e*N+n,simd::load(conv+2*stride+n)+simd::load(viscous+vs+e*N+n));
            sink.template put<3>(output,elements*N,at,e*N+n,simd::load(conv+3*stride+n)+simd::load(viscous+2*vs+e*N+n));
            sink.template put<4>(output,elements*N,at,e*N+n,simd::load(conv+4*stride+n)+simd::load(viscous+3*vs+e*N+n)+forcing*simd::load(primitive+e*N+n));
        }
    }
    tick(4);
}
}
