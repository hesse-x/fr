// Fused FP64 SoA p=3 utilities: FMA contraction enabled, no fast math.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include <omp.h>
#include "avx2.hpp"
using simd::Vec;

extern "C" double flux_force(const double* rhs,const double* weights,
                              const double* scales,int64_t cells,int threads) {
    std::vector<double> partial(threads,0.);
    #pragma omp parallel num_threads(threads)
    {
        Vec sum=0.;
        #pragma omp for schedule(static)
        for(int64_t e=0;e<cells;++e)
            for(int n=0;n<64;n+=4)
                sum=sum+(simd::load(rhs+cells*64+e*64+n)*simd::load(weights+n))*scales[e];
        double lane[4];
        simd::store(lane,sum);
        partial[omp_get_thread_num()]=((lane[0]+lane[1])+lane[2])+lane[3];
    }
    double sum=0.;
    for(int i=0;i<threads;++i) sum+=partial[i];
    return -sum;
}

extern "C" void flux_rk_update(const double* original,const double* stage,
                                const double* rhs,double* output,int64_t count,
                                double force,double coefficient,int combined,int threads) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int64_t at=0;at<count;at+=4) {
        Vec u=simd::load(stage+count+at)/simd::load(stage+at);
        for(int c=0;c<5;++c) {
            Vec derivative=simd::load(rhs+c*count+at);
            if(c==1) derivative=derivative+force;
            if(c==4) derivative=derivative+force*u;
            Vec value=simd::load(stage+c*count+at);
            if(combined) value=simd::load(original+c*count+at)*(2./3.)+value*(1./3.);
            simd::store(output+c*count+at,value+coefficient*derivative);
        }
    }
}

extern "C" void flux_rk_finalize(const double* original,const double* stage,
    const double* pending,double* output,int64_t count,double force,double coefficient,
    int combined,int threads) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int64_t at=0;at<count;at+=4) {
        Vec u=simd::load(stage+count+at)/simd::load(stage+at);
        Vec momentum=simd::load(pending+count+at)+force;
        Vec energy=simd::load(pending+at)+force*u;
        Vec m=simd::load(stage+count+at),e=simd::load(stage+4*count+at);
        if(combined) {
            m=simd::load(original+count+at)*(2./3.)+m*(1./3.);
            e=simd::load(original+4*count+at)*(2./3.)+e*(1./3.);
        }
        simd::store(output+count+at,m+coefficient*momentum);
        simd::store(output+4*count+at,e+coefficient*energy);
    }
}

extern "C" void state_scan(const double* state,const double* widths,int64_t cells,
                            double gamma,int threads,double* result) {
    double minimum=std::numeric_limits<double>::infinity(),maximum=0.;
    int valid=1;
    #pragma omp parallel num_threads(threads) reduction(min:minimum) reduction(max:maximum) reduction(&:valid)
    {
        Vec rho_min=std::numeric_limits<double>::infinity(),rate_max=0.;
        #pragma omp for schedule(static)
        for(int64_t e=0;e<cells;++e) {
            for(int n=0;n<64;n+=4) {
                int64_t at=e*64+n,count=cells*64;
                Vec rho=simd::load(state+at),mx=simd::load(state+count+at);
                Vec my=simd::load(state+2*count+at),mz=simd::load(state+3*count+at);
                Vec energy=simd::load(state+4*count+at);
                Vec u=mx/rho,v=my/rho,w=mz/rho;
                Vec pressure=(gamma-1)*(energy-.5*rho*((u*u+v*v)+w*w));
                __m256d good=_mm256_and_pd(_mm256_cmp_pd(rho.value,_mm256_setzero_pd(),_CMP_GT_OQ),
                                         _mm256_cmp_pd(pressure.value,_mm256_setzero_pd(),_CMP_GT_OQ));
                for(Vec field : {rho,mx,my,mz,energy})
                    good=_mm256_and_pd(good,_mm256_cmp_pd(simd::abs(field).value,
                                            _mm256_set1_pd(std::numeric_limits<double>::max()),_CMP_LE_OQ));
                valid &= _mm256_movemask_pd(good)==15;
                Vec sound=simd::sqrt(gamma*pressure/rho);
                Vec rate=(Vec(0.)+(simd::abs(u)+sound)/widths[e*3]);
                rate=rate+(simd::abs(v)+sound)/widths[e*3+1];
                rate=rate+(simd::abs(w)+sound)/widths[e*3+2];
                rho_min=Vec(_mm256_min_pd(rho_min.value,rho.value));
                rate_max=simd::maximum(rate_max,rate);
            }
        }
        double r[4],a[4];
        simd::store(r,rho_min); simd::store(a,rate_max);
        for(int i=0;i<4;++i) { minimum=std::min(minimum,r[i]); maximum=std::max(maximum,a[i]); }
    }
    result[0]=minimum; result[1]=maximum; result[2]=valid;
}

namespace {
struct Primitive {
    Vec rho,u,v,w,pressure,temperature;
};
inline Primitive primitives(const double* state,int64_t count,int64_t at,double gamma) {
    Primitive p;
    p.rho=simd::load(state+at);
    p.u=simd::load(state+count+at)/p.rho;
    p.v=simd::load(state+2*count+at)/p.rho;
    p.w=simd::load(state+3*count+at)/p.rho;
    p.pressure=(gamma-1)*(simd::load(state+4*count+at)-.5*p.rho*((p.u*p.u+p.v*p.v)+p.w*p.w));
    p.temperature=p.pressure/p.rho;
    return p;
}
inline void accumulate(double* dst,Vec previous,Vec current,double weight) {
    Vec value=simd::load(dst)+previous*weight;
    simd::store(dst,value+current*weight);
}
}

extern "C" void mean_accumulate(const double* previous,const double* current,
                                 double* integral,int64_t count,double gamma,
                                 double weight,int threads) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int64_t at=0;at<count;at+=4) {
        const Primitive a=primitives(previous,count,at,gamma);
        const Primitive b=primitives(current,count,at,gamma);
        // Match Channel.field_moments order and two sequential trapezoidal adds.
        // The only persistent statistics buffer is the SoA time integral.
        accumulate(integral+at, a.rho,b.rho,weight);
        accumulate(integral+count+at, a.u,b.u,weight);
        accumulate(integral+2*count+at, a.v,b.v,weight);
        accumulate(integral+3*count+at, a.w,b.w,weight);
        accumulate(integral+4*count+at, a.pressure,b.pressure,weight);
        accumulate(integral+5*count+at, a.temperature,b.temperature,weight);
        accumulate(integral+6*count+at, a.u*a.u,b.u*b.u,weight);
        accumulate(integral+7*count+at, a.v*a.v,b.v*b.v,weight);
        accumulate(integral+8*count+at, a.w*a.w,b.w*b.w,weight);
        accumulate(integral+9*count+at, a.u*a.v,b.u*b.v,weight);
        accumulate(integral+10*count+at, a.u*a.w,b.u*b.w,weight);
        accumulate(integral+11*count+at, a.v*a.w,b.v*b.w,weight);
    }
}

namespace {
template<bool Sample>
void finish_impl(const double* original,const double* stage,const double* pending,
    double* output,const double* widths,double* integral,int64_t cells,
    double force,double coefficient,int combined,double gamma,double weight,
    int threads,double* result) {
    int64_t count=cells*64;
    double minimum=std::numeric_limits<double>::infinity(),maximum=0.;
    int valid=1;
    #pragma omp parallel num_threads(threads) reduction(min:minimum) reduction(max:maximum) reduction(&:valid)
    {
        Vec rho_min=std::numeric_limits<double>::infinity(),rate_max=0.;
        #pragma omp for schedule(static)
        for(int64_t cell=0;cell<cells;++cell) for(int n=0;n<64;n+=4) {
            int64_t at=cell*64+n;
            Vec stage_u=simd::load(stage+count+at)/simd::load(stage+at);
            Vec dm=simd::load(pending+count+at)+force;
            Vec de=simd::load(pending+at)+force*stage_u;
            Vec m=simd::load(stage+count+at),e=simd::load(stage+4*count+at);
            if(combined) {
                m=simd::load(original+count+at)*(2./3.)+m*(1./3.);
                e=simd::load(original+4*count+at)*(2./3.)+e*(1./3.);
            }
            Vec mx=m+coefficient*dm,energy=e+coefficient*de;
            simd::store(output+count+at,mx);
            simd::store(output+4*count+at,energy);
            Vec rho=simd::load(output+at),my=simd::load(output+2*count+at),mz=simd::load(output+3*count+at);
            Vec u=mx/rho,v=my/rho,w=mz/rho;
            Vec pressure=(gamma-1)*(energy-.5*rho*((u*u+v*v)+w*w));
            __m256d good=_mm256_and_pd(_mm256_cmp_pd(rho.value,_mm256_setzero_pd(),_CMP_GT_OQ),
                                     _mm256_cmp_pd(pressure.value,_mm256_setzero_pd(),_CMP_GT_OQ));
            for(Vec field : {rho,mx,my,mz,energy})
                good=_mm256_and_pd(good,_mm256_cmp_pd(simd::abs(field).value,
                    _mm256_set1_pd(std::numeric_limits<double>::max()),_CMP_LE_OQ));
            valid &= _mm256_movemask_pd(good)==15;
            Vec sound=simd::sqrt(gamma*pressure/rho);
            Vec rate=Vec(0.)+(simd::abs(u)+sound)/widths[cell*3];
            rate=rate+(simd::abs(v)+sound)/widths[cell*3+1];
            rate=rate+(simd::abs(w)+sound)/widths[cell*3+2];
            rho_min=Vec(_mm256_min_pd(rho_min.value,rho.value));
            rate_max=simd::maximum(rate_max,rate);
            if constexpr(Sample) {
                const Primitive a=primitives(original,count,at,gamma);
                accumulate(integral+at,a.rho,rho,weight);
                accumulate(integral+count+at,a.u,u,weight);
                accumulate(integral+2*count+at,a.v,v,weight);
                accumulate(integral+3*count+at,a.w,w,weight);
                accumulate(integral+4*count+at,a.pressure,pressure,weight);
                accumulate(integral+5*count+at,a.temperature,pressure/rho,weight);
                accumulate(integral+6*count+at,a.u*a.u,u*u,weight);
                accumulate(integral+7*count+at,a.v*a.v,v*v,weight);
                accumulate(integral+8*count+at,a.w*a.w,w*w,weight);
                accumulate(integral+9*count+at,a.u*a.v,u*v,weight);
                accumulate(integral+10*count+at,a.u*a.w,u*w,weight);
                accumulate(integral+11*count+at,a.v*a.w,v*w,weight);
            }
        }
        double r[4],a[4];
        simd::store(r,rho_min); simd::store(a,rate_max);
        for(int i=0;i<4;++i) { minimum=std::min(minimum,r[i]); maximum=std::max(maximum,a[i]); }
    }
    result[0]=minimum; result[1]=maximum; result[2]=valid;
}
}

extern "C" void flux_rk_finish(const double* original,const double* stage,const double* pending,
    double* output,const double* widths,double* integral,int64_t cells,
    double force,double coefficient,int combined,double gamma,double weight,
    int threads,double* result) {
    if(integral) finish_impl<true>(original,stage,pending,output,widths,integral,cells,
                                 force,coefficient,combined,gamma,weight,threads,result);
    else finish_impl<false>(original,stage,pending,output,widths,integral,cells,
                           force,coefficient,combined,gamma,weight,threads,result);
}
