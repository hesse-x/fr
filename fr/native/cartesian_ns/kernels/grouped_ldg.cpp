#include "viscous_kernel.hpp"
#include "scratch.hpp"
#include <mutex>
extern "C" void grouped_ldg(
    const double* input,const double* convection,double* output,int64_t elements,
    const int64_t* offsets,const int64_t* primitive_ids,
    const int64_t* active_meta,const int64_t* owned_meta,
    const double* widths,const double* L,const double* R,
    const double* primal,const double* dual,const double* gl,const double* gr,
    double gamma,double mu,double kappa,double wall_temperature,double forcing,
    int groups,int max_owned,int max_active,int max_primitive,
    double* timing) {
    using namespace fr_viscous;
    if(timing) std::fill_n(timing,5,0.);
    std::mutex timing_mutex;
    at::parallel_for(0, groups, 1, [&](int64_t begin, int64_t end) {
        const int capacity=(4*max_primitive+16*max_active+5*max_owned)*N;
        auto& scratch=fr_scratch::buffer(capacity,1);
        double local[5]={};
        for(int64_t g=begin;g<end;++g) {
            fr_viscous::group<false>(input,convection,output,elements,g,offsets,primitive_ids,
                active_meta,owned_meta,widths,L,R,primal,dual,gl,gr,gamma,mu,kappa,
                wall_temperature,forcing,scratch.data(),timing ? local:nullptr);
        }
        if(timing) {
            std::lock_guard<std::mutex> lock(timing_mutex);
            for(int i=0;i<5;++i) timing[i]+=local[i];
        }
    });
}
