// Convective residual stays inside the worker's group scratch, never in global U-sized storage.
#include "convection_kernel.hpp"
#include "viscous_kernel.hpp"
#include "scratch.hpp"
#include <mutex>

template<bool RK>
static void fused_impl(const double* input,double* output,int64_t elements,
    const void* const* conv,const void* const* ldg,const double* constants,
    int groups,int max_cells,int max_faces,int max_owned,int max_active,int max_primitive,
    double* timing,const double* original=nullptr,double* pending=nullptr,
    double coefficient=0.,int combined=0) {
    if(timing) std::fill_n(timing,8,0.);
    std::mutex timing_mutex;
    at::parallel_for(0, groups, 1, [&](int64_t begin, int64_t end) {
        int conv_capacity=fr_convection::scratch_capacity(max_cells,max_faces);
        int ldg_capacity=(4*max_primitive+(16+fr_sgs::extra_fields)*max_active+5*max_owned)*64;
        auto& conv_scratch=fr_scratch::buffer(conv_capacity,0);
        auto& ldg_scratch=fr_scratch::buffer(ldg_capacity,1);
        double local[8]={};
        for(int64_t g=begin;g<end;++g) {
            auto value=fr_convection::group(input,elements,g,
                static_cast<const int64_t*>(conv[0]),static_cast<const int64_t*>(conv[1]),
                static_cast<const int64_t*>(conv[2]),static_cast<const int64_t*>(conv[3]),
                static_cast<const int64_t*>(conv[4]),static_cast<const double*>(conv[5]),
                static_cast<const double*>(conv[6]),static_cast<const double*>(conv[7]),
                static_cast<const double*>(conv[8]),static_cast<const double*>(conv[9]),
                static_cast<const double*>(conv[10]),static_cast<const double*>(conv[11]),
                constants[0],conv_scratch.data(),timing ? local:nullptr,
                static_cast<const double*>(conv[12]),static_cast<const double*>(conv[13]));
            auto finish=[&](auto sink) {
            fr_viscous::group<true,decltype(sink)>(input,value.result,output,elements,g,
                static_cast<const int64_t*>(ldg[0]),static_cast<const int64_t*>(ldg[1]),
                static_cast<const int64_t*>(ldg[2]),static_cast<const int64_t*>(ldg[3]),
                static_cast<const double*>(ldg[4]),static_cast<const double*>(ldg[5]),
                static_cast<const double*>(ldg[6]),static_cast<const double*>(ldg[7]),
                static_cast<const double*>(ldg[8]),static_cast<const double*>(ldg[9]),
                static_cast<const double*>(ldg[10]),constants[0],constants[1],constants[2],
                constants[3],constants[4],ldg_scratch.data(),timing ? local+3:nullptr,
                sink,nullptr,{constants[5],constants[6],constants[7]});
            };
            if constexpr(RK) finish(fr_output::PartialRK{original,value.packed,pending,
                                                        value.count*64,coefficient,combined});
            else finish(fr_output::Raw{});
        }
        if constexpr(RK) _mm_sfence();
        if(timing) {
            std::lock_guard<std::mutex> lock(timing_mutex);
            for(int i=0;i<8;++i) timing[i]+=local[i];
        }
    });
}

extern "C" void grouped_fused(const double* input,double* output,int64_t elements,
    const void* const* conv,const void* const* ldg,const double* constants,
    int groups,int max_cells,int max_faces,int max_owned,int max_active,int max_primitive,
    double* timing) {
    fused_impl<false>(input,output,elements,conv,ldg,constants,groups,max_cells,max_faces,
                      max_owned,max_active,max_primitive,timing);
}

extern "C" void grouped_fused_rk(const double* input,double* output,int64_t elements,
    const void* const* conv,const void* const* ldg,const double* constants,
    int groups,int max_cells,int max_faces,int max_owned,int max_active,int max_primitive,
    const double* original,double* pending,double coefficient,int combined) {
    fused_impl<true>(input,output,elements,conv,ldg,constants,groups,max_cells,max_faces,
                     max_owned,max_active,max_primitive,nullptr,original,pending,
                     coefficient,combined);
}
