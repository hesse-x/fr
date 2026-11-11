#include "convection_kernel.hpp"
#include "scratch.hpp"
#include <mutex>
extern "C" void grouped_convection(
    const double* input, double* output, int64_t elements,
    const int64_t* group_offsets, const int64_t* cells,
    const int64_t* face_offsets, const int64_t* faces, const int64_t* cell_faces,
    const double* widths, const double* L, const double* R, const double* matrix,
    const double* gl, const double* gr, const double* reference, double gamma,
    int groups, int max_cells, int max_faces, double* timing) {
    using namespace fr_convection;
    if(timing) std::fill_n(timing,4,0.);
    std::mutex timing_mutex;
    at::parallel_for(0, groups, 1, [&](int64_t begin, int64_t end) {
        const int capacity=fr_convection::scratch_capacity(max_cells,max_faces);
        auto& scratch=fr_scratch::buffer(capacity,0);
        double local[4]={};
        for(int64_t g=begin;g<end;++g) {
            auto value=fr_convection::group(input,elements,g,group_offsets,cells,
                face_offsets,faces,cell_faces,widths,L,R,matrix,gl,gr,reference,gamma,
                scratch.data(),timing ? local:nullptr);
            double started=timing ? fr_parallel::wall_time():0.;
            fr_convection::scatter(value.result,output,elements,g,group_offsets,cells);
            if(timing) local[3]+=fr_parallel::wall_time()-started;
        }
        if(timing) {
            std::lock_guard<std::mutex> lock(timing_mutex);
            for(int i=0;i<4;++i) timing[i]+=local[i];
        }
    });
}
