#include "convection_kernel.hpp"
#include "scratch.hpp"
extern "C" void grouped_convection(
    const double* input, double* output, int64_t elements,
    const int64_t* group_offsets, const int64_t* cells,
    const int64_t* face_offsets, const int64_t* faces, const int64_t* cell_faces,
    const double* widths, const double* L, const double* R, const double* matrix,
    const double* gl, const double* gr, const double* reference, double gamma,
    int groups, int max_cells, int max_faces, int threads, double* timing) {
    using namespace fr_convection;
    if(timing) std::fill_n(timing,4,0.);
    #pragma omp parallel num_threads(threads)
    {
        const int capacity=fr_convection::scratch_capacity(max_cells,max_faces);
        auto& scratch=fr_scratch::buffer(capacity,0);
        double local[4]={};
        #pragma omp for schedule(static)
        for(int g=0;g<groups;++g) {
            auto value=fr_convection::group(input,elements,g,group_offsets,cells,
                face_offsets,faces,cell_faces,widths,L,R,matrix,gl,gr,reference,gamma,
                scratch.data(),timing ? local:nullptr);
            double started=timing ? omp_get_wtime():0.;
            fr_convection::scatter(value.result,output,elements,g,group_offsets,cells);
            if(timing) local[3]+=omp_get_wtime()-started;
        }
        if(timing) {
            #pragma omp critical
            for(int i=0;i<4;++i) timing[i]+=local[i];
        }
    }
}
