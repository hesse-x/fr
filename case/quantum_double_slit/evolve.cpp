// Tensor-product real symmetric Hamiltonian, Chebyshev exponential action.
// Python supplies CSR one-dimensional kinetic matrices and Bessel coefficients.
#include <complex>
#include <vector>
#include <algorithm>
#include <cstdint>
using C = std::complex<double>;
extern "C" void advance(int nx, int ny, const int64_t* xp, const int64_t* xi,
 const double* xv, const int64_t* yp, const int64_t* yi, const double* yv,
 const double* potential, double center, double radius, const C* coefficients,
 int order, int steps, C* state, int threads) {
 const long size = long(nx)*ny;
 std::vector<C> a(size), b(size), c(size), result(size);
 // One OpenMP team for the whole exponential. Each row finishes its
 // Hamiltonian, recurrence and accumulation together, before leaving cache.
 C *pa=a.data(), *pb=b.data(), *pc=c.data();
 #pragma omp parallel num_threads(threads) shared(pa,pb,pc)
 {
 for (int step=0; step<steps; ++step) {
   #pragma omp for schedule(static)
   for (long j=0;j<size;++j) pa[j]=state[j];
   for (int k=1;k<=order;++k) {
    const C* in=k==1 ? pa : pb;
    C* out=k==1 ? pb : pc;
    #pragma omp for schedule(static)
    for (int i=0; i<nx; ++i) {
     const long base=long(i)*ny;
     for (int j=0; j<ny; ++j) out[base+j]=(potential[base+j]-center)*in[base+j];
     for (long k=xp[i]; k<xp[i+1]; ++k) {
       const C* src=in+xi[k]*ny;
       for (int j=0; j<ny; ++j) out[base+j]+=xv[k]*src[j];
     }
     for (int j=0; j<ny; ++j) {
       C z=out[base+j];
       for (long k=yp[j]; k<yp[j+1]; ++k) z+=yv[k]*in[base+yi[k]];
       const long index=base+j;
       if (k==1) {
         out[index]=z/radius;
         result[index]=coefficients[0]*pa[index]+coefficients[1]*out[index];
       } else {
         out[index]=2.*z/radius-pa[index];
         result[index]+=coefficients[k]*out[index];
       }
     }
    }
    #pragma omp single
    { if (k>1) { std::swap(pa,pb); std::swap(pb,pc); } }
   }
   #pragma omp for schedule(static)
   for (long j=0;j<size;++j) state[j]=result[j];
 }
 }
}
