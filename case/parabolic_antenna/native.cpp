// Constant-coefficient Maxwell FR. Python owns all FP64 operators and geometry.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <omp.h>

struct Solver {
    int nx, ny, nz, n, cells, nodes, threads;
    std::vector<double> scale;
    int boundary_kind;
    double voltage_amplitude;
    std::vector<int64_t> feed_ids;
    std::vector<double> feed_coeff, feed_lift;
    std::vector<double> D,L,R,gl,gr,rate,drive,physical,traces,faces,stage,k,acc;
    std::vector<uint8_t> solid;
    Solver(int x,int y,int z,int q,int nt,const double* h,const double* op,
           const double* damping,const double* source,const uint8_t* mask,
           int boundary,const int64_t* ids,const double* coeff,const double* lift,int feed_count,double amplitude)
      :nx(x),ny(y),nz(z),n(q),cells(x*y*z),nodes(q*q*q),threads(nt),
       scale(cells*3),boundary_kind(boundary),voltage_amplitude(amplitude),
       feed_ids(feed_count),feed_coeff(feed_count),feed_lift(feed_count),D(op,op+q*q), L(op+q*q,op+q*q+q), R(op+q*q+q,op+q*q+2*q),
       gl(op+q*q+2*q,op+q*q+3*q), gr(op+q*q+3*q,op+q*q+4*q),
       rate(damping,damping+cells*nodes*3),drive(source,source+cells*nodes),
       physical(cells*nodes*6),traces(cells*6*q*q*6),faces(traces.size()),
       stage(cells*nodes*18),k(stage.size()),acc(stage.size()),solid(mask,mask+cells) {
        for(int e=0;e<cells;++e) for(int a=0;a<3;++a) scale[e*3+a]=2/h[e*3+a];
        if(feed_count) {
            std::copy(ids,ids+feed_count,feed_ids.begin());
            std::copy(coeff,coeff+feed_count,feed_coeff.begin());
            std::copy(lift,lift+feed_count,feed_lift.begin());
        }
    }
    int point(int i,int j,int l) const { return (i*n+j)*n+l; }
    int line(int a,int v,int b,int c) const {
        return a==0 ? point(v,b,c) : a==1 ? point(b,v,c) : point(b,c,v);
    }
    size_t face(int e,int a,int s,int b,int c) const {
        return (((size_t(e)*3+a)*2+s)*n*n+b*n+c)*6;
    }
    // F_E=-axis cross H; F_H=axis cross E.
    static double flux(const double* u,int a,int v) {
        const int j=(a+1)%3, l=(a+2)%3;
        if(v==j) return u[3+l];
        if(v==l) return -u[3+j];
        if(v==3+j) return -u[l];
        if(v==3+l) return u[j];
        return 0.;
    }
    void rhs(const double* u,double* out,double t) {
        if(n==4) rhs_impl<4>(u,out,t); else if(n==3) rhs_impl<3>(u,out,t); else rhs_impl<0>(u,out,t);
        project(out,signal(t,true));
    }
    template<int Q> void rhs_impl(const double* u,double* out,double t) {
        const int n=Q?Q:this->n;
        auto point_local=[=](int i,int j,int l) { return (i*n+j)*n+l; };
        auto line_local=[=](int a,int v,int b,int c) {
            return a==0?point_local(v,b,c):a==1?point_local(b,v,c):point_local(b,c,v);
        };
        auto face_local=[=](int e,int a,int s,int b,int c) -> size_t {
            return (((size_t(e)*3+a)*2+s)*n*n+b*n+c)*6;
        };
        const int total=cells*nodes;
        #pragma omp parallel for num_threads(threads) schedule(static)
        for(int p=0;p<total;++p)
            for(int v=0;v<6;++v)
                physical[size_t(p)*6+v]=u[size_t(p)*18+v]+u[size_t(p)*18+6+v]+u[size_t(p)*18+12+v];
        #pragma omp parallel for num_threads(threads) schedule(static)
        for(int e=0;e<cells;++e) {
            for(int a=0;a<3;++a) for(int s=0;s<2;++s)
                for(int b=0;b<n;++b) for(int c=0;c<n;++c) {
                    double* tr=&traces[face_local(e,a,s,b,c)];
                    for(int v=0;v<6;++v) tr[v]=0.;
                    const auto& vector=s?R:L;
                    for(int q=0;q<n;++q) {
                        const double* p=&physical[(size_t(e)*nodes+line_local(a,q,b,c))*6];
                        for(int v=0;v<6;++v) tr[v]+=vector[q]*p[v];
                    }
                }
        }
        #pragma omp parallel for num_threads(threads) schedule(static)
        for(int e=0;e<cells;++e) {
            const int location[3]={e/(ny*nz),(e/nz)%ny,e%nz};
            const int sizes[3]={nx,ny,nz}, strides[3]={ny*nz,nz,1};
            for(int a=0;a<3;++a) for(int s=0;s<2;++s) {
                const int neighbor=e+(s?1:-1)*strides[a];
                const bool boundary=s ? location[a]==sizes[a]-1 : location[a]==0;
                const bool pec=boundary || (!boundary && solid[neighbor]);
                for(int b=0;b<n;++b) for(int c=0;c<n;++c) {
                    const double* own=&traces[face_local(e,a,s,b,c)];
                    double ghost[6];
                    const double* other;
                    if(boundary && boundary_kind==1) {
                        std::fill(ghost,ghost+6,0.);
                        other=ghost; // Incoming characteristic zero: Silver-Muller.
                    } else if(pec) {
                        for(int v=0;v<6;++v)
                            ghost[v]=((v<3 && v!=a) || v==a+3) ? -own[v] : own[v];
                        other=ghost;
                    } else other=&traces[face_local(neighbor,a,1-s,b,c)];
                    double* f=&faces[face_local(e,a,s,b,c)];
                    const double* minus=s?own:other;
                    const double* plus=s?other:own;
                    for(int v=0;v<6;++v)
                        f[v]=.5*(flux(minus,a,v)+flux(plus,a,v))
                             -.5*((v%3==a)?0.:plus[v]-minus[v]);
                }
            }
        }
        const double ramp=std::pow(std::sin(1.57079632679489661923*std::min(t/2.,1.)),2);
        const double excitation=ramp*std::cos(6.28318530717958647693*t)/3.;
        #pragma omp parallel for num_threads(threads) schedule(static)
        for(int e=0;e<cells;++e) {
            if(solid[e]) {
                std::fill(out+size_t(e)*nodes*18,out+size_t(e+1)*nodes*18,0.);
                continue;
            }
            for(int i=0;i<n;++i) for(int j=0;j<n;++j) for(int l=0;l<n;++l) {
                const int index[3]={i,j,l};
                const size_t p=size_t(e)*nodes+point_local(i,j,l);
                for(int a=0;a<3;++a) {
                    const int b=a==0?j:i, c=a==2?j:l;
                    const int q=index[a];
                    const double* fl=&traces[face_local(e,a,0,b,c)];
                    const double* fr=&traces[face_local(e,a,1,b,c)];
                    const double* common_l=&faces[face_local(e,a,0,b,c)];
                    const double* common_r=&faces[face_local(e,a,1,b,c)];
                    double derivative[6]={0.,0.,0.,0.,0.,0.};
                    for(int m=0;m<n;++m) {
                        const double coefficient=D[q*n+m];
                        const double* value=&physical[(size_t(e)*nodes+line_local(a,m,b,c))*6];
                        for(int v=0;v<6;++v) derivative[v]+=coefficient*value[v];
                    }
                    const size_t offset=p*18+a*6;
                    for(int v=0;v<6;++v)
                        out[offset+v]=-rate[p*3+a]*u[offset+v]-(v==2?excitation*drive[p]:0.);
                    const int j=(a+1)%3, l=(a+2)%3;
                    const double factor=scale[e*3+a];
                    out[offset+j]-=factor*(derivative[3+l]+gl[q]*(common_l[j]-fl[3+l])+gr[q]*(common_r[j]-fr[3+l]));
                    out[offset+l]-=factor*(-derivative[3+j]+gl[q]*(common_l[l]+fl[3+j])+gr[q]*(common_r[l]+fr[3+j]));
                    out[offset+3+j]-=factor*(-derivative[l]+gl[q]*(common_l[3+j]+fl[l])+gr[q]*(common_r[3+j]+fr[l]));
                    out[offset+3+l]-=factor*(derivative[j]+gl[q]*(common_l[3+l]-fl[j])+gr[q]*(common_r[3+l]-fr[j]));
                }
            }
        }
    }
    double line_voltage(const double* u) const {
        double value=0.;
        for(size_t i=0;i<feed_ids.size();++i) {
            const size_t p=size_t(feed_ids[i])*18;
            value+=feed_coeff[i]*(u[p+2]+u[p+8]+u[p+14]);
        }
        return value;
    }
    double signal(double t,bool derivative=false) const {
        const double pi=3.14159265358979323846;
        const double ramp=std::pow(std::sin(pi*std::min(t/2.,1.)/2),2);
        const double dramp=t>=0. && t<2. ? pi/4*std::sin(pi*t/2) : 0.;
        return voltage_amplitude*(derivative ? dramp*std::cos(2*pi*t)-2*pi*ramp*std::sin(2*pi*t) : ramp*std::cos(2*pi*t));
    }
    void project(double* u,double target) const {
        if(feed_ids.empty()) return;
        const double delta=target-line_voltage(u);
        for(size_t i=0;i<feed_ids.size();++i) {
            const size_t p=size_t(feed_ids[i])*18;
            const double change=delta*feed_lift[i]/3;
            u[p+2]+=change; u[p+8]+=change; u[p+14]+=change;
        }
    }
    void step(const double* u,double* out,double t,double dt) {
        const size_t count=stage.size();
        if(!feed_ids.empty()) {
            std::copy(u,u+count,stage.data());
            project(stage.data(),signal(t));
        }
        for(int s=0;s<4;++s) {
            rhs((s||!feed_ids.empty())?stage.data():u,k.data(),t+(s==0?0.:s==3?dt:dt/2));
            const double weight=(s==0||s==3)?1.:2.;
            const double factor=s==2?dt:dt/2;
            #pragma omp parallel for num_threads(threads) schedule(static)
            for(size_t p=0;p<count;++p) {
                if(s==0) acc[p]=k[p]; else acc[p]+=weight*k[p];
                if(s<3) stage[p]=u[p]+factor*k[p];
                else out[p]=u[p]+dt/6*acc[p];
            }
            if(s<3) project(stage.data(),signal(t+(s==2?dt:dt/2)));
            else project(out,signal(t+dt));
        }
    }
};
extern "C" {
void* antenna_create(int x,int y,int z,int n,int threads,const double* h,const double* op,
                     const double* rate,const double* drive,const uint8_t* solid,
                     int boundary,const int64_t* ids,const double* coeff,const double* lift,int feed_count,double amplitude) {
    try { return new Solver(x,y,z,n,threads,h,op,rate,drive,solid,boundary,ids,coeff,lift,feed_count,amplitude); } catch(...) { return nullptr; }
}
double antenna_voltage(void* ptr,const double* u) { return static_cast<Solver*>(ptr)->line_voltage(u); }
void antenna_destroy(void* ptr) { delete static_cast<Solver*>(ptr); }
void antenna_rhs(void* ptr,const double* u,double* out,double t) { static_cast<Solver*>(ptr)->rhs(u,out,t); }
void antenna_step(void* ptr,const double* u,double* out,double t,double dt) { static_cast<Solver*>(ptr)->step(u,out,t,dt); }
}
