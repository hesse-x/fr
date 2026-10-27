// CPU FP64 curved-element BR1 Navier--Stokes and SSPRK(4,3).
// ATen owns every buffer; no raw-pointer Python ABI or external state.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {
using at::Tensor;
using Tensors = c10::List<Tensor>;

struct Plan {
  int64_t ne, ns, nq, nf, faces, edges;
  const double *B, *FB, *gx, *gy, *lift, *normal, *penalty, *far;
  const double *base, *h, *weight;
  const int64_t *ids, *neighbor, *tag;
  double mu, heat, diff;
  Plan(const Tensor& u, const Tensors& d, double viscosity, double pr,
       double diffusion): mu(viscosity), heat(viscosity*3.5/pr), diff(diffusion) {
    TORCH_CHECK(d.size()==14, "expected 14 mesh tensors");
    TORCH_CHECK(u.device().is_cpu() && u.scalar_type()==at::kDouble &&
                u.is_contiguous() && u.dim()==2 && u.size(1)==4,
                "state must be contiguous CPU float64 (ndof,4)");
    for (int i=0;i<14;++i) {
      auto t=d.get(i);
      TORCH_CHECK(t.device().is_cpu() && t.is_contiguous(), "mesh tensors must be contiguous CPU tensors");
      TORCH_CHECK(t.scalar_type()==((i>=9 && i<=11)?at::kLong:at::kDouble), "invalid mesh dtype");
    }
    auto b=d.get(0), f=d.get(1), g=d.get(2), n=d.get(5);
    TORCH_CHECK(b.dim()==2 && f.dim()==3 && g.dim()==3 && n.dim()==3, "invalid mesh ranks");
    ne=g.size(0); ns=b.size(1); nq=b.size(0); nf=f.size(1); faces=n.size(0); edges=f.size(0);
    TORCH_CHECK(ne>0 && ns>0 && nq>0 && nf>0 && (edges==3 || edges==4) && f.size(2)==ns &&
                g.size(1)==ns && g.size(2)==nq && n.size(1)==nf && n.size(2)==2 &&
                u.size(0)==ne*ns, "incompatible mesh dimensions");
    auto shape=[&](int i,std::initializer_list<int64_t> sizes) {
      TORCH_CHECK(d.get(i).sizes()==at::IntArrayRef(sizes), "invalid mesh shape at index ",i);
    };
    shape(3,{ne,ns,nq}); shape(4,{ne,ns,edges*nf}); shape(6,{faces,nf});
    shape(7,{4}); shape(8,{ne}); shape(9,{ne,edges}); shape(10,{faces,nf});
    shape(11,{faces}); shape(12,{ne}); shape(13,{ne,nq});
    B=b.const_data_ptr<double>(); FB=f.const_data_ptr<double>();
    gx=g.const_data_ptr<double>(); gy=d.get(3).const_data_ptr<double>();
    lift=d.get(4).const_data_ptr<double>(); normal=n.const_data_ptr<double>();
    penalty=d.get(6).const_data_ptr<double>(); far=d.get(7).const_data_ptr<double>();
    base=d.get(8).const_data_ptr<double>(); ids=d.get(9).const_data_ptr<int64_t>();
    neighbor=d.get(10).const_data_ptr<int64_t>(); tag=d.get(11).const_data_ptr<int64_t>();
    h=d.get(12).const_data_ptr<double>(); weight=d.get(13).const_data_ptr<double>();
  }
};

inline void primitive(const double* u,double* w) {
  w[0]=u[1]/u[0]; w[1]=u[2]/u[0];
  w[2]=.4*(u[3]-.5*(u[1]*w[0]+u[2]*w[1]))/u[0];
}
inline double pressure(const double* u) {
  return .4*(u[3]-.5*(u[1]*u[1]+u[2]*u[2])/u[0]);
}
inline bool positive(const double* u) {
  return std::isfinite(u[0]) && std::isfinite(u[1]) && std::isfinite(u[2]) &&
         std::isfinite(u[3]) && u[0]>0 && pressure(u)>0;
}
inline void stress(const double* g,double mu,double& xx,double& xy,double& yy) {
  double div=(2./3)*(g[0]+g[3]);
  xx=mu*(2*g[0]-div); xy=mu*(g[1]+g[2]); yy=mu*(2*g[3]-div);
}
inline void viscous_normal(const double* w,const double* g,double nx,double ny,
                           const Plan& p,double* f) {
  double xx,xy,yy; stress(g,p.mu,xx,xy,yy);
  f[0]=0; f[1]=xx*nx+xy*ny; f[2]=xy*nx+yy*ny;
  f[3]=w[0]*f[1]+w[1]*f[2]+p.heat*(g[4]*nx+g[5]*ny);
}
inline double euler_normal(const double* u,double nx,double ny,double* f) {
  double pr=pressure(u), un=(u[1]*nx+u[2]*ny)/u[0];
  for(int v=0;v<4;++v) f[v]=u[v]*un;
  f[1]+=pr*nx; f[2]+=pr*ny; f[3]+=pr*un;
  return std::abs(un)+std::sqrt(1.4*pr/u[0]);
}
inline void outer_state(const double* u,double nx,double ny,const Plan& p,double* r) {
  double w[3]; primitive(u,w);
  double un=w[0]*nx+w[1]*ny, unf=p.far[1]*nx+p.far[2]*ny;
  double rp=un+5*std::sqrt(1.4*w[2]), rm=unf-5;
  double ub=.5*(rp+rm), cb=.1*(rp-rm);
  double entropy=un<0?1/1.4:pressure(u)/std::pow(u[0],1.4);
  double rho=std::pow(cb*cb/(1.4*entropy),2.5), pr=entropy*std::pow(rho,1.4);
  double vx=(un<0?p.far[1]-unf*nx:w[0]-un*nx)+ub*nx;
  double vy=(un<0?p.far[2]-unf*ny:w[1]-un*ny)+ub*ny;
  r[0]=rho; r[1]=rho*vx; r[2]=rho*vy; r[3]=pr/.4+.5*rho*(vx*vx+vy*vy);
}

struct Workspace {
  Tensor w,l,wl,g,gl;
  Workspace(const Plan& p,const Tensor& u) {
    auto o=u.options(); w=at::empty({p.ne*p.ns,3},o);
    l=at::empty({p.faces*p.nf,4},o); wl=at::empty({p.faces*p.nf,3},o);
    g=at::empty({p.ne*p.ns,6},o); gl=at::empty({p.faces*p.nf,6},o);
  }
};

void rhs_kernel(const Tensor& U,Tensor& out,const Plan& p,Workspace& ws) {
  const double* u=U.const_data_ptr<double>(); double* result=out.mutable_data_ptr<double>();
  double *w=ws.w.mutable_data_ptr<double>(), *l=ws.l.mutable_data_ptr<double>();
  double *wl=ws.wl.mutable_data_ptr<double>(), *g=ws.g.mutable_data_ptr<double>();
  double *gl=ws.gl.mutable_data_ptr<double>();
  // Own-cell interpolation. All neighbor reads happen after this barrier.
  at::parallel_for(0,p.ne,16,[&](int64_t begin,int64_t end) {
    for(int64_t e=begin;e<end;++e) {
      for(int64_t i=0;i<p.ns;++i) primitive(u+(e*p.ns+i)*4,w+(e*p.ns+i)*3);
      for(int f=0;f<p.edges;++f) for(int64_t q=0;q<p.nf;++q) {
        int64_t k=p.ids[e*p.edges+f]*p.nf+q;
        std::fill_n(l+k*4,4,0.); std::fill_n(wl+k*3,3,0.);
        for(int64_t i=0;i<p.ns;++i) {
          double b=p.FB[(f*p.nf+q)*p.ns+i];
          for(int v=0;v<4;++v) l[k*4+v]+=b*u[(e*p.ns+i)*4+v];
          for(int v=0;v<3;++v) wl[k*3+v]+=b*w[(e*p.ns+i)*3+v];
        }
      }
    }
  });
  // BR1 gradient and its own-cell traces. Scratch is local to each worker.
  at::parallel_for(0,p.ne,16,[&](int64_t begin,int64_t end) {
    std::vector<double> qv(p.nq*3), trace(p.edges*p.nf*3);
    for(int64_t e=begin;e<end;++e) {
      std::fill(qv.begin(),qv.end(),0.);
      for(int64_t q=0;q<p.nq;++q) for(int64_t i=0;i<p.ns;++i)
        for(int v=0;v<3;++v) qv[q*3+v]+=p.B[q*p.ns+i]*w[(e*p.ns+i)*3+v];
      for(int f=0;f<p.edges;++f) for(int64_t q=0;q<p.nf;++q) {
        int64_t face=p.ids[e*p.edges+f], k=face*p.nf+q, other=p.neighbor[k];
        for(int v=0;v<3;++v) {
          double t=.5*(wl[k*3+v]+wl[other*3+v]);
          if(p.tag[face]==2) t=v<2?0:wl[k*3+v];
          if(p.tag[face]==1) t=v<2?p.far[v+1]:1/1.4;
          trace[(f*p.nf+q)*3+v]=t;
        }
      }
      for(int64_t i=0;i<p.ns;++i) for(int v=0;v<3;++v) for(int a=0;a<2;++a) {
        double value=0; const double* D=a==0?p.gx:p.gy;
        for(int64_t q=0;q<p.nq;++q) value-=D[(e*p.ns+i)*p.nq+q]*qv[q*3+v];
        for(int f=0;f<p.edges;++f) for(int64_t q=0;q<p.nf;++q) {
          int64_t k=p.ids[e*p.edges+f]*p.nf+q;
          value-=p.lift[(e*p.ns+i)*p.edges*p.nf+f*p.nf+q]*trace[(f*p.nf+q)*3+v]*p.normal[k*2+a];
        }
        g[(e*p.ns+i)*6+v*2+a]=value;
      }
      for(int f=0;f<p.edges;++f) for(int64_t q=0;q<p.nf;++q) {
        int64_t k=p.ids[e*p.edges+f]*p.nf+q;
        std::fill_n(gl+k*6,6,0.);
        for(int64_t i=0;i<p.ns;++i) for(int v=0;v<6;++v)
          gl[k*6+v]+=p.FB[(f*p.nf+q)*p.ns+i]*g[(e*p.ns+i)*6+v];
      }
    }
  });
  // Volume and face fluxes consumed immediately by the local derivative/lift.
  at::parallel_for(0,p.ne,16,[&](int64_t begin,int64_t end) {
    std::vector<double> fx(p.nq*4),fy(p.nq*4),common(p.edges*p.nf*4);
    for(int64_t e=begin;e<end;++e) {
      for(int64_t q=0;q<p.nq;++q) {
        double uq[4]={},wq[3]={},gq[6]={};
        for(int64_t i=0;i<p.ns;++i) {
          double b=p.B[q*p.ns+i];
          for(int v=0;v<4;++v) uq[v]+=b*u[(e*p.ns+i)*4+v];
          for(int v=0;v<3;++v) wq[v]+=b*w[(e*p.ns+i)*3+v];
          for(int v=0;v<6;++v) gq[v]+=b*g[(e*p.ns+i)*6+v];
        }
        euler_normal(uq,1,0,fx.data()+q*4); euler_normal(uq,0,1,fy.data()+q*4);
        double vx[4],vy[4]; viscous_normal(wq,gq,1,0,p,vx); viscous_normal(wq,gq,0,1,p,vy);
        for(int v=0;v<4;++v) {fx[q*4+v]-=vx[v]; fy[q*4+v]-=vy[v];}
      }
      for(int f=0;f<p.edges;++f) for(int64_t q=0;q<p.nf;++q) {
        int64_t face=p.ids[e*p.edges+f],k=face*p.nf+q,other=p.neighbor[k];
        double nx=p.normal[k*2],ny=p.normal[k*2+1],r[4],fl[4],fr[4],vl[4],vr[4];
        std::copy_n(l+other*4,4,r);
        if(p.tag[face]==1) outer_state(l+k*4,nx,ny,p,r);
        double speed=std::max(euler_normal(l+k*4,nx,ny,fl),euler_normal(r,nx,ny,fr));
        double* c=common.data()+(f*p.nf+q)*4;
        for(int v=0;v<4;++v) c[v]=.5*(fl[v]+fr[v])-speed*.5*(r[v]-l[k*4+v]);
        if(p.tag[face]==2) {
          double pr=pressure(l+k*4), un=(l[k*4+1]*nx+l[k*4+2]*ny)/l[k*4];
          double wp=pr+l[k*4]*un*un+l[k*4]*(std::abs(un)+std::sqrt(1.4*pr/l[k*4]))*un;
          c[0]=c[3]=0; c[1]=wp*nx; c[2]=wp*ny;
        }
        viscous_normal(wl+k*3,gl+k*6,nx,ny,p,vl);
        viscous_normal(wl+other*3,gl+other*6,nx,ny,p,vr);
        double jump[3];
        for(int v=0;v<3;++v) {
          jump[v]=wl[other*3+v]-wl[k*3+v];
          if(p.tag[face]==1) jump[v]=(v<2?p.far[v+1]:1/1.4)-wl[k*3+v];
          if(p.tag[face]==2) jump[v]=v<2?-wl[k*3+v]:0.;
        }
        double visc[4]; for(int v=0;v<4;++v) visc[v]=.5*(vl[v]+vr[v]);
        visc[1]+=p.penalty[k]*jump[0]; visc[2]+=p.penalty[k]*jump[1];
        visc[3]+=p.penalty[k]*(p.heat/p.mu)*jump[2];
        if(p.tag[face]==2) visc[3]=0.;
        for(int v=0;v<4;++v) c[v]-=visc[v];
      }
      for(int64_t i=0;i<p.ns;++i) for(int v=0;v<4;++v) {
        double value=0;
        for(int64_t q=0;q<p.nq;++q) value+=p.gx[(e*p.ns+i)*p.nq+q]*fx[q*4+v]+p.gy[(e*p.ns+i)*p.nq+q]*fy[q*4+v];
        for(int64_t f=0;f<p.edges*p.nf;++f) value+=p.lift[(e*p.ns+i)*p.edges*p.nf+f]*common[f*4+v];
        result[(e*p.ns+i)*4+v]=value;
      }
    }
  });
}

Tensor rhs(const Tensor& u,const Tensors& data,double mu,double pr,double diffusion) {
  Plan p(u,data,mu,pr,diffusion); Workspace ws(p,u); auto out=at::empty_like(u);
  rhs_kernel(u,out,p,ws); return out;
}

Tensor timestep(const Tensor& U,const Plan& p,double cfl,bool local) {
  auto dt=at::empty({p.ne,1,1},U.options()); auto d=dt.mutable_data_ptr<double>();
  auto u=U.const_data_ptr<double>();
  at::parallel_for(0,p.ne,16,[&](int64_t begin,int64_t end) {
    for(int64_t e=begin;e<end;++e) {
      double speed=0,rho=INFINITY;
      for(int64_t q=0;q<p.nq;++q) {
        double v[4]={};
        for(int64_t i=0;i<p.ns;++i) for(int a=0;a<4;++a) v[a]+=p.B[q*p.ns+i]*u[(e*p.ns+i)*4+a];
        rho=std::min(rho,v[0]);
        speed=std::max(speed,std::hypot(v[1],v[2])/v[0]+std::sqrt(1.4*pressure(v)/v[0]));
      }
      d[e]=std::min(cfl*p.base[e]/speed,cfl*p.diff*p.h[e]*p.h[e]*rho);
    }
  });
  if(!local) {double minimum=*std::min_element(d,d+p.ne); std::fill_n(d,p.ne,minimum);}
  return dt;
}

std::tuple<Tensor,Tensor,Tensor> step(const Tensor& u,const Tensor& cfl,const Tensor& first,
                const Tensors& data,double mu,double pr,double diffusion,bool local) {
  Plan p(u,data,mu,pr,diffusion);
  TORCH_CHECK(first.sizes()==u.sizes() && first.scalar_type()==at::kDouble && first.device().is_cpu() && first.is_contiguous(), "invalid first RHS");
  TORCH_CHECK(cfl.numel()==1 && cfl.device().is_cpu() && cfl.scalar_type()==at::kDouble, "CFL must be CPU float64 scalar");
  double cf=cfl.item<double>(); TORCH_CHECK(std::isfinite(cf) && cf>0, "CFL must be positive");
  Workspace ws(p,u); auto dt=timestep(u,p,cf,local);
  auto a=at::empty_like(u),b=at::empty_like(u),f=at::empty_like(u);
  const double *base=u.const_data_ptr<double>(), *d=dt.const_data_ptr<double>();
  auto update=[&](Tensor& target,const Tensor& state,const Tensor& force,int stage) {
    double* out=target.mutable_data_ptr<double>(); const double* v=state.const_data_ptr<double>();
    const double* r=force.const_data_ptr<double>();
    at::parallel_for(0,p.ne,16,[&](int64_t begin,int64_t end) {
      for(int64_t e=begin;e<end;++e) for(int64_t j=e*p.ns*4;j<(e+1)*p.ns*4;++j)
        out[j]=stage==2?(2./3)*base[j]+(1./3)*v[j]+d[e]/6*r[j]:v[j]+.5*d[e]*r[j];
    });
  };
  update(a,u,first,0); rhs_kernel(a,f,p,ws);
  update(b,a,f,1); rhs_kernel(b,f,p,ws);
  update(a,b,f,2); rhs_kernel(a,f,p,ws);
  update(b,a,f,3); rhs_kernel(b,f,p,ws);
  return {b,dt,f};
}
bool valid(const Tensor& U,const Tensors& data,double mu,double pr,double diffusion) {
  Plan p(U,data,mu,pr,diffusion); const double* u=U.const_data_ptr<double>();
  auto flags=at::empty({p.ne},U.options().dtype(at::kByte));
  auto flag=flags.mutable_data_ptr<uint8_t>();
  at::parallel_for(0,p.ne,16,[&](int64_t begin,int64_t end) {
    for(int64_t e=begin;e<end;++e) {
      bool ok=true;
      for(int64_t i=0;i<p.ns;++i) ok=ok && positive(u+(e*p.ns+i)*4);
      for(int64_t q=0;q<p.nq;++q) {
        double v[4]={};
        for(int64_t i=0;i<p.ns;++i) for(int a=0;a<4;++a) v[a]+=p.B[q*p.ns+i]*u[(e*p.ns+i)*4+a];
        ok=ok && positive(v);
      }
      for(int f=0;f<p.edges;++f) for(int64_t q=0;q<p.nf;++q) {
        double v[4]={};
        for(int64_t i=0;i<p.ns;++i) for(int a=0;a<4;++a) v[a]+=p.FB[(f*p.nf+q)*p.ns+i]*u[(e*p.ns+i)*4+a];
        ok=ok && positive(v);
      }
      flag[e]=ok;
    }
  });
  return std::all_of(flag,flag+p.ne,[](uint8_t b){return b!=0;});
}

double residual_norm(const Tensor& F,const Tensors& data,double mu,double pr,double diffusion) {
  Plan p(F,data,mu,pr,diffusion); auto f=F.const_data_ptr<double>();
  auto sums=at::zeros({p.ne,5},F.options()); auto sum=sums.mutable_data_ptr<double>();
  at::parallel_for(0,p.ne,16,[&](int64_t begin,int64_t end) {
    for(int64_t e=begin;e<end;++e) for(int64_t q=0;q<p.nq;++q) {
      double value[4]={},w=p.weight[e*p.nq+q];
      for(int64_t i=0;i<p.ns;++i) for(int v=0;v<4;++v) value[v]+=p.B[q*p.ns+i]*f[(e*p.ns+i)*4+v];
      for(int v=0;v<4;++v) sum[e*5+v]+=value[v]*value[v]*w;
      sum[e*5+4]+=w;
    }
  });
  double total[5]={};
  for(int64_t e=0;e<p.ne;++e) for(int v=0;v<5;++v) total[v]+=sum[e*5+v];
  double result=0;
  for(int v=0;v<4;++v) {
    if(!std::isfinite(total[v])) return std::numeric_limits<double>::quiet_NaN();
    result=std::max(result,std::sqrt(total[v]/total[4]));
  }
  return result;
}
} // namespace

TORCH_LIBRARY(cylinder_flow, m) {
  m.def("rhs(Tensor u, Tensor[] data, float mu, float pr, float diffusion) -> Tensor");
  m.def("step(Tensor u, Tensor cfl, Tensor first_rhs, Tensor[] data, float mu, float pr, float diffusion, bool local) -> (Tensor, Tensor, Tensor)");
  m.def("valid(Tensor u, Tensor[] data, float mu, float pr, float diffusion) -> bool");
  m.def("residual_norm(Tensor u, Tensor[] data, float mu, float pr, float diffusion) -> float");
}
TORCH_LIBRARY_IMPL(cylinder_flow, CPU, m) {
  m.impl("rhs",rhs); m.impl("step",step); m.impl("valid",valid); m.impl("residual_norm",residual_norm);
}
