// p=3 Gauss FR/Rusanov and paired LDG on conforming mapped hexes.
// SSPRK43 uses the same stages as the Cartesian solver.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <omp.h>
#include "time.hpp"
#include "hex_modal_filter.hpp"
#include "cartesian_ns/kernels/round_jet_boundary.hpp"
#ifndef FR_HEX_FUSED_VOLUME
#define FR_HEX_FUSED_VOLUME 1
#endif
namespace {
constexpr int N=64,F=16,Q=4;
inline int node(int a,int f,int i){return a==0?16*i+f:a==1?16*(f/4)+4*i+f%4:16*(f/4)+4*(f%4)+i;}
inline int faceq(int a,int n){return a==0?n%16:a==1?(n/16)*4+n%4:n/4;}
inline int along(int a,int n){return a==0?n/16:a==1?(n/4)%4:n%4;}
inline void primitive(const double* u,double g,double* v,double& p,double& sound){
    double kinetic=0;for(int x=0;x<3;++x){v[x]=u[x+1]/u[0];kinetic+=v[x]*v[x];}
    p=(g-1)*(u[4]-.5*u[0]*kinetic);sound=std::sqrt(g*p/u[0]);
}
inline bool valid(const double* u,double g){
    for(int c=0;c<5;++c)if(!std::isfinite(u[c]))return false;
    if(!(u[0]>0))return false;
    return u[4]>.5*(u[1]*u[1]+u[2]*u[2]+u[3]*u[3])/u[0];
}
inline void inviscid(const double* u,const double* v,double p,int x,double* f){
    for(int c=0;c<5;++c)f[c]=u[c]*v[x];f[x+1]+=p;f[4]+=p*v[x];
}
inline void exterior(const double* u,const double* far,const double* n,int tag,double g,double* out){
    double v[3],p,a;primitive(u,g,v,p,a);double un=0;for(int x=0;x<3;++x)un+=v[x]*n[x];
    if(tag==1){std::copy(u,u+5,out);for(int x=0;x<3;++x)out[x+1]-=2*u[0]*un*n[x];return;}
    double vf[3],pf,af;primitive(far,g,vf,pf,af);double uf=0;for(int x=0;x<3;++x)uf+=vf[x]*n[x];
    if(un>=a){std::copy(u,u+5,out);return;}if(un<=-a){std::copy(far,far+5,out);return;}
    double ub=.5*(un+2*a/(g-1)+uf-2*af/(g-1));
    double ab=.25*(g-1)*(un+2*a/(g-1)-uf+2*af/(g-1));
    if(!(ab>0)){std::copy(far,far+5,out);return;}
    bool incoming=ub<0;double entropy=incoming?pf/std::pow(far[0],g):p/std::pow(u[0],g);
    double rho=std::pow(ab*ab/(g*entropy),1/(g-1)),pb=entropy*std::pow(rho,g),energy=pb/(g-1);
    out[0]=rho;for(int x=0;x<3;++x){double vel=(incoming?vf[x]-uf*n[x]:v[x]-un*n[x])+ub*n[x];out[x+1]=rho*vel;energy+=.5*rho*vel*vel;}
    out[4]=energy;
}

constexpr int jet_ambient=3, jet_inlet=4, jet_outlet=5;
inline bool jet_characteristic_3d(int tag, const double* jet){
    return jet && ((tag==jet_outlet && jet[4]>=2.) || (tag==jet_ambient && (jet[4]==3. || jet[4]==4.))
                   || (tag==jet_inlet && jet[4]==4.));
}
inline bool jet_legacy_lateral_backflow(int tag,const double* jet){
    return tag==jet_ambient && jet[4]==3.;
}
inline void jet_exterior(const double* u,const double* far,const double* normal,
                         int tag,double gamma,const double* jet,int64_t face,int point,
                         double time,double acoustic_weight,double* out){
    double target[5];
    const bool acoustic_3d=jet_characteristic_3d(tag,jet);
    if(tag==jet_inlet){
        double velocity[3],weight;
        fr_round_jet::nozzle(jet,face,point,time,velocity,weight);
        target[0]=1.;
        target[4]=jet[0]/(gamma-1);
        for(int d=0;d<3;++d){
            target[d+1]=velocity[d];
            target[4]+=.5*velocity[d]*velocity[d];
        }
        far=target;
    }
    bool blend_outflow=false;
    if(tag==jet_outlet || acoustic_3d){
        double velocity[3],pressure,sound;
        primitive(u,gamma,velocity,pressure,sound);
        const double un=velocity[0]*normal[0]+velocity[1]*normal[1]+velocity[2]*normal[2];
        if(acoustic_3d){
            if(fr_round_jet::uses_acoustic_correction(un,sound,jet_legacy_lateral_backflow(tag,jet))){
                if(acoustic_weight==1.){
                    std::copy(u,u+5,out);
                    return;
                }
                blend_outflow=true;
            }
        }else if(jet[4]==1. && std::abs(un)<sound){
            std::copy(u,u+5,out);
            return;
        }
        if(!acoustic_3d && un>=0.){
            std::copy(u,u+5,target);
            target[4]+=(jet[0]-pressure)/(gamma-1);
            far=target;
        }
    }
    exterior(u,far,normal,tag,gamma,out);
    if(blend_outflow)
        for(int c=0;c<5;++c)out[c]+=acoustic_weight*(u[c]-out[c]);
}
}

#if FR_HEX_FUSED_VOLUME
#include "hex_fr_optimized.hpp"
#include "hex_fr_grouped.hpp"
#endif

extern "C" void mapped_hex_stage(int64_t count,const double* original,double* state,const double* rhs,double dt,int stage,int threads){
    #pragma omp parallel for num_threads(threads) schedule(dynamic,65536)
    for(int64_t i=0;i<count;++i)state[i]=fr_time::SSPRK43::update(original[i],stage==0?original[i]:state[i],rhs[i],dt,stage);
}


extern "C" int mapped_jet_rhs(int64_t ne,const double* input,double* output,void** data,
    double gamma,double mu,double prandtl,int threads,double* scan,
    const double* original,double* stage_state,double dt,int rkstage,int scan_stage,void** plan){
    if(plan[17]){
        if(rkstage<0 || scan_stage) return mapped_group::evaluate<true,true,false,true,true,true>(
            ne,input,output,data,gamma,mu,prandtl,-1.,threads,scan,original,stage_state,dt,rkstage,plan);
        return mapped_group::evaluate<true,false,false,true,true,true>(
            ne,input,output,data,gamma,mu,prandtl,-1.,threads,scan,original,stage_state,dt,rkstage,plan);
    }
    if(rkstage<0 || scan_stage) return mapped_group::evaluate<true,true,false,true>(
        ne,input,output,data,gamma,mu,prandtl,-1.,threads,scan,original,stage_state,dt,rkstage,plan);
    return mapped_group::evaluate<true,false,false,true>(
        ne,input,output,data,gamma,mu,prandtl,-1.,threads,scan,original,stage_state,dt,rkstage,plan);
}
