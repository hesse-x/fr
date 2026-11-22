#pragma once
#include <type_traits>
#include "hex_wale_pipeline.hpp"
#include "hex_trace_map.hpp"
// Group-local FR/LDG: compact directional halos, no global face work arrays.
namespace mapped_group {
using namespace mapped_simd;
template<bool Rect,int Mask=511,bool JetAxes=false> struct CellGeometry {
    const double *inverse_j,*c,*s,*box;int64_t at;
    CellGeometry(const double* inverseJ,const double* C,const double* S,int64_t e,const int64_t* indices,const double* constants):
        inverse_j(inverseJ),c(C),s(S),box(nullptr),at(e){
        if constexpr(Rect)box=constants+e*8;
        else if(indices)at=indices[e];
    }
    static constexpr int physical_axis(int a){return JetAxes?(a+1)%3:a;}
    static constexpr bool nonzero(int a,int x){return (Mask>>(a*3+x))&1;}
    Vec inverse(int n)const{if constexpr(Rect)return box[1];else return simd::load(inverse_j+at*64+n);}
    Vec cofactor(int a,int x,int n)const{if constexpr(Rect)return physical_axis(a)==x?Vec(box[2+a]):Vec(0.);else return nonzero(a,x)?simd::load(c+(at*9+a*3+x)*64+n):Vec(0.);}
    Vec surface(int f,int x,int q)const{if constexpr(Rect)return physical_axis(f/2)==x?Vec((f%2?1.:-1.)*box[2+f/2]):Vec(0.);else return simd::load(s+((at*6+f)*3+x)*16+q);}
    Vec norm2(int n)const{Vec value=0.;
        if constexpr(Rect){for(int a=0;a<3;++a)value=value+box[5+a]*box[5+a];}
        else{Vec inv=inverse(n);for(int a=0;a<3;++a)for(int x=0;x<3;++x)if(nonzero(a,x)){Vec v=cofactor(a,x,n)*inv;value=value+v*v;}}
        return value;}
};
inline Vec local_trace(const double* field,int components,int slot,const int64_t* neighbor,int q,int c){
    const double* base=field+(int64_t(slot)*components+c)*16;
    int first=int(neighbor[q]%16),last=int(neighbor[q+3]%16);
    if(last-first==3)return simd::load(base+first);
    if(last-first==-3)return Vec(_mm256_permute4x64_pd(simd::load(base+last).value,0x1b));
    return Vec(_mm256_i64gather_pd(base,_mm256_setr_epi64x(first,neighbor[q+1]%16,neighbor[q+2]%16,last),8));
}

template<bool SoA,bool Scan,bool Rotating=false,bool Shared=false,bool Compact=false,bool JetAxes=false>
int evaluate(int64_t ne,const double* input,double* output,void** data,double gamma,double mu,double prandtl,
    double wall_temperature,int threads,double* scan,const double* original,double* stage_state,double dt,int rkstage,void** plan){
    const auto inverseJ=(const double*)data[0],C=(const double*)data[1],S=(const double*)data[2];
    const auto neighbor=(const int64_t*)data[3],tags=(const int64_t*)data[4],choose=(const int64_t*)data[5];
    const auto D=(const double*)data[6],L=(const double*)data[7],R=(const double*)data[8],gl=(const double*)data[9],gr=(const double*)data[10],far=(const double*)data[11];
    const auto mortar_slots=(const int64_t*)data[25];
    const auto mortar_trace=(const double*)data[26];
    const auto boundary_index=(const int64_t*)data[16];auto boundary_flux=(double*)data[17],timing=(double*)data[18];
    const auto jet=(const double*)data[27];
    const double jet_time=*(const double*)data[28];
    const auto records=(const int64_t*)plan[0],pids=(const int64_t*)plan[1],uslots=(const int64_t*)plan[2],wslots=(const int64_t*)plan[3],vslots=(const int64_t*)plan[4];
    const auto wneighbors=(const int64_t*)plan[5],uneighbors=(const int64_t*)plan[6],vsources=(const int64_t*)plan[7];
    auto workspace=(double*)plan[8],boundaryU=(double*)plan[9],boundaryV=(double*)plan[10];
    const auto sizes=(const int64_t*)plan[13];
    const int64_t* compact_indices=nullptr;const double* compact_constants=nullptr;
    if constexpr(Compact){compact_indices=(const int64_t*)plan[17];compact_constants=(const double*)plan[18];}
    const auto face_codes=(const uint32_t*)plan[19];
    const int64_t count=ne*64;const double kappa=mu*gamma/((gamma-1)*prandtl);
    double vf[3],pf,af;primitive(far,gamma,vf,pf,af);double farW[4]={vf[0],vf[1],vf[2],pf/far[0]},reference[3][5],weak[16];
    for(int x=0;x<3;++x)inviscid(far,vf,pf,x,reference[x]);
    for(int i=0;i<4;++i)for(int j=0;j<4;++j)weak[i*4+j]=-D[i*4+j]+gl[i]*L[j]+gr[i]*R[j];
    double mark=omp_get_wtime(),max_adv=0,max_vis=0;int invalid=0;
    // Schedule ICV is local to the calling OpenMP task; restore it afterwards.
    // Dynamic chunks are consecutive groups, with thread-private scratch and
    // immutable input halos. Each owned cell and boundary face has one writer.
    omp_sched_t old_schedule;int old_chunk;omp_get_schedule(&old_schedule,&old_chunk);
    omp_set_schedule(sizes[3]?omp_sched_dynamic:omp_sched_static,sizes[3]?int(sizes[4]):0);
    #pragma omp parallel for num_threads(threads) schedule(runtime) reduction(|:invalid) reduction(max:max_adv,max_vis)
    for(int g=0;g<sizes[0];++g){
        const int64_t* r=records+g*14;const int no=r[0],na=r[1],np=r[2],nu=r[3],nw=r[4],nv=r[5];
        const auto cells=pids+r[6],u_slot=uslots+r[7],w_slot=wslots+r[8],v_slot=vslots+r[9];
        const auto w_other=wneighbors+r[10],u_other=uneighbors+r[11],v_source=vsources+r[12];
        double* wn=workspace+omp_get_thread_num()*sizes[1];
        double* un=wn+na*4*64;double* traceU=un+no*5*64;
        double* traceW=traceU+nu*5*16;double* traceV=traceW+nw*4*16;
        double* partial=traceV+nv*6*16;
        const int64_t* face_sources=nullptr;const int64_t* face_peers=nullptr;
        if constexpr(Shared){
            int64_t offset=((const int64_t*)plan[15])[g]*6;
            face_sources=(const int64_t*)plan[14]+offset;face_peers=(const int64_t*)plan[16]+offset;
        }
        int bad=0;
        for(int pidx=0;pidx<np;++pidx){
            int64_t e=cells[pidx];
            auto primitive_cell=[&](auto tag){constexpr bool Rect=decltype(tag)::value;
            CellGeometry<Rect,511,JetAxes> geo(inverseJ,C,S,e,compact_indices,compact_constants);


            alignas(32) double U[5][64],W[4][64];Vec advmax=0.,vismax=0.;
            for(int n=0;n<64;n+=4){Vec u[5],w[4],pressure,sound;
                for(int c=0;c<5;++c){u[c]=state_load<SoA>(input,count,e*64+n,c);simd::store(U[c]+n,u[c]);}
                if(!primitive_vectors(u,gamma,w,pressure,sound))bad=1;
                for(int c=0;c<4;++c)simd::store(W[c]+n,w[c]);
                if constexpr(Scan){if(pidx<no){Vec invJ=geo.inverse(n),adv=0.,vis=0.;
                    if constexpr(Rect){for(int axis=0;axis<3;++axis){Vec h=geo.box[5+axis];adv=adv+.5*(simd::abs(w[geo.physical_axis(axis)])*h+sound*h);vis=vis+.25*h*h;}}
                    else for(int axis=0;axis<3;++axis){Vec magnitude=0.,velocity=0.;for(int x=0;x<3;++x){Vec m=geo.cofactor(axis,x,n);magnitude=magnitude+m*m;velocity=velocity+m*w[x];}
                        magnitude=simd::sqrt(magnitude);Vec h=magnitude*invJ;adv=adv+.5*(simd::abs(velocity)*invJ+sound*h);vis=vis+.25*h*h;}
                    advmax=simd::maximum(advmax,adv);vismax=simd::maximum(vismax,vis*std::max(4*mu/3,kappa*(gamma-1))/u[0]);}}
            }
            if constexpr(Scan){if(pidx<no){max_adv=std::max(max_adv,maximum(advmax));max_vis=std::max(max_vis,maximum(vismax));}}
            if(pidx<na)std::copy(&W[0][0],&W[0][0]+4*64,wn+pidx*4*64);
            if(pidx<no)std::copy(&U[0][0],&U[0][0]+5*64,un+pidx*5*64);
            for(int f=0;f<6;++f){int us=int(u_slot[pidx*6+f]),ws=int(w_slot[pidx*6+f]);
                if(us<0&&ws<0)continue;
                for(int q=0;q<16;q+=4){const double* vector=f%2?R:L;
                    if(us>=0){Vec u[5],w[4],pressure,sound;
                        for(int c=0;c<5;++c){u[c]=simd::trace(U[c],f/2,q,vector);simd::store(traceU+(us*5+c)*16+q,u[c]);}
                        if(!primitive_vectors(u,gamma,w,pressure,sound))bad=1;}
                    if(ws>=0)for(int c=0;c<4;++c)simd::store(traceW+(ws*4+c)*16+q,simd::trace(W[c],f/2,q,vector));
                }
            }
            };
            if constexpr(Compact){if(compact_indices[e]<0)primitive_cell(std::true_type{});else primitive_cell(std::false_type{});}
            else primitive_cell(std::false_type{});
        }
        if(bad){invalid=1;continue;}
        for(int aidx=0;aidx<na;++aidx){
            const int64_t e=cells[aidx];const bool owned=aidx<no;
            auto active_cell=[&](auto tag){constexpr int Mask=decltype(tag)::value;constexpr bool Rect=Mask==0;
            CellGeometry<Rect,Rect?511:Mask,JetAxes> geo(inverseJ,C,S,e,compact_indices,compact_constants);


            const double (*W)[64]=reinterpret_cast<const double(*)[64]>(wn+aidx*4*64);
            const double (*U)[64]=reinterpret_cast<const double(*)[64]>(un+aidx*5*64);
            alignas(32) double pressure[64],G[4][3][64],derivative[4][64];
            if constexpr(!Rect)std::fill(&G[0][0][0],&G[0][0][0]+4*3*64,0.);
            if(owned)for(int n=0;n<64;n+=4){Vec rho=simd::load(U[0]+n),vx=simd::load(W[0]+n),vy=simd::load(W[1]+n),vz=simd::load(W[2]+n);
                simd::store(pressure+n,(gamma-1)*(simd::load(U[4]+n)-.5*rho*((vx*vx+vy*vy)+vz*vz)));}
        // Cartesian derivatives map to one physical component. Write that
        // component directly, avoiding a temporary derivative round trip.
        if constexpr(Rect){
            for(int axis=0;axis<3;++axis)for(int c=0;c<4;++c)for(int q=0;q<16;q+=4){
                Vec lines[4],d[4];simd::load_lines(W[c],axis,q,lines);
                for(int i=0;i<4;++i){d[i]=0.;for(int j=0;j<4;++j)d[i]=d[i]+D[i*4+j]*lines[j];d[i]=geo.box[2+axis]*d[i];}
                    simd::store_lines(G[c][geo.physical_axis(axis)],axis,q,d);
            }
        }else if constexpr(Mask!=511){
            auto direction=[&](auto axis_tag){constexpr int axis=decltype(axis_tag)::value;
                for(int c=0;c<4;++c)for(int q=0;q<16;q+=4){Vec lines[4],d[4];simd::load_lines(W[c],axis,q,lines);
                    for(int i=0;i<4;++i){d[i]=0.;for(int j=0;j<4;++j)d[i]=d[i]+D[i*4+j]*lines[j];}
                    simd::store_lines(derivative[c],axis,q,d);
                }
                for(int n=0;n<64;n+=4)for(int c=0;c<4;++c){Vec d=simd::load(derivative[c]+n);
                    if constexpr(Mask&(1<<(axis*3+0)))simd::store(G[c][0]+n,simd::load(G[c][0]+n)+geo.cofactor(axis,0,n)*d);
                    if constexpr(Mask&(1<<(axis*3+1)))simd::store(G[c][1]+n,simd::load(G[c][1]+n)+geo.cofactor(axis,1,n)*d);
                    if constexpr(Mask&(1<<(axis*3+2)))simd::store(G[c][2]+n,simd::load(G[c][2]+n)+geo.cofactor(axis,2,n)*d);
                }
            };
            direction(std::integral_constant<int,0>{});direction(std::integral_constant<int,1>{});direction(std::integral_constant<int,2>{});
        }else{
        for(int axis=0;axis<3;++axis){
            for(int c=0;c<4;++c)for(int q=0;q<16;q+=4){Vec lines[4],d[4];simd::load_lines(W[c],axis,q,lines);
                for(int i=0;i<4;++i){d[i]=0.;for(int j=0;j<4;++j)d[i]=d[i]+D[i*4+j]*lines[j];}simd::store_lines(derivative[c],axis,q,d);}
            for(int n=0;n<64;n+=4)for(int c=0;c<4;++c){Vec d=simd::load(derivative[c]+n);
                if constexpr(Rect)simd::store(G[c][axis]+n,geo.cofactor(axis,axis,n)*d);
                else for(int x=0;x<3;++x)if(geo.nonzero(axis,x))simd::store(G[c][x]+n,simd::load(G[c][x]+n)+geo.cofactor(axis,x,n)*d);}
        }
        }
        for(int f=0;f<6;++f){
            int tag=int(tags[e*6+f]);if(!tag&&choose[e*6+f])continue;
            int own_slot=int(w_slot[aidx*6+f]),other_slot=int(w_other[aidx*6+f]);
            const auto nb=face_codes+(e*6+f)*4;
            const int64_t mortar=mortar_slots&&tag ? mortar_slots[boundary_index[e*6+f]] : -1;
            for(int q=0;q<16;q+=4){Vec delta[4];const TracePermutation map(nb[q/4]);
                double inlet_velocity[3][4],inlet_weight[4];
                if(tag==jet_inlet)for(int k=0;k<4;++k){
                    double velocity[3],weight;
                    fr_round_jet::nozzle(jet,boundary_index[e*6+f],q+k,jet_time,velocity,weight);
                    inlet_weight[k]=weight;
                    for(int d=0;d<3;++d)inlet_velocity[d][k]=velocity[d];
                }
                for(int c=0;c<4;++c){Vec own=simd::load(traceW+(own_slot*4+c)*16+q),common;
                    if(mortar>=0)common=simd::load(mortar_trace+(mortar*4+c)*16+q);
                    else if(tag==1)common=c<3?Vec(0.):wall_temperature<0?own:Vec(wall_temperature);
                    else if(tag==2){
                        if constexpr(Rotating){const double* ff=(const double*)data[24];int64_t at=boundary_index[e*6+f]*16+q;
                            Vec rho=state_load<false>(ff,0,at,0);
                            if(c<3)common=state_load<false>(ff,0,at,c+1)/rho;
                            else{Vec kin=0.;for(int d=0;d<3;++d){Vec m=state_load<false>(ff,0,at,d+1);kin=kin+m*m;}
                                common=(gamma-1)*(state_load<false>(ff,0,at,4)-.5*kin/rho)/rho;}
                        }else common=farW[c];
                    }else if(tag==jet_inlet){
                        const Vec target=c<3?simd::load(inlet_velocity[c]):Vec(jet[0]);
                        common=own+simd::load(inlet_weight)*(target-own);
                    }else if(tag)common=own;
                    else common=packed_trace(traceW,4,other_slot,map,c);
                    delta[c]=common-own;}
                for(int c=0;c<4;++c){int begin=Rect?geo.physical_axis(f/2):0,end=Rect?begin+1:3;
                    for(int x=begin;x<end;++x){Vec correction=geo.surface(f,x,q)*delta[c],line[4];simd::load_lines(G[c][x],f/2,q,line);
                    for(int i=0;i<4;++i)line[i]=line[i]+(f%2?gr[i]:-gl[i])*correction;simd::store_lines(G[c][x],f/2,q,line);}}
            }
        }
        alignas(32) double fv[5][3][64],heat[3][64],flux[3][5][64],result[5][64]={};
        alignas(32) double turbulent_pipeline[64];
        if constexpr(fr_sgs::compiled_model==fr_sgs::Model::WALE){
            for(int n=0;n<64;n+=4){Vec invJ=geo.inverse(n);
                for(int c=0;c<4;++c)for(int x=0;x<3;++x)simd::store(G[c][x]+n,simd::load(G[c][x]+n)*invJ);}
            wale_pipeline<4,SoA,Rotating>(data,e,count,input,&G[0][0][0],turbulent_pipeline);
        }
        for(int n=0;n<64;n+=4){Vec invJ=geo.inverse(n);
            if constexpr(fr_sgs::compiled_model!=fr_sgs::Model::WALE)
                for(int c=0;c<4;++c)for(int x=0;x<3;++x)simd::store(G[c][x]+n,simd::load(G[c][x]+n)*invJ);
            Vec effective_mu,effective_kappa;
            if constexpr(Compact){Vec measure=0.;if constexpr(Scan){if(owned)measure=geo.norm2(n);}
                if constexpr(fr_sgs::compiled_model==fr_sgs::Model::WALE)
                    transport_pipeline<Scan,true>(data,e,n,turbulent_pipeline,state_load<SoA>(input,count,e*64+n,0),invJ,gamma,mu,kappa,owned,effective_mu,effective_kappa,max_vis,measure);
                else transport<Scan,Rotating,true>(data,e,n,&G[0][0][0],state_load<SoA>(input,count,e*64+n,0),invJ,gamma,mu,kappa,owned,effective_mu,effective_kappa,max_vis,measure);}
            else{
                if constexpr(fr_sgs::compiled_model==fr_sgs::Model::WALE)
                    transport_pipeline<Scan>(data,e,n,turbulent_pipeline,state_load<SoA>(input,count,e*64+n,0),invJ,gamma,mu,kappa,owned,effective_mu,effective_kappa,max_vis);
                else transport<Scan,Rotating>(data,e,n,&G[0][0][0],state_load<SoA>(input,count,e*64+n,0),invJ,gamma,mu,kappa,owned,effective_mu,effective_kappa,max_vis);
            }
            Vec div=(simd::load(G[0][0]+n)+simd::load(G[1][1]+n))+simd::load(G[2][2]+n);
            for(int x=0;x<3;++x){simd::store(fv[0][x]+n,0.);Vec energy=0.;
                for(int c=0;c<3;++c){Vec stress=effective_mu*(simd::load(G[c][x]+n)+simd::load(G[x][c]+n)-(x==c?2./3.*div:Vec(0.)));simd::store(fv[c+1][x]+n,stress);energy=energy+simd::load(W[c]+n)*stress;}
                Vec h=effective_kappa*simd::load(G[3][x]+n);simd::store(heat[x]+n,h);simd::store(fv[4][x]+n,energy+h);}
            if(owned)for(int c=0;c<5;++c){Vec uc=simd::load(U[c]+n),p=simd::load(pressure+n),physical[3];
                for(int x=0;x<3;++x){Vec velocity=simd::load(W[x]+n);physical[x]=uc*velocity;if(c==x+1)physical[x]=physical[x]+p;if(c==4)physical[x]=physical[x]+p*velocity;physical[x]=physical[x]-reference[x][c]-simd::load(fv[c][x]+n);}
                if constexpr(!Rect&&Mask!=511){
                    auto transform=[&](auto axis_tag){constexpr int axis=decltype(axis_tag)::value;Vec cf=0.;
                        if constexpr(Mask&(1<<(axis*3+0)))cf=cf+geo.cofactor(axis,0,n)*physical[0];
                        if constexpr(Mask&(1<<(axis*3+1)))cf=cf+geo.cofactor(axis,1,n)*physical[1];
                        if constexpr(Mask&(1<<(axis*3+2)))cf=cf+geo.cofactor(axis,2,n)*physical[2];
                        simd::store(flux[axis][c]+n,cf);
                    };
                    transform(std::integral_constant<int,0>{});transform(std::integral_constant<int,1>{});transform(std::integral_constant<int,2>{});
                }else{
                for(int axis=0;axis<3;++axis){Vec cf=0.;
                    if constexpr(Rect){int x=geo.physical_axis(axis);cf=geo.cofactor(axis,x,n)*physical[x];}
                    else for(int x=0;x<3;++x)if(geo.nonzero(axis,x))cf=cf+geo.cofactor(axis,x,n)*physical[x];simd::store(flux[axis][c]+n,cf);}
                }}
        }
        for(int f=0;f<6;++f){int slot=int(v_slot[aidx*6+f]);if(slot<0)continue;
            for(int q=0;q<16;q+=4){const double* vector=f%2?R:L;
                for(int c=0;c<5;++c){Vec v=0.;
                    if constexpr(Rect){int x=geo.physical_axis(f/2);v=geo.surface(f,x,q)*simd::trace(fv[c][x],f/2,q,vector);}
                    else for(int x=0;x<3;++x)v=v+geo.surface(f,x,q)*simd::trace(fv[c][x],f/2,q,vector);simd::store(traceV+(slot*6+c)*16+q,v);}
                Vec h=0.;
                if constexpr(Rect){int x=geo.physical_axis(f/2);h=geo.surface(f,x,q)*simd::trace(heat[x],f/2,q,vector);}
                else for(int x=0;x<3;++x)h=h+geo.surface(f,x,q)*simd::trace(heat[x],f/2,q,vector);simd::store(traceV+(slot*6+5)*16+q,h);}
        }
        if(owned){
        for(int axis=0;axis<3;++axis)for(int c=0;c<5;++c)for(int q=0;q<16;q+=4){Vec lines[4],r[4];simd::load_lines(flux[axis][c],axis,q,lines);simd::load_lines(result[c],axis,q,r);
            for(int i=0;i<4;++i)for(int j=0;j<4;++j)r[i]=r[i]+weak[i*4+j]*lines[j];simd::store_lines(result[c],axis,q,r);}
            std::copy(&result[0][0],&result[0][0]+5*64,partial+aidx*5*64);
        }
            };
            if constexpr(Compact){
                if(compact_indices[e]<0)active_cell(std::integral_constant<int,0>{});
                else switch(((const int64_t*)plan[20])[e]){
                    case 107:active_cell(std::integral_constant<int,107>{});break;
                    case 179:active_cell(std::integral_constant<int,179>{});break;
                    case 309:active_cell(std::integral_constant<int,309>{});break;
                    default:active_cell(std::integral_constant<int,511>{});break;
                }
            }else active_cell(std::integral_constant<int,511>{});
        }
        auto finish=[&](int oidx,const double (*result)[64],const auto& geo){
            const int64_t e=cells[oidx];

        for(int n=0;n<64;n+=4){Vec source[5]={};const Vec invJ=geo.inverse(n);
            if constexpr(Rotating){const double* o=(const double*)data[22];const double* gv=(const double*)data[23]+e*3*64;
                Vec m[3],v[3],acc[3];for(int d=0;d<3;++d){m[d]=state_load<SoA>(input,count,e*64+n,d+1);v[d]=simd::load(gv+d*64+n);}
                for(int d=0;d<3;++d){int j=(d+1)%3,k=(d+2)%3;acc[d]=o[k]*v[j]-o[j]*v[k];
                    source[d+1]=2.*(o[k]*m[j]-o[j]*m[k])+state_load<SoA>(input,count,e*64+n,0)*acc[d];}
                source[4]=m[0]*acc[0]+m[1]*acc[1]+m[2]*acc[2];}
        for(int c=0;c<5;++c){Vec value=simd::load(result[c]+n)*invJ;
            if constexpr(Rotating)value=value+source[c];if(!finite(value))invalid=1;
            if(rkstage<0)state_store<SoA>(output,count,e*64+n,c,value);
            else state_store<SoA>(stage_state,count,e*64+n,c,fr_time::SSPRK43::update(state_load<SoA>(original,count,e*64+n,c),state_load<SoA>(input,count,e*64+n,c),value,dt,rkstage));}}
        };
        for(int oidx=0;oidx<no;++oidx){
            const int64_t e=cells[oidx];
            auto owned_cell=[&](auto tag){constexpr bool Rect=decltype(tag)::value;
            CellGeometry<Rect,511,JetAxes> geo(inverseJ,C,S,e,compact_indices,compact_constants);
            alignas(32) double result_storage[5][64],shared_face[5][16];
            double (*result)[64]=Shared?reinterpret_cast<double(*)[64]>(partial+oidx*5*64):result_storage;
            if constexpr(!Shared)std::copy(partial+oidx*5*64,partial+(oidx+1)*5*64,&result[0][0]);
            for(int f=0;f<6;++f){
                int source=0;
                if constexpr(Shared){source=int(face_sources[oidx*6+f]);if(source<0)continue;}
                const int tag=int(tags[e*6+f]);
                alignas(32) double acoustic[5][16]={};
                const bool acoustic_3d=jet_characteristic_3d(tag,jet);
                const bool acoustic_face=acoustic_3d || (tag==jet_outlet && jet[4]);
                const double* face_metric=acoustic_3d ? (const double*)data[29]
                    +boundary_index[e*6+f]*16*fr_round_jet::acoustic_metric_size : nullptr;
                if(acoustic_face){
                    alignas(32) double gradient[5][16];
                    const double* state=un+oidx*5*64;
                    const auto relaxation=acoustic_3d?(const double*)data[30]:nullptr;
                    const double* derivative=f%2?jet+6:relaxation+2;
                    for(int c=0;c<5;++c)for(int q=0;q<16;q+=4)
                        simd::store(gradient[c]+q,simd::trace(state+c*64,f/2,q,derivative));
                    const double* face_state=traceU+u_slot[oidx*6+f]*5*16;
                    if(acoustic_3d){
                        if(jet_legacy_lateral_backflow(tag,jet))
                            fr_round_jet::acoustic_outlet_3d<true>(jet,face_state,&gradient[0][0],
                                D,f/2,face_metric,relaxation,acoustic,gamma);
                        else
                            fr_round_jet::acoustic_outlet_3d<false>(jet,face_state,&gradient[0][0],
                                D,f/2,face_metric,relaxation,acoustic,gamma);
                    }else fr_round_jet::acoustic_outlet(jet,face_state,&gradient[0][0],
                                                       &acoustic[0][0],gamma);
                }
                for(int q=0;q<16;q+=4){
                const TracePermutation map(face_codes[(e*6+f)*4+q/4]);
                int own_slot=int(u_slot[oidx*6+f]),other_slot=int(u_other[oidx*6+f]),vslot=int(v_source[oidx*6+f]);
                Vec s[3],normal[3],magnitude=0.;
                if constexpr(Rect){magnitude=geo.box[2+f/2];for(int x=0;x<3;++x){s[x]=geo.surface(f,x,q);normal[x]=x==geo.physical_axis(f/2)?Vec(f%2?1.:-1.):Vec(0.);}}
                else{for(int x=0;x<3;++x){s[x]=geo.surface(f,x,q);magnitude=magnitude+s[x]*s[x];}
                    magnitude=simd::sqrt(magnitude);
                    for(int x=0;x<3;++x)normal[x]=s[x]/magnitude;}
            Vec um[5],up[5],wm[4],wp[4],pm,pp,am,ap;
            for(int c=0;c<5;++c)um[c]=simd::load(traceU+(own_slot*5+c)*16+q);
            if(!tag){for(int c=0;c<5;++c)up[c]=packed_trace(traceU,5,other_slot,map,c);}
            else {double own[5][4],n[3][4],ghost[5][4];for(int c=0;c<5;++c)simd::store(own[c],um[c]);for(int x=0;x<3;++x)simd::store(n[x],normal[x]);
                for(int k=0;k<4;++k){double u[5],unit[3],out[5];for(int c=0;c<5;++c)u[c]=own[c][k];for(int x=0;x<3;++x)unit[x]=n[x][k];
                    const double* ff=far;if constexpr(Rotating)ff=(const double*)data[24]+(boundary_index[e*6+f]*16+q+k)*5;
                    const double weight=face_metric ?
                        face_metric[(q+k)*fr_round_jet::acoustic_metric_size+13] : 1.;
                    jet_exterior(u,ff,unit,tag,gamma,jet,boundary_index[e*6+f],q+k,jet_time,weight,out);
                    for(int c=0;c<5;++c)ghost[c][k]=out[c];}
                for(int c=0;c<5;++c)up[c]=simd::load(ghost[c]);}
            primitive_unchecked(um,gamma,wm,pm,am);
            if(tag){if(!primitive_vectors(up,gamma,wp,pp,ap))invalid=1;}else primitive_unchecked(up,gamma,wp,pp,ap);
            Vec unm=0.,unp=0.;
            if constexpr(Rect){int x=geo.physical_axis(f/2);unm=wm[x]*(f%2?1.:-1.);unp=wp[x]*(f%2?1.:-1.);}
            else for(int x=0;x<3;++x){unm=unm+wm[x]*normal[x];unp=unp+wp[x]*normal[x];}
            Vec speed=simd::maximum(simd::abs(unm)+am,simd::abs(unp)+ap);
            for(int c=0;c<5;++c){Vec fm=um[c]*unm,fp=up[c]*unp;if(c>0&&c<4){fm=fm+pm*normal[c-1];fp=fp+pp*normal[c-1];}if(c==4){fm=fm+pm*unm;fp=fp+pp*unp;}
                Vec visc=(!tag&&choose[e*6+f])?-1.*packed_trace(traceV,6,vslot,map,c):simd::load(traceV+(vslot*6+c)*16+q);
                if(tag==1&&c==4)visc=wall_temperature<0?Vec(0.):simd::load(traceV+(vslot*6+5)*16+q);
                Vec common=magnitude*(.5*(fm+fp)-.5*speed*(up[c]-um[c]))-visc;
                if(acoustic_face)common=common+magnitude*simd::load(acoustic[c]+q);
                if constexpr(Shared){if(source>0)simd::store(shared_face[c]+q,common);}
                if(tag){int64_t b=boundary_index[e*6+f];state_store<false>(boundary_flux,0,b*16+q,c,common);
                    simd::store(boundaryU+(b*5+c)*16+q,um[c]);simd::store(boundaryV+(b*6+c)*16+q,simd::load(traceV+(vslot*6+c)*16+q));}
                for(int x=0;x<3;++x)common=common-s[x]*reference[x][c];
                Vec line[4];simd::load_lines(result[c],f/2,q,line);for(int i=0;i<4;++i)line[i]=line[i]-(f%2?gr[i]:-gl[i])*common;simd::store_lines(result[c],f/2,q,line);
            }
            if(tag)simd::store(boundaryV+(boundary_index[e*6+f]*6+5)*16+q,simd::load(traceV+(vslot*6+5)*16+q));
                }
                if constexpr(Shared){if(source>0){
                    // Both cells belong to this worker. Accumulate the same
                    // flux into both local residuals immediately, without a
                    // group-wide face cache or any neighboring-group writes.
                    int peer=int(face_peers[oidx*6+f]),j=peer/6,ff=peer%6;
                    int64_t other=cells[j];
                    const int64_t* mapping=neighbor+(other*6+ff)*16;
                    const double* other_surface=S+(other*6+ff)*3*16;
                    double (*other_result)[64]=reinterpret_cast<double(*)[64]>(partial+j*5*64);
                    for(int q=0;q<16;q+=4)for(int c=0;c<5;++c){
                        Vec common=-1.*local_trace(&shared_face[0][0],5,0,mapping,q,c);
                        for(int x=0;x<3;++x)common=common-simd::load(other_surface+x*16+q)*reference[x][c];
                        Vec line[4];simd::load_lines(other_result[c],ff/2,q,line);
                        for(int i=0;i<4;++i)line[i]=line[i]-(ff%2?gr[i]:-gl[i])*common;
                        simd::store_lines(other_result[c],ff/2,q,line);
                    }
                }}
            }
        if constexpr(!Shared)finish(oidx,result,geo);
            };
            if constexpr(Compact){if(compact_indices[e]<0)owned_cell(std::true_type{});else owned_cell(std::false_type{});}
            else owned_cell(std::false_type{});
        }
        if constexpr(Shared)for(int oidx=0;oidx<no;++oidx){
            const int64_t e=cells[oidx];
            const auto result=reinterpret_cast<const double(*)[64]>(partial+oidx*5*64);
            if constexpr(Compact){
                if(compact_indices[e]<0){
                    CellGeometry<true,511,JetAxes> geo(inverseJ,C,S,e,compact_indices,compact_constants);
                    finish(oidx,result,geo);
                    continue;
                }
            }
            CellGeometry<false> geo(inverseJ,C,S,e,compact_indices,compact_constants);
            finish(oidx,result,geo);
        }
    }
    omp_set_schedule(old_schedule,old_chunk);
    timing[0]=omp_get_wtime()-mark;timing[1]=timing[2]=0.;
    if constexpr(Scan){scan[0]=max_adv;scan[1]=max_vis;}
    return invalid?1:0;
}
}

