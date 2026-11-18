#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <omp.h>
#ifdef __AVX2__
#include "cartesian_ns/kernels/avx2.hpp"
#endif

// Independent p=3 tensor-product implementation of the concentration/filter
// construction. F and sensor matrices are runtime data; no copied HiFiLES code.
namespace hex_modal {
inline int line_node(int axis,int n,int j){
    const int stride=axis==0?16:axis==1?4:1;
    return n+(j-(n/stride)%4)*stride;
}
inline int64_t offset(int64_t count,int64_t e,int n,int c,int soa){
    return soa ? c*count+e*64+n:(e*64+n)*5+c;
}

inline double sensor_strength(const double* rho,const double* sensor){
#ifdef __AVX2__
    simd::Vec strength=0.;
    for(int axis=0;axis<3;++axis)for(int q=0;q<16;q+=4){
        simd::Vec lines[4];simd::load_lines(rho,axis,q,lines);
        for(int i=0;i<4;++i){simd::Vec value=0.;
            for(int j=0;j<4;++j)value=value+sensor[i*4+j]*lines[j];
            strength=simd::maximum(strength,8.*simd::abs(value)*value*value);
        }
    }
    double values[4];simd::store(values,strength);
    return std::max(std::max(values[0],values[1]),std::max(values[2],values[3]));
#else
    double strength=0.;
    for(int axis=0;axis<3;++axis)for(int n=0;n<64;++n){
        int stride=axis==0?16:axis==1?4:1,row=(n/stride)%4;double value=0.;
        for(int j=0;j<4;++j)value+=sensor[row*4+j]*rho[line_node(axis,n,j)];
        strength=std::max(strength,8.*std::abs(value)*value*value);
    }
    return strength;
#endif
}
}

extern "C" int mapped_hex_modal_filter(int64_t ne,double* state,const double* J,
    const double* weights,const double* filter,const double* sensor,int soa,
    double threshold,int threads,double* statistics,double* cell_sensor,int64_t* hits){
    const int64_t count=ne*64;
    int64_t changed=0,invalid=0;double maximum_sensor=0.,maximum_error=0.;
    #pragma omp parallel for num_threads(threads) schedule(dynamic,256) reduction(+:changed,invalid) reduction(max:maximum_sensor,maximum_error)
    for(int64_t e=0;e<ne;++e){
        double rho[64],strength=0.;bool bad=false;
        for(int n=0;n<64;++n){rho[n]=state[hex_modal::offset(count,e,n,0,soa)];bad|=!std::isfinite(rho[n]);}
        if(bad){++invalid;cell_sensor[e]=NAN;continue;}
        strength=hex_modal::sensor_strength(rho,sensor);
        cell_sensor[e]=strength;maximum_sensor=std::max(maximum_sensor,strength);
        if(!(strength>threshold))continue;
        double value[5][64],temporary[64],average[5]={},volume=0.;
        for(int n=0;n<64;++n){double w=J[e*64+n]*weights[n];volume+=w;
            for(int c=0;c<5;++c){double u=state[hex_modal::offset(count,e,n,c,soa)];
                value[c][n]=u;average[c]+=w*u;bad|=!std::isfinite(u);}}
        if(bad || !(volume>0)){++invalid;continue;}
        for(int c=0;c<5;++c)average[c]/=volume;
        for(int axis=0;axis<3;++axis)for(int c=0;c<5;++c){
            for(int n=0;n<64;++n){int stride=axis==0?16:axis==1?4:1,row=(n/stride)%4;double sum=0.;
                for(int j=0;j<4;++j)sum+=filter[row*4+j]*value[c][hex_modal::line_node(axis,n,j)];temporary[n]=sum;}
            std::copy(temporary,temporary+64,value[c]);
        }
        for(int c=0;c<5;++c){
            double after=0.;for(int n=0;n<64;++n)after+=J[e*64+n]*weights[n]*value[c][n];
            double correction=average[c]-after/volume,check=0.;
            for(int n=0;n<64;++n){double u=value[c][n]+correction;
                state[hex_modal::offset(count,e,n,c,soa)]=u;check+=J[e*64+n]*weights[n]*u;}
            maximum_error=std::max(maximum_error,std::abs(check/volume-average[c]));
        }
        ++changed;++hits[e];
    }
    statistics[0]=double(changed);statistics[1]=maximum_sensor;
    statistics[2]=double(invalid);statistics[3]=maximum_error;
    return invalid ? 1:0;
}
