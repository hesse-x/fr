#pragma once
// Included inside fr_convection, after numerical/solve_face. All buffers are
// cell/face-local; quadrature nodes never change the solution DOFs.
#if FR_NS_VOLUME_AA || FR_NS_SURFACE_AA
constexpr int AQ=FR_NS_NQ;
template<int Input,int Output>
inline void tensor3(const double* src,double* dst,const double* matrix) {
    double first[Output*Input*Input],second[Output*Output*Input];
    for(int a=0;a<Output;++a) for(int b=0;b<Input;++b) for(int c=0;c<Input;++c) {
        double value=0.;
        for(int i=0;i<Input;++i) value+=matrix[a*Input+i]*src[(i*Input+b)*Input+c];
        first[(a*Input+b)*Input+c]=value;
    }
    for(int a=0;a<Output;++a) for(int b=0;b<Output;++b) for(int c=0;c<Input;++c) {
        double value=0.;
        for(int j=0;j<Input;++j) value+=matrix[b*Input+j]*first[(a*Input+j)*Input+c];
        second[(a*Output+b)*Input+c]=value;
    }
    for(int a=0;a<Output;++a) for(int b=0;b<Output;++b) for(int c=0;c<Output;++c) {
        double value=0.;
        for(int k=0;k<Input;++k) value+=matrix[c*Input+k]*second[(a*Output+b)*Input+k];
        dst[(a*Output+b)*Output+c]=value;
    }
}
template<int Input,int Output>
inline void tensor2(const double* src,double* dst,const double* matrix) {
    double first[Output*Input];
    for(int a=0;a<Output;++a) for(int b=0;b<Input;++b) {
        double value=0.;
        for(int i=0;i<Input;++i) value+=matrix[a*Input+i]*src[i*Input+b];
        first[a*Input+b]=value;
    }
    for(int a=0;a<Output;++a) for(int b=0;b<Output;++b) {
        double value=0.;
        for(int j=0;j<Input;++j) value+=matrix[b*Input+j]*first[a*Input+j];
        dst[a*Output+b]=value;
    }
}
#endif
#if FR_NS_VOLUME_AA
inline void volume_overintegrated(const double* packed,double* flux,int count,
    const double* reference,double gamma,const double* interpolation,const double* projection) {
    constexpr int Points=AQ*AQ*AQ,Stride=(Points+3)/4*4;
    const int stride=count*64;
    for(int e=0;e<count;++e) {
        alignas(32) double state[5*Stride],qflux[15*Stride];
        for(int c=0;c<5;++c) {
            tensor3<4,AQ>(packed+c*stride+e*64,state+c*Stride,interpolation);
            // Inactive SIMD lanes copy a valid state; their flux is discarded.
            for(int n=Points;n<Stride;++n) state[c*Stride+n]=state[c*Stride];
        }
        for(int n=0;n<Stride;n+=4) {
            Vec rho=simd::load(state+n),u[3],speed2=0.;
            for(int a=0;a<3;++a) { u[a]=simd::load(state+(a+1)*Stride+n)/rho; speed2=speed2+u[a]*u[a]; }
            Vec energy=simd::load(state+4*Stride+n);
            Vec pressure=(gamma-1)*(energy-.5*rho*speed2);
            for(int axis=0;axis<3;++axis) for(int c=0;c<5;++c) {
                Vec value=simd::load(state+c*Stride+n)*u[axis];
                if(c==axis+1) value=value+pressure;
                if(c==4) value=value+pressure*u[axis];
                simd::store(qflux+(axis*5+c)*Stride+n,value);
            }
        }
        for(int axis=0;axis<3;++axis) for(int c=0;c<5;++c) {
            double projected[64];
            tensor3<AQ,4>(qflux+(axis*5+c)*Stride,projected,projection);
            for(int n=0;n<64;++n)
                flux[(axis*5+c)*stride+e*64+n]=projected[n]-reference[axis*5+c];
        }
    }
}
#endif
#if FR_NS_SURFACE_AA
inline void surface_overintegrated(int axis,const double* left,const double* right,double* output,
    const double* reference,double gamma,const double* interpolation,const double* projection) {
    constexpr int Points=AQ*AQ,Stride=(Points+3)/4*4;
    alignas(32) double qleft[5*Stride],qright[5*Stride],qflux[5*Stride];
    for(int c=0;c<5;++c) {
        tensor2<4,AQ>(left+c*16,qleft+c*Stride,interpolation);
        tensor2<4,AQ>(right+c*16,qright+c*Stride,interpolation);
        for(int n=Points;n<Stride;++n) {
            qleft[c*Stride+n]=qleft[c*Stride];
            qright[c*Stride+n]=qright[c*Stride];
        }
    }
    solve_face<Stride>(axis,qleft,qright,qflux,reference,gamma);
    for(int c=0;c<5;++c) tensor2<AQ,4>(qflux+c*Stride,output+c*16,projection);
}
#endif
