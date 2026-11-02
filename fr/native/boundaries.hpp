#pragma once
#include <cmath>

namespace fr_boundary {
// Cartesian Maxwell state (Ex,Ey,Ez,Hx,Hy,Hz). Pointers may alias.
inline void maxwell_pec(const double* inside, int axis, double* ghost) {
    for(int v=0;v<6;++v)
        ghost[v]=((v<3 && v!=axis) || v==axis+3) ? -inside[v] : inside[v];
}
inline void zero_incoming(double* ghost, int variables) {
    for(int v=0;v<variables;++v) ghost[v]=0.;
}

// BR1 primitive traces for a normalized far field and an adiabatic no-slip
// wall. Tags are chosen by the mesh adapter, never assumed by the kernel.
inline double primitive_trace(double inside, double neighbor, double far,
                              int component, bool wall, bool outer) {
    if(wall) return component<2?0.:inside;
    if(outer) return far;
    return .5*(inside+neighbor);
}
inline double primitive_jump(double inside, double neighbor, double far,
                             int component, bool wall, bool outer) {
    if(wall) return component<2?-inside:0.;
    return (outer?far:neighbor)-inside;
}
// 2D gamma=1.4, normalized rho_inf=c_inf=1; matches Python's callback.
inline void euler_characteristic(const double* u,double nx,double ny,
                                 const double* far,double* r) {
    double vx=u[1]/u[0], vy=u[2]/u[0];
    double pressure=.4*(u[3]-.5*(u[1]*vx+u[2]*vy));
    double un=vx*nx+vy*ny, unf=far[1]*nx+far[2]*ny;
    double rp=un+5*std::sqrt(1.4*pressure/u[0]), rm=unf-5;
    double ub=.5*(rp+rm), cb=.1*(rp-rm);
    double entropy=un<0?1/1.4:pressure/std::pow(u[0],1.4);
    double rho=std::pow(cb*cb/(1.4*entropy),2.5), pr=entropy*std::pow(rho,1.4);
    vx=(un<0?far[1]-unf*nx:vx-un*nx)+ub*nx;
    vy=(un<0?far[2]-unf*ny:vy-un*ny)+ub*ny;
    r[0]=rho; r[1]=rho*vx; r[2]=rho*vy; r[3]=pr/.4+.5*rho*(vx*vx+vy*vy);
}
}
