#pragma once

#include <cmath>
#include <cstdint>

namespace fr_round_jet {
constexpr int points=16;
constexpr int acoustic_metric_size=14;

// Four nozzle parameters, outlet flag, inverse right lift, four derivative
// weights, then [inlet cell][y or z][face point] coordinates.
inline void nozzle(const double* jet,int64_t cell,int f,double time,
                   double (&velocity)[3],double& weight) {
    const double* coords=jet+10+cell*2*points;
    const double y=coords[f],z=coords[points+f];
    const double radius=std::hypot(y,z),angle=std::atan2(z,y);
    const double shape=std::tanh((radius-jet[3])/jet[1]);
    const double scaled=time/.35,ramp=-std::expm1(-scaled*scaled);
    const double phase=6.2831853071795864769*time;
    const double perturb=jet[2]*ramp*(1-shape*shape)*radius/jet[3];
    weight=.5*(1-shape);
    velocity[0]=weight*ramp+perturb*std::cos(4*angle-phase);
    velocity[1]=perturb*std::sin(3*angle+phase);
    velocity[2]=perturb*std::cos(5*angle-1.7*phase);
}

inline void conservative(double density,const double (&v)[3],double pressure,
                         double gamma,double* face,int f) {
    face[f]=density;
    for(int c=0;c<3;++c) face[(c+1)*points+f]=density*v[c];
    face[4*points+f]=pressure/(gamma-1)+.5*density*(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
}

inline void exterior(const double* jet,int64_t cell,int axis,int side,double time,
                     const double* inside,double* ghost,double gamma) {
    for(int f=0;f<points;++f) {
        const double rho=inside[f];
        const double v[3]={inside[points+f]/rho,inside[2*points+f]/rho,inside[3*points+f]/rho};
        const double p=(gamma-1)*(inside[4*points+f]-.5*rho*(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]));
        const double normal=side*v[axis],sound=std::sqrt(gamma*p/rho);
        if(normal>=sound || (axis==0 && side>0 && jet[4] && std::abs(normal)<sound)) {
            for(int c=0;c<5;++c) ghost[c*points+f]=inside[c*points+f];
            continue;
        }
        double target[3]={},target_rho=1.;
        if(axis==0 && side<0) {
            double weight;
            nozzle(jet,cell,f,time,target,weight);
        } else if(axis==0 && normal>=0.) {
            for(int c=0;c<3;++c) target[c]=v[c];
            target_rho=rho;
        }
        if(normal<=-sound) {
            conservative(target_rho,target,jet[0],gamma,ghost,f);
            continue;
        }
        const double scale=2/(gamma-1);
        const double plus=normal+scale*sound;
        const double minus=side*target[axis]-scale*std::sqrt(gamma*jet[0]/target_rho);
        const double speed=.5*(plus+minus),c=.5*(plus-minus)/scale;
        const double entropy=speed<0. ? jet[0]/std::pow(target_rho,gamma) : p/std::pow(rho,gamma);
        const double density=std::pow(c*c/(gamma*entropy),1/(gamma-1));
        double velocity[3];
        for(int i=0;i<3;++i) velocity[i]=speed<0. ? target[i] : v[i];
        velocity[axis]=side*speed;
        conservative(density,velocity,entropy*std::pow(density,gamma),gamma,ghost,f);
    }
}

inline void acoustic_outlet(const double* jet,const double* state,const double* gradient,
                            double* flux,double gamma) {
    for(int f=0;f<points;++f) {
        const double rho=state[f];
        const double u=state[points+f]/rho,v=state[2*points+f]/rho,w=state[3*points+f]/rho;
        const double energy=state[4*points+f],speed2=u*u+v*v+w*w;
        const double p=(gamma-1)*(energy-.5*rho*speed2),c=std::sqrt(gamma*p/rho);
        if(std::abs(u)>=c) continue;
        const double drho=gradient[f],dmx=gradient[points+f];
        const double dp=(gamma-1)*(gradient[4*points+f]-
            (u*dmx+v*gradient[2*points+f]+w*gradient[3*points+f])+.5*speed2*drho);
        // Set L_- to zero in the locally one-dimensional Euler update.
        const double lift=-jet[5]*(u-c)*(dp-c*(dmx-u*drho))/(2*c*c);
        flux[f]+=lift;
        flux[points+f]+=lift*(u-c);
        flux[2*points+f]+=lift*v;
        flux[3*points+f]+=lift*w;
        flux[4*points+f]+=lift*((energy+p)/rho-c*u);
        if(u<0.) {
            const double ambient_rho=std::pow(p/jet[0],1/gamma);
            flux[f]+=u*(ambient_rho-rho);
            flux[points+f]+=u*(ambient_rho*u-state[points+f]);
            flux[2*points+f]-=u*state[2*points+f];
            flux[3*points+f]-=u*state[3*points+f];
            flux[4*points+f]+=u*(p/(gamma-1)+.5*ambient_rho*u*u-energy);
        }
    }
}

inline bool uses_acoustic_correction(double un, double sound, bool correct_backflow) {
    return std::abs(un) < sound && (correct_backflow || un >= 0.);
}

// Correct incoming normal Euler characteristics using physical derivatives.
// The mapped FR lift converts their change at the trace into a common face flux.
template<bool CorrectBackflow>
inline void acoustic_outlet_3d(const double* jet, const double* state, const double* normal_gradient,
                               const double* differentiation, int axis, const double* metric,
                               const double* relaxation, double flux[5][16], double gamma) {
    const int tangent[2] = {axis == 0 ? 1 : 0, axis == 2 ? 1 : 2};
    for (int q = 0; q < points; ++q) {
        const double* mapping = metric + acoustic_metric_size*q;
        const double weight = mapping[13];
        if (weight == 0.) continue;
        const double* normal = mapping + 10;
        const double rho = state[q];
        const double velocity[3] = {state[points+q]/rho, state[2*points+q]/rho, state[3*points+q]/rho};
        const double un = velocity[0]*normal[0] + velocity[1]*normal[1] + velocity[2]*normal[2];
        const double energy = state[4*points+q];
        const double speed2 = velocity[0]*velocity[0] + velocity[1]*velocity[1] + velocity[2]*velocity[2];
        const double p = (gamma-1)*(energy-.5*rho*speed2);
        const double c2 = gamma*p/rho, c = std::sqrt(c2);
        if (!uses_acoustic_correction(un, c, CorrectBackflow)) continue;

        double gradient[3][5];
        for (int component = 0; component < 5; ++component) {
            const double* values = state + component*points;
            double reference[3];
            reference[axis] = normal_gradient[component*points+q];
            double d0 = 0., d1 = 0.;
            for (int i = 0; i < 4; ++i) {
                d0 += differentiation[4*(q/4)+i]*values[4*i+q%4];
                d1 += differentiation[4*(q%4)+i]*values[4*(q/4)+i];
            }
            reference[tangent[0]] = d0;
            reference[tangent[1]] = d1;
            for (int d = 0; d < 3; ++d)
                gradient[d][component] = mapping[d]*reference[0] + mapping[3+d]*reference[1]
                                       + mapping[6+d]*reference[2];
        }

        double div_tangent = 0., transverse = 0., dp_normal = 0.;
        double drho_normal = 0., rho_du_normal[3] = {};
        for (int d = 0; d < 3; ++d) {
            const double* g = gradient[d];
            const double dp = (gamma-1)*(g[4]-velocity[0]*g[1]-velocity[1]*g[2]
                                        -velocity[2]*g[3]+.5*speed2*g[0]);
            double rho_dun = 0.;
            for (int j = 0; j < 3; ++j) {
                const double rho_du = g[j+1]-velocity[j]*g[0];
                rho_dun += normal[j]*rho_du;
                rho_du_normal[j] += normal[d]*rho_du;
            }
            div_tangent += (g[d+1]-velocity[d]*g[0]-normal[d]*rho_dun)/rho;
            transverse += (velocity[d]-un*normal[d])*(dp-c*rho_dun);
            dp_normal += normal[d]*dp;
            if constexpr(CorrectBackflow) drho_normal += normal[d]*g[0];
        }
        transverse += gamma*p*div_tangent;
        const double rho_dun_normal = normal[0]*rho_du_normal[0] + normal[1]*rho_du_normal[1]
                                     +normal[2]*rho_du_normal[2];
        const double mach = std::abs(un)/c;
        const double incoming = (un-c)*(dp_normal-c*rho_dun_normal);
        const double target = relaxation[0]*c*(1-mach*mach)*(p-jet[0])
                            - relaxation[1]*(1-mach)*transverse;
        const double scale = weight*mapping[9];
        const double acoustic = scale*(target-incoming)/(2*c2);
        flux[0][q] = acoustic;
        for (int j = 0; j < 3; ++j)
            flux[j+1][q] = acoustic*(velocity[j]-c*normal[j]);
        flux[4][q] = acoustic*((energy+p)/rho-c*un);

        if constexpr(CorrectBackflow) {
            if (un < 0.) {
                // On the lateral boundary, zero the incoming entropy and shear waves.
                const double contact = -scale*un*(drho_normal-dp_normal/c2);
                flux[0][q] += contact;
                flux[4][q] += .5*speed2*contact;
                for (int j = 0; j < 3; ++j) {
                    const double shear = -scale*un*(rho_du_normal[j]-normal[j]*rho_dun_normal);
                    flux[j+1][q] += contact*velocity[j] + shear;
                    flux[4][q] += velocity[j]*shear;
                }
            }
        }
    }
}
} // namespace fr_round_jet
