#pragma once

// Allocation and parallel execution belong to the caller. These routines only
// schedule stages; callbacks may fuse updates into an existing spatial kernel.
namespace fr_time {
inline double rk4_update(double original, double derivative, double& accumulated,
                         int stage, double dt, double weight, double next_dt) {
    if(stage==0) accumulated=derivative;
    else accumulated+=weight*derivative;
    return stage<3 ? original+next_dt*derivative : original+dt/6*accumulated;
}
struct SSPRK43 {
    static constexpr int stages = 4;
    static constexpr double time[4] = {0., .5, 1., .5};
    static constexpr double force[4] = {.5, .5, 1./6., .5};
    template<class T>
    static inline T update(T original, T state, T rhs, double dt, int stage) {
        return stage == 2 ? (2./3.)*original+(1./3.)*state+dt/6*rhs
                          : state+.5*dt*rhs;
    }
};

// evaluate(stage,time) then update(stage,weight,next_dt). The final update
// writes the accepted state; project is called after each stage update.
template<class Evaluate, class Update, class Project>
inline void rk4(double t, double dt, Evaluate evaluate, Update update, Project project) {
    for(int s=0;s<4;++s) {
        evaluate(s,t+(s==0?0.:s==3?dt:dt/2));
        update(s,(s==0||s==3)?1.:2.,s==2?dt:dt/2);
        project(s,t+((s==2||s==3)?dt:dt/2));
    }
}

// First derivative can be supplied by the caller. evaluate_after also computes
// the accepted state's derivative for residual monitoring and next-step reuse.
template<class Update, class EvaluateAfter>
inline void ssprk43(Update update, EvaluateAfter evaluate_after) {
    for(int s=0;s<SSPRK43::stages;++s) {
        update(s);
        evaluate_after(s);
    }
}
}
