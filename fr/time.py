"""Explicit integrators. rhs(U,t) may depend on stage time."""
import math
import torch


def rk4(rhs, U, t, dt):
    k1 = rhs(U,t)
    k2 = rhs(U+dt/2*k1,t+dt/2)
    k3 = rhs(U+dt/2*k2,t+dt/2)
    k4 = rhs(U+dt*k3,t+dt)
    return U+dt/6*(k1+2*k2+2*k3+k4)


def ssprk43(rhs, U, t, dt):
    """Four-stage third-order SSP RK, SSP coefficient 2.

    Stage abscissae: 0, 1/2, 1, 1/2. Strong stability inherits the
    forward-Euler property only when dt <= 2*dt_FE.
    """
    u1 = U+dt/2*rhs(U,t)
    u2 = u1+dt/2*rhs(u1,t+dt/2)
    u3 = 2/3*U+1/3*u2+dt/6*rhs(u2,t+dt)
    return u3+dt/2*rhs(u3,t+dt/2)


METHODS = {'rk4': rk4, 'ssprk43': ssprk43}


@torch.no_grad()
def integrate(rhs, U, final_time, *, dt, method='ssprk43', start_time=0., validate=None, prepare_step=None, monitor=None):
    """dt is a positive constant or callable(U,t) returning a time step.

    A custom stepper callable(rhs,U,t,dt) can replace the method string.
    prepare_step(actual_dt,t) is called after truncation, before RK stages.
    monitor(U,t) observes the initial state and every accepted step, including final.
    """
    step = METHODS[method] if isinstance(method,str) else method
    if not math.isfinite(final_time) or final_time < start_time:
        raise ValueError('finite final_time >= start_time required')
    U = U.clone()
    t, steps = float(start_time), 0
    if monitor is not None:
        monitor(U,t)
    while t < final_time:
        if validate is not None and not validate(U):
            raise RuntimeError('Invalid state before time step')
        proposed = float(dt(U,t) if callable(dt) else dt)
        if not math.isfinite(proposed) or proposed <= 0:
            raise ValueError('time step must be finite and positive')
        h = min(proposed,final_time-t)
        if t+h == t:
            raise ValueError('time step too small to advance time')
        if prepare_step is not None:
            prepare_step(h, t)
        U = step(rhs,U,t,h)
        t = min(t+h,final_time)
        steps += 1
        if monitor is not None:
            monitor(U,t)
    if validate is not None and not validate(U):
        raise RuntimeError('Invalid final state')
    return U, {'time': t, 'steps': steps}
