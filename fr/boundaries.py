"""Composable exterior-state callbacks, with positive-axis flux convention."""
import torch

def extrapolate(coords, inside, axis, side, t):
    return inside


def prescribed(state):
    return lambda coords, inside, axis, side, t: state(coords, t)


def per_face(callbacks, default=extrapolate):
    return lambda coords, inside, axis, side, t: callbacks.get((axis, side), default)(coords, inside, axis, side, t)


def acoustic_wall(coords, inside, axis, side, t):
    ghost = inside.clone()
    ghost[..., axis+1] *= -1
    return ghost


def normal_extrapolate(coords,inside,normal,tag,t):
    return inside


def per_tag(callbacks,default=normal_extrapolate):
    """Dispatch an unstructured-mesh boundary by its integer physical tag."""
    return lambda coords,inside,normal,tag,t:callbacks.get(tag,default)(coords,inside,normal,tag,t)


def euler_slip_wall(coords, inside, normal, tag, t):
    """Reflect normal momentum; preserve density and total energy."""
    ghost = inside.clone()
    ghost[...,1:3] -= 2*(inside[...,1:3]*normal).sum(-1)[...,None]*normal
    return ghost


def maxwell_pec(coords, inside, axis, side, t):
    """Cartesian PEC: E_t and H_n odd, E_n and H_t even."""
    ghost = inside.clone()
    ghost[...,:3] = -inside[...,:3]
    ghost[...,axis] = inside[...,axis]
    ghost[...,axis+3] = -inside[...,axis+3]
    return ghost


def euler_characteristic(inside, normal, far, equation):
    """2D gamma=1.4 characteristic far field, rho_inf=c_inf=1.

    Preserve outgoing entropy/tangential velocity; prescribe incoming values.
    Far contains the conservative normalized freestream state.
    """
    rho,vel,p=equation.primitive(inside)
    c=(1.4*p/rho).sqrt()
    un=(vel*normal).sum(-1)
    vinf=far[1:3]
    unf=(vinf*normal).sum(-1)
    rp=un+5*c
    rm=unf-5
    ub=.5*(rp+rm); cb=.1*(rp-rm)
    incoming=un<0
    entropy=torch.where(incoming,torch.full_like(rho,1/1.4),p/rho**1.4)
    rb=(cb**2/(1.4*entropy))**2.5
    pb=entropy*rb**1.4
    tangent=torch.where(incoming[...,None],vinf-unf[...,None]*normal,vel-un[...,None]*normal)
    vb=tangent+ub[...,None]*normal
    return torch.cat((rb[...,None],rb[...,None]*vb,(pb/.4+.5*rb*(vb**2).sum(-1))[...,None]),-1)
