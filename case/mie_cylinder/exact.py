"""Fourier-Bessel reference for a homogeneous dielectric circular cylinder.

TMz, incident Ez=exp(i*k*x), exp(-i*omega*t), outgoing H^(1).
Only reference evaluation is performed here; the numerical PDE solver is separate.
"""
import math
import torch


def bessel_j(order,x):
    term=(x/2)**order/math.factorial(order)
    result=term.clone()
    for k in range(1,65):
        term=term*(-x*x/4)/(k*(order+k))
        result=result+term
    return result


def cylindrical_functions(x,terms):
    if (x<=0).any(): raise ValueError('positive Bessel arguments required')
    J=torch.stack([bessel_j(n,x) for n in range(terms+2)])
    ys=[torch.special.bessel_y0(x),torch.special.bessel_y1(x)]
    for n in range(1,terms+1): ys.append(2*n/x*ys[-1]-ys[-2])
    Y=torch.stack(ys)
    H=J.to(torch.complex128)+1j*Y
    dJ=torch.stack([-J[1]]+[.5*(J[n-1]-J[n+1]) for n in range(1,terms+1)])
    dH=torch.stack([-H[1]]+[.5*(H[n-1]-H[n+1]) for n in range(1,terms+1)])
    return J[:terms+1],H[:terms+1],dJ,dH


class CylinderSeries:
    def __init__(self,radius=.5,wavelength=2.,epsilon_r=4.,mu_r=1.,terms=16):
        self.radius,self.k,self.terms=radius,2*math.pi/wavelength,terms
        self.k_inside=self.k*math.sqrt(epsilon_r*mu_r)
        ka=torch.tensor(self.k*radius,dtype=torch.float64)
        J,H,dJ,dH=cylindrical_functions(ka,terms)
        A,_,dA,_=cylindrical_functions(torch.tensor(self.k_inside*radius,dtype=torch.float64),terms)
        q=self.k_inside/mu_r
        denominator=self.k*dH*A-q*H*dA
        self.coefficients=(q*J*dA-self.k*dJ*A)/denominator
        self.inside_coefficients=self.k*(J*dH-dJ*H)/denominator

    def scattered(self,x,y):
        radius=torch.hypot(x,y)
        if (radius<self.radius*(1-1e-10)).any(): raise ValueError('scattered reference is defined outside cylinder')
        _,H,_,_=cylindrical_functions(self.k*radius,self.terms)
        theta=torch.atan2(y,x)
        result=self.coefficients[0]*H[0]
        for n in range(1,self.terms+1):
            result=result+2*(1j**n)*self.coefficients[n]*H[n]*torch.cos(n*theta)
        return result

    def angular_width(self,theta):
        far=self.coefficients[0]*torch.ones_like(theta,dtype=torch.complex128)
        for n in range(1,self.terms+1): far=far+2*self.coefficients[n]*torch.cos(n*theta)
        return 2/(math.pi*self.k)*far.abs().square()

    def total_width(self):
        return (4/self.k*(self.coefficients[0].abs().square()+2*self.coefficients[1:].abs().square().sum())).item()
