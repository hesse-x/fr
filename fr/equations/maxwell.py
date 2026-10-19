"""Source-free Maxwell equations in a uniform, isotropic lossless material.

State: (Ex,Ey,Ez,Hx,Hy,Hz). All fluxes point along the positive coordinate
axis. Material interfaces, conductivity and divergence cleaning are absent.
"""
import math
import torch


class Maxwell:
    def __init__(self,epsilon=1.,mu=1.):
        if not all(math.isfinite(v) and v>0 for v in (epsilon,mu)):
            raise ValueError('epsilon and mu must be finite and positive')
        self.epsilon,self.mu=epsilon,mu
        self.dim,self.nvars=3,6
        self.c=1/math.sqrt(epsilon*mu)
        self.impedance=math.sqrt(mu/epsilon)

    def flux(self,U,axis):
        if U.shape[-1]!=6 or axis not in (0,1,2):
            raise ValueError('Maxwell requires six fields and axis=0/1/2')
        # F_E = -e_axis x H / epsilon; F_H = e_axis x E / mu.
        j,k=(axis+1)%3,(axis+2)%3
        F=torch.zeros_like(U)
        F[...,j]=U[...,3+k]/self.epsilon
        F[...,k]=-U[...,3+j]/self.epsilon
        F[...,3+j]=-U[...,k]/self.mu
        F[...,3+k]=U[...,j]/self.mu
        return F

    def speed(self,U,axis):
        return torch.full_like(U[...,0],self.c)

    def normal_speed(self,U,normal):
        return torch.full_like(U[...,0],self.c)

    def absolute_normal_jacobian_action(self,jump,minus,plus,normal):
        if normal.shape[-1]==2:
            normal=torch.cat((normal,torch.zeros_like(normal[...,:1])),-1)
        out=jump.clone()
        for offset in (0,3):
            out[...,offset:offset+3]-=(jump[...,offset:offset+3]*normal).sum(-1)[...,None]*normal
        return self.c*out

    def roe_flux(self,minus,plus,axis):
        """Exact characteristic upwind/Roe flux for constant epsilon,mu.

        The eigenvalues are (-c,-c,0,0,c,c). Thus |A_axis|=c*P_t,
        where P_t removes both normal components. Unlike scalar Rusanov,
        the two zero-speed normal fields receive no artificial penalty.
        """
        return .5*(minus.flux+plus.flux)-.5*self.absolute_jacobian_action(plus.state-minus.state,minus.state,plus.state,axis)

    def absolute_jacobian_action(self,jump,minus,plus,axis):
        transverse=jump.clone()
        transverse[...,axis]=0
        transverse[...,axis+3]=0
        return self.c*transverse

    @staticmethod
    def pec_boundary(coords,inside,axis,side,t):
        """PEC exterior state: E_t odd, E_n even; H_t even, H_n odd.

        The average traces enforce E_t=0 and H_n=0 on either box wall.
        """
        ghost=inside.clone()
        ghost[...,:3]=-inside[...,:3]
        ghost[...,axis]=inside[...,axis]
        ghost[...,axis+3]=-inside[...,axis+3]
        return ghost

    def energy_density(self,U):
        return .5*(self.epsilon*(U[...,:3]**2).sum(-1)+self.mu*(U[...,3:]**2).sum(-1))

    @staticmethod
    def valid(U):
        return bool(torch.isfinite(U).all())


class MaxwellTMz:
    """2D TMz projection of the public Maxwell model: (Ez,Hx,Hy).

    material_flux supplies the bare curl flux for a capacity formulation
    diag(epsilon,mu,mu) U_t + div(F_bare) = 0.
    """
    def __init__(self):
        self.full = Maxwell()
        self.dim,self.nvars,self.c=2,3,1.
        self.components=(2,3,4)
        eye=torch.zeros(3,6,dtype=torch.float64)
        eye[torch.arange(3),torch.tensor(self.components)]=1.
        self.matrices=tuple(self.full.flux(eye,a)[:,self.components].T.contiguous() for a in range(2))

    def flux(self,U,axis):
        return U@self.matrices[axis].to(device=U.device,dtype=U.dtype).T

    def speed(self,U,axis):
        return torch.ones_like(U[...,0])

    def absolute_jacobian_action(self,jump,minus,plus,axis):
        out=jump.clone()
        out[...,axis+1]=0
        return out

    @staticmethod
    def absolute_normal_jacobian_action(jump,minus,plus,normal):
        out=jump.clone()
        out[...,1:]-=(jump[...,1:]*normal).sum(-1)[...,None]*normal
        return out

    @staticmethod
    def material_flux(minus,plus,axis,epsilon_minus,epsilon_plus,mu_minus=1.,mu_plus=1.):
        """Exact impedance Riemann flux at a planar dielectric face."""
        zm=(mu_minus/epsilon_minus)**.5
        zp=(mu_plus/epsilon_plus)**.5
        sign=1 if axis==0 else -1
        tangent=2 if axis==0 else 1
        em,ep=minus.state[...,0],plus.state[...,0]
        hm,hp=sign*minus.state[...,tangent],sign*plus.state[...,tangent]
        electric=(zp*em+zm*ep+zm*zp*(hp-hm))/(zm+zp)
        magnetic=(zm*hm+zp*hp+(ep-em))/(zm+zp)
        out=torch.zeros_like(minus.state)
        out[...,0]=-magnetic
        out[...,tangent]=-sign*electric
        return out

    @staticmethod
    def pec_boundary(coords,inside,axis,side,t):
        ghost=inside.clone()
        ghost[...,0]*=-1
        ghost[...,axis+1]*=-1
        return ghost

    @staticmethod
    def valid(U):
        return bool(torch.isfinite(U).all())

    @staticmethod
    def interface_state(minus,plus,normal,epsilon_minus,epsilon_plus,mu_minus=1.,mu_plus=1.):
        """TMz impedance Riemann state for an arbitrary unit face normal."""
        zm=(mu_minus/epsilon_minus)**.5
        zp=(mu_plus/epsilon_plus)**.5
        nx,ny=normal.unbind(-1)
        hm=nx*minus[...,2]-ny*minus[...,1]
        hp=nx*plus[...,2]-ny*plus[...,1]
        electric=(zp*minus[...,0]+zm*plus[...,0]+zm*zp*(hp-hm))/(zm+zp)
        tangent=(zm*hm+zp*hp+(plus[...,0]-minus[...,0]))/(zm+zp)
        longitudinal=.5*(nx*(minus[...,1]+plus[...,1])+ny*(minus[...,2]+plus[...,2]))
        return torch.stack((electric,nx*longitudinal-ny*tangent,ny*longitudinal+nx*tangent),-1)

    @staticmethod
    def normal_pec_boundary(coords,inside,normal,tag,t):
        ghost=inside.clone()
        ghost[...,0]=-inside[...,0]
        projection=(inside[...,1:]*normal).sum(-1)
        ghost[...,1:]=inside[...,1:]-2*projection[...,None]*normal
        return ghost

    @staticmethod
    def normal_speed(U,normal):
        return torch.ones_like(U[...,0])
