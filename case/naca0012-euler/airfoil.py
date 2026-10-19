"""Airfoil Euler FR-DG physics with explicit global or local time stepping."""
import math
import torch
from fr import Mesh2D, MeshFR2D
from fr.equations import Euler


class Airfoil:
    def __init__(self, mesh, p=3, mach=.3, alpha=0., quadrature_order=None):
        self.eq=Euler(2)
        self.mach=mach
        self.alpha=alpha
        a=math.radians(alpha)
        self.direction=torch.tensor([math.cos(a),math.sin(a)],dtype=torch.float64)
        self.far=torch.tensor([1.,mach*math.cos(a),mach*math.sin(a),1/1.4/.4+.5*mach**2],dtype=torch.float64)
        self.mesh=Mesh2D.read(mesh)
        if set(self.mesh.face_tags[self.mesh.boundary].tolist())!={1,2}:
            raise ValueError('expected farfield=1 and airfoil=2')
        self.s=MeshFR2D(self.mesh,p=p,flux=self.eq.flux,
                       numerical_flux=self.numerical_flux,boundary=self.boundary,
                       speed=self.eq.normal_speed,quadrature_order=quadrature_order)
        self.wall=self.mesh.boundary & (self.mesh.face_tags==2)
        self.wall_ids=torch.where(self.wall)[0]
        self.outer_ids=torch.where(self.mesh.boundary & (self.mesh.face_tags==1))[0]
        if len(self.s.blocks)!=1: raise ValueError('airfoil run expects homogeneous triangles')
        s=self.s; b=s.blocks[0]
        inv=torch.linalg.inv(b['mass'])
        self.Gx=inv@b['Gx']; self.Gy=inv@b['Gy']
        self.lift=-torch.einsum('eij,fqj,q,efq->efiq',inv,b['face_B'],s.fw,s.face_jacobian[b['ids']])
        # One contiguous matrix per element, avoiding a lift permutation each RHS.
        self.lift_matrix=self.lift.permute(0,2,1,3).flatten(2).contiguous()
        face_cell=torch.empty(self.mesh.nfaces,dtype=torch.long)
        face_local=torch.empty_like(face_cell)
        face_cell[b['ids']]=torch.arange(b['ne'])[:,None].expand_as(b['ids'])
        face_local[b['ids']]=torch.arange(b['ids'].shape[1])[None,:].expand_as(b['ids'])
        self.wall_cells=face_cell[self.wall_ids]
        self.wall_basis=b['face_B'][face_local[self.wall_ids]]
        self.base_step=b['areas']/((2*p+1)*(s.face_jacobian[b['ids']]*s.fw).sum((1,2)))

    def boundary(self,coords,U,n,tag,t):
        if tag==2:
            outside=U.clone()
            outside[...,1:3]-=2*(U[...,1:3]*n).sum(-1)[...,None]*n
            return outside
        rho,vel,p=self.eq.primitive(U)
        c=(1.4*p/rho).sqrt()
        un=(vel*n).sum(-1)
        vinf=self.far[1:3]
        unf=(vinf*n).sum(-1)
        rp=un+5*c
        rm=unf-5
        ub=.5*(rp+rm); cb=.1*(rp-rm)
        incoming=un<0
        entropy=torch.where(incoming,torch.full_like(rho,1/1.4),p/rho**1.4)
        rb=(cb**2/(1.4*entropy))**2.5
        pb=entropy*rb**1.4
        tangent=torch.where(incoming[...,None],vinf-unf[...,None]*n,vel-un[...,None]*n)
        vb=tangent+ub[...,None]*n
        return torch.cat((rb[...,None],rb[...,None]*vb,(pb/.4+.5*rb*(vb**2).sum(-1))[...,None]),-1)

    def numerical_flux(self,L,R,n):
        e=self.eq
        speed=torch.maximum(e.normal_speed(L.state,n),e.normal_speed(R.state,n))
        F=.5*(L.flux+R.flux)-.5*speed[...,None]*(R.state-L.state)
        wallflux=torch.zeros_like(F)
        wallflux[...,1:3]=self.wall_pressure(L.state,n)[...,None]*n
        return torch.where((L.tag==2)[...,None],wallflux,F)

    @staticmethod
    def flux_xy(U):
        rho=U[...,0]; u=U[...,1]/rho; v=U[...,2]/rho
        p=.4*(U[...,3]-.5*(U[...,1]*u+U[...,2]*v))
        z=torch.zeros_like(p)
        Fx=U*u[...,None]+torch.stack((z,p,z,p*u),-1)
        Fy=U*v[...,None]+torch.stack((z,z,p,p*v),-1)
        return Fx,Fy

    def wall_pressure(self,U,n):
        # Exact algebraic form of Rusanov with reflected normal momentum.
        # Includes acoustic impedance damping absent from a central p*n wall.
        rho,vel,p=self.eq.primitive(U)
        un=(vel*n).sum(-1)
        speed=un.abs()+(1.4*p/rho).sqrt()
        return p+rho*un**2+rho*speed*un

    def common(self,L,R):
        s=self.s; R=R.clone()
        R[self.outer_ids]=self.boundary(None,L[self.outer_ids],s.normals[self.outer_ids],1,0.)
        n=s.normals
        def normal_flux(U):
            rho,vel,p=self.eq.primitive(U)
            un=(vel*n).sum(-1)
            speed=un.abs()+(1.4*p/rho).sqrt()
            z=torch.zeros_like(p)
            F=U*un[...,None]+torch.stack((z,p*n[...,0],p*n[...,1],p*un),-1)
            return F,speed
        fl,sl=normal_flux(L); fr,sr=normal_flux(R)
        speed=torch.maximum(sl,sr)
        F=.5*(fl+fr)-.5*speed[...,None]*(R-L)
        wn=n[self.wall_ids]
        pressure=self.wall_pressure(L[self.wall_ids],wn)
        z=torch.zeros_like(pressure)
        F[self.wall_ids]=torch.stack((z,pressure*wn[...,0],pressure*wn[...,1],z),-1)
        return F

    def rhs(self,U):
        s=self.s; b=s.blocks[0]
        V=U.reshape(b['ne'],b['ns'],4)
        Q=b['B']@V
        fx,fy=self.flux_xy(Q)
        L=s._states(U)
        R=s.neighbor_states(L)
        common=self.common(L,R)
        result=self.Gx@fx+self.Gy@fy+self.lift_matrix@common[b['ids']].reshape(b['ne'],-1,4)
        return result.reshape(-1,4)

    def local_dt(self,U,cfl):
        b=self.s.blocks[0]
        rho,vel,p=self.eq.primitive(b['B']@U.reshape(b['ne'],b['ns'],4))
        speed=(vel.norm(dim=-1)+(1.4*p/rho).sqrt()).amax(1)
        return (cfl*self.base_step/speed)[:,None,None]

    def valid(self,U):
        if not self.eq.valid(U): return False
        if not self.eq.valid(self.s._states(U)): return False
        return all(self.eq.valid(b['B']@U[b['offset']].reshape(b['ne'],b['ns'],4)) for b in self.s.blocks)

    def loads(self,U):
        s=self.s; ids=self.wall_ids
        # Loads need only wall traces, not interpolation on every mesh face.
        V=U.reshape(s.blocks[0]['ne'],s.blocks[0]['ns'],4)
        wall_states=self.wall_basis@V[self.wall_cells]
        p=self.wall_pressure(wall_states,s.normals[ids])
        # Fluid-domain outward normals point into the solid: body force is +p*n.
        force=((p-1/1.4)[...,None]*s.normals[ids]*s.face_jacobian[ids][...,None]*s.fw[None,:,None]).sum((0,1))/(.5*self.mach**2)
        cd=float(force@self.direction)
        cl=float(force@torch.stack((-self.direction[1],self.direction[0])))
        return cl,cd,p

    def rk4(self,U,dt,rhs=None):
        f=self.rhs if rhs is None else rhs
        V=U.reshape(-1,self.s.blocks[0]['ns'],4)
        def g(X):return f(X.reshape_as(U)).reshape_as(V)
        k1=g(V); k2=g(V+.5*dt*k1); k3=g(V+.5*dt*k2); k4=g(V+dt*k3)
        return (V+dt/6*(k1+2*k2+2*k3+k4)).reshape_as(U)

    def ssprk43(self,U,dt,rhs=None):
        from fr.time import ssprk43
        f=self.rhs if rhs is None else rhs
        V=U.reshape(-1,self.s.blocks[0]['ns'],4)
        return ssprk43(lambda X,t:f(X.reshape_as(U)).reshape_as(V),V,0.,dt).reshape_as(U)

    def make_step(self,method='ssprk43',time_mode='global',*,compile=False):
        """Build one adaptive RK step; CFL stays a scalar tensor across calls."""
        if method not in ('ssprk43','rk4') or time_mode not in ('global','local'):
            raise ValueError('unknown RK method or time mode')
        advance=getattr(self,method)
        def step(U,cfl):
            dt=self.local_dt(U,cfl)
            if time_mode=='global':dt=dt.amin().expand_as(dt)
            return advance(U,dt),dt
        return torch.compile(step,fullgraph=True) if compile else step
