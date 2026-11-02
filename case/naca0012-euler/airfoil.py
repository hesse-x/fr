"""Current NACA geometry, physical boundaries and pressure-load postprocessing."""
import math
import torch
from fr.boundaries import euler_characteristic, euler_slip_wall
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
        if len(self.s.blocks)!=1: raise ValueError('airfoil run expects homogeneous triangles or quadrilaterals')
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
            return euler_slip_wall(coords,U,n,tag,t)
        return euler_characteristic(U,n,self.far,self.eq)

    def numerical_flux(self,L,R,n):
        e=self.eq
        speed=torch.maximum(e.normal_speed(L.state,n),e.normal_speed(R.state,n))
        F=.5*(L.flux+R.flux)-.5*speed[...,None]*(R.state-L.state)
        wallflux=torch.zeros_like(F)
        wallflux[...,1:3]=self.wall_pressure(L.state,n)[...,None]*n
        return torch.where((L.tag==2)[...,None],wallflux,F)

    def wall_pressure(self,U,n):
        # Exact algebraic form of Rusanov with reflected normal momentum.
        # Includes acoustic impedance damping absent from a central p*n wall.
        rho,vel,p=self.eq.primitive(U)
        un=(vel*n).sum(-1)
        speed=un.abs()+(1.4*p/rho).sqrt()
        return p+rho*un**2+rho*speed*un

    def common(self,L,R):
        """Boundary-consistent face flux for current FW-H sampling."""
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
