"""Curvilinear mixed-element FR-DG (triangle VCJH c=0 member).

Physical state is a flat (degrees_of_freedom, variables) tensor. Geometric
mass matrices and face lifting are preintegrated. Fluxes are evaluated with
volume overintegration; no c>0 VCJH family is claimed.
"""
from dataclasses import dataclass
import math
import torch
from .reference import Basis, gmsh_reference_nodes, quadrature, edge_points
from .operators import operators


@dataclass(frozen=True)
class NormalFace:
    state: torch.Tensor
    flux: torch.Tensor
    coords: tuple
    normal: torch.Tensor
    region: torch.Tensor
    tag: torch.Tensor


def normal_central(minus,plus,normal):
    return .5*(minus.flux+plus.flux)


def normal_rusanov_flux(speed):
    """speed(U,unit_normal) returns a normal spectral radius."""
    def callback(m,p,n):
        a=torch.maximum(speed(m.state,n),speed(p.state,n))
        return .5*(m.flux+p.flux)-.5*a[...,None]*(p.state-m.state)
    return callback


class MeshFR2D:
    def __init__(self,mesh,p=3,*,flux,numerical_flux,boundary=None,source=None,
                 speed=None,device='cpu',quadrature_order=None):
        self.mesh,self.p,self.dim=mesh,p,2
        self.flux,self.numerical_flux,self.boundary,self.source,self.speed=flux,numerical_flux,boundary,source,speed
        self.device=device
        geometry_order=max(b.order for b in mesh.blocks)
        nq=p+geometry_order+2 if quadrature_order is None else quadrature_order
        if nq<p+1: raise ValueError('quadrature_order must be at least p+1')
        s,w=operators(nq-1,device)[:2]
        self.fw=w/2
        s=s/2+.5
        self.blocks=[]
        coords=[]
        self.ndof=0
        self.face_coordinates=torch.empty(mesh.nfaces,nq,2,dtype=torch.float64,device=device)
        self.normals=torch.empty_like(self.face_coordinates)
        self.face_jacobian=torch.empty(mesh.nfaces,nq,dtype=torch.float64,device=device)
        self.regions=torch.empty(mesh.nfaces,nq,dtype=torch.long,device=device)
        self.nodal_regions=[]
        self.minimum_jacobian=math.inf
        for bi,block in enumerate(mesh.blocks):
            basis=Basis(block.kind,p,device=device)
            geometry=Basis(block.kind,block.order,nodes=gmsh_reference_nodes(block.kind,block.order),device=device)
            X=block.geometry.to(device)
            q,qw=quadrature(block.kind,nq,device)
            B,Br,Bs=(basis.evaluate(q,a) for a in (None,0,1))
            def evaluate(points,derivative=None):
                return torch.einsum('qi,eik->eqk',geometry.evaluate(points,derivative),X)
            xy=evaluate(q)
            xr,xs=evaluate(q,0),evaluate(q,1)
            jac=xr[...,0]*xs[...,1]-xr[...,1]*xs[...,0]
            fp,ft=edge_points(block.kind,s)
            face_B=torch.stack([basis.evaluate(f) for f in fp])
            fxy=torch.stack([evaluate(f) for f in fp],1)
            tangent=torch.stack([evaluate(f,0)*t[0]+evaluate(f,1)*t[1] for f,t in zip(fp,ft)],1)
            fj=tangent.norm(dim=-1)
            # Check nodes, edges and integration points, not just the centroid.
            check=torch.cat((basis.nodes,geometry.nodes,q,fp.flatten(0,1)))
            cr,cs=evaluate(check,0),evaluate(check,1)
            cj=cr[...,0]*cs[...,1]-cr[...,1]*cs[...,0]
            if not torch.isfinite(cj).all() or (cj<=0).any() or (fj<=0).any():
                raise ValueError('inverted, folded or degenerate element geometry')
            self.minimum_jacobian=min(self.minimum_jacobian,cj.min().item())
            normal=torch.stack((tangent[...,1],-tangent[...,0]),-1)/fj[...,None]
            ids=mesh.face_ids[bi].to(device)
            self.face_coordinates[ids]=fxy
            self.normals[ids]=normal
            self.face_jacobian[ids]=fj
            self.regions[ids]=block.regions.to(device)[:,None,None].expand_as(fj)
            weight=jac*qw
            mass=torch.einsum('qi,eq,qj->eij',B,weight,B)
            chol=torch.linalg.cholesky(mass)
            Gx=Br.T[None,:,:]*(qw[None,:]*xs[...,1])[:,None,:]-Bs.T[None,:,:]*(qw[None,:]*xr[...,1])[:,None,:]
            Gy=-Br.T[None,:,:]*(qw[None,:]*xs[...,0])[:,None,:]+Bs.T[None,:,:]*(qw[None,:]*xr[...,0])[:,None,:]
            ne,ns=len(X),len(basis.nodes)
            region=block.regions.to(device)
            data=dict(basis=basis,geometry=geometry,X=X,B=B,face_B=face_B,xy=xy,weight=weight,
                      chol=chol,mass=mass,Gx=Gx,Gy=Gy,ids=ids,ne=ne,ns=ns,
                      region=region,offset=slice(self.ndof,self.ndof+ne*ns),areas=weight.sum(-1))
            self.blocks.append(data)
            self.ndof+=ne*ns
            coords.append(evaluate(basis.nodes).reshape(-1,2))
            self.nodal_regions.append(region[:,None].expand(ne,ns).reshape(-1))
        coordinates=torch.cat(coords)
        self.coords=tuple(coordinates.unbind(-1))
        self.nodal_regions=torch.cat(self.nodal_regions)
        self.shape=(self.ndof,)
        self.volume=sum(b['areas'].sum().item() for b in self.blocks)
        self.neighbor=mesh.neighbor.to(device)
        self.flip=mesh.flip.to(device)
        self.tags=mesh.face_tags.to(device)[:,None].expand(mesh.nfaces,nq)
        mask=mesh.boundary.to(device)
        self.boundary_groups=[(int(tag),torch.where(mask&(mesh.face_tags.to(device)==tag))[0])
                              for tag in torch.unique(mesh.face_tags.to(device)[mask]).tolist()]
        if self.boundary_groups and boundary is None:
            raise ValueError('mesh exterior requires boundary(coords,inside,normal,tag,t)')
        other=self.face_coordinates[self.neighbor]
        other=torch.where(self.flip[:,None,None],other.flip(1),other)
        interior=~mask
        scale=max(1.,self.face_coordinates.abs().max().item())
        if interior.any() and (self.face_coordinates[interior]-other[interior]).abs().max()>1e-10*scale:
            raise ValueError('neighboring curved edges do not match; hanging/nonconforming faces are unsupported')
        other_n=self.normals[self.neighbor]
        other_n=torch.where(self.flip[:,None,None],other_n.flip(1),other_n)
        if interior.any() and (self.normals[interior]+other_n[interior]).abs().max()>1e-8:
            raise ValueError('shared face normals are not opposite')

    def _states(self,U):
        if U.ndim!=2 or U.shape[0]!=self.ndof: raise ValueError('expected state (ndof,variables)')
        faces=torch.empty(*self.tags.shape,U.shape[-1],dtype=U.dtype,device=U.device)
        for b in self.blocks:
            V=U[b['offset']].reshape(b['ne'],b['ns'],-1)
            faces[b['ids']]=torch.einsum('fqi,eiv->efqv',b['face_B'],V)
        return faces

    def rhs(self,U,t=0.):
        inside=self._states(U)
        outside=inside[self.neighbor]
        outside=torch.where(self.flip[:,None,None],outside.flip(1),outside)
        outside=outside.clone()
        for tag,indices in self.boundary_groups:
            xy=self.face_coordinates[indices]
            outside[indices]=self.boundary(tuple(xy.unbind(-1)),inside[indices],self.normals[indices],tag,t)
        def face(state,region):
            F=sum(self.normals[...,a,None]*self.flux(state,a) for a in (0,1))
            return NormalFace(state,F,tuple(self.face_coordinates.unbind(-1)),self.normals,region,self.tags)
        other_regions=self.regions[self.neighbor]
        common=self.numerical_flux(face(inside,self.regions),face(outside,other_regions),self.normals)
        result=torch.empty_like(U)
        for b in self.blocks:
            V=U[b['offset']].reshape(b['ne'],b['ns'],-1)
            Q=b['B']@V
            residual=b['Gx']@self.flux(Q,0)+b['Gy']@self.flux(Q,1)
            ids=b['ids']
            surface=torch.einsum('fqi,efqv,q,efq->eiv',b['face_B'],common[ids],self.fw,self.face_jacobian[ids])
            residual=residual-surface
            if self.source is not None:
                S=self.source(tuple(b['xy'].unbind(-1)),Q,t,b['region'][:,None].expand_as(b['weight']))
                residual=residual+torch.einsum('qi,eq,eqv->eiv',b['B'],b['weight'],S)
            derivative=torch.cholesky_solve(residual,b['chol'])
            result[b['offset']]=derivative.reshape(-1,U.shape[-1])
        return result

    def integration(self,U):
        """Yield physical quadrature coordinates, states, weights and region tags."""
        for b in self.blocks:
            V=U[b['offset']].reshape(b['ne'],b['ns'],-1)
            yield tuple(b['xy'].unbind(-1)),b['B'].to(U.dtype)@V,b['weight'],b['region'][:,None].expand_as(b['weight'])

    def integral(self,U):
        return sum((V*weights[...,None]).sum((0,1)) for coords,V,weights,region in self.integration(U))

    def norm(self,U):
        squared=sum((V.abs().square()*weights[...,None]).sum((0,1)) for coords,V,weights,region in self.integration(U))
        return (squared/self.volume).sqrt()

    def timestep(self,U,cfl=.15):
        if self.speed is None: raise ValueError('normal spectral speed required for CFL')
        if not math.isfinite(cfl) or cfl<=0: raise ValueError('positive finite CFL required')
        states=self._states(U)
        speed=self.speed(states,self.normals)
        if not torch.isfinite(speed).all() or (speed<0).any(): raise ValueError('invalid normal speed')
        rate=0.
        for b in self.blocks:
            ids=b['ids']
            value=(speed[ids]*self.face_jacobian[ids]*self.fw).sum((1,2))/b['areas']
            rate=max(rate,value.max().item())
        return cfl/((2*self.p+1)*rate) if rate else 1e30

    def write_svg(self,path):
        """Geometry preview with sampled curved edges, independent of plotting libraries."""
        from pathlib import Path
        extent=self.face_coordinates.flatten(0,1)
        lo,hi=extent.min(0).values,extent.max(0).values
        size=(hi-lo).max().item()
        def point(x,y): return (20+760*(x-lo[0].item())/size,780-760*(y-lo[1].item())/size)
        parts=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 800"><rect width="800" height="800" fill="white"/>']
        s=torch.linspace(0,1,20,dtype=torch.float64,device=self.device)
        for b in self.blocks:
            fp,_=edge_points(b['basis'].kind,s)
            for e in range(b['ne']):
                for f,refs in enumerate(fp):
                    face=int(b['ids'][e,f])
                    if int(self.neighbor[face])<face: continue
                    xy=b['geometry'].evaluate(refs)@b['X'][e]
                    points=' '.join(f'{u:.3f},{v:.3f}' for u,v in (point(x,y) for x,y in xy.tolist()))
                    color='#c33' if bool(self.mesh.boundary[face]) else '#586779'
                    parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="0.7"/>')
        parts.append('</svg>')
        Path(path).write_text('\n'.join(parts))


def make_normal_numerical_flux(name,equation):
    """Interface treatment for physical unit normals on unstructured meshes."""
    if callable(name): return name
    if name=='central': return normal_central
    if name=='rusanov': return normal_rusanov_flux(equation.normal_speed)
    if name in ('roe','upwind') and hasattr(equation,'absolute_normal_jacobian_action'):
        return lambda m,p,n:.5*(m.flux+p.flux)-.5*equation.absolute_normal_jacobian_action(p.state-m.state,m.state,p.state,n)
    raise ValueError(f'normal flux {name!r} is unavailable for this equation')
