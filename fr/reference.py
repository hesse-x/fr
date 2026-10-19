"""Nodal polynomial bases and quadrature for triangles and quadrilaterals."""
import torch
from .operators import operators


def gmsh_reference_nodes(kind, order):
    """Gmsh complete-element ordering, returned on the unit reference cell."""
    if not isinstance(order,int) or not 1<=order<=5:
        raise ValueError('geometry order must be an integer in 1..5')
    if kind=='quad':
        def walk(lo,hi):
            if lo>hi: return []
            if lo==hi: return [(lo,lo)]
            out=[(lo,lo),(hi,lo),(hi,hi),(lo,hi)]
            out += [(i,lo) for i in range(lo+1,hi)]
            out += [(hi,i) for i in range(lo+1,hi)]
            out += [(i,hi) for i in range(hi-1,lo,-1)]
            out += [(lo,i) for i in range(hi-1,lo,-1)]
            return out+walk(lo+1,hi-1)
        points=walk(0,order)
    elif kind=='tri':
        def walk(p,a=0,b=0):
            if p<0: return []
            if p==0: return [(a,b)]
            out=[(a,b),(a+p,b),(a,b+p)]
            out += [(a+i,b) for i in range(1,p)]
            out += [(a+p-i,b+i) for i in range(1,p)]
            out += [(a,b+p-i) for i in range(1,p)]
            return out+walk(p-3,a+1,b+1)
        points=walk(order)
    else:
        raise ValueError('kind must be tri or quad')
    return torch.tensor(points,dtype=torch.float64)/order


class Basis:
    def __init__(self, kind, p, nodes=None, device='cpu'):
        if kind not in ('tri','quad') or not isinstance(p,int) or not 0<=p<=5:
            raise ValueError('tri/quad and polynomial order 0..5 required')
        self.kind,self.p=kind,p
        self.powers=[(i,j) for j in range(p+1) for i in range(p+1) if kind=='quad' or i+j<=p]
        if nodes is None:
            if p==0:
                nodes=torch.tensor([[1/3,1/3] if kind=='tri' else [.5,.5]],dtype=torch.float64)
            elif kind=='tri':
                nodes=torch.tensor([(i/p,j/p) for j in range(p+1) for i in range(p+1-j)],dtype=torch.float64)
            else:
                z=operators(p,'cpu')[0]/2+.5
                nodes=torch.cartesian_prod(z,z)
        self.nodes=nodes.to(device=device,dtype=torch.float64)
        self.inverse=torch.linalg.inv(self.monomials(self.nodes))

    def monomials(self,points,derivative=None):
        x,y=points.unbind(-1)
        columns=[]
        for i,j in self.powers:
            if derivative==0:
                value=torch.zeros_like(x) if i==0 else i*x**(i-1)*y**j
            elif derivative==1:
                value=torch.zeros_like(x) if j==0 else j*x**i*y**(j-1)
            else:
                value=x**i*y**j
            columns.append(value)
        return torch.stack(columns,-1)

    def evaluate(self,points,derivative=None):
        return self.monomials(points,derivative)@self.inverse


def quadrature(kind,n,device='cpu'):
    z,w=operators(n-1,device)[:2]
    z,w=z/2+.5,w/2
    a,b=torch.meshgrid(z,z,indexing='ij')
    weights=w[:,None]*w[None,:]
    if kind=='tri':
        points=torch.stack((a,(1-a)*b),-1)
        weights=weights*(1-a)
    else:
        points=torch.stack((a,b),-1)
    return points.reshape(-1,2),weights.flatten()


def edge_points(kind,s):
    z=torch.zeros_like(s)
    o=torch.ones_like(s)
    if kind=='tri':
        points=(torch.stack((s,z),-1),torch.stack((1-s,s),-1),torch.stack((z,1-s),-1))
        tangents=((1.,0.),(-1.,1.),(0.,-1.))
    else:
        points=(torch.stack((s,z),-1),torch.stack((o,s),-1),torch.stack((1-s,o),-1),torch.stack((z,1-s),-1))
        tangents=((1.,0.),(0.,1.),(-1.,0.),(0.,-1.))
    return torch.stack(points),torch.tensor(tangents,dtype=s.dtype,device=s.device)
