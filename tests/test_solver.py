import unittest
import torch
from case.ringleb import RinglebCase, ringleb
from fr.equations import flux
from fr.operators import operators

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


class Verification(unittest.TestCase):
    def test_polynomial_derivative_and_radau(self):
        for p in range(5):
            z,w,D,L,R,gl,gr = operators(p,'cpu')
            for k in range(p+1):
                exact = k*z**(k-1) if k else torch.zeros_like(z)
                torch.testing.assert_close(D@z**k,exact,atol=2e-13,rtol=2e-13)
            self.assertAlmostEqual((w*gl).sum().item(),-1,places=13)
            self.assertAlmostEqual((w*gr).sum().item(),1,places=13)

    def test_exact_euler_pde(self):
        # Independent Cartesian finite difference, no discrete FR operator.
        x,y = torch.meshgrid(torch.linspace(-.5,0,9),torch.linspace(1.5,2,9),indexing='ij')
        eps = 1e-5
        div = (flux(ringleb(x+eps,y),0)[0]-flux(ringleb(x-eps,y),0)[0])/(2*eps)
        div += (flux(ringleb(x,y+eps),1)[0]-flux(ringleb(x,y-eps),1)[0])/(2*eps)
        self.assertLess(div.abs().max().item(),2e-8)

    def test_global_conservation(self):
        s = RinglebCase(3,3)
        U = s.exact * (1+.001*torch.sin(s.x)[...,None])
        rhs = s.rhs(U)
        vol = (rhs*(s.w[:,None]*s.w[None,:])[None,None,:,:,None]).sum((0,1,2,3))*s.hx*s.hy/4
        # Independently reconstruct the outward net numerical boundary flux.
        from fr.equations import rusanov
        boundary = torch.zeros(4)
        for axis,h in ((0,s.hy),(1,s.hx)):
            A = U if axis == 0 else U.permute(1,0,3,2,4)
            l = torch.einsum('i,abijv->abjv',s.L,A)[0]
            r = torch.einsum('i,abijv->abjv',s.R,A)[-1]
            bl,br = s.bc[axis]
            net = rusanov(r,br,axis)-rusanov(bl,l,axis)
            boundary += (net*s.w[None,:,None]).sum((0,1))*h/2
        torch.testing.assert_close(vol,-boundary,atol=1e-13,rtol=1e-11)

    def test_third_order_ringleb(self):
        errors=[]
        for n in (4,8):
            s=RinglebCase(n,2)
            _,stats=s.run(.05)
            errors.append(stats['l2'][0])
        self.assertGreater(errors[0]/errors[1],6.)


if __name__ == '__main__':
    unittest.main()
