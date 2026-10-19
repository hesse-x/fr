"""LDG identities, diffusion consistency and Schrodinger regression tests."""
import math
import unittest
import torch
from fr import CartesianFR, CartesianLDG, central, integrate
from case.schrodinger import SchrodingerCase, NAMES


def grid(dim, periodic=True, n=3, p=3):
    return CartesianFR((n,)*dim, ((0.,1.),)*dim, p, flux=lambda U,a: U,
                       numerical_flux=central, periodic=periodic,
                       boundary=lambda c,U,a,s,t: U)


class SecondOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_adjoint_and_diffusion_energy(self):
        torch.manual_seed(17)
        for dim in (1,2,3):
            for periodic in (True,False):
                s = grid(dim,periodic,n=2)
                op = CartesianLDG(s,lambda c,U,a,side,t: torch.zeros_like(U))
                U = torch.randn(*s.shape,2,dtype=torch.float64)
                Q = torch.randn(*s.shape,2,dim,dtype=torch.float64)
                G = op.gradient(U)
                lhs = s.integral(U*op.divergence(Q))
                rhs = s.integral((G*Q).sum(-1))
                torch.testing.assert_close(lhs,-rhs,atol=3e-12,rtol=3e-12)
                torch.testing.assert_close(s.integral(U*op.diffusion(U)),
                                           -s.integral(G.square().sum(-1)),atol=3e-11,rtol=3e-12)

    def test_polynomial_nonzero_boundary(self):
        s = grid(2,False)
        boundary = lambda coords,U,a,side,t: (coords[0]**2+3*coords[1]**2+t)[...,None]
        op = CartesianLDG(s,boundary)
        U = (s.coords[0]**2+3*s.coords[1]**2+.7)[...,None]
        torch.testing.assert_close(op.diffusion(U,.7),torch.full_like(U,8.),atol=3e-11,rtol=0.)

    def test_coupled_constitutive_and_heat_decay(self):
        s = grid(1,True,n=8)
        op = CartesianLDG(s)
        U = torch.sin(2*math.pi*s.coords[0])[...,None]
        coupled = op.diffusion(U,constitutive=lambda c,U,G,t: 2*G)
        torch.testing.assert_close(coupled,2*op.diffusion(U))
        result,_ = integrate(lambda U,t: op.diffusion(U),U,.001,dt=1e-5,method='rk4')
        self.assertLess(float(s.norm(result-U*math.exp(-4*math.pi**2*.001))[0]),.002)
        self.assertLess(float(s.integral(result.square())[0]),float(s.integral(U.square())[0]))

    def test_coupled_tensor_flux(self):
        s = grid(2,False)
        def values(coords):
            x,y = coords
            return torch.stack((x*x+y*y,x*y),-1)
        op = CartesianLDG(s,lambda c,U,a,side,t: values(c))
        def constitutive(coords,U,G,t):
            F = torch.zeros_like(G)
            F[...,0,0] = G[...,0,0]+2*G[...,1,1]
            F[...,0,1] = G[...,0,1]
            F[...,1,0] = 3*G[...,0,0]
            F[...,1,1] = 4*G[...,1,1]
            return F
        result = op.diffusion(values(s.coords),constitutive=constitutive)
        torch.testing.assert_close(result,torch.full_like(result,6.),atol=5e-11,rtol=0.)

    def test_periodic_complex_plane_wave_convergence(self):
        errors = []
        for n in (4,8,16):
            s = grid(1,True,n=n)
            op = CartesianLDG(s)
            U = torch.exp(2j*math.pi*s.coords[0])[...,None]
            err = op.diffusion(U)+4*math.pi**2*U
            errors.append(float(s.integral(err.abs().square())[0].sqrt()))
        self.assertGreater(math.log2(errors[1]/errors[2]),1.8)

    def test_schrodinger_semidiscrete_conservation(self):
        for name in NAMES:
            case = SchrodingerCase(name,n=4,p=2)
            U = case.initial
            rhs = case.equation.rhs(U)
            mass_rate = case.grid.integral((2*U*rhs).sum(-1,keepdim=True))[0]
            self.assertLess(abs(float(mass_rate)),1e-11)

    def test_analytic_cases_converge(self):
        for name in ('gaussian','harmonic','well'):
            errors=[]
            for n in ((24,48) if name!='well' else (4,8)):
                case = SchrodingerCase(name,n=n)
                _,stats = case.run(.02,plot=False)
                errors.append(stats['complex_l2_error'])
                self.assertLess(abs(stats['probability_change']),1e-6)
            self.assertLess(errors[1],errors[0]/5,name)

    def test_all_cases_smoke(self):
        for name in NAMES:
            case = SchrodingerCase(name,n=4,p=2)
            U,stats = case.run(.001,plot=False)
            self.assertTrue(bool(torch.isfinite(U).all()))
            if name in ('barrier','double-slit'):
                total = sum(stats[k] for k in ('left_probability','screen_probability','right_probability'))
                self.assertAlmostEqual(total,stats['probability_final']/stats['probability_initial'],places=12)


if __name__ == '__main__':
    unittest.main()
