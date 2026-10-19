import math
import unittest
import torch
from fr import make_equation,Mesh2D,MeshFR2D,normal_central
from case.cylinder_exact import CylinderSeries,cylindrical_functions
from case.mie_cylinder import CylinderScattering


class CylinderScatteringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_normal_riemann_reduces_to_characteristic_upwind(self):
        equation=make_equation('maxwell-tmz')
        torch.manual_seed(5)
        m,p=torch.randn(12,3,dtype=torch.float64),torch.randn(12,3,dtype=torch.float64)
        angle=torch.linspace(0.,2*math.pi,12,dtype=torch.float64)
        n=torch.stack((angle.cos(),angle.sin()),-1)
        star=equation.interface_state(m,p,n,1.,1.)
        flux=lambda U:sum(n[...,a,None]*equation.flux(U,a) for a in (0,1))
        jump=p-m
        transverse=jump.clone()
        transverse[...,1:]-=(jump[...,1:]*n).sum(-1)[...,None]*n
        torch.testing.assert_close(flux(star),.5*(flux(m)+flux(p)-transverse),atol=1e-13,rtol=1e-13)
        incoming=torch.tensor([1.,0.,-1.],dtype=torch.float64)
        state=equation.interface_state(incoming,torch.zeros_like(incoming),torch.tensor([1.,0.]),1.,4.)
        torch.testing.assert_close(state,torch.tensor([2/3,0.,-4/3],dtype=torch.float64))

    def test_analytic_transparency_and_interface_conditions(self):
        transparent=CylinderSeries(epsilon_r=1.)
        self.assertLess(transparent.coefficients.abs().max().item(),1e-14)
        series=CylinderSeries()
        x=torch.tensor(series.k*series.radius,dtype=torch.float64)
        J,H,dJ,dH=cylindrical_functions(x,series.terms)
        A,_,dA,_=cylindrical_functions(torch.tensor(series.k_inside*series.radius,dtype=torch.float64),series.terms)
        a,b=series.coefficients,series.inside_coefficients
        torch.testing.assert_close(J+a*H,b*A,atol=1e-10,rtol=1e-10)
        torch.testing.assert_close(series.k*(dJ+a*dH),series.k_inside*b*dA,atol=1e-9,rtol=1e-9)
        theta=torch.linspace(-math.pi,math.pi,4001,dtype=torch.float64)
        self.assertAlmostEqual(torch.trapezoid(series.angular_width(theta),theta).item(),series.total_width(),places=8)

    def test_no_contrast_produces_zero_scattering(self):
        mesh=Mesh2D.rectangle(2,bounds=((-2.5,2.5),)*2)
        mesh.blocks[0].regions[:]=2
        case=CylinderScattering(mesh,epsilon_r=1.)
        torch.testing.assert_close(case.rhs(case.initial,5.),torch.zeros_like(case.initial),atol=0.,rtol=0.)

    def test_curved_maxwell_pec_energy_rate(self):
        mesh=Mesh2D.rectangle(2,kind='mixed',jitter=.15).mapped(lambda x,y:(x+.03*torch.sin(math.pi*x)*torch.sin(math.pi*y),y),3)
        eq=make_equation('maxwell-tmz')
        solver=MeshFR2D(mesh,flux=eq.flux,numerical_flux=normal_central,boundary=eq.normal_pec_boundary)
        torch.manual_seed(12)
        U=torch.randn(solver.ndof,3,dtype=torch.float64)
        rhs=solver.rhs(U)
        rate=0.
        for b in solver.blocks:
            V=U[b['offset']].reshape(b['ne'],b['ns'],3)
            D=rhs[b['offset']].reshape_as(V)
            rate+=(V*(b['mass']@D)).sum().item()
        self.assertLess(abs(rate),1e-9)


if __name__=='__main__': unittest.main()
