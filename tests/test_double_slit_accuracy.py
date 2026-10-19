"""Independent reference, diagonalized LDG and detector extraction checks."""
import unittest
import torch
from case.double_slit_accuracy import ScreenEvolution, basis, interpolate_y, fringe_metrics
from case.schrodinger import SchrodingerCase
from fr import integrate


class DoubleSlitAccuracyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fd_basis_and_boundary_eigenvalues(self):
        x,w,e,Q=basis('fd',32,24.,-12.)
        torch.testing.assert_close(Q.T@Q,torch.eye(32,dtype=Q.dtype),atol=2e-14,rtol=0.)
        h=24/32
        L=torch.diag(torch.full((32,),-2/h**2,dtype=Q.dtype))
        L+=torch.diag(torch.full((31,),1/h**2,dtype=Q.dtype),1)
        L+=torch.diag(torch.full((31,),1/h**2,dtype=Q.dtype),-1)
        L[0,0]-=1/h**2;L[-1,-1]-=1/h**2
        torch.testing.assert_close(-.5*L@Q,Q*e,atol=2e-14,rtol=2e-14)

    def test_screen_kick_matches_physical_potential(self):
        m=ScreenEvolution('fr',32)
        torch.manual_seed(5)
        z=torch.randn(m.initial.shape,dtype=torch.complex128)
        step=.023
        result=m.physical(m.kick(z,complex(torch.exp(torch.tensor(-30j*step,dtype=torch.complex128)))-1))
        mask=((m.x>=0)&(m.x<.75))[:,None]&~(((m.y>1)&(m.y<2))|((m.y>-2)&(m.y<-1)))[None,:]
        exact=m.physical(z)*torch.exp(-30j*step*mask.to(torch.float64))
        torch.testing.assert_close(result,exact,atol=1e-11,rtol=1e-12)

    def test_split_evolution_matches_production_rk4(self):
        c=SchrodingerCase('double-slit',n=32)
        m=ScreenEvolution('fr',32)
        z,_=m.evolve(.01,.00025)
        psi=m.physical(z)
        U,_=integrate(c.equation.rhs,c.initial,.01,dt=c.timestep()/8,method='rk4')
        actual=torch.complex(U[...,0],U[...,1]).permute(0,2,1,3).reshape(128,128)
        self.assertLess(float((psi-actual).abs().max()),2e-8)

    def test_detector_interpolates_amplitude(self):
        samples=torch.linspace(-5,5,100,dtype=torch.float64)
        for kind in ('fr','fd'):
            x,w,k,Q=basis(kind,32,16.,-8.)
            values=(1+x+2j*x*x)[None,:]
            interpolated=interpolate_y(values,x,samples,kind,32)
            torch.testing.assert_close(interpolated,(1+samples+2j*samples*samples)[None,:],atol=2e-13,rtol=0.)

    def test_known_fringe_position_and_visibility(self):
        y=torch.linspace(-5,5,2001,dtype=torch.float64)
        f=2+torch.cos(2*torch.pi*y/2.5)
        result=fringe_metrics(y,f)
        self.assertAlmostEqual(result['peaks']['right']['position'],2.5,places=10)
        self.assertAlmostEqual(result['peaks']['left']['position'],-2.5,places=10)
        self.assertAlmostEqual(result['contrast']['right'],.5,places=10)


if __name__=='__main__':unittest.main()
