import math
import unittest
import torch
from case.isentropic_vortex import IsentropicVortex, IsentropicVortexCase


torch.set_num_threads(1)


class IsentropicVortexVerification(unittest.TestCase):
    def test_center_entropy_and_far_field(self):
        for gamma in (1.4,1.67):
            vortex=IsentropicVortex(gamma=gamma)
            x=torch.tensor([0.,.7,20.],dtype=torch.float64)
            y=torch.tensor([0.,-.4,20.],dtype=torch.float64)
            rho,vel,p=vortex.fields(x,y)
            T0=1-(gamma-1)*25*math.e/(8*gamma*math.pi**2)
            self.assertAlmostEqual(rho[0].item(),T0**(1/(gamma-1)),places=13)
            torch.testing.assert_close(p/rho**gamma,torch.ones_like(p))
            torch.testing.assert_close(vel[0],torch.ones(2,dtype=torch.float64))
            self.assertEqual(rho[-1].item(),1.)
            self.assertEqual(p[-1].item(),1.)

    def test_unsteady_continuous_euler_residual(self):
        # Independent x,y,t finite differences of the translating exact field.
        vortex=IsentropicVortex(velocity=(.7,-.4),center=(.2,.3))
        x,y=torch.meshgrid(torch.linspace(-1.3,1.7,7,dtype=torch.float64),
                           torch.linspace(-1.1,1.9,7,dtype=torch.float64),indexing='ij')
        t,eps=.37,1e-5
        eq=vortex.equation
        exact=vortex.exact
        div=(exact(x,y,t+eps)-exact(x,y,t-eps))/(2*eps)
        div+=(eq.flux(exact(x+eps,y,t),0)-eq.flux(exact(x-eps,y,t),0))/(2*eps)
        div+=(eq.flux(exact(x,y+eps,t),1)-eq.flux(exact(x,y-eps,t),1))/(2*eps)
        self.assertLess(div.abs().max().item(),2e-8)

    def test_periodic_translation_and_return(self):
        vortex=IsentropicVortex(center=(9.95,9.95))
        x=torch.tensor([-10.,-9.9,0.,9.9,10.],dtype=torch.float64)
        y=x.clone()
        periods=(20.,20.)
        torch.testing.assert_close(vortex.exact(x,y,0.,periods),vortex.exact(x,y,20.,periods),atol=1e-13,rtol=0)
        torch.testing.assert_close(vortex.exact(x,y,.1,periods),vortex.exact(x+20,y-20,.1,periods),atol=1e-13,rtol=0)
        # Galilean translation, including crossing the periodic seam.
        torch.testing.assert_close(vortex.exact(x,y,.1,periods),vortex.exact(x-.1,y-.1,0.,periods),atol=1e-13,rtol=0)

    def test_periodic_conservation_and_moving_reference(self):
        case=IsentropicVortexCase(n=8,vortex=IsentropicVortex(center=(9.95,9.95)))
        U,stats=case.run(.1)
        self.assertTrue(case.equation.valid(U))
        self.assertLess(max(abs(v) for v in stats['conservation_change']),1e-10)
        self.assertLess(stats['l2'][0],.003)
        self.assertAlmostEqual(stats['center'][0],-9.95)
        torch.testing.assert_close(case.exact,case.exact_at(.1))
        self.assertGreater(case.norm(case.exact-case.initial)[0].item(),1e-3)
        # A repeated run starts from the stored initial state, not the last reference.
        _,zero=case.run(0.)
        self.assertEqual(zero['l2'],[0.,0.,0.,0.])

    def test_spatial_convergence(self):
        errors=[]
        for n in (8,16):
            case=IsentropicVortexCase(n=n,p=3)
            _,stats=case.run(.1)
            errors.append(stats['l2'][0])
        self.assertGreater(errors[0]/errors[1],10.)

    def test_zero_strength_is_uniform(self):
        case=IsentropicVortexCase(n=2,vortex=IsentropicVortex(beta=0.))
        U,stats=case.run(.05)
        torch.testing.assert_close(U,case.initial,atol=1e-13,rtol=0)
        self.assertLess(stats['l2'][0],1e-13)

    def test_invalid_parameters(self):
        for kwargs in ({'beta':20.},{'gamma':1.},{'velocity':(1.,)},
                       {'center':(0.,float('nan'))},{'beta':float('inf')}):
            with self.assertRaises(ValueError):
                IsentropicVortex(**kwargs)
        with self.assertRaises(ValueError):
            IsentropicVortexCase(length=0.)


if __name__=='__main__':
    unittest.main()
