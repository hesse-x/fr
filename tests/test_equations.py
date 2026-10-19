import math
import unittest
import torch
from fr import CartesianFR, integrate, make_equation, make_numerical_flux
from fr.equations import Euler, LinearizedEuler, Maxwell, Acoustics, ConvectedAcoustics


class SharedEquationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_factory_and_compatibility(self):
        from fr.maxwell import Maxwell as OldMaxwell
        from fr.acoustics import Acoustics as OldAcoustics
        self.assertIs(OldMaxwell,Maxwell)
        self.assertIs(OldAcoustics,Acoustics)
        for name,args,nvars in [('euler',{'dim':2},4),('linearized-euler',{},4),('maxwell',{},6)]:
            eq=make_equation(name,**args)
            self.assertEqual(eq.nvars,nvars)
        with self.assertRaises(ValueError): make_equation('unknown')
        callback=lambda m,p,a:m.flux
        self.assertIs(make_numerical_flux(callback,Euler(2)),callback)
        with self.assertRaises(ValueError): make_numerical_flux('roe',Euler(2))

    def test_linearized_euler_is_nonlinear_flux_derivative(self):
        # Constant primitive-to-conservative Jacobian maps the two systems.
        for dim in (1,2,3):
            mean=torch.tensor([.3,-.2,.1][:dim],dtype=torch.float64)
            rho0,p0,gamma=1.2,.8,1.4
            lin=LinearizedEuler(dim,mean=mean.tolist(),rho0=rho0,p0=p0,gamma=gamma)
            nonlinear=Euler(dim,gamma)
            torch.manual_seed(dim)
            W=torch.randn(7,dim+2,dtype=torch.float64)
            def conservative(rho,vel,pressure):
                return torch.cat((rho[...,None],rho[...,None]*vel,
                                  (pressure/(gamma-1)+.5*rho*vel.square().sum(-1))[...,None]),-1)
            def transform(delta):
                return torch.cat((delta[...,:1],rho0*delta[...,1:-1]+delta[...,:1]*mean,
                                  (delta[...,-1]/(gamma-1)+.5*mean.square().sum()*delta[...,0]
                                   +rho0*(mean*delta[...,1:-1]).sum(-1))[...,None]),-1)
            eps=1e-6
            plus=conservative(rho0+eps*W[...,0],mean+eps*W[...,1:-1],p0+eps*W[...,-1])
            minus=conservative(rho0-eps*W[...,0],mean-eps*W[...,1:-1],p0-eps*W[...,-1])
            for axis in range(dim):
                derivative=(nonlinear.flux(plus,axis)-nonlinear.flux(minus,axis))/(2*eps)
                torch.testing.assert_close(transform(lin.flux(W,axis)),derivative,atol=2e-9,rtol=2e-9)

    def test_acoustic_subspace_and_entropy_mode(self):
        for dim in (1,2,3):
            mean=(.3,-.2,.1)[:dim]
            acoustic=ConvectedAcoustics(mean,rho=1.2,c=1.4)
            lin=LinearizedEuler(dim,mean=mean,rho0=1.2,p0=1.2*1.4**2/1.4)
            U=torch.randn(9,dim+1,dtype=torch.float64)
            embedded=lin.from_acoustic(U)
            torch.testing.assert_close(lin.acoustic_state(embedded),U)
            entropy=torch.zeros(dim+2,dtype=torch.float64)
            entropy[0]=2.
            for axis in range(dim):
                torch.testing.assert_close(lin.flux(embedded,axis),lin.from_acoustic(acoustic.flux(U,axis)))
                torch.testing.assert_close(lin.absolute_jacobian_action(embedded,None,None,axis),
                                           lin.from_acoustic(acoustic.absolute_jacobian_action(U,None,None,axis)))
                torch.testing.assert_close(lin.flux(entropy,axis),mean[axis]*entropy)
                torch.testing.assert_close(lin.absolute_jacobian_action(entropy,None,None,axis),abs(mean[axis])*entropy)

    def test_entropy_wave_through_common_solver(self):
        eq=make_equation('linearized-euler',dim=2,mean=(.3,-.2))
        mesh=CartesianFR((4,4),((0.,1.),)*2,p=3,flux=eq.flux,
                         numerical_flux=make_numerical_flux('roe',eq),speed=eq.speed,periodic=True)
        x,y=mesh.coords
        U=torch.zeros(*mesh.shape,eq.nvars,dtype=torch.float64)
        U[...,0]=torch.sin(2*math.pi*(x+y))
        V,_=integrate(mesh.rhs,U,.1,dt=mesh.timestep(U),validate=eq.valid)
        exact=U.clone()
        exact[...,0]=torch.sin(2*math.pi*(x+y-.01))
        self.assertLess(mesh.norm(V-exact).max().item(),.001)
        self.assertEqual(V[...,1:].abs().max().item(),0.)


if __name__ == '__main__': unittest.main()
